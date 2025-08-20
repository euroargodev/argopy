import fsspec
from typing import List, Union, Dict, Literal
from pathlib import Path
import json
import logging
from packaging import version

from ..utils import to_list
from . import memorystore, filestore

log = logging.getLogger("argopy.stores.kerchunk")


try:
    from kerchunk.hdf import SingleHdf5ToZarr
    from kerchunk.netCDF3 import NetCDF3ToZarr

    HAS_KERCHUNK = True
except ModuleNotFoundError:
    HAS_KERCHUNK = False
    SingleHdf5ToZarr, NetCDF3ToZarr = None, None

try:
    import dask

    HAS_DASK = True
except ModuleNotFoundError:
    HAS_DASK = False
    dask = None
    import concurrent.futures


class ArgoKerchunker:
    """
    Argo netcdf file kerchunk helper

    This class is for expert users who wish to test lazy access to remote netcdf files.
    It is designed to be used through one of the **argopy** stores inheriting from :class:`ArgoStoreProto`.

    The `kerchunk <https://fsspec.github.io/kerchunk/>`_ library is required only if you
    need to extract zarr data from a netcdf file, i.e. execute :meth:`ArgoKerchunker.translate`.

    Notes
    -----
    According to `AWS <https://docs.aws.amazon.com/whitepapers/latest/s3-optimizing-performance-best-practices/use-byte-range-fetches.html>`_,
    typical sizes for byte-range requests are 8 MB or 16 MB.

    If you intend to compute kerchunk zarr data on-demand, we don't recommend to use this method on mono or multi
    profile files that are only a few MB in size, because (ker)-chunking creates a significant performance overhead.

    Warnings
    --------
    We noticed that kerchunk zarr data for Rtraj files can be insanely larger than the netcdf file itself.
    This could go from 10Mb to 228Mb !

    Examples
    --------
    .. code-block:: python
        :caption: :class:`ArgoKerchunker` API

        # Use default memory store to manage kerchunk zarr data:
        ak = ArgoKerchunker(store='memory')

        # Use a local file store to keep track of zarr kerchunk data (for later
        # reuse or sharing):
        ak = ArgoKerchunker(store='local', root='kerchunk_data_folder')

        # Use a remote file store to keep track of zarr kerchunk data (for later
        # reuse or sharing):
        fs = fsspec.filesystem('dir',
                               path='s3://.../kerchunk_data_folder/',
                               target_protocol='s3')
        ak = ArgoKerchunker(store=fs)

        # Methods:
        ak.supported(ncfile)
        ak.translate(ncfiles)
        ak.to_reference(ncfile)
        ak.pprint(ncfile)

    .. code-block:: python
        :caption: Loading one file lazily

        # Let's consider a remote Argo netcdf file from a s3 server supporting lazy access
        # (i.e. support byte range requests):
        ncfile = "argo-gdac-sandbox/pub/dac/coriolis/6903090/6903090_prof.nc"

        # Simply open the netcdf file lazily:
        from argopy.stores import s3store
        ds = s3store().open_dataset(ncfile, lazy=True)

        # You can also do it with the GDAC fs:
        from argopy.stores import gdacfs
        ds = gdacfs('s3').open_dataset("dac/coriolis/6903090/6903090_prof.nc", lazy=True)

    .. code-block:: python
        :caption: Translate and save references for a batch of netcdf files

        # Create an instance that will save netcdf to zarr references on a local
        # folder at "~/kerchunk_data_folder":
        ak = ArgoKerchunker(store='local', root='~/kerchunk_data_folder')

        # Get a dummy list of netcdf files:
        from argopy import ArgoIndex
        idx = ArgoIndex(host='s3').search_lat_lon_tim([-70, -55, 30, 45,
                                                       '2025-01-01', '2025-02-01'])
        ncfiles = [af.ls_dataset()['prof'] for af in idx.iterfloats()]

        # Translate and save references for this batch of netcdf files:
        # (done in parallel, possibly using a Dask client if available)
        ak.translate(ncfiles, fs=idx.fs['src'], chunker='auto')

    """

    def __init__(
        self,
        store: Union[Literal["memory", "local"], fsspec.AbstractFileSystem] = "memory",
        root: Union[Path, str] = ".",
        preload: bool = True,
        inline_threshold: int = 0,
        max_chunk_size: int = 0,
        storage_options: Dict = None,
    ):
        """

        Parameters
        ----------
        store: str, default='memory'
            Kerchunk data store, i.e. the file system used to load from and/or save to kerchunk json files
        root: Path, str, default='.'
            Use to specify a local folder to base the store
        preload: bool, default=True
            Indicate if kerchunk references already on the store should be preloaded or not.
        inline_threshold: int, default=0
            Byte size below which an array will be embedded in the output. Use 0 to disable inlining.

            This argument is passed to :class:`kerchunk.netCDF3.NetCDF3ToZarr` or :class:`kerchunk.hdf.SingleHdf5ToZarr`
        max_chunk_size: int, default=0
            How big a chunk can be before triggering subchunking. If 0, there is no
            subchunking, and there is never subchunking for coordinate/dimension arrays.
            E.g., if an array contains 10,000bytes, and this value is 6000, there will
            be two output chunks, split on the biggest available dimension.

            This argument is passed to :class:`kerchunk.netCDF3.NetCDF3ToZarr` only.
        storage_options: dict, default=None
            This argument is passed to :class:`kerchunk.netCDF3.NetCDF3ToZarr` or :class:`kerchunk.hdf.SingleHdf5ToZarr`
            during translation. These in turn, will pass options to fsspec when opening netcdf file.

        """
        # Instance file system to load/save kerchunk json files
        if store == "memory":
            self.fs = memorystore()
        elif store == "local":
            root = Path(root).expanduser()
            if root.name == "":
                self.fs = filestore()
            else:
                root.mkdir(parents=True, exist_ok=True)
                self.fs = fsspec.filesystem("dir", path=root, target_protocol="local")

        elif isinstance(store, fsspec.AbstractFileSystem):
            self.fs = store

        # Passed to fsspec when opening netcdf file:
        self.storage_options = storage_options if storage_options is not None else {}

        # List of processed files register:
        self.kerchunk_references = {}
        if preload:
            self.update_kerchunk_references_from_store()

        self.inline_threshold = inline_threshold
        """inline_threshold: int
        Byte size below which an array will be embedded in the output. Use 0 to disable inlining.
        """

        self.max_chunk_size = max_chunk_size
        """
        max_chunk_size: int
        How big a chunk can be before triggering subchunking. If 0, there is no
        subchunking, and there is never subchunking for coordinate/dimension arrays.
        E.g., if an array contains 10,000bytes, and this value is 6000, there will
        be two output chunks, split on the biggest available dimension. [TBC]
        """

    def __repr__(self):
        summary = ["<argopy.kerchunker>"]
        summary.append("- kerchunk data store: %s" % str(self.fs))
        summary.append(
            "- Inline threshold: %i (byte size below which an array will be embedded in the output)"
            % self.inline_threshold
        )
        summary.append(
            "- Maximum chunk size: %i (how big a chunk can be before triggering sub-chunking)"
            % self.max_chunk_size
        )
        n = len(self.kerchunk_references)
        summary.append("- %i dataset%s listed in store" % (n, "s" if n > 1 else ""))
        return "\n".join(summary)

    @property
    def store_path(self):
        """Absolute path to the reference store, including protocol"""
        p = getattr(self.fs, "path", str(Path(".").absolute()))
        # Ensure the protocol is included for non-local files:
        if self.fs.fs.protocol[0] == "s3":
            p = "s3://" + fsspec.core.split_protocol(p)[-1]
        return p

    def _ncfile2jsfile(self, ncfile):
        """Convert a netcdf file path to a data store file path for kerchunk zarr reference (json data)"""
        return Path(ncfile).name.replace(".nc", "_kerchunk.json")

    def _ncfile2ncref(self, ncfile: Union[str, Path], fs=None):
        """Convert a netcdf file path to a key used in internal kerchunk_references"""
        # return fs.full_path(fs.info(str(ncfile))['name'], protocol=True)
        return fs.full_path(str(ncfile), protocol=True)

    def _magic2chunker(self, ncfile, fs):
        """Get a netcdf file path chunker alias: 'cdf3' or 'hdf5'

        This is based on the file binary magic value.

        Raises
        ------
        :class:`ValueError` if file not recognized
        """
        magic = fs.open(ncfile).read(3)
        if magic == b"CDF":
            return "cdf3"
        elif magic == b"\x89HD":
            return "hdf5"
        else:
            raise ValueError("No chunker for this magic: '%s')" % magic)

    def nc2reference(
        self,
        ncfile: Union[str, Path],
        fs=None,
        chunker: Literal["auto", "cdf3", "hdf5"] = "auto",
    ):
        """Compute reference data for a netcdf file (kerchunk json data)

        This method is intended to be used internally, since it's not using the kerchunk reference store.

        Users should rather use the :meth:`to_reference` method to avoid to recompute reference data
        when available on the :class:`ArgoKerchunker` instance.

        Parameters
        ----------
        ncfile : Union[str, Path]
            Path to a netcdf file to process
        fs: None
            An **argopy** file store, inheriting from :class:`ArgoStoreProto`.
        chunker : Literal['auto', 'cdf3', 'hdf5'] = 'auto'
            Define the kerchunker formatter to use. Two formatter are available: :class:`kerchunk.netCDF3.NetCDF3ToZarr` or :class:`kerchunk.hdf.SingleHdf5ToZarr`:

            - 'auto': detect and select formatter for each netcdf of the ncfiles
            - 'cdf3': impose use of :class:`kerchunk.netCDF3.NetCDF3ToZarr`
            - 'hdf5': impose use of :class:`kerchunk.hdf.SingleHdf5ToZarr`

        Returns
        -------
        dict
        """
        chunker = self._magic2chunker(ncfile, fs) if chunker == "auto" else chunker
        ncfile_full = self._ncfile2ncref(ncfile, fs=fs)

        storage_options = self.storage_options.copy()
        if fs.protocol == 'ftp' and version.parse(fsspec.__version__) < version.parse("2024.10.0"):
            # We need https://github.com/fsspec/filesystem_spec/pull/1673
            storage_options.pop('host', None)
            storage_options.pop('port', None)

        if chunker == "cdf3":
            chunks = NetCDF3ToZarr(
                ncfile_full,
                inline_threshold=self.inline_threshold,
                max_chunk_size=self.max_chunk_size,
                storage_options=storage_options,
            )
        elif chunker == "hdf5":
            chunks = SingleHdf5ToZarr(
                ncfile_full,
                inline_threshold=self.inline_threshold,
                storage_options=storage_options,
            )

        kerchunk_data = chunks.translate()

        kerchunk_jsfile = self._ncfile2jsfile(ncfile)

        with self.fs.open(kerchunk_jsfile, "wb") as f:
            f.write(json.dumps(kerchunk_data).encode())

        return ncfile_full, kerchunk_jsfile, kerchunk_data

    def update_kerchunk_references_from_store(self):
        """Load kerchunk data already on store"""
        for f in self.fs.glob("*_kerchunk.json"):
            with self.fs.open(f, "r") as file:
                kerchunk_data = json.load(file)
                for k, v in kerchunk_data["refs"].items():
                    if k != ".zgroup" and "/0" in k:
                        if Path(v[0]).suffix == ".nc":
                            self.kerchunk_references.update({v[0]: f})
                            break

    def translate(
        self,
        ncfiles: Union[str, Path, List],
        fs=None,
        chunker: Literal["first", "auto", "cdf3", "hdf5"] = "first",
    ):
        """Translate netcdf file(s) into kerchunk reference data

        Kerchunk data are saved on the :class:`ArgoKerchunker` instance store.

        Once translated, netcdf file reference data are internally registered in the :attr:`ArgoKerchunker.kerchunk_references` attribute.

        If more than 1 netcdf file is provided, the translation is executed in parallel:

        - if `Dask <https://www.dask.org>`_ is available we use :class:`dask.delayed`/:meth:`dask.compute`,
        - otherwise we use a :class:`concurrent.futures.ThreadPoolExecutor`.

        Parameters
        ----------
        ncfiles : Union[str, Path, List]
            One or more netcdf files to translate
        fs: None
            An **argopy** file store, inheriting from :class:`ArgoStoreProto`.
        chunker : Literal['first', 'auto', 'cdf3', 'hdf5'] = 'first'
            Define the kerchunker formatter to use. Two formatter are available: :class:`kerchunk.netCDF3.NetCDF3ToZarr` or :class:`kerchunk.hdf.SingleHdf5ToZarr`:

            - 'first': detect and select formatter from the first netcdf file type
            - 'auto': detect and select formatter for each netcdf of the nc files
            - 'cdf3': impose use of :class:`kerchunk.netCDF3.NetCDF3ToZarr` for all nc files
            - 'hdf5': impose use of :class:`kerchunk.hdf.SingleHdf5ToZarr` for all nc files

        Returns
        -------
        List(dict)

        See Also
        --------
        :meth:`ArgoKerchunker.to_reference`

        """
        if not HAS_KERCHUNK:
            raise ModuleNotFoundError("This method requires the 'kerchunk' library")

        ncfiles = to_list(ncfiles)
        for i, f in enumerate(ncfiles):
            ncfiles[i] = str(f)

        #
        chunker = self._magic2chunker(ncfiles[0], fs) if chunker == "first" else chunker

        #
        if HAS_DASK:
            tasks = [
                dask.delayed(self.nc2reference)(uri, fs=fs, chunker=chunker)
                for uri in ncfiles
            ]
            results = dask.compute(tasks)
            results = results[0]  # ?
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_url = {
                    executor.submit(self.nc2reference, uri, fs=fs, chunker=chunker): uri
                    for uri in ncfiles
                }
                futures = concurrent.futures.as_completed(future_to_url)

                results = []
                for future in futures:
                    results.append(future.result())

        for result in results:
            ncfile, kerchunk_jsfile, _ = result
            self.kerchunk_references.update({ncfile: kerchunk_jsfile})

        return results

    def to_reference(self, ncfile: Union[str, Path], fs=None, overwrite: bool = False):
        """Return zarr reference data for a given netcdf file path

        Return data from the instance store if available, otherwise trigger :meth:`ArgoKerchunker.translate` (which save
        data on the instance data store).

        This is the method to use in **argopy** file store methods :meth:`ArgoStoreProto.open_dataset` to implement laziness.

        Parameters
        ----------
        ncfile : Union[str, Path]
            Path to netcdf file to process
        fs: None
            An **argopy** file store, inheriting from :class:`ArgoStoreProto`.

        Returns
        -------
        dict

        See Also
        --------
        :meth:`ArgoKerchunker.translate`
        """
        if overwrite:
            self.translate(ncfile, fs=fs)
        elif self._ncfile2ncref(ncfile, fs=fs) not in self.kerchunk_references:
            if self.fs.exists(self._ncfile2jsfile(ncfile)):
                self.kerchunk_references.update(
                    {self._ncfile2ncref(ncfile, fs=fs): self._ncfile2jsfile(ncfile)}
                )
            else:
                self.translate(ncfile, fs=fs)

        # Read and load the kerchunk JSON file:
        kerchunk_jsfile = self.kerchunk_references[self._ncfile2ncref(ncfile, fs=fs)]
        with self.fs.open(kerchunk_jsfile, "r") as file:
            kerchunk_data = json.load(file)

        # Ensure that reference data corresponds to the target netcdf file:
        if not overwrite:
            target_ok = False
            for key, value in kerchunk_data["refs"].items():
                if key not in [".zgroup", ".zattrs"] and "0." in key:
                    if value[0] == self._ncfile2ncref(ncfile, fs=fs):
                        target_ok = True
                        break
            if not target_ok:
                kerchunk_data = self.to_reference(ncfile, overwrite=True, fs=fs)

        return kerchunk_data

    def pprint(self, ncfile: Union[str, Path], params: List[str] = None, fs=None):
        """Pretty print kerchunk json data for a netcdf file"""
        params = to_list(params) if params is not None else []
        kerchunk_data = self.to_reference(ncfile, fs=fs)

        # Pretty print JSON data
        keys_to_select = [".zgroup", ".zattrs", ".zmetadata"]
        data_to_print = {}
        for key, value in kerchunk_data["refs"].items():
            if key in keys_to_select:
                if isinstance(value, str):
                    data_to_print[key] = json.loads(value)
                else:
                    data_to_print[key] = value
            for p in params:
                if p == key.split("/")[0]:
                    if isinstance(value, str):
                        data_to_print[key] = json.loads(value)
                    else:
                        data_to_print[key] = value

        print(json.dumps(data_to_print, indent=4))

    def supported(self, ncfile: Union[str, Path], fs=None) -> bool:
        """Check if a netcdf file can be accessed through byte ranges

        For non-local files, the absolute path toward the netcdf file must include the file protocol to return
        a correct answer.

        Known Argo GDAC supporting byte ranges:

        - ftp://ftp.ifremer.fr/ifremer/argo
        - s3://argo-gdac-sandbox/pub
        - https://usgodae.org/pub/outgoing/argo
        - https://argo-gdac-sandbox.s3-eu-west-3.amazonaws.com/pub

        Not supporting:

        - https://data-argo.ifremer.fr

        Parameters
        ----------
        ncfile: str, Path
            Absolute path toward the netcdf file to assess for lazy support, must include protocol for non-local files.
        """
        try:
            return fs.first(ncfile) is not None
        except Exception:
            log.debug(f"Could not read {ncfile} with {fs}")
            return False
