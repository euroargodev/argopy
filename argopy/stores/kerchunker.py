import fsspec
import xarray as xr
from typing import List, Union, Dict, Literal
from pathlib import Path
from urllib.parse import urlparse
from fsspec.core import split_protocol
import json
import logging
import aiohttp

from ..utils import to_list
from . import memorystore, filestore

log = logging.getLogger("argopy.stores.kerchunk")


try:
    from kerchunk.hdf import SingleHdf5ToZarr
    from kerchunk.netCDF3 import NetCDF3ToZarr

    HAS_KERCHUNK = True
except ModuleNotFoundError:
    # log.debug("argopy missing 'kerchunk' to translate netcdf file")
    HAS_KERCHUNK = False
    SingleHdf5ToZarr, NetCDF3ToZarr = None, None

try:
    import dask

    HAS_DASK = True
except ModuleNotFoundError:
    # log.debug("argopy missing 'dask' to improve performances of 'ArgoKerchunker'")
    HAS_DASK = False
    dask = None
    import concurrent.futures


class ArgoKerchunker:
    """
    Argo netcdf file kerchunk helper

    This class is for expert users who wish to test lazy access to remote netcdf files. If you need to compute kerchunk
    zarr data on-demand, we don't recommend to use this method as it shows poor performances on mono or multi profile files.
    It is more efficient to compute kerchunk zarr data in batch, and then to provide these data to users.

    The `kerchunk <https://fsspec.github.io/kerchunk/>`_ library is required only if you start from scratch and
    need to extract zarr data from a netcdf file, i.e. execute :meth:`ArgoKerchunker.translate`.

    .. code-block:: python
        :caption: API

        # Default store to manage zarr kerchunk data
        ak = ArgoKerchunker(store='memory')
        # Custom local storage folder:
        ak = ArgoKerchunker(store='local', root='kerchunk_data_folder')
        # or remote:
        ak = ArgoKerchunker(store=fsspec.filesystem('dir', path='s3://.../kerchunk_data_folder/', target_protocol='s3'))

        # Methods:
        ak.supported(ncfile)
        ak.translate(ncfile)  # takes 1 or a list of uris, requires the kerchunk library
        ak.to_kerchunk(ncfile)  # Take 1 netcdf file uri, return kerchunk json data (translate or load from store)
        ak.pprint(ncfile)

        # Return lazy xarray dataset of a netcdf file, using zarr engine from kerchunk data
        ak.open_dataset(ncfile)

    .. code-block:: python
        :caption: Examples

        # Create an instance that will save netcdf to zarr translation data on a local folder "kerchunk_data_folder" for later re-use (or sharing):
        ak = ArgoKerchunker(store='local', root='kerchunk_data_folder')

        # Let's consider a remote Argo netcdf file from a server supporting lazy access
        # (i.e. support byte range requests):
        ncfile = "s3://argo-gdac-sandbox/pub/dac/coriolis/6903090/6903090_prof.nc"

        # Simply open the netcdf file lazily:
        # (ArgoKerchunker will handle zarr data generation and xarray syntax)
        ds = ak.open_dataset(ncfile)


    """

    def __init__(
        self,
        store: Union[Literal["memory", "local"], fsspec.AbstractFileSystem] = "memory",
        root: Union[Path, str] = ".",
        preload: bool = True,
        inline_threshold: int = 0,
        max_chunk_size: int = 0,
        remote_options: Dict = None,
    ):
        """

        Parameters
        ----------
        store: str, default='memory'
            Kerchunk data store, i.e. the file system to use to load from and/or save to kerchunk json files
        root: Path, str, default='.'
            Use to specify a local folder to base the store
        preload: bool, default=True
            Indicate if kerchunk references already on the store should be loaded or not.
        inline_threshold: int, default=0
            Byte size below which an array will be embedded in the output. Use 0 to disable inlining.

            This argument is passed to :class:`kerchunk.netCDF3.NetCDF3ToZarr` or :class:`kerchunk.hdf.SingleHdf5ToZarr`
        max_chunk_size: int, default=0
            How big a chunk can be before triggering subchunking. If 0, there is no
            subchunking, and there is never subchunking for coordinate/dimension arrays.
            E.g., if an array contains 10,000bytes, and this value is 6000, there will
            be two output chunks, split on the biggest available dimension.

            This argument is passed to :class:`kerchunk.netCDF3.NetCDF3ToZarr`.
        remote_options: dict, default=None
            Options passed to fsspec when opening netcdf file

            This argument is passed to :class:`kerchunk.netCDF3.NetCDF3ToZarr` or :class:`kerchunk.hdf.SingleHdf5ToZarr`
            during translation, and in `backend_kwargs` of :meth:`xarray.open_dataset`.

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
        self.remote_options = remote_options if remote_options is not None else {}

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
        summary.append("- %i dataset%s listed in %s" % (n, "s" if n > 1 else "", self.store_path))
        return "\n".join(summary)

    @property
    def store_path(self):
        p = getattr(self.fs, 'path', str(Path('.').absolute()))
        # Ensure the protocol is included for non-local files:
        if self.fs.fs.protocol[0] == 'ftp':
            p = "ftp://" + self.fs.fs.host + fsspec.core.split_protocol(p)[-1]
        if self.fs.fs.protocol[0] == 's3':
            p = "s3://" + fsspec.core.split_protocol(p)[-1]
        return p

    def _ncfile2jsfile(self, ncfile):
        return Path(ncfile).name.replace(".nc", "_kerchunk.json")

    def _tojson(self, ncfile: Union[str, Path]):
        ncfile = str(ncfile)

        magic = self._magic(ncfile)[0:3]
        if magic == b"CDF":
            chunks = NetCDF3ToZarr(
                ncfile,
                inline_threshold=self.inline_threshold,
                max_chunk_size=self.max_chunk_size,
                storage_options=self.remote_options,
            )
        elif magic == b"\x89HD":
            chunks = SingleHdf5ToZarr(
                ncfile,
                inline_threshold=self.inline_threshold,
                storage_options=self.remote_options,
            )

        kerchunk_data = chunks.translate()

        kerchunk_jsfile = self._ncfile2jsfile(ncfile)

        with self.fs.open(kerchunk_jsfile, "wb") as f:
            f.write(json.dumps(kerchunk_data).encode())

        return ncfile, kerchunk_jsfile, kerchunk_data

    def update_kerchunk_references_from_store(self):
        """Load kerchunk data already on store"""
        for f in self.fs.glob("*_kerchunk.json"):
            with self.fs.open(f, "r") as file:
                kerchunk_data = json.load(file)
                for k, v in kerchunk_data['refs'].items():
                    if k != '.zgroup' and '/0' in k:
                        if Path(v[0]).suffix == ".nc":
                            self.kerchunk_references.update({v[0]: f})
                            break

    def translate(self, ncfiles: Union[str, Path, List]):
        """Compute kerchunk data for one or a list of netcdf files

        Kerchunk data are saved with the instance file store.

        Once translated, netcdf file reference data are internally registered in the :attr:`ArgoKerchunker.kerchunk_references` attribute.

        If more than 1 netcdf file is provided, the translation is executed in parallel:

        - if `Dask <https://www.dask.org>`_ is available we use :class:`dask.delayed`/:meth:`dask.compute`,
        - otherwise we use a :class:`concurrent.futures.ThreadPoolExecutor`.

        See Also
        --------
        :meth:`ArgoKerchunker.to_kerchunk`

        """
        if not HAS_KERCHUNK:
            raise ModuleNotFoundError("This method requires the 'kerchunk' library")

        ncfiles = to_list(ncfiles)
        for i, f in enumerate(ncfiles):
            ncfiles[i] = str(f)

        if HAS_DASK:
            tasks = [dask.delayed(self._tojson)(uri) for uri in ncfiles]
            results = dask.compute(tasks)
            results = results[0]  # ?
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_url = {
                    executor.submit(
                        self._tojson,
                        uri,
                    ): uri
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

    def to_kerchunk(self, ncfile: Union[str, Path], overwrite: bool = False):
        """Return json kerchunk data for a given netcdf file

        If data are found on the instance file store, load it, otherwise triggers :meth:`ArgoKerchunker.translate` and
        save data on instance file store.

        See Also
        --------
        :meth:`ArgoKerchunker.translate`
        """
        if overwrite:
            self.translate(ncfile)
        elif str(ncfile) not in self.kerchunk_references:
            if self.fs.exists(self._ncfile2jsfile(ncfile)):
                self.kerchunk_references.update({str(ncfile): self._ncfile2jsfile(ncfile)})
            else:
                self.translate(ncfile)

        # Read and load the kerchunk JSON file:
        kerchunk_jsfile = self.kerchunk_references[str(ncfile)]
        with self.fs.open(kerchunk_jsfile, "r") as file:
            kerchunk_data = json.load(file)

        # Ensure that the loaded kerchunk data corresponds to the ncfile target
        if not overwrite:
            target_ok = False
            for key, value in kerchunk_data["refs"].items():
                if key not in [".zgroup", ".zattrs"] and "0." in key:
                    if value[0] == ncfile:
                        target_ok = True
                        break
            if not target_ok:
                kerchunk_data = self.to_kerchunk(ncfile, overwrite=True)

        return kerchunk_data

    def pprint(self, ncfile: Union[str, Path], params: List[str] = None):
        """Pretty print kerchunk json data for a netcdf file"""
        params = to_list(params) if params is not None else []
        kerchunk_data = self.to_kerchunk(ncfile)

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

    def open_dataset(self, ncfile: Union[str, Path], **kwargs) -> xr.Dataset:
        """Open a netcdf file lazily using zarr engine and kerchunk data

        Parameters
        ----------
        ncfile: str, Path
            Netcdf file to open lazily

        **kwargs:
            Other kwargs are passed to :meth:`ArgoKerchunker.to_kerchunk`

        Returns
        -------
        :class:`xarray.Dataset`
        """
        remote_protocol = split_protocol(ncfile)[0]
        return xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": self.to_kerchunk(ncfile, **kwargs),
                    "remote_protocol": remote_protocol,
                    "remote_options": self.remote_options,
                },
            },
        )

    def _magic(self, ncfile: Union[str, Path]) -> str:
        """Read first 4 bytes of a netcdf file

        Return None if the netcdf file cannot be open

        Parameters
        ----------
        ncfile: str, Path

        Raises
        ------
        :class:`aiohttp.ClientResponseError`
        """
        protocol = split_protocol(str(ncfile))[0]
        if protocol == 'ftp':
            opts = {'host': urlparse(ncfile).hostname,  # host eg: ftp.ifremer.fr
                    'port': 0 if urlparse(ncfile).port is None else urlparse(ncfile).port}
        else:
            opts = {}
        fs = fsspec.filesystem(protocol, **opts)

        def is_read(fs, uri):
            try:
                fs.ls(uri)
                return True
            except aiohttp.ClientResponseError:
                raise
            except:
                return False

        if is_read(fs, str(ncfile)):
            try:
                return fs.open(str(ncfile)).read(4)
            except:  # noqa: E722
                return None
        else:
            return None

    def supported(self, ncfile: Union[str, Path]) -> bool:
        """Check if a netcdf file can be accessed through byte ranges

        For non-local files, the absolute path toward the netcdf file must include the file protocol to return
        a correct answer.

        Argo GDAC supporting byte ranges:
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
        return self._magic(ncfile) is not None
