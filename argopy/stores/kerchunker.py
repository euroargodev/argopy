import fsspec
import xarray as xr
from typing import List, Union, Dict, Literal
from pathlib import Path
from fsspec.core import split_protocol
import json
import logging

# from tempfile import TemporaryDirectory
# from fsspec.implementations.dirfs import DirFileSystem

from ..stores import memorystore, filestore
from ..utils import to_list

log = logging.getLogger("argopy.stores.kerchunk")

try:
    from kerchunk.hdf import SingleHdf5ToZarr
    from kerchunk.netCDF3 import NetCDF3ToZarr

    HAS_KERCHUNK = True
except ModuleNotFoundError:
    log.debug("argopy missing 'kerchunk' to translate netcdf file")
    HAS_KERCHUNK = False
    SingleHdf5ToZarr, NetCDF3ToZarr = None, None

try:
    import dask

    HAS_DASK = True
except ModuleNotFoundError:
    log.debug("argopy missing 'dask' to improve performances of 'ArgoKerchunker'")
    HAS_DASK = False
    dask = None
    import concurrent.futures


class ArgoKerchunker:
    """
    Argo netcdf file kerchunk helper

    Note that the `kerchunk <https://fsspec.github.io/kerchunk/>`_ library is required if you need to extract
    zarr data from netcdf file(s), a.k.a. :meth:`argopy.stores.ArgoKerchunker.translate`.

    Examples
    --------
    .. code-block:: python
        :caption: API examples

        >>> ncfile = "s3://argo-gdac-sandbox/pub/dac/coriolis/6903090/6903090_prof.nc"

        >>> ak = ArgoKerchunker(store='memory')  # default
        >>> ak = ArgoKerchunker(store='local', root='.')
        >>> ak = ArgoKerchunker(store='local', root='/kerchunk_data_folder')  # Custom local storage folder
        >>> ak = ArgoKerchunker(store=fsspec.filesystem('dir', path='s3://.../kerchunk_data_folder/', target_protocol='s3'))

        >>> ak.supported(ncfile)
        >>> ak.translate(ncfile)  # takes 1 or a list of uris, requires the kerchunk library
        >>> ak.to_kerchunk(ncfile)  # Take 1 netcdf file uri, return kerchunk json data (translate or load from store)
        >>> ak.pprint(ncfile)

        >>> ak.open_dataset(ncfile)  #  Return xarray dataset of the netcdf file using zarr engine from kerchunk data

    """

    def __init__(
        self,
        store: Literal["memory", "local"] = "memory",
        root: Union[Path, str] = ".",
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
            during translation, and in `backend_kwargs` of :class:`xarray.open_dataset`.

        """
        # Instance file system to load/save kerchunk json files
        if store == "memory":
            self.fs = memorystore()
        elif store == "local":
            if Path(root).name == "":
                self.fs = filestore()
            else:
                self.fs = fsspec.filesystem("dir", path=root, target_protocol="local")
        elif isinstance(store, fsspec.AbstractFileSystem):
            self.fs = store

        # Passed to fsspec when opening netcdf file:
        self.remote_options = remote_options if remote_options is not None else {}

        # List of processed files register:
        self.kerchunk_references = {}

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
        summary.append("- %i reference%s loaded" % (n, "s" if n > 0 else ""))
        return "\n".join(summary)

    def _ncfile2jsfile(self, ncfile):
        return Path(ncfile).name.replace(".nc", ".json")

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

    def translate(self, ncfiles: List):
        """Compute kerchunk data for one or a list of netcdf files

        Kerchunk data are saved with the instance file store

        Once translated, netcdf file reference data are internally registered in ``kerchunk_references``
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

        Load data from instance file store, translate if necessary
        """
        if overwrite:
            self.translate(ncfile)
        elif str(ncfile) not in self.kerchunk_references:
            if self.fs.exists(self._ncfile2jsfile(ncfile)):
                self.kerchunk_references.update({ncfile: self._ncfile2jsfile(ncfile)})
            else:
                self.translate(ncfile)

        # Read and load the kerchunk JSON file:
        kerchunk_jsfile = self.kerchunk_references[ncfile]
        with self.fs.open(kerchunk_jsfile, "r") as file:
            kerchunk_data = json.load(file)

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

    def open_dataset(self, ncfile: Union[str, Path]) -> xr.Dataset:
        """Open a netcdf file using zarr engine from kerchunk data

        Parameters
        ----------
        ncfile: str, Path

        Returns
        -------
        :class:`xr.Dataset`
        """
        remote_protocol = split_protocol(ncfile)[0]
        return xr.open_dataset(
            "reference://",
            engine="zarr",
            backend_kwargs={
                "consolidated": False,
                "storage_options": {
                    "fo": self.to_kerchunk(ncfile),
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
        """
        fs = fsspec.filesystem(split_protocol(str(ncfile))[0])
        try:
            return fs.open(str(ncfile)).read(4)
        except:  # noqa: E722
            return None

    def supported(self, ncfile: Union[str, Path]) -> bool:
        """Check if a netcdf file can be accessed through byte ranges

        Argo GDAC supporting byte ranges:
        - s3://argo-gdac-sandbox/pub/idx
        - https://argo-gdac-sandbox.s3-eu-west-3.amazonaws.com/pub/idx

        Not supporting:
        - https://data-argo.ifremer.fr

        Parameters
        ----------
        ncfile: str, Path
        """
        return self._magic(ncfile) is not None
