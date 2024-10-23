import fsspec
import xarray as xr
from typing import List, Union
from pathlib import Path
from fsspec.core import split_protocol
import json
import logging

# from tempfile import TemporaryDirectory
# from fsspec.implementations.dirfs import DirFileSystem

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
    log.debug("argopy missing 'dask' to improve performances of 'ArgoKerchunker' ")
    HAS_DASK = False
    dask = None
    import concurrent.futures


from ..stores import memorystore, filestore
from ..utils import to_list


class ArgoKerchunker:
    """
    Argo netcdf file kerchunk helper

    Note that the 'kerchunk' library is required if you need to extract
    zarr data from netcdf file(s).

    Examples
    --------
    .. code-block:: python
        :caption:

        >>> ncfile = "s3://argo-gdac-sandbox/pub/dac/coriolis/6903090/6903090_prof.nc"

        >>> ak = ArgoKerchunker(store='local')
        >>> ak = ArgoKerchunker(store='memory')  # default

        >>> ak.supported(ncfile)
        >>> ak.translate(ncfile)  # takes 1 or a list of uris
        >>> ak.pprint(ncfile)

        >>> ak.open_dataset(ncfile)

    """

    def __init__(
        self,
        store: Union["memory", "local"] = "memory",
        inline_threshold: int = 0,
        max_chunk_size: int = 0,
    ):
        # File system to load/save kerchunk json files
        if store == "memory":
            self.fs = memorystore()
        elif store == "local":
            self.fs = filestore()
        elif isinstance(store, fsspec.AbstractFileSystem):
            self.fs = store

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

    def _tojson(self, ncfile: Union[str, Path]):
        if not HAS_KERCHUNK:
            raise ModuleNotFoundError("Requires the 'kerchunk' library")

        ncfile = str(ncfile)

        magic = self._magic(ncfile)[0:3]
        if magic == b"CDF":
            chunks = NetCDF3ToZarr(
                ncfile,
                inline_threshold=self.inline_threshold,
                max_chunk_size=self.max_chunk_size,
            )
        elif magic == b"\x89HD":
            chunks = SingleHdf5ToZarr(
                ncfile,
                inline_threshold=self.inline_threshold,
                max_chunk_size=self.max_chunk_size,
            )

        zjs = chunks.translate()

        single_kerchunk_reference = Path(ncfile).name.replace(".nc", ".json")

        with self.fs.open(single_kerchunk_reference, "wb") as f:
            f.write(json.dumps(zjs).encode())

        return ncfile, single_kerchunk_reference, zjs

    def translate(self, ncfiles: List):
        """Compute kerchunks for one or a list of netcdf files"""
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
            ncfile, single_kerchunk_reference, _ = result
            self.kerchunk_references.update({ncfile: single_kerchunk_reference})

        return results

    def load_kerchunk(self, ncfile):
        """Return json kerchunk data for a given netcdf file"""

        if ncfile not in self.kerchunk_references:
            if self.fs.exists(ncfile):
                single_kerchunk_reference = self.fs.cat(ncfile)
                self.kerchunk_references.update({ncfile: single_kerchunk_reference})
            else:
                self.translate(ncfile)

        # Read and load the kerchunk JSON file:
        single_kerchunk_reference = self.kerchunk_references[ncfile]
        with self.fs.open(single_kerchunk_reference, "r") as file:
            data = json.load(file)

        return data

    def pprint(self, ncfile: Union[str, Path], params: List[str] = None):
        """Pretty print kerchunk json data for a netcdf file"""
        ncfile = str(ncfile)

        if ncfile not in self.kerchunk_references:
            self.translate(ncfile)

        params = to_list(params) if params is not None else []

        # Read and load the kerchunk JSON file:
        single_kerchunk_reference = self.kerchunk_references[ncfile]
        with self.fs.open(single_kerchunk_reference, "r") as file:
            data = json.load(file)

        # Pretty print JSON data
        keys_to_select = [".zgroup", ".zattrs", ".zmetadata"]
        data_to_print = {}
        for key, value in data["refs"].items():
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
        """Open a netcdf Argo file with kerchunk"""
        ncfile = str(ncfile)

        if ncfile not in self.kerchunk_references:
            self.translate(ncfile)

        single_kerchunk_reference = self.kerchunk_references[ncfile]

        remote_protocol = split_protocol(ncfile)[0]

        fs_single = fsspec.filesystem(
            "reference", fo=single_kerchunk_reference, remote_protocol=remote_protocol
        )

        single_map = fs_single.get_mapper("")

        return xr.open_dataset(
            single_map, engine="zarr", backend_kwargs={"consolidated": False}
        )

    def _magic(self, ncfile: Union[str, Path]) -> str:
        fs = fsspec.filesystem(split_protocol(str(ncfile))[0])
        try:
            return fs.open(str(ncfile)).read(4)
        except:
            return None

    def supported(self, ncfile: Union[str, Path]) -> bool:
        """Check if a netcdf file can be accessed through byte ranges

        Argo GDAC supporting byte ranges:
        - s3://argo-gdac-sandbox/pub/idx
        - https://argo-gdac-sandbox.s3-eu-west-3.amazonaws.com/pub/idx

        Not supporting:
        - https://data-argo.ifremer.fr
        """
        fs = fsspec.filesystem(split_protocol(str(ncfile))[0])
        try:
            fs.open(str(ncfile)).read(4)
            return True
        except:
            return False
