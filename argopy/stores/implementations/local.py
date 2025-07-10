import json
import xarray as xr
import pandas as pd
import types
import concurrent.futures
import multiprocessing
import logging
import io
from typing import Literal, Any
import fsspec
from pathlib import Path
import warnings
from netCDF4 import Dataset

from ...options import OPTIONS
from ...errors import InvalidMethod, DataNotFound

from ..spec import ArgoStoreProto
from ..filesystems import has_distributed, distributed
from ..filesystems import tqdm

log = logging.getLogger("argopy.stores.implementation.local")


class filestore(ArgoStoreProto):
    """Argo local file system

    Relies on :class:`fsspec.implementations.local.LocalFileSystem`
    """

    protocol = "file"

    def open_json(self, url, errors: Literal['raise', 'silent', 'ignore'] = 'raise', **kwargs) -> Any:
        """Open and process a json document from a path

        Steps performed:

        1. Path is open from ``url`` with :class:`filestore.open` and then
        2. Create a JSON with :func:`json.loads`.

        Each steps can be passed specifics arguments (see Parameters below).

        Parameters
        ----------
        path: str
            Path to resources passed to :func:`json.loads`
        errors: str, default: ``raise``
            Define how to handle errors:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console and return None
                - ``silent``:  Do not stop processing and do not issue log message, return None

        kwargs: dict

            - ``open_opts`` key dictionary is passed to :class:`filestore.open`
            - ``js_opts`` key dictionary is passed to :func:`json.loads`

        Returns
        -------
        Any

        See Also
        --------
        :class:`filestore.open_mfjson`
        """
        open_opts = {}
        if "open_opts" in kwargs:
            open_opts.update(kwargs["open_opts"])

        js_opts = {}
        if "js_opts" in kwargs:
            js_opts.update(kwargs["js_opts"])

        with self.open(url, **open_opts) as of:
            js = json.load(of, **js_opts)

        if len(js) == 0:
            if errors == "raise":
                raise DataNotFound("No data return by %s" % url)

            elif errors == "ignore":
                log.debug("No data return by %s" % url)
                return None

            else:
                return None

        return js

    def open_dataset(
            self,
            path,
            errors: Literal["raise", "ignore", "silent"] = "raise",
            lazy: bool = False,
            xr_opts: dict = {},
            **kwargs,
    ) -> xr.Dataset:
        """Create a :class:`xarray.Dataset` from a local path pointing to a netcdf file

        Parameters
        ----------
        path: str
            The local path of the netcdf file to open

        errors:

        lazy: bool, default=False
            Define if we should try to open the netcdf dataset lazily or not

        xr_opts:
            Arguments to be passed to :func:`xarray.open_dataset`

        Returns
        -------
        :class:`xarray.Dataset`
        """
        def load_in_memory(path, errors="raise", xr_opts={}):
            """
            Returns
            -------
            tuple: (data, _) or (None, _) if errors == "ignore"
            """
            try:
                data = self.fs.cat_file(path)

                if data[0:3] != b"CDF" and data[0:3] != b"\x89HD":
                    raise TypeError(
                        "We didn't get a CDF or HDF5 binary data as expected ! We get: %s"
                        % data
                    )
                if data[0:3] == b"\x89HD":
                    data = io.BytesIO(data)

                return data, None
            except FileNotFoundError as e:
                if errors == "raise":
                    raise e
                elif errors == "ignore":
                    log.error("FileNotFoundError raised from: %s" % path)
            return None, None

        def load_lazily(path, errors="raise", xr_opts={}, akoverwrite: bool = False):
            from .. import ArgoKerchunker

            if "ak" not in kwargs:
                self.ak = ArgoKerchunker(
                    store="local", root=Path(OPTIONS["cachedir"]).joinpath("kerchunk")
                )
            else:
                self.ak = kwargs["ak"]

            if self.ak.supported(path):
                xr_opts = {
                    "engine": "zarr",
                    "backend_kwargs": {
                        "consolidated": False,
                        "storage_options": {
                            "fo": self.ak.to_kerchunk(path, overwrite=akoverwrite),  # codespell:ignore
                            "remote_protocol": fsspec.core.split_protocol(path)[0],
                        },
                    },
                }
                return "reference://", xr_opts
            else:
                warnings.warn(
                    "This path does not support byte range requests so we cannot load it lazily, falling back on "
                    "loading in memory."
                )
                log.debug("This path does not support byte range requests: %s" % path)
                return load_in_memory(path, errors=errors, xr_opts=xr_opts)

        netCDF4 = kwargs.get("netCDF4", False)
        if lazy and netCDF4:
            if errors == "raise":
                raise ValueError("Cannot return a netCDF4.Dataset object in lazy mode")
            elif errors == "ignore":
                log.error("Cannot return a netCDF4.Dataset object in lazy mode")
                return None

        if not lazy:
            target, _ = load_in_memory(path, errors=errors, xr_opts=xr_opts)
        else:
            target, xr_opts = load_lazily(
                path,
                errors=errors,
                xr_opts=xr_opts,
                akoverwrite=kwargs.get("akoverwrite", False),
            )

        if target is not None:
            if not netCDF4:
                ds = xr.open_dataset(target, **xr_opts)

                if "source" not in ds.encoding:
                    if isinstance(path, str):
                        ds.encoding["source"] = path

            else:
                target = target if isinstance(target, bytes) else target.getbuffer()
                ds = Dataset(None, memory=target, diskless=True, mode='r')

            self.register(path)
            return ds

        elif errors == "raise":
            raise DataNotFound(path)

        elif errors == "ignore":
            log.error("DataNotFound from: %s" % path)
            return None

    def _mfprocessor(
        self,
        url,
        preprocess=None,
        preprocess_opts={},
        open_dataset_opts={},
        *args,
        **kwargs,
    ):
        # Load data
        ds = self.open_dataset(url, **open_dataset_opts)
        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(
            preprocess, types.MethodType
        ):
            ds = preprocess(ds, **preprocess_opts)
        return ds

    def open_mfdataset(
        self,  # noqa: C901
        urls,
        concat_dim="row",
        max_workers: int = 6,
        method: str = "thread",
        progress: bool = False,
        concat: bool = True,
        preprocess=None,
        preprocess_opts={},
        open_dataset_opts={},
        errors: str = "ignore",
        *args,
        **kwargs,
    ):
        """Open multiple urls as a single xarray dataset.

        This is a version of the ``open_dataset`` method that is able to handle a list of urls/paths
        sequentially or in parallel.

        Use a Threads Pool by default for parallelization.

        Parameters
        ----------
        urls: list(str)
            List of url/path to open
        concat_dim: str
            Name of the dimension to use to concatenate all datasets (passed to :class:`xarray.concat`)
        max_workers: int
            Maximum number of threads or processes
        method: str
            The parallelization method to execute calls asynchronously:
                - ``thread`` (Default): use a pool of at most ``max_workers`` threads
                - ``process``: use a pool of at most ``max_workers`` processes
                - (XFAIL) a :class:`distributed.client.Client` object (:class:`distributed.client.Client`)

            Use 'seq' to simply open data sequentially
        progress: bool
            Display a progress bar (True by default)
        preprocess: callable (optional)
            If provided, call this function on each dataset prior to concatenation
        errors: str
            Should it 'raise' or 'ignore' errors. Default: 'ignore'

        Returns
        -------
        :class:`xarray.Dataset`

        """
        if not isinstance(urls, list):
            urls = [urls]

        results = []
        if method in ["thread", "process"]:
            if method == "thread":
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                )
            else:
                if max_workers == 6:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers
                )

            with ConcurrentExecutor as executor:
                future_to_url = {
                    executor.submit(
                        self._mfprocessor,
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    ): url
                    for url in urls
                }
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(
                        futures, total=len(urls), disable="disable" in [progress]
                    )

                for future in futures:
                    data = None
                    # url = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as e:
                        if errors == "ignore":
                            log.debug(
                                "Ignored error with this file: %s\nException raised: %s"
                                % (future_to_url[future], str(e.args))
                            )
                            pass
                        else:
                            raise
                    finally:
                        results.append(data)

        elif has_distributed and isinstance(method, distributed.client.Client):
            # Use a dask client:

            if progress:
                from dask.diagnostics import ProgressBar

                with ProgressBar():
                    futures = method.map(
                        self._mfprocessor,
                        urls,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    )
                    results = method.gather(futures)
            else:
                futures = method.map(
                    self._mfprocessor,
                    urls,
                    preprocess=preprocess,
                    preprocess_opts=preprocess_opts,
                    open_dataset_opts=open_dataset_opts,
                    *args,
                    **kwargs,
                )
                results = method.gather(futures)

        elif method in ["seq", "sequential"]:
            if progress:
                urls = tqdm(urls, total=len(urls), disable="disable" in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor(
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    )
                except Exception as e:
                    if errors == "ignore":
                        log.debug(
                            "Ignored error with this url: %s\nException raised: %s"
                            % (url, str(e.args))
                        )
                        pass
                    else:
                        raise
                finally:
                    results.append(data)

        else:
            raise InvalidMethod(method)

        # Post-process results
        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            if concat:
                # ds = xr.concat(results, dim=concat_dim, data_vars='all', coords='all', compat='override')
                ds = xr.concat(
                    results,
                    dim=concat_dim,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                )
                return ds
            else:
                return results
        else:
            raise DataNotFound(urls)

    def read_csv(self, path, **kwargs):
        """Return a pandas.dataframe from a path that is a csv resource

        Parameters
        ----------
        Path: str
            Path to csv resources passed to :func:`pandas.read_csv`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        log.debug("Reading csv: %s" % path)
        with self.open(path) as of:
            df = pd.read_csv(of, **kwargs)
        return df
