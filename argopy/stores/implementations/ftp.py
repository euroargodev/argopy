import logging
import types
import xarray as xr
import concurrent.futures
import multiprocessing
import io
import fsspec
import warnings
from typing import Literal
from netCDF4 import Dataset

from ...errors import InvalidMethod, DataNotFound
from ...utils.transform import drop_variables_not_in_all_datasets
from ..filesystems import has_distributed, distributed
from ..filesystems import tqdm
from .http import httpstore


log = logging.getLogger("argopy.stores.implementation.ftp")


class ftpstore(httpstore):
    """Argo ftp file system

    Inherits from :class:`argopy.stores.httpstore` but relies on :class:`fsspec.implementations.ftp.FTPFileSystem`
    """

    protocol = "ftp"

    @property
    def host(self):
        return self.fs.fs.host if self.fs.protocol == "dir" else self.fs.host

    @property
    def port(self):
        return self.fs.fs.port if self.fs.protocol == "dir" else self.fs.port

    def open_dataset(
        self,
        url: str,
        errors: Literal["raise", "ignore", "silent"] = "raise",
        lazy: bool = False,
        xr_opts: dict = {},
        **kwargs,
    ):
        """Create a :class:`xarray.Dataset` from an url pointing to a netcdf file

        Parameters
        ----------
        url: str
            The remote URL of the netcdf file to open

        errors: Literal, default: ``raise``
            Define how to handle errors raised during data fetching:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message

        lazy: bool, default=False
            Define if we should try to load netcdf file lazily or not

            **If this is set to False (default)** opening is done in 2 steps:
                1. Download from ``url`` raw binary data with :class:`ftpstore.fs.cat_file`,
                2. Create a :class:`xarray.Dataset` with :func:`xarray.open_dataset`.

            Each functions can be passed specifics arguments with ``dwn_opts`` and  ``xr_opts`` (see below).

            **If this is set to True**, use a :class:`ArgoKerchunker` instance to access
            the netcdf file lazily. You can provide a specific :class:`ArgoKerchunker` instance with the ``ak`` argument (see below).

        xr_opts: dict, default={}
             Options passed to :func:`xarray.open_dataset`. This will be ignored if the ``netCDF4` option is set to True.

        Other Parameters
        ----------------
        ak: :class:`ArgoKerchunker`, optional
            :class:`ArgoKerchunker` instance to use if ``lazy=True``.

        akoverwrite: bool, optional
            Determine if kerchunk data should be overwritten or not. This is passed to :meth:`ArgoKerchunker.to_kerchunk`.

        netCDF4: bool, optional, default=False
            Return a :class:`netCDF4.Dataset` object instead of a :class:`xarray.Dataset`

        Returns
        -------
        :class:`xarray.Dataset` or :class:`netCDF4.Dataset`

        Raises
        ------
        :class:`TypeError`
            Raised if data returned by ``url`` are not CDF or HDF5 binary data.

        :class:`DataNotFound`
            Raised if ``errors`` is set to ``raise`` and url returns no data.

        See Also
        --------
        :func:`httpstore.open_mfdataset`, :class:`ArgoKerchunker`

        """

        def load_in_memory(url, errors="raise", xr_opts={}):
            """Download url content and return data along with xarray option to open it

            Returns
            -------
            tuple: (data, xr_opts) or (None, None) if errors == "ignore"
            """

            try:
                this_url = self.fs._strip_protocol(url)
                data = self.fs.cat_file(this_url)
                if data is None:
                    if errors == "raise":
                        raise DataNotFound(url)
                    elif errors == "ignore":
                        log.error("DataNotFound: %s" % url)
                    return None, None
            except Exception:
                log.debug("Error with: %s" % url)
                # except aiohttp.ClientResponseError as e:
                raise

            if data[0:3] != b"CDF" and data[0:3] != b"\x89HD":
                raise TypeError(
                    "We didn't get a CDF or HDF5 binary data as expected ! We get: %s"
                    % data
                )
            if data[0:3] == b"\x89HD":
                data = io.BytesIO(data)

            return data, xr_opts

        def load_lazily(url, errors="raise", xr_opts={}, akoverwrite: bool = False):
            """Check if url support lazy access and return kerchunk data along with xarray option to open it lazily

            Otherwise, download url content and return data along with xarray option to open it.

            Returns
            -------
            tuple:
                If the url support lazy access:
                    ("reference://", xr_opts)
                else:
                    (data, xr_opts) or (None, None) if errors == "ignore"
            """
            from .. import ArgoKerchunker

            if "ak" not in kwargs:
                self.ak = ArgoKerchunker()
                self.ak.storage_options = {"host": self.host, "port": self.port}
            else:
                self.ak = kwargs["ak"]

            if self.ak.supported(url, fs=self):
                xr_opts = {
                    "engine": "zarr",
                    "backend_kwargs": {
                        "consolidated": False,
                        "storage_options": {
                            "fo": self.ak.to_reference(
                                url, overwrite=akoverwrite, fs=self
                            ),  # codespell:ignore
                            "remote_protocol": fsspec.core.split_protocol(url)[0],
                            "remote_options": self.ak.storage_options,
                        },
                    },
                }
                return "reference://", xr_opts
            else:
                warnings.warn(
                    "This url does not support byte range requests so we cannot load it lazily, falling back on loading in memory.\n(url='%s')"
                    % url
                )
                log.debug(
                    "This url does not support byte range requests: %s"
                    % self.full_path(url)
                )
                return load_in_memory(url, errors=errors, xr_opts=xr_opts)

        netCDF4 = kwargs.get("netCDF4", False)
        if lazy and netCDF4:
            if errors == "raise":
                raise ValueError("Cannot return a netCDF4.Dataset object in lazy mode")
            elif errors == "ignore":
                log.error("Cannot return a netCDF4.Dataset object in lazy mode")
                return None

        if not lazy:
            target, _ = load_in_memory(url, errors=errors, xr_opts=xr_opts)
        else:
            target, xr_opts = load_lazily(
                url,
                errors=errors,
                xr_opts=xr_opts,
                akoverwrite=kwargs.get("akoverwrite", False),
            )

        if target is not None:
            if not netCDF4:
                ds = xr.open_dataset(target, **xr_opts)

                if "source" not in ds.encoding:
                    if isinstance(url, str):
                        ds.encoding["source"] = self.full_path(url)

            else:
                target = target if isinstance(target, bytes) else target.getbuffer()
                ds = Dataset(None, memory=target, diskless=True, mode="r")

            self.register(url)
            return ds

        elif errors == "raise":
            raise DataNotFound(url)

        elif errors == "ignore":
            log.error("DataNotFound from: %s" % url)
            return None

    def _mfprocessor_dataset(
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
        max_workers: int = 6,
        method: str = "sequential",
        progress: bool = False,
        concat: bool = True,
        concat_dim="row",
        preprocess=None,
        preprocess_opts={},
        open_dataset_opts={},
        errors: str = "ignore",
        *args,
        **kwargs,
    ):
        """Open multiple ftp urls as a single xarray dataset.

        This is a version of the :class:`argopy.stores.ftpstore.open_dataset` method that is able
        to handle a list of urls/paths sequentially or in parallel.

        Use a Threads Pool by default for parallelization.

        Parameters
        ----------
        urls: list(str)
            List of url/path to open
        max_workers: int, default: 6
            Maximum number of threads or processes
        method: str, default: ``thread``
            The parallelization method to execute calls asynchronously:

                - ``seq`` (default): open data sequentially, no parallelization applied
                - ``process``: use a pool of at most ``max_workers`` processes
                - :class:`distributed.client.Client`: Experimental, expect this method to fail !
        progress: bool, default: False
            Display a progress bar
        concat: bool, default: True
            Concatenate results in a single :class:`xarray.Dataset` or not (in this case, function will return a
            list of :class:`xarray.Dataset`)
        concat_dim: str, default: ``row``
            Name of the dimension to use to concatenate all datasets (passed to :class:`xarray.concat`)
        preprocess: callable (optional)
            If provided, call this function on each dataset prior to concatenation
        preprocess_opts: dict (optional)
            If ``preprocess`` is provided, pass this as options
        errors: str, default: ``ignore``
            Define how to handle errors raised during data URIs fetching:

                - ``raise``: Raise any error encountered
                - ``ignore`` (default): Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message
        Other args and kwargs: other options passed to :class:`argopy.stores.httpstore.open_dataset`.

        Returns
        -------
        output: :class:`xarray.Dataset` or list of :class:`xarray.Dataset`

        """
        strUrl = lambda x: x.replace("ftps://", "").replace("ftp://", "")  # noqa: E731

        if not isinstance(urls, list):
            urls = [urls]

        results = []
        failed = []
        if method in ["process"]:
            if max_workers == 6:
                max_workers = multiprocessing.cpu_count()
            ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            )

            with ConcurrentExecutor as executor:
                future_to_url = {
                    executor.submit(
                        self._mfprocessor_dataset,
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
                    try:
                        data = future.result()
                    except Exception:
                        failed.append(future_to_url[future])
                        if errors == "ignore":
                            log.debug(
                                "Ignored error with this url: %s"
                                % strUrl(future_to_url[future])
                            )
                            # See fsspec.http logger for more
                            pass
                        elif errors == "silent":
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
                        self._mfprocessor_dataset,
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
                    self._mfprocessor_dataset,
                    urls,
                    preprocess=preprocess,
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
                    data = self._mfprocessor_dataset(
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    )
                except Exception:
                    failed.append(url)
                    if errors == "ignore":
                        log.debug(
                            "Ignored error with this url: %s" % strUrl(url)
                        )  # See fsspec.http logger for more
                        pass
                    elif errors == "silent":
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
                results = drop_variables_not_in_all_datasets(results)
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
