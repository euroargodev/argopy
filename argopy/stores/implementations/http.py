import json

import xarray as xr
import pandas as pd
import types
import concurrent.futures
import multiprocessing
import logging
from typing import Union, Any, List, Literal
from collections.abc import Callable
import aiohttp
import fsspec
import time
import warnings
import io
from functools import lru_cache
from netCDF4 import Dataset

from ...errors import InvalidMethod, DataNotFound
from ...utils import Registry, UriCName
from ...utils import has_aws_credentials
from ...utils import (
    drop_variables_not_in_all_datasets,
    fill_variables_not_in_all_datasets,
)
from ...utils.monitored_threadpool import MyThreadPoolExecutor as MyExecutor
from ..spec import ArgoStoreProto
from ..filesystems import has_distributed, distributed
from ..filesystems import tqdm


log = logging.getLogger("argopy.stores.implementation.http")


class httpstore(ArgoStoreProto):
    """Argo http file system

    Relies on :class:`fsspec.implementations.http.HTTPFileSystem`

    This store intends to make argopy safer to failures from http requests and to provide higher levels methods to
    work with our datasets. Key methods are:

    - :class:`httpstore.download_url`
    - :class:`httpstore.open_dataset`
    - :class:`httpstore.open_json`
    - :class:`httpstore.open_mfdataset`
    - :class:`httpstore.open_mfjson`
    - :class:`httpstore.read_csv`

    """

    protocol = "http"

    def __init__(self, *args, **kwargs):
        # Create a registry that will be used to keep track of all URLs accessed by this store
        self.urls_registry = Registry(name="Accessed URLs")
        super().__init__(*args, **kwargs)

    def open(self, path, *args, **kwargs):
        path = self.curateurl(path)
        return super().open(path, *args, **kwargs)

    def exists(self, path, *args, **kwargs):
        path = self.curateurl(path)
        return super().exists(path, *args, **kwargs)

    def curateurl(self, url) -> str:
        """Register and possibly manipulate an url before it's accessed

        This method should be called anytime an url is accessed

        Parameters
        ----------
        url: str
            URL to register and curate

        Returns
        -------
        url: str
            Registered and curated URL
        """
        self.urls_registry.commit(url)
        return url

    def download_url(
        self, url, max_attempt: int = 5, cat_opts: dict = {}, errors: str = "raise"
    ) -> Any:
        """Resilient URL data downloader

        This is basically a :func:`fsspec.implementations.http.HTTPFileSystem.cat_file` that is able to handle a 429 "Too many requests" error from a server, by waiting and sending requests several time.

        Parameters
        ----------
        url: str
            URL to download
        max_attempt: int, default = 5
            Maximum number of attempts to perform before failing
        cat_opts: dict, default = {}
            Options to be passed to the HTTPFileSystem cat_file method
        errors: str, default: ``raise``
            Define how to handle errors:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message

        """

        def make_request(
            ffs,
            url,
            n_attempt: int = 1,
            max_attempt: int = 5,
            cat_opts: dict = {},
            errors: str = "raise",
        ):
            data = None
            if n_attempt <= max_attempt:
                try:
                    data = ffs.cat_file(url, **cat_opts)
                except FileNotFoundError as e:
                    if errors == "raise":
                        raise e
                    elif errors == "ignore":
                        log.error("FileNotFoundError raised from: %s" % url)
                except aiohttp.ClientResponseError as e:
                    if e.status == 413:
                        if errors == "raise":
                            raise e
                        elif errors == "ignore":
                            log.error(
                                "Error %i (Payload Too Large) raised with %s"
                                % (e.status, url)
                            )

                    elif e.status == 429:
                        retry_after = int(e.headers.get("Retry-After", 5))
                        log.debug(
                            f"Error {e.status} (Too many requests). Retry after {retry_after} seconds. Tentative {n_attempt}/{max_attempt}"
                        )
                        time.sleep(retry_after)
                        n_attempt += 1
                        make_request(ffs, url, n_attempt=n_attempt, cat_opts=cat_opts)
                    else:
                        # Handle other client response errors
                        print(f"Error: {e}")
                except aiohttp.ClientError as e:
                    if errors == "raise":
                        raise e
                    elif errors == "ignore":
                        log.error("Error: {e}")
                except fsspec.FSTimeoutError as e:
                    if errors == "raise":
                        raise e
                    elif errors == "ignore":
                        log.error("Error: {e}")
            else:
                if errors == "raise":
                    raise ValueError(
                        f"Error: All attempts failed to download this url: {url}"
                    )
                elif errors == "ignore":
                    log.error("Error: All attempts failed to download this url: {url}")

            return data, n_attempt

        url = self.curateurl(url)
        data, n = make_request(
            self.fs,
            url,
            max_attempt=max_attempt,
            cat_opts=cat_opts,
            errors=errors,
        )

        if data is None:
            if errors == "raise":
                raise FileNotFoundError(url)
            elif errors == "ignore":
                log.error("FileNotFoundError: %s" % url)
        else:
            self.register(url)
        return data

    def open_dataset(
        self,
        url: str,
        errors: Literal["raise", "ignore", "silent"] = "raise",
        lazy: bool = False,
        dwn_opts: dict = {},
        xr_opts: dict = {},
        **kwargs,
    ) -> xr.Dataset:
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
                1. Download from ``url`` raw binary data with :class:`httpstore.download_url`,
                2. Create a :class:`xarray.Dataset` with :func:`xarray.open_dataset`.

            Each functions can be passed specifics arguments with ``dwn_opts`` and  ``xr_opts`` (see below).

            **If this is set to True**, use a :class:`ArgoKerchunker` instance to access
            the netcdf file lazily. You can provide a specific :class:`ArgoKerchunker` instance with the ``ak`` argument (see below).

        dwn_opts: dict, default={}
             Options passed to :func:`httpstore.download_url`

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

        def load_in_memory(url, errors="raise", dwn_opts={}, xr_opts={}):
            """Download url content and return data along with xarray option to open it

            Returns
            -------
            tuple: (data, xr_opts) or (None, None) if errors == "ignore"
            """
            data = self.download_url(url, **dwn_opts)
            if data is None:
                if errors == "raise":
                    raise DataNotFound(url)
                elif errors == "ignore":
                    log.error("DataNotFound: %s" % url)
                return None, None

            if b"Not Found: Your query produced no matching results" in data:
                if errors == "raise":
                    raise DataNotFound(url)
                elif errors == "ignore":
                    log.error("DataNotFound from [%s]: %s" % (url, data))
                return None, None

            if data[0:3] != b"CDF" and data[0:3] != b"\x89HD":
                raise TypeError(
                    "We didn't get a CDF or HDF5 binary data as expected ! We get: %s"
                    % data
                )
            if data[0:3] == b"\x89HD":
                data = io.BytesIO(data)

            return data, xr_opts

        def load_lazily(
            url, errors="raise", dwn_opts={}, xr_opts={}, akoverwrite: bool = False
        ):
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
                if self.protocol == 's3':
                    storage_options = {'anon': not has_aws_credentials()}
                else:
                    storage_options = {}
                self.ak.storage_options = storage_options
            else:
                self.ak = kwargs["ak"]

            if self.ak.supported(url, fs=self):
                xr_opts = {
                    "engine": "zarr",
                    "backend_kwargs": {
                        "consolidated": False,
                        "storage_options": {
                            "fo": self.ak.to_reference(url,
                                                       overwrite=akoverwrite,
                                                       fs=self),  # codespell:ignore
                            "remote_protocol": fsspec.core.split_protocol(url)[0],
                            "remote_options": self.ak.storage_options
                        },
                    },
                }
                return "reference://", xr_opts
            else:
                warnings.warn(
                    "This url does not support byte range requests so we cannot load it lazily, falling back on loading in memory."
                )
                log.debug("This url does not support byte range requests: %s" % self.full_path(url))
                return load_in_memory(
                    url, errors=errors, dwn_opts=dwn_opts, xr_opts=xr_opts
                )

        netCDF4 = kwargs.get("netCDF4", False)
        if lazy and netCDF4:
            if errors == "raise":
                raise ValueError("Cannot return a netCDF4.Dataset object in lazy mode")
            elif errors == "ignore":
                log.error("Cannot return a netCDF4.Dataset object in lazy mode")
                return None

        if not lazy:
            target, _ = load_in_memory(
                url, errors=errors, dwn_opts=dwn_opts, xr_opts=xr_opts
            )
        else:
            target, xr_opts = load_lazily(
                url,
                errors=errors,
                dwn_opts=dwn_opts,
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
                ds = Dataset(None, memory=target, diskless=True, mode='r')

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
        open_dataset_opts: dict = {},
        preprocess: Callable = None,
        preprocess_opts: dict = {},
    ) -> xr.Dataset:
        """Single URL dataset processor

        Internal method sent to a worker by :class:`httpstore.open_mfdataset` and responsible for dealing with a single URL.

        1. Open the dataset with :class:`httpstore.open_dataset`
        2. Pre-process the dataset with the ``preprocess`` function given in arguments

        Parameters
        ----------
        url: str
            URI to process
        open_dataset_opts: dict, default: {}
            Set of arguments passed to :class:`httpstore.open_dataset`
        preprocess: :class:`Typing.Callable`, default: None
            Pre-processing function
        preprocess_opts: dict, default: {}
            Options to be passed to the pre-processing function

        Returns
        -------
        :class:`xarray.Dataset`
        """
        # Load data
        ds = self.open_dataset(url, **open_dataset_opts)

        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(
            preprocess, types.MethodType
        ):
            ds = preprocess(ds, **preprocess_opts)
        return ds

    def _open_mfdataset_from_erddap(
        self,
        urls: list,
        concat_dim: str = "rows",
        max_workers: int = 6,
        preprocess: Callable = None,
        preprocess_opts: dict = None,
        concat: bool = True,
        progress: bool = True,
        compute_details: bool = False,
        *args,
        **kwargs,
    ):
        """
        Method used by :class:`httpstore.open_mfdataset` dedicated to handle the case where we need to
        create a dataset from multiples erddap urls download/preprocessing and need a visual feedback of the
        procedure up to the final merge.

        - httpstore.open_dataset is distributed is handle by a pool of threads

        """
        strUrl = lambda x: x.replace("https://", "").replace(  # noqa: E731
            "http://", ""
        )

        @lru_cache
        def task_fct(url):
            try:
                ds = self.open_dataset(url)
                ds.attrs["Fetched_url"] = url
                ds.attrs["Fetched_constraints"] = UriCName(url).cname
                return ds, True
            except FileNotFoundError:
                log.debug("task_fct: This url returned no data: %s" % strUrl(url))
                return DataNotFound(url), True
            except Exception as e:
                log.debug(
                    "task_fct: Unexpected error when opening the remote dataset '%s':\n'%s'"
                    % (strUrl(url), str(e))
                )
                return None, False

        def postprocessing_fct(obj, **kwargs):
            if isinstance(obj, xr.Dataset):
                try:
                    ds = preprocess(obj, **kwargs)
                    return ds, True
                except Exception as e:
                    log.debug(
                        "postprocessing_fct: Unexpected error when post-processing dataset: '%s'"
                        % str(e)
                    )
                    return None, False

            elif isinstance(obj, DataNotFound):
                return obj, True

            elif obj is None:
                # This is because some un-expected Exception was raised in task_fct(url)
                return None, False

            else:
                log.debug("postprocessing_fct: Unexpected object: '%s'" % type(obj))
                return None, False

        def finalize(obj_list, **kwargs):
            try:
                # Read list of datasets from the list of objects:
                ds_list = [v for v in dict(sorted(obj_list.items())).values()]
                # Only keep non-empty results:
                ds_list = [
                    r
                    for r in ds_list
                    if (r is not None and not isinstance(r, DataNotFound))
                ]
                # log.debug(ds_list)
                if len(ds_list) > 0:
                    if "data_vars" in kwargs and kwargs["data_vars"] == "all":
                        # log.info('fill_variables_not_in_all_datasets')
                        ds_list = fill_variables_not_in_all_datasets(
                            ds_list, concat_dim=concat_dim
                        )
                    else:
                        # log.info('drop_variables_not_in_all_datasets')
                        ds_list = drop_variables_not_in_all_datasets(ds_list)

                    log.info("Nb of dataset to concat: %i" % len(ds_list))
                    # log.debug(concat_dim)
                    # for ds in ds_list:
                    #     log.debug(ds[concat_dim])
                    log.info(
                        "Dataset sizes before concat: %s"
                        % [len(ds[concat_dim]) for ds in ds_list]
                    )
                    ds = xr.concat(
                        ds_list,
                        dim=concat_dim,
                        data_vars="minimal",
                        coords="minimal",
                        compat="override",
                    )
                    log.info("Dataset size after concat: %i" % len(ds[concat_dim]))
                    return ds, True
                else:
                    ds_list = [v for v in dict(sorted(obj_list.items())).values()]
                    # Is the ds_list full of None or DataNotFound ?
                    if len([r for r in ds_list if (r is None)]) == len(ds_list):
                        log.debug("finalize: An error occurred with all URLs !")
                        return (
                            ValueError(
                                "An un-expected error occurred with all URLs, check log file for more "
                                "information"
                            ),
                            True,
                        )
                    elif len(
                        [r for r in ds_list if isinstance(r, DataNotFound)]
                    ) == len(ds_list):
                        log.debug("finalize: All URLs returned DataNotFound !")
                        return DataNotFound("All URLs returned DataNotFound !"), True
            except Exception as e:
                log.debug(
                    "finalize: Unexpected error when finalize request: '%s'" % str(e)
                )
                return None, False

        if ".nc" in urls[0]:
            task_legend = {
                "w": "Downloading netcdf from the erddap",
                "p": "Formatting xarray dataset",
                "c": "Callback",
                "f": "Failed or No Data",
            }
        else:
            task_legend = {"w": "Working", "p": "Post-processing", "c": "Callback"}

        if concat:
            finalize_fct = finalize
        else:
            finalize_fct = None

        run = MyExecutor(
            max_workers=max_workers,
            task_fct=task_fct,
            postprocessing_fct=postprocessing_fct,
            postprocessing_fct_kwargs=preprocess_opts,
            finalize_fct=finalize_fct,
            finalize_fct_kwargs=kwargs["final_opts"] if "final_opts" in kwargs else {},
            task_legend=task_legend,
            final_legend={
                "task": "Processing data chunks",
                "final": "Merging chunks of xarray dataset",
            },
            show=progress,
        )
        results, failed = run.execute(urls, list_failed=True)

        if concat:
            # results = Union[xr.DataSet, DataNotFound, None]
            if isinstance(results, xr.Dataset):
                if not compute_details:
                    return results
                else:
                    return results, failed, len(results)
            elif results is None:
                raise DataNotFound("An error occurred while finalizing the dataset")
            else:
                raise results

        elif len(failed) == len(urls):
            raise ValueError(
                "Errors happened with all URLs, this could be due to an internal impossibility to read returned content"
            )

        else:
            if len([r for r in results if r == DataNotFound]) == len(urls):
                raise DataNotFound("All URLs returned DataNotFound !")
            else:
                if not compute_details:
                    return results
                else:
                    return results, failed, len(results)

    def open_mfdataset(
        self,  # noqa: C901
        urls,
        max_workers: int = 6,
        method: str = "thread",
        progress: Union[bool, str] = False,
        concat: bool = True,
        concat_dim: str = "row",
        concat_method: Literal["drop", "fill"] = "drop",
        preprocess: Callable = None,
        preprocess_opts: dict = {},
        open_dataset_opts: dict = {},
        errors: Literal["ignore", "raise", "silent"] = "ignore",
        compute_details: bool = False,
        *args,
        **kwargs,
    ) -> Union[xr.Dataset, List[xr.Dataset]]:
        """Download and process multiple urls as a single or a collection of :class:`xarray.Dataset`

        This is a version of the :class:`httpstore.open_dataset` method that is able to
        handle a list of urls sequentially or in parallel.

        This method uses a :class:`concurrent.futures.ThreadPoolExecutor` by default for parallelization. See the ``method`` parameter below for more options.

        Parameters
        ----------
        urls: list(str)
            List of url/path to open
        max_workers: int, default: 6
            Maximum number of threads or processes
        method: str, default: ``thread``
            Define the parallelization method:
                - ``thread`` (default): based on :class:`concurrent.futures.ThreadPoolExecutor` with a pool of at most ``max_workers`` threads
                - ``process``: based on :class:`concurrent.futures.ProcessPoolExecutor` with a pool of at most ``max_workers`` processes
                - :class:`distributed.client.Client`: use a Dask client
                - ``sequential``/``seq``: open data sequentially in a simple loop, no parallelization applied
                - ``erddap``: provides a detailed progress bar for erddap URLs, otherwise based on a :class:`concurrent.futures.ThreadPoolExecutor` with a pool of at most ``max_workers``
        progress: bool, default: False
            Display a progress bar
        concat: bool, default: True
            Concatenate results in a single :class:`xarray.Dataset` or not (in this case, function will return a
            list of :class:`xarray.Dataset`)
        concat_dim: str, default: ``row``
            Name of the dimension to use to concatenate all datasets (passed to :func:`xarray.concat`)
        preprocess: :class:`collections.abc.Callable` (optional)
            If provided, call this function on each dataset prior to concatenation
        preprocess_opts: dict (optional)
            Options passed to the ``preprocess`` :class:`collections.abc.Callable`, if any.
        errors: str, default: ``ignore``
            Define how to handle errors raised during data URIs fetching:
                - ``ignore`` (default): Do not stop processing, simply issue a debug message in logging console
                - ``raise``: Raise any error encountered
                - ``silent``:  Do not stop processing and do not issue log message

        Returns
        -------
        :class:`xarray.Dataset` or list of :class:`xarray.Dataset`

        See Also
        --------
        :class:`httpstore.open_dataset`

        Notes
        -----
        For the :class:`distributed.client.Client` and :class:`concurrent.futures.ProcessPoolExecutor` to work appropriately, the pre-processing :class:`collections.abc.Callable` must be serializable. This can be checked with:

        >>> from distributed.protocol import serialize
        >>> from distributed.protocol.serialize import ToPickle
        >>> serialize(ToPickle(preprocess_function))
        """
        strUrl = lambda x: x.replace("https://", "").replace(  # noqa: E731
            "http://", ""
        )

        if not isinstance(urls, list):
            urls = [urls]

        urls = [self.curateurl(url) for url in urls]

        if "lazy" in open_dataset_opts and open_dataset_opts["lazy"] and concat:
            warnings.warn(
                "Lazy opening and concatenate multiple netcdf files is not yet supported. Ignoring the 'lazy' option."
            )
            open_dataset_opts["lazy"] = False

        results = []
        failed = []

        ################################
        if method == "erddap":
            return self._open_mfdataset_from_erddap(
                urls=urls,
                concat_dim=concat_dim,
                max_workers=max_workers,
                preprocess=preprocess,
                preprocess_opts=preprocess_opts,
                concat=concat,
                progress=progress,
                compute_details=compute_details,
                *args,
                **kwargs,
            )

        ################################
        elif method == "thread":
            ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(
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

        ################################
        elif method == "process":
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

        ################################
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
                    )
                    results = method.gather(futures)
            else:
                futures = method.map(
                    self._mfprocessor_dataset,
                    urls,
                    preprocess=preprocess,
                    preprocess_opts=preprocess_opts,
                    open_dataset_opts=open_dataset_opts,
                )
                results = method.gather(futures)

        ################################
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

        ################################
        else:
            raise InvalidMethod(method)

        ################################
        # Post-process results
        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            if concat:
                # ds = xr.concat(results, dim=concat_dim, data_vars='all', coords='all', compat='override')
                if concat_method == "drop":
                    results = drop_variables_not_in_all_datasets(results)
                elif concat_method == "fill":
                    results = fill_variables_not_in_all_datasets(results)
                ds = xr.concat(
                    results,
                    dim=concat_dim,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                )
                if not compute_details:
                    return ds
                else:
                    return ds, failed, len(results)
            else:
                return results
        elif len(failed) == len(urls):
            raise ValueError(
                "Errors happened with all URLs, this could be due to an internal impossibility to read returned content."
            )
        else:
            raise DataNotFound(urls)

    def read_csv(self, url, **kwargs):
        """Read a comma-separated values (csv) url into Pandas DataFrame.

        Parameters
        ----------
        url: str
        **kwargs: Arguments passed to :class:`pandas.read_csv`

        Returns
        -------
        :class:`pandas.DataFrame`

        """
        url = self.curateurl(url)
        # log.debug("Opening/reading csv from: %s" % url)
        with self.open(url) as of:
            df = pd.read_csv(of, **kwargs)

        self.register(url)
        return df

    def open_json(self, url: str, errors: Literal['raise', 'silent', 'ignore'] = 'raise', **kwargs) -> Any:
        """Download and process a json document from an url

        Steps performed:

        1. Download from ``url`` raw data with :class:`httpstore.download_url` and then
        2. Create a JSON with :func:`json.loads`.

        Each steps can be passed specifics arguments (see Parameters below).

        Parameters
        ----------
        url: str
            Path to resources passed to :class:`httpstore.download_url`
        errors: str, default: ``raise``
            Define how to handle errors:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console and return None
                - ``silent``:  Do not stop processing and do not issue log message, return None

        kwargs: dict

            - ``dwn_opts`` key dictionary is passed to :class:`httpstore.download_url`
            - ``js_opts`` key dictionary is passed to :func:`json.loads`

        Returns
        -------
        Any

        See Also
        --------
        :class:`httpstore.open_mfjson`
        """
        dwn_opts = {}
        if "dwn_opts" in kwargs:
            dwn_opts.update(kwargs["dwn_opts"])
        data = self.download_url(url, **dwn_opts)

        if len(data) == 0:
            if errors == "raise":
                raise DataNotFound("No data return by %s" % url)

            elif errors == "ignore":
                log.debug("No data return by %s" % url)
                return None

            else:
                return None

        js_opts = {}
        if "js_opts" in kwargs:
            js_opts.update(kwargs["js_opts"])
        js = json.loads(data, **js_opts)
        if len(js) == 0:
            if errors == "raise":
                raise DataNotFound("No data loadable from %s, although the url return some data: '%s'" % (url, data))

            elif errors == "ignore":
                log.debug("No data loaded from %s, although the url return some data" % url)
                return None

            else:
                return None

        return js

    def _mfprocessor_json(
        self,
        url,
        open_json_opts: dict = {},
        preprocess: Callable = None,
        preprocess_opts: dict = {},
        url_follow: bool = False,
        *args,
        **kwargs,
    ):
        """Single URL json processor

        Internal method sent to a worker by :class:`httpstore.open_mfjson` and responsible for dealing with a single URL.

        1. Open the json with :class:`httpstore.open_json`
        2. Pre-process the json with the ``preprocess`` function given in arguments

        Parameters
        ----------
        url: str
            URI to process
        open_json_opts: dict, default: {}
            Set of arguments passed to :class:`httpstore.open_json`
        preprocess: :class:`collections.abc.Callable`, default: None
            Pre-processing function
        preprocess_opts: dict, default: {}
            Options to be passed to the pre-processing function

        Returns
        -------
        Anything as returned by the ``preprocess`` :class:`collections.abc.Callable`
        """
        # Load data
        data = self.open_json(url, **open_json_opts)

        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(
            preprocess, types.MethodType
        ):
            if url_follow:
                data = preprocess(data, url=url, **preprocess_opts)
            else:
                data = preprocess(data, **preprocess_opts)
        return data

    def open_mfjson(
        self,  # noqa: C901
        urls,
        max_workers: int = 6,
        method: str = "thread",
        progress: Union[bool, str] = False,
        preprocess=None,
        preprocess_opts={},
        open_json_opts={},
        url_follow=False,
        errors: str = "ignore",
        *args,
        **kwargs,
    ):
        """Download and process a collection of JSON documents from urls

        This is a version of the :class:`httpstore.open_json` method that is able to
        handle a list of urls sequentially or in parallel.

        This method uses a :class:`concurrent.futures.ThreadPoolExecutor` by default for parallelization. See
        ``method`` parameters below for more options.

        Parameters
        ----------
        urls: list(str)
        max_workers: int
            Maximum number of threads or processes.
        method: str, default: ``thread``
            Define the parallelization method:
                - ``thread`` (default): based on :class:`concurrent.futures.ThreadPoolExecutor` with a pool of at most ``max_workers`` threads
                - ``process``: based on :class:`concurrent.futures.ProcessPoolExecutor` with a pool of at most ``max_workers`` processes
                - :class:`distributed.client.Client`: use a Dask client
                - ``sequential``/``seq``: open data sequentially in a simple loop, no parallelization applied
        progress: bool, default: False
            Display a progress bar if possible
        preprocess: :class:`collections.abc.Callable` (optional)
            If provided, call this function on each dataset prior to concatenation
        preprocess_opts: dict (optional)
            Options passed to the ``preprocess`` :class:`collections.abc.Callable`, if any.
        url_follow: bool, False
            Follow the URL to the preprocess method as ``url`` argument.
        errors: str, default: ``ignore``
            Define how to handle errors raised during data URIs fetching:
                - ``ignore`` (default): Do not stop processing, simply issue a debug message in logging console
                - ``raise``: Raise any error encountered
                - ``silent``:  Do not stop processing and do not issue log message

        Returns
        -------
        list()

        Notes
        -----
        For the :class:`distributed.client.Client` and :class:`concurrent.futures.ProcessPoolExecutor` to work appropriately, the pre-processing :class:`collections.abc.Callable` must be serializable. This can be checked with:

        >>> from distributed.protocol import serialize
        >>> from distributed.protocol.serialize import ToPickle
        >>> serialize(ToPickle(preprocess_function))
        """
        strUrl = lambda x: x.replace("https://", "").replace(  # noqa: E731
            "http://", ""
        )

        if not isinstance(urls, list):
            urls = [urls]

        urls = [self.curateurl(url) for url in urls]

        results = []
        failed = []
        ################################
        if method == "thread":
            ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            )

            with ConcurrentExecutor as executor:
                future_to_url = {
                    executor.submit(
                        self._mfprocessor_json,
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_json_opts=open_json_opts,
                        url_follow=url_follow,
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

        ################################
        elif method == "process":
            if max_workers == 6:
                max_workers = multiprocessing.cpu_count()
            ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            )

            with ConcurrentExecutor as executor:
                future_to_url = {
                    executor.submit(
                        self._mfprocessor_json,
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_json_opts=open_json_opts,
                        url_follow=url_follow,
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

        ################################
        elif has_distributed and isinstance(method, distributed.client.Client):

            if progress:
                from dask.diagnostics import ProgressBar

                with ProgressBar():
                    futures = method.map(
                        self._mfprocessor_json,
                        urls,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_json_opts=open_json_opts,
                        url_follow=url_follow,
                        *args,
                        **kwargs,
                    )
                    results = method.gather(futures)
            else:
                futures = method.map(
                    self._mfprocessor_json,
                    urls,
                    preprocess=preprocess,
                    preprocess_opts=preprocess_opts,
                    open_json_opts=open_json_opts,
                    url_follow=url_follow,
                    *args,
                    **kwargs,
                )
                results = method.gather(futures)

        ################################
        elif method in ["seq", "sequential"]:
            if progress:
                # log.debug("We asked for a progress bar !")
                urls = tqdm(urls, total=len(urls), disable="disable" in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_json(
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_json_opts=open_json_opts,
                        url_follow=url_follow,
                        *args,
                        **kwargs,
                    )
                except Exception:
                    failed.append(url)
                    if errors == "ignore":
                        log.debug("Ignored error with this url: %s" % strUrl(url))
                        # See fsspec.http logger for more
                        pass
                    elif errors == "silent":
                        pass
                    else:
                        raise
                finally:
                    results.append(data)

        ################################
        else:
            raise InvalidMethod(method)

        ################################
        # Post-process results
        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            return results
        else:
            raise DataNotFound(urls)
