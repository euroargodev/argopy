"""
Argo specific layer on top of fsspec
This allows for file/memory/http/ftp stores with Argo dedicated methods

fs = filestore()
fs = memorystore()
fs = httpstore()
fs = ftpstore()
fs = httpstore_erddap_auth(payload = {"user": None, "password": None})

fs.open_dataset
fs.open_json

fs.open_mfdataset
fs.open_mfjson

fs.read_csv

"""

import os
import types
import warnings

import xarray as xr
import pandas as pd
import fsspec
import aiohttp
import shutil
import pickle  # nosec B403 only used with internal files/assets
import json
import io
from pathlib import Path
import time
import tempfile
import logging
from packaging import version
from typing import Union, Any, List
from collections.abc import Callable
from urllib.parse import urlparse, parse_qs
from functools import lru_cache
from abc import ABC, abstractmethod
import concurrent.futures
import multiprocessing

from ..options import OPTIONS
from ..errors import (
    FileSystemHasNoCache,
    CacheFileNotFound,
    DataNotFound,
    InvalidMethod,
    ErddapHTTPUnauthorized,
    ErddapHTTPNotFound,
)
from ..utils.transform import (
    drop_variables_not_in_all_datasets,
    fill_variables_not_in_all_datasets,
)
from ..utils.monitored_threadpool import MyThreadPoolExecutor as MyExecutor
from ..utils.accessories import Registry
from ..utils.format import UriCName


log = logging.getLogger("argopy.stores")

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    log.debug("argopy needs tqdm installed to display progress bars")

    def tqdm(fct, **kw):
        return fct


try:
    import distributed

    has_distributed = True
except ModuleNotFoundError:
    log.debug("argopy needs distributed to use Dask cluster/client")
    has_distributed = False
    distributed = None


def new_fs(
    protocol: str = "",
    cache: bool = False,
    cachedir: str = OPTIONS["cachedir"],
    cache_expiration: int = OPTIONS["cache_expiration"],
    **kwargs,
):
    """Create a new fsspec file system

    Parameters
    ----------
    protocol: str (optional)
    cache: bool (optional)
        Use a filecache system on top of the protocol. Default: False
    cachedir: str
        Define path to cache directory.
    **kwargs: (optional)
        Other arguments passed to :func:`fsspec.filesystem`

    """
    # Merge default FSSPEC kwargs with user defined kwargs:
    default_fsspec_kwargs = {"simple_links": True, "block_size": 0}
    if protocol == "http":
        client_kwargs = {
            "trust_env": OPTIONS["trust_env"]
        }  # Passed to aiohttp.ClientSession
        if "client_kwargs" in kwargs:
            client_kwargs = {**client_kwargs, **kwargs["client_kwargs"]}
            kwargs.pop("client_kwargs")
        default_fsspec_kwargs = {
            **default_fsspec_kwargs,
            **{"client_kwargs": {**client_kwargs}},
        }
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    elif protocol == "ftp":
        default_fsspec_kwargs = {
            **default_fsspec_kwargs,
            **{"block_size": 1000 * (2**20)},
        }
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    elif protocol == "s3":
        default_fsspec_kwargs.pop("simple_links")
        default_fsspec_kwargs.pop("block_size")
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    else:
        # Merge default with user arguments:
        fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    # Create filesystem:
    if not cache:
        fs = fsspec.filesystem(protocol, **fsspec_kwargs)
        cache_registry = None
        log_msg = (
            "Opening a fsspec [file] system for '%s' protocol with options: %s"
            % (protocol, str(fsspec_kwargs))
        )
    else:
        # https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/cached.html#WholeFileCacheFileSystem
        fs = fsspec.filesystem(
            "filecache",
            target_protocol=protocol,
            target_options={**fsspec_kwargs},
            cache_storage=cachedir,
            expiry_time=cache_expiration,
            cache_check=10,
        )

        cache_registry = Registry(
            name="Cache"
        )  # Will hold uri cached by this store instance
        log_msg = (
            "Opening a fsspec [filecache, storage='%s'] system for '%s' protocol with options: %s"
            % (cachedir, protocol, str(fsspec_kwargs))
        )

    if protocol == "file" and os.path.sep != fs.sep:
        # For some reason (see https://github.com/fsspec/filesystem_spec/issues/937), the property fs.sep is
        # not '\' under Windows. So, using this dirty fix to overwrite it:
        fs.sep = os.path.sep
        # fsspec folks recommend to use posix internally. But I don't see how to handle this. So keeping this fix
        # because it solves issues with failing tests under Windows. Enough at this time.
        # todo: Revisit this choice in a while

    # log_msg = "%s\n[sys sep=%s] vs [fs sep=%s]" % (log_msg, os.path.sep, fs.sep)
    # log.warning(log_msg)
    log.debug(log_msg)
    # log_argopy_callerstack()
    return fs, cache_registry


class argo_store_proto(ABC):
    """Argo Abstract File System

    Provide a prototype for Argo file systems

    Should this class inherits from :class:`fsspec.spec.AbstractFileSystem` ?
    """

    protocol = ""
    """str: File system name, one in fsspec.registry.known_implementations"""

    def __init__(self, cache: bool = False, cachedir: str = "", **kwargs):
        """Create a file storage system for Argo data

        Parameters
        ----------
        cache: bool (False)
        cachedir: str (from OPTIONS)
        **kwargs: (optional)
            Other arguments are passed to :func:`fsspec.filesystem`

        """
        self.cache = cache
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self._fsspec_kwargs = {**kwargs}
        self.fs, self.cache_registry = new_fs(
            self.protocol, self.cache, self.cachedir, **self._fsspec_kwargs
        )

    def open(self, path, *args, **kwargs):
        self.register(path)
        # log.debug("Opening path: %s" % path)
        return self.fs.open(path, *args, **kwargs)

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def exists(self, path, *args):
        return self.fs.exists(path, *args)

    def info(self, path, *args, **kwargs):
        return self.fs.info(path, *args, **kwargs)

    def expand_path(self, path):
        if self.protocol != "http" and self.protocol != "https":
            return self.fs.expand_path(path)
        else:
            return [path]

    def store_path(self, uri):
        path = uri
        path = self.expand_path(path)[0]
        if not path.startswith(self.fs.target_protocol) and version.parse(
            fsspec.__version__
        ) <= version.parse("0.8.3"):
            path = self.fs.target_protocol + "://" + path
        return path

    def register(self, uri):
        """Keep track of files open with this instance"""
        if self.cache:
            path = self.store_path(uri)
            if path not in self.cache_registry:
                self.cache_registry.commit(path)

    @property
    def cached_files(self):
        # See https://github.com/euroargodev/argopy/issues/294
        if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
            return self.fs.cached_files
        else:
            return self.fs._metadata.cached_files

    def cachepath(self, uri: str, errors: str = "raise"):
        """Return path to cached file for a given URI"""
        if not self.cache:
            if errors == "raise":
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        elif uri is not None:
            store_path = self.store_path(uri)
            self.fs.load_cache()  # Read set of stored blocks from file and populate self.fs.cached_files
            if store_path in self.cached_files[-1]:
                return os.path.sep.join(
                    [self.cachedir, self.cached_files[-1][store_path]["fn"]]
                )
            elif errors == "raise":
                raise CacheFileNotFound(
                    "No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri)
                )
        else:
            raise CacheFileNotFound(
                "No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri)
            )

    def _clear_cache_item(self, uri):
        """Remove metadata and file for fsspec cache uri"""
        fn = os.path.join(self.fs.storage[-1], "cache")
        self.fs.load_cache()  # Read set of stored blocks from file and populate self.cached_files
        cache = self.cached_files[-1]

        # Read cache metadata:
        if os.path.exists(fn):
            if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
                with open(fn, "rb") as f:
                    cached_files = pickle.load(
                        f
                    )  # nosec B301 because files controlled internally
            else:
                with open(fn, "r") as f:
                    cached_files = json.load(f)
        else:
            cached_files = cache

        # Build new metadata without uri to delete, and delete corresponding cached file:
        cache = {}
        for k, v in cached_files.items():
            if k != uri:
                cache[k] = v.copy()
            else:
                # Delete file:
                os.remove(os.path.join(self.fs.storage[-1], v["fn"]))
                # log.debug("Removed %s -> %s" % (uri, v['fn']))

        # Update cache metadata file:
        if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
            with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
                pickle.dump(cache, f)
            shutil.move(f.name, fn)
        else:
            with fsspec.utils.atomic_write(fn, mode="w") as f:
                json.dump(cache, f)

    def clear_cache(self):
        """Remove cache files and entry from uri open with this store instance"""
        if self.cache:
            for uri in self.cache_registry:
                # log.debug("Removing from cache %s" % uri)
                self._clear_cache_item(uri)
            self.cache_registry.clear()  # Reset registry

    @abstractmethod
    def open_dataset(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_csv(self):
        raise NotImplementedError("Not implemented")


class filestore(argo_store_proto):
    """Argo local file system

    Relies on :class:`fsspec.implementations.local.LocalFileSystem`
    """

    protocol = "file"

    def open_json(self, url, **kwargs):
        """Return a json from a path, or verbose errors

        Parameters
        ----------
        path: str
            Path to resources passed to :func:`json.loads`
        *args, **kwargs:
            Other arguments passed to :func:`json.loads`

        Returns
        -------
        json

        """
        with self.open(url) as of:
            js = json.load(of, **kwargs)
        if len(js) == 0:
            js = None
        return js

    def open_dataset(self, path, *args, **kwargs):
        """Return a xarray.dataset from a path.

        Parameters
        ----------
        path: str
            Path to resources passed to xarray.open_dataset
        *args, **kwargs:
            Other arguments are passed to :func:`xarray.open_dataset`

        Returns
        -------
        :class:`xarray.DataSet`
        """
        xr_opts = {}
        if "xr_opts" in kwargs:
            xr_opts.update(kwargs["xr_opts"])

        with self.open(path) as of:
            # log.debug("Opening dataset: '%s'" % path)  # Redundant with fsspec logger
            ds = xr.open_dataset(of, *args, **xr_opts)
            ds.load()
        if "source" not in ds.encoding:
            if isinstance(path, str):
                ds.encoding["source"] = path
        return ds.copy()

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


class memorystore(filestore):
    """Argo in-memory file system (global)

    Note that this inherits from :class:`argopy.stores.filestore`, not the:class:`argopy.stores.argo_store_proto`.

    Relies on :class:`fsspec.implementations.memory.MemoryFileSystem`
    """

    protocol = "memory"

    def exists(self, path, *args):
        """Check if path can be open or not

        Special handling for memory store

        The fsspec.exists() will return False even if the path is in cache.
        Here we bypass this in order to return True if the path is in cache.
        This assumes that the goal of fs.exists is to determine if we can load the path or not.
        If the path is in cache, it can be loaded.
        """
        guess = self.fs.exists(path, *args)
        if not guess:
            try:
                self.cachepath(path)
                return True
            except CacheFileNotFound:
                pass
            except FileSystemHasNoCache:
                pass
        return guess


class httpstore(argo_store_proto):
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

        return data

    def open_dataset(
        self, url, errors: str = "raise", lazy: bool = False, dwn_opts: dict = {}, xr_opts: dict = {}, **kwargs
    ) -> xr.Dataset:
        """Create a :class:`xarray.Dataset` from an url pointing to a netcdf file

        Parameters
        ----------
        url: str
            The remote URL of the netcdf file to open

        errors: str, default: ``raise``
            Define how to handle errors raised during data fetching:
                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message

        lazy: bool, default=False
            Define if we should try to load netcdf file lazily or not.

            **If this is set to False (default)** opening is done in 2 steps:
                1. Download from ``url`` raw binary data with :class:`httpstore.download_url`,
                2. Create a :class:`xarray.Dataset` with :func:`xarray.open_dataset`.

            Each functions can be passed specifics arguments with ``dwn_opts`` and  ``xr_opts`` (see below).

            **If this is set to True**, we'll try to use a :class:`argopy.stores.ArgoKerchunker` to access
            the netcdf file using zarr data from it.

        dwn_opts: dict, default={}
             Options passed to :func:`httpstore.download_url` if not in lazy mode.

        xr_opts: dict, default={}
             Options passed to :func:`xarray.open_dataset` if not in lazy mode.

        Returns
        -------
        :class:`xarray.Dataset`

        Raises
        ------
        :class:`TypeError` if data returned by ``url`` are not CDF or HDF5 binary data.

        :class:`DataNotFound` if ``errors`` is set to ``raise`` and url returns no data.

        See Also
        --------
        :func:`httpstore.open_mfdataset`
        """
        def load_in_memory(url, errors, dwn_opts, xr_opts):
            data = self.download_url(url, **dwn_opts)
            if data is None:
                if errors == "raise":
                    raise DataNotFound(url)
                elif errors == "ignore":
                    log.error("DataNotFound: %s" % url)
                return None

            if b"Not Found: Your query produced no matching results" in data:
                if errors == "raise":
                    raise DataNotFound(url)
                elif errors == "ignore":
                    log.error("DataNotFound from [%s]: %s" % (url, data))
                return None

            if data[0:3] != b"CDF" and data[0:3] != b"\x89HD":
                raise TypeError(
                    "We didn't get a CDF or HDF5 binary data as expected ! We get: %s"
                    % data
                )
            if data[0:3] == b"\x89HD":
                data = io.BytesIO(data)

            return data, xr_opts

        def load_lazily(url, errors, dwn_opts, xr_opts):
            from . import ArgoKerchunker

            if "ak" not in kwargs:
                self.ak = ArgoKerchunker(
                    store="local", root=Path(OPTIONS["cachedir"]).joinpath("kerchunk")
                )
            else:
                self.ak = kwargs["ak"]

            if self.ak.supported(url):
                xr_opts = {
                    "engine": "zarr",
                    "backend_kwargs": {
                        "consolidated": False,
                        "storage_options": {
                            "fo": self.ak.to_kerchunk(url),
                            "remote_protocol": fsspec.core.split_protocol(url)[0],
                        },
                    },
                }
                return "reference://", xr_opts
            else:
                warnings.warn(
                    "This url does not support byte range requests so we cannot load lazily, hence falling back on loading in memory"
                )
                return load_in_memory(url, errors=errors, dwn_opts=dwn_opts, xr_opts=xr_opts)

        if not lazy:
            target = load_in_memory(url, errors=errors, dwn_opts=dwn_opts, xr_opts=xr_opts)
        else:
            target, xr_opts = load_lazily(url, errors=errors, dwn_opts=dwn_opts, xr_opts=xr_opts)

        ds = xr.open_dataset(target, **xr_opts)

        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url

        self.register(url)
        return ds

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
        preprocess: Callable = None,
        preprocess_opts: dict = {},
        open_dataset_opts: dict = {},
        errors: str = "ignore",
        compute_details: bool = False,
        *args,
        **kwargs,
    ) -> Union[xr.Dataset, List[xr.Dataset]]:
        """Download and process multiple urls as a single or a collection of :class:`xarray.Dataset`

        This is a version of the :class:`httpstore.open_dataset` method that is able to
        handle a list of urls sequentially or in parallel.

        This method uses a :class:`concurrent.futures.ThreadPoolExecutor` by default for parallelization. See
        ``method`` parameters below for more options.

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
                results = drop_variables_not_in_all_datasets(results)
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
        return df

    def open_json(self, url: str, **kwargs) -> Any:
        """Download and process a json document from an url

        Steps performed:

        1. Download from ``url`` raw data with :class:`httpstore.download_url` and then
        2. Create a JSON with :func:`json.loads`.

        Each functions can be passed specifics arguments (see Parameters below).

        Parameters
        ----------
        url: str
        kwargs: dict

            - ``dwn_opts`` key is passed to :class:`httpstore.download_url`
            - ``js_opts`` key is passed to :func:`json.loads`

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

        js_opts = {}
        if "js_opts" in kwargs:
            js_opts.update(kwargs["js_opts"])
        js = json.loads(data, **js_opts)
        if len(js) == 0:
            js = None

        self.register(url)
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


class ftpstore(httpstore):
    """Argo ftp file system

    Relies on :class:`fsspec.implementations.ftp.FTPFileSystem`
    """

    protocol = "ftp"

    def open_dataset(self, url, *args, **kwargs):
        """Open and decode a xarray dataset from an ftp url

        Parameters
        ----------
        url: str

        Returns
        -------
        :class:`xarray.Dataset`
        """
        try:
            this_url = self.fs._strip_protocol(url)
            data = self.fs.cat_file(this_url)
        except Exception:
            log.debug("Error with: %s" % url)
            # except aiohttp.ClientResponseError as e:
            raise

        xr_opts = {}
        if "xr_opts" in kwargs:
            xr_opts.update(kwargs["xr_opts"])
        ds = xr.open_dataset(data, *args, **xr_opts)

        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url
        self.register(this_url)
        self.register(url)
        return ds

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


class httpstore_erddap_auth(httpstore):
    async def get_auth_client(self, **kwargs):
        session = aiohttp.ClientSession(**kwargs)

        async with session.post(self._login_page, data=self._login_payload) as resp:
            resp_query = dict(parse_qs(urlparse(str(resp.url)).query))

            if resp.status == 404:
                raise ErddapHTTPNotFound(
                    "Error %s: %s. This erddap server does not support log-in"
                    % (resp.status, resp.reason)
                )

            elif resp.status == 200:
                has_expected = (
                    "message" in resp_query
                )  # only available when there is a form page response
                if has_expected:
                    message = resp_query["message"][0]
                    if "failed" in message:
                        raise ErddapHTTPUnauthorized(
                            "Error %i: %s (%s)" % (401, message, self._login_payload)
                        )
                else:
                    raise ErddapHTTPUnauthorized(
                        "This erddap server does not support log-in with a user/password"
                    )

            else:
                log.debug("resp.status", resp.status)
                log.debug("resp.reason", resp.reason)
                log.debug("resp.headers", resp.headers)
                log.debug("resp.url", urlparse(str(resp.url)))
                log.debug("resp.url.query", resp_query)
                data = await resp.read()
                log.debug("data", data)

        return session

    def __init__(
        self,
        cache: bool = False,
        cachedir: str = "",
        login: str = None,
        payload: dict = {"user": None, "password": None},
        auto: bool = True,
        **kwargs,
    ):
        if login is None:
            raise ValueError("Invalid login url")
        else:
            self._login_page = login

        self._login_auto = (
            auto  # Should we try to log-in automatically at instantiation ?
        )

        self._login_payload = payload.copy()
        if "user" in self._login_payload and self._login_payload["user"] is None:
            self._login_payload["user"] = OPTIONS["user"]
        if (
            "password" in self._login_payload
            and self._login_payload["password"] is None
        ):
            self._login_payload["password"] = OPTIONS["password"]

        fsspec_kwargs = {**kwargs, **{"get_client": self.get_auth_client}}
        super().__init__(cache=cache, cachedir=cachedir, **fsspec_kwargs)

        if auto:
            assert isinstance(self.connect(), bool)

    # def __repr__(self):
    #     # summary = ["<httpstore_erddap_auth.%i>" % id(self)]
    #     summary = ["<httpstore_erddap_auth>"]
    #     summary.append("login page: %s" % self._login_page)
    #     summary.append("login data: %s" % (self._login_payload))
    #     if hasattr(self, '_connected'):
    #         summary.append("connected: %s" % (self._connected))
    #     else:
    #         summary.append("connected: ?")
    #     return "\n".join(summary)

    def _repr_html_(self):
        td_title = (
            lambda title: '<td colspan="2"><div style="vertical-align: middle;text-align:left"><strong>%s</strong></div></td>'
            % title
        )  # noqa: E731
        tr_title = lambda title: "<thead><tr>%s</tr></thead>" % td_title(  # noqa: E731
            title
        )
        a_link = lambda url, txt: '<a href="%s">%s</a>' % (url, txt)  # noqa: E731
        td_key = (  # noqa: E731
            lambda prop: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>'
            % str(prop)
        )
        td_val = (
            lambda label: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>'
            % str(label)
        )  # noqa: E731
        tr_tick = lambda key, value: "<tr>%s%s</tr>" % (  # noqa: E731
            td_key(key),
            td_val(value),
        )
        td_vallink = (
            lambda url, label: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>'
            % a_link(url, label)
        )
        tr_ticklink = lambda key, url, value: "<tr>%s%s</tr>" % (  # noqa: E731
            td_key(key),
            td_vallink(url, value),
        )

        html = []
        html.append("<table style='border-collapse:collapse;border-spacing:0'>")
        html.append("<thead>")
        html.append(tr_title("httpstore_erddap_auth"))
        html.append("</thead>")
        html.append("<tbody>")
        html.append(tr_ticklink("login page", self._login_page, self._login_page))
        payload = self._login_payload.copy()
        payload["password"] = "*" * len(payload["password"])
        html.append(tr_tick("login data", payload))
        if hasattr(self, "_connected"):
            html.append(tr_tick("connected", "✅" if self._connected else "⛔"))
        else:
            html.append(tr_tick("connected", "?"))
        html.append("</tbody>")
        html.append("</table>")

        html = "\n".join(html)
        return html

    def connect(self):
        try:
            payload = self._login_payload.copy()
            payload["password"] = "*" * len(payload["password"])
            log.info("Try to log-in to '%s' page with %s" % (self._login_page, payload))
            self.fs.info(self._login_page)
            self._connected = True
        except ErddapHTTPUnauthorized:
            self._connected = False
        except:  # noqa: E722
            raise
        return self._connected

    @property
    def connected(self):
        if not hasattr(self, "_connected"):
            self.connect()
        return self._connected


def httpstore_erddap(url: str = "", cache: bool = False, cachedir: str = "", **kwargs):
    erddap = OPTIONS["erddap"] if url == "" else url
    login_page = "%s/login.html" % erddap.rstrip("/")
    login_store = httpstore_erddap_auth(
        cache=cache, cachedir=cachedir, login=login_page, auto=False, **kwargs
    )
    try:
        login_store.connect()
        keep = True
    except ErddapHTTPNotFound:
        keep = False
        pass

    if keep:
        return login_store
    else:
        return httpstore(cache=cache, cachedir=cachedir, **kwargs)


class s3store(httpstore):
    """Argo s3 file system

    Inherits from :class:`httpstore` but will rely on :class:`s3fs.S3FileSystem` through
    the fsspec 's3' protocol specification.

    By default, this store will use AWS credentials available in the environment.

    If you want to force an anonymous session, you should use the `anon=True` option.

    In order to avoid a *no credentials found error*, you can use:

    >>> from argopy.utils import has_aws_credentials
    >>> fs = s3store(anon=not has_aws_credentials())

    """

    protocol = "s3"
