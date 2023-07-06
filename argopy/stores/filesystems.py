import os
import types
import numpy as np
import xarray as xr
import pandas as pd
import fsspec
import aiohttp
import shutil
import pickle  # nosec B403 only used with internal files/assets
import json
import time
import tempfile
import warnings
import logging
from packaging import version
from typing import Union
from urllib.parse import urlparse, parse_qs

import concurrent.futures
import multiprocessing

from ..options import OPTIONS
from ..errors import FileSystemHasNoCache, CacheFileNotFound, DataNotFound, \
    InvalidMethod, ErddapHTTPUnauthorized, ErddapHTTPNotFound
from abc import ABC, abstractmethod
from ..utilities import Registry, log_argopy_callerstack, drop_variables_not_in_all_datasets

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


def new_fs(protocol: str = '',
           cache: bool = False,
           cachedir: str = OPTIONS['cachedir'],
           cache_expiration: int = OPTIONS['cache_expiration'],
           **kwargs):
    """ Create a new fsspec file system

    Parameters
    ----------
    protocol: str (optional)
    cache: bool (optional)
        Use a filecache system on top of the protocol. Default: False
    cachedir: str
        Define path to cache directory.
    **kwargs: (optional)
        Other arguments passed to :class:`fsspec.filesystem`

    """
    # Load default FSSPEC kwargs:
    default_fsspec_kwargs = {'simple_links': True, "block_size": 0}
    if protocol == 'http':
        default_fsspec_kwargs = {**default_fsspec_kwargs,
                                     **{"client_kwargs": {"trust_env": OPTIONS['trust_env']}}}
    elif protocol == 'ftp':
        default_fsspec_kwargs = {**default_fsspec_kwargs,
                                     **{"block_size": 1000 * (2 ** 20)}}
    # Merge default with user arguments:
    fsspec_kwargs = {**default_fsspec_kwargs, **kwargs}

    # Create filesystem:
    if not cache:
        fs = fsspec.filesystem(protocol, **fsspec_kwargs)
        cache_registry = None
        log_msg = "Opening a fsspec [file] system for '%s' protocol with options: %s" % \
                  (protocol, str(fsspec_kwargs))
    else:
        # https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/cached.html#WholeFileCacheFileSystem
        fs = fsspec.filesystem("filecache",
                               target_protocol=protocol,
                               target_options={**fsspec_kwargs},
                               cache_storage=cachedir,
                               expiry_time=cache_expiration, cache_check=10)

        # We use a refresh rate for cache of 1 day,
        # since this is the update frequency of the Ifremer erddap
        cache_registry = Registry(name='Cache')  # Will hold uri cached by this store instance
        log_msg = "Opening a fsspec [filecache, storage='%s'] system for '%s' protocol with options: %s" % \
                  (cachedir, protocol, str(fsspec_kwargs))

    if protocol == 'file' and os.path.sep != fs.sep:
        # For some reasons (see https://github.com/fsspec/filesystem_spec/issues/937), the property fs.sep is
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
    """ Argo Abstract File System

        Provide a prototype for Argo file systems

        Should this class inherits from fsspec.spec.AbstractFileSystem ?
    """
    protocol = ''
    """str: File system name, one in fsspec.registry.known_implementations"""

    def __init__(self,
                 cache: bool = False,
                 cachedir: str = "",
                 **kwargs):
        """ Create a file storage system for Argo data

            Parameters
            ----------
            cache: bool (False)
            cachedir: str (from OPTIONS)
            **kwargs: (optional)
                Other arguments passed to :class:`fsspec.filesystem`

        """
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        self._fsspec_kwargs = {**kwargs}
        self.fs, self.cache_registry = new_fs(self.protocol,
                                              self.cache,
                                              self.cachedir,
                                              **self._fsspec_kwargs)

    def open(self, path, *args, **kwargs):
        self.register(path)
        # log.debug("Opening path: %s" % path)
        return self.fs.open(path, *args, **kwargs)

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def exists(self, path, *args):
        return self.fs.exists(path, *args)

    def expand_path(self, path):
        if self.protocol != "http" and self.protocol != "https":
            return self.fs.expand_path(path)
        else:
            return [path]

    def store_path(self, uri):
        path = uri
        path = self.expand_path(path)[0]
        if not path.startswith(self.fs.target_protocol) and version.parse(fsspec.__version__) <= version.parse("0.8.3"):
            path = self.fs.target_protocol + "://" + path
        return path

    def register(self, uri):
        """ Keep track of files open with this instance """
        if self.cache:
            path = self.store_path(uri)
            if path not in self.cache_registry:
                self.cache_registry.commit(path)

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        elif uri is not None:
            store_path = self.store_path(uri)
            self.fs.load_cache()  # Read set of stored blocks from file and populate self.fs.cached_files
            if store_path in self.fs.cached_files[-1]:
                return os.path.sep.join([self.cachedir, self.fs.cached_files[-1][store_path]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))
        else:
            raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    def _clear_cache_item(self, uri):
        """ Open fsspec cache registry (pickle file) and remove entry for uri

        """
        # See the "save_cache()" method in:
        # https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/cached.html#WholeFileCacheFileSystem
        fn = os.path.join(self.fs.storage[-1], "cache")
        self.fs.load_cache()  # Read set of stored blocks from file and populate self.fs.cached_files
        cache = self.fs.cached_files[-1]
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                cached_files = pickle.load(f)  # nosec B301 because files controlled internally
        else:
            cached_files = cache
        cache = {}
        for k, v in cached_files.items():
            if k != uri:
                cache[k] = v.copy()
            else:
                os.remove(os.path.join(self.fs.storage[-1], v['fn']))
                # log.debug("Removed %s -> %s" % (uri, v['fn']))
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump(cache, f)
        shutil.move(f.name, fn)

    def clear_cache(self):
        """ Remove cache files and entry from uri open with this store instance """
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
    protocol = 'file'

    def open_json(self, url, **kwargs):
        """ Return a json from a path, or verbose errors

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
        with self.open(path) as of:
            # log.debug("Opening dataset: '%s'" % path)  # Redundant with fsspec logger
            ds = xr.open_dataset(of, *args, **kwargs)
            ds.load()
        if "source" not in ds.encoding:
            if isinstance(path, str):
                ds.encoding["source"] = path
        return ds.copy()

    def _mfprocessor(self, url, preprocess=None, *args, **kwargs):
        # Load data
        ds = self.open_dataset(url, *args, **kwargs)
        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(preprocess, types.MethodType):
            ds = preprocess(ds)
        return ds

    def open_mfdataset(self,  # noqa: C901
                       urls,
                       concat_dim='row',
                       max_workers: int = 6,
                       method: str = 'thread',
                       progress: bool = False,
                       concat: bool = True,
                       preprocess=None,
                       errors: str = 'ignore',
                       *args, **kwargs):
        """ Open multiple urls as a single xarray dataset.

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
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                if max_workers == 6:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor, url,
                                                 preprocess=preprocess, *args, **kwargs): url
                                 for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls), disable='disable' in [progress])

                for future in futures:
                    data = None
                    # url = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as e:
                        if errors == 'ignore':
                            log.debug(
                                "Ignored error with this file: %s\nException raised: %s"
                                % (future_to_url[future], str(e.args)))
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
                    futures = method.map(self._mfprocessor, urls, preprocess=preprocess, *args, **kwargs)
                    results = method.gather(futures)
            else:
                futures = method.map(self._mfprocessor, urls, preprocess=preprocess, *args, **kwargs)
                results = method.gather(futures)

        elif method in ['seq', 'sequential']:
            if progress:
                urls = tqdm(urls, total=len(urls), disable='disable' in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor(url, preprocess=preprocess, *args, **kwargs)
                except Exception as e:
                    if errors == 'ignore':
                        log.debug(
                            "Ignored error with this url: %s\nException raised: %s" % (url, str(e.args)))
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
                ds = xr.concat(results, dim=concat_dim, data_vars='minimal', coords='minimal', compat='override')
                return ds
            else:
                return results
        else:
            raise DataNotFound(urls)

    def read_csv(self, path, **kwargs):
        """ Return a pandas.dataframe from a path that is a csv resource

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
    """ Argo in-memory file system (global)

        Note that this inherits from :class:`argopy.stores.filestore`, not the:class:`argopy.stores.argo_store_proto`.

        Relies on :class:`fsspec.implementations.memory.MemoryFileSystem`
    """
    protocol = 'memory'

    def exists(self, path, *args):
        """ Check if path can be open or not

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

        This store intends to make argopy: safer to failures from http requests and to provide higher levels methods to
        work with our datasets

        This store is primarily used by the Erddap/Argovis data/index fetchers
    """
    protocol = "http"

    def curateurl(self, url):
        """Possibly replace server of a given url by a local argopy option value

        This is intended to be used by tests and dev
        """
        return url
        # if OPTIONS["server"] is not None:
        #     # log.debug("Replaced '%s' with '%s'" % (urlparse(url).netloc, OPTIONS["netloc"]))
        #
        #     if urlparse(url).scheme == "":
        #         patternA = "//%s" % (urlparse(url).netloc)
        #     else:
        #         patternA = "%s://%s" % (urlparse(url).scheme, urlparse(url).netloc)
        #
        #     patternB = "%s://%s" % (urlparse(OPTIONS["server"]).scheme, urlparse(OPTIONS["server"]).netloc)
        #     log.debug(patternA)
        #     log.debug(patternB)
        #
        #     new_url = url.replace(patternA, patternB)
        #     # log.debug(url)
        #     # log.debug(new_url)
        #     return new_url
        # else:
        #     # log.debug("'%s' left unchanged" % urlparse(url).netloc)
        #     log.debug(url)
        #     return url

    def download_url(self,
                     url,
                     n_attempt: int = 1,
                     max_attempt: int = 5,
                     *args,
                     **kwargs):

        def make_request(ffs, url, n_attempt: int = 1, max_attempt: int = 5, *args, **kwargs):
            data = None
            if n_attempt <= max_attempt:
                try:
                    data = ffs.cat_file(url, *args, **kwargs)
                except aiohttp.ClientResponseError as e:
                    if e.status == 413:
                        log.debug(f"Error %i (Payload Too Large) raised with %s" % (e.status, url))
                        raise
                    elif e.status == 429:
                        retry_after = int(e.headers.get('Retry-After', 5))
                        log.debug(
                            f"Error {e.status} (Too many requests). Retry after {retry_after} seconds. Tentative {n_attempt}/{max_attempt}")
                        time.sleep(retry_after)
                        n_attempt += 1
                        make_request(ffs, url, n_attempt=n_attempt, *args, **kwargs)
                    else:
                        # Handle other client response errors
                        print(f"Error: {e}")
                except aiohttp.ClientError as e:
                    # Handle other request exceptions
                    # print(f"Error: {e}")
                    raise
            else:
                raise ValueError(f"Error: All attempts failed to download this url: {url}")
            return data, n_attempt

        url = self.curateurl(url)
        data, n = make_request(self.fs, url, n_attempt=n_attempt, max_attempt=max_attempt, *args, **kwargs)

        return data

    def open_dataset(self, url, *args, **kwargs):
        """ Open and decode a xarray dataset from an url

        Parameters
        ----------
        url: str

        Returns
        -------
        :class:`xarray.Dataset`
        """
        data = self.download_url(url, *args, **kwargs)

        if data[0:3] != b'CDF':
            raise TypeError("We didn't get a CDF binary data as expected ! We get: %s" % data)

        ds = xr.open_dataset(data, *args, **kwargs)

        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url

        self.register(url)
        return ds

    def _mfprocessor_dataset(self, url, preprocess=None, preprocess_opts={}, *args, **kwargs):
        # Load data
        ds = self.open_dataset(url, *args, **kwargs)

        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(preprocess, types.MethodType):
            ds = preprocess(ds, **preprocess_opts)
        return ds

    def open_mfdataset(self,  # noqa: C901
                       urls,
                       max_workers: int = 6,
                       method: str = 'thread',
                       progress: Union[bool, str] = False,
                       concat: bool = True,
                       concat_dim='row',
                       preprocess=None,
                       preprocess_opts={},
                       errors: str = 'ignore',
                       compute_details: bool = False,
                       *args,
                       **kwargs):
        """ Open multiple urls as a single xarray dataset.

            This is a version of the :class:`argopy.stores.httpstore.open_dataset` method that is able to
            handle a list of urls/paths sequentially or in parallel.

            Use a Threads Pool by default for parallelization.

            Parameters
            ----------
            urls: list(str)
                List of url/path to open
            max_workers: int, default: 6
                Maximum number of threads or processes
            method: str, default: ``thread``
                The parallelization method to execute calls asynchronously:

                    - ``thread`` (default): use a pool of at most ``max_workers`` threads
                    - ``process``: use a pool of at most ``max_workers`` processes
                    - :class:`distributed.client.Client`: Experimental, expect this method to fail !
                    - ``seq``: open data sequentially, no parallelization applied
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
        strUrl = lambda x: x.replace("https://", "").replace("http://", "")  # noqa: E731

        if not isinstance(urls, list):
            urls = [urls]

        urls = [self.curateurl(url) for url in urls]

        results = []
        failed = []
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                if max_workers == 6:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor_dataset,
                                                 url,
                                                 preprocess=preprocess,
                                                 preprocess_opts=preprocess_opts, *args, **kwargs): url
                                 for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls), disable='disable' in [progress])

                for future in futures:
                    data = None
                    try:
                        data = future.result()
                    except Exception:
                        failed.append(future_to_url[future])
                        if errors == 'ignore':
                            log.debug("Ignored error with this url: %s" % strUrl(future_to_url[future]))
                            # See fsspec.http logger for more
                            pass
                        elif errors == 'silent':
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
                    futures = method.map(self._mfprocessor_dataset,
                                         urls,
                                         preprocess=preprocess, *args, **kwargs)
                    results = method.gather(futures)
            else:
                futures = method.map(self._mfprocessor_dataset,
                                     urls,
                                     preprocess=preprocess, *args, **kwargs)
                results = method.gather(futures)

        elif method in ['seq', 'sequential']:
            if progress:
                urls = tqdm(urls, total=len(urls), disable='disable' in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_dataset(url,
                                                     preprocess=preprocess,
                                                     preprocess_opts=preprocess_opts,
                                                     *args,
                                                     **kwargs)
                except Exception:
                    failed.append(url)
                    if errors == 'ignore':
                        log.debug("Ignored error with this url: %s" % strUrl(url))  # See fsspec.http logger for more
                        pass
                    elif errors == 'silent':
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
                ds = xr.concat(results,
                               dim=concat_dim,
                               data_vars='minimal',
                               coords='minimal',
                               compat='override')
                if not compute_details:
                    return ds
                else:
                    return ds, failed, len(results)
            else:
                return results
        elif len(failed) == len(urls):
            raise ValueError("Errors happened with all URLs, this could be due to an internal impossibility to read returned content.")
        else:
            raise DataNotFound(urls)

    def read_csv(self, url, **kwargs):
        """ Read a comma-separated values (csv) url into Pandas DataFrame.

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

    def open_json(self, url, **kwargs):
        """ Return a json from an url, or verbose errors

            Parameters
            ----------
            url: str

            Returns
            -------
            json

        """
        data = self.download_url(url)
        js = json.loads(data, **kwargs)
        if len(js) == 0:
            js = None
        self.register(url)
        return js

    def _mfprocessor_json(self, url, preprocess=None, url_follow=False, *args, **kwargs):
        # Load data
        data = self.open_json(url, **kwargs)
        # Pre-process
        if data is None:
            raise DataNotFound(url)
        elif isinstance(preprocess, types.FunctionType) or isinstance(preprocess, types.MethodType):
            if url_follow:
                data = preprocess(data, url=url, **kwargs)
            else:
                data = preprocess(data)
        return data

    def open_mfjson(self,  # noqa: C901
                    urls,
                    max_workers: int = 6,
                    method: str = 'thread',
                    progress: Union[bool, str] = False,
                    preprocess=None,
                    url_follow=False,
                    errors: str = 'ignore',
                    *args, **kwargs):
        """ Open multiple json urls

            This is a parallelized version of ``open_json``.
            Use a Threads Pool by default for parallelization.

            Parameters
            ----------
            urls: list(str)
            max_workers: int
                Maximum number of threads or processes.
            method:
                The parallelization method to execute calls asynchronously:
                    - 'thread' (Default): use a pool of at most ``max_workers`` threads
                    - 'process': use a pool of at most ``max_workers`` processes
                    - (XFAIL) Dask client object: use a Dask distributed client object

                Use 'seq' to simply open data sequentially
            progress: bool
                Display a progress bar (True by default, not for dask client method)
            preprocess: (callable, optional)
                If provided, call this function on each json set
            url_follow: bool, False
                Follow the URL to the preprocess method as ``url`` argument.

            Returns
            -------
            list()
        """
        strUrl = lambda x: x.replace("https://", "").replace("http://", "")  # noqa: E731

        if not isinstance(urls, list):
            urls = [urls]

        urls = [self.curateurl(url) for url in urls]

        results = []
        failed = []
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                if max_workers == 6:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor_json, url,
                                                 preprocess=preprocess, url_follow=url_follow, *args, **kwargs): url for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls), disable='disable' in [progress])

                for future in futures:
                    data = None
                    try:
                        data = future.result()
                    except Exception:
                        failed.append(future_to_url[future])
                        if errors == 'ignore':
                            log.debug("Ignored error with this url: %s" % strUrl(future_to_url[future]))
                            # See fsspec.http logger for more
                            pass
                        elif errors == 'silent':
                            pass
                        else:
                            raise
                    finally:
                        results.append(data)

        # elif type(method) == distributed.client.Client:
        #     # Use a dask client:
        #     futures = method.map(self._mfprocessor_json, urls, preprocess=preprocess, *args, **kwargs)
        #     results = method.gather(futures)

        elif method in ['seq', 'sequential']:
            if progress:
                # log.debug("We asked for a progress bar !")
                urls = tqdm(urls, total=len(urls), disable='disable' in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_json(url, preprocess=preprocess, url_follow=url_follow, *args, **kwargs)
                except Exception:
                    failed.append(url)
                    if errors == 'ignore':
                        log.debug("Ignored error with this url: %s" % strUrl(url))
                        # See fsspec.http logger for more
                        pass
                    elif errors == 'silent':
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
            return results
        else:
            raise DataNotFound(urls)


class ftpstore(httpstore):
    """ Argo ftp file system

        Relies on :class:`fsspec.implementations.ftp.FTPFileSystem`
    """
    protocol = 'ftp'

    def open_dataset(self, url, *args, **kwargs):
        """ Open and decode a xarray dataset from an ftp url

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

        ds = xr.open_dataset(data, *args, **kwargs)
        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url
        self.register(this_url)
        self.register(url)
        return ds

    def _mfprocessor_dataset(self, url, preprocess=None, preprocess_opts={}, *args, **kwargs):
        # Load data
        ds = self.open_dataset(url, *args, **kwargs)
        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(preprocess, types.MethodType):
            ds = preprocess(ds, **preprocess_opts)
        return ds

    def open_mfdataset(self,  # noqa: C901
                       urls,
                       max_workers: int = 6,
                       method: str = 'seq',
                       progress: bool = False,
                       concat: bool = True,
                       concat_dim='row',
                       preprocess=None,
                       preprocess_opts={},
                       errors: str = 'ignore',
                       *args, **kwargs):
        """ Open multiple ftp urls as a single xarray dataset.

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
        if method in ['process']:
            if max_workers == 6:
                max_workers = multiprocessing.cpu_count()
            ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor_dataset, url,
                                                 preprocess=preprocess,
                                                 preprocess_opts=preprocess_opts, *args, **kwargs): url for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls), disable='disable' in [progress])

                for future in futures:
                    data = None
                    try:
                        data = future.result()
                    except Exception:
                        failed.append(future_to_url[future])
                        if errors == 'ignore':
                            log.debug("Ignored error with this url: %s" % strUrl(future_to_url[future]))
                            # See fsspec.http logger for more
                            pass
                        elif errors == 'silent':
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
                    futures = method.map(self._mfprocessor_dataset,
                                         urls,
                                         preprocess=preprocess,
                                         *args, **kwargs)
                    results = method.gather(futures)
            else:
                futures = method.map(self._mfprocessor_dataset,
                                     urls,
                                     preprocess=preprocess,
                                     *args, **kwargs)
                results = method.gather(futures)

        elif method in ['seq', 'sequential']:
            if progress:
                urls = tqdm(urls, total=len(urls), disable='disable' in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_dataset(url,
                                                     preprocess=preprocess,
                                                     preprocess_opts=preprocess_opts,
                                                     *args, **kwargs)
                except Exception:
                    failed.append(url)
                    if errors == 'ignore':
                        log.debug("Ignored error with this url: %s" % strUrl(url))  # See fsspec.http logger for more
                        pass
                    elif errors == 'silent':
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
                ds = xr.concat(results,
                               dim=concat_dim,
                               data_vars='minimal',
                               coords='minimal',
                               compat='override')
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
                    "Error %s: %s. This erddap server does not support log-in" % (resp.status, resp.reason))

            elif resp.status == 200:
                has_expected = 'message' in resp_query  # only available when there is a form page response
                if has_expected:
                    message = resp_query['message'][0]
                    if 'failed' in message:
                        raise ErddapHTTPUnauthorized("Error %i: %s (%s)" % (401, message, self._login_payload))
                else:
                    raise ErddapHTTPUnauthorized("This erddap server does not support log-in with a user/password")

            else:
                log.debug('resp.status', resp.status)
                log.debug('resp.reason', resp.reason)
                log.debug('resp.headers', resp.headers)
                log.debug('resp.url', urlparse(str(resp.url)))
                log.debug('resp.url.query', resp_query)
                data = await resp.read()
                log.debug('data', data)

        return session

    def __init__(self,
                 cache: bool = False,
                 cachedir: str = "",
                 login: str = None,
                 payload: dict = {"user": None, "password": None},
                 auto: bool = True,
                 **kwargs):

        if login is None:
            raise ValueError("Invalid login url")
        else:
            self._login_page = login

        self._login_auto = auto  # Should we try to log-in automatically at instantiation ?

        self._login_payload = payload.copy()
        if "user" in self._login_payload and self._login_payload['user'] is None:
            self._login_payload['user'] = OPTIONS['user']
        if "password" in self._login_payload and self._login_payload['password'] is None:
            self._login_payload['password'] = OPTIONS['password']

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
        td_title = lambda title: '<td colspan="2"><div style="vertical-align: middle;text-align:left"><strong>%s</strong></div></td>' % title  # noqa: E731
        tr_title = lambda title: "<thead><tr>%s</tr></thead>" % td_title(title)  # noqa: E731
        a_link = lambda url, txt: '<a href="%s">%s</a>' % (url, txt)
        td_key = lambda prop: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>' % str(prop)  # noqa: E731
        td_val = lambda label: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>' % str(label)  # noqa: E731
        tr_tick = lambda key, value: '<tr>%s%s</tr>' % (td_key(key), td_val(value))  # noqa: E731
        td_vallink = lambda url, label: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>' % a_link(url, label)  # noqa: E731
        tr_ticklink = lambda key, url, value: '<tr>%s%s</tr>' % (td_key(key), td_vallink(url, value))  # noqa: E731

        html = []
        html.append("<table style='border-collapse:collapse;border-spacing:0'>")
        html.append("<thead>")
        html.append(tr_title("httpstore_erddap_auth"))
        html.append("</thead>")
        html.append("<tbody>")
        html.append(tr_ticklink("login page", self._login_page, self._login_page))
        html.append(tr_tick("login data", self._login_payload))
        if hasattr(self, '_connected'):
            html.append(tr_tick("connected", "✅" if self._connected else "⛔"))
        else:
            html.append(tr_tick("connected", "?"))
        html.append("</tbody>")
        html.append("</table>")

        html = "\n".join(html)
        return html

    def connect(self):
        try:
            log.info("Try to log-in to '%s' page with %s data ..." % (self._login_page, self._login_payload))
            self.fs.info(self._login_page)
            self._connected = True
        except ErddapHTTPUnauthorized:
            self._connected = False
        except:  #noqa: E722
            raise
        return self._connected

    @property
    def connected(self):
        if not hasattr(self, '_connected'):
            self.connect()
        return self._connected


def httpstore_erddap(url: str = "",
                     cache: bool = False,
                     cachedir: str = "",
                     **kwargs):

    login_page = "%s/login.html" % url.rstrip("/")
    login_store = httpstore_erddap_auth(cache=cache, cachedir=cachedir, login=login_page, auto=False, **kwargs)
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
