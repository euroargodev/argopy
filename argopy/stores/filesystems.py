import os
import types
import xarray as xr
import pandas as pd
import fsspec
import shutil
import pickle
import json
import tempfile
import warnings
import logging
from packaging import version

import concurrent.futures
import multiprocessing


try:
    from tqdm import tqdm
except ModuleNotFoundError:
    warnings.warn("argopy needs tqdm installed to display progress bars")

    def tqdm(fct, **kw):
        return fct


from argopy.options import OPTIONS
from argopy.errors import FileSystemHasNoCache, CacheFileNotFound, DataNotFound, \
    InvalidMethod
from abc import ABC, abstractmethod


log = logging.getLogger("argopy.stores")


def new_fs(protocol: str = '', cache: bool = False, cachedir: str = OPTIONS['cachedir'], **kwargs):
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
    default_filesystem_kwargs = {'simple_links': True, "block_size": 0}
    if protocol == 'http':
        default_filesystem_kwargs = {**default_filesystem_kwargs,
                                     **{"client_kwargs": {"trust_env": OPTIONS['trust_env']}}}
    filesystem_kwargs = {**default_filesystem_kwargs, **kwargs}

    if not cache:
        fs = fsspec.filesystem(protocol, **filesystem_kwargs)
        cache_registry = None
        log.debug("Opening a fsspec [file] system for '%s' protocol with options: %s" %
                  (protocol, str(filesystem_kwargs)))
    else:
        fs = fsspec.filesystem("filecache",
                               target_protocol=protocol,
                               target_options={**filesystem_kwargs},
                               cache_storage=cachedir,
                               expiry_time=86400, cache_check=10)
        # We use a refresh rate for cache of 1 day,
        # since this is the update frequency of the Ifremer erddap
        cache_registry = []  # Will hold uri cached by this store instance
        log.debug("Opening a fsspec [filecache] system for '%s' protocol with options: %s" %
                  (protocol, str(filesystem_kwargs)))
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
        self._filesystem_kwargs = {**kwargs}
        self.fs, self.cache_registry = new_fs(self.protocol,
                                              self.cache,
                                              self.cachedir,
                                              **self._filesystem_kwargs)

    def open(self, path, *args, **kwargs):
        self.register(path)
        return self.fs.open(path, *args, **kwargs)

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def exists(self, path, *args):
        return self.fs.exists(path, *args)

    def expand_path(self, path):
        if self.protocol != "http":
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
            self.cache_registry.append(self.store_path(uri))

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        else:
            store_path = self.store_path(uri)
            self.fs.load_cache()  # Read set of stored blocks from file and populate self.fs.cached_files
            if store_path in self.fs.cached_files[-1]:
                return os.path.sep.join([self.cachedir, self.fs.cached_files[-1][store_path]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    def _clear_cache_item(self, uri):
        """ Open fsspec cache registry (pickle file) and remove entry for uri

        """
        # See the "save_cache()" method in:
        # https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/cached.html#WholeFileCacheFileSystem
        fn = os.path.join(self.fs.storage[-1], "cache")
        cache = self.fs.cached_files[-1]
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                cached_files = pickle.load(f)
        else:
            cached_files = cache
        cache = {}
        for k, v in cached_files.items():
            if k != uri:
                cache[k] = v.copy()
            else:
                os.remove(os.path.join(self.fs.storage[-1], v['fn']))
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            pickle.dump(cache, f)
        shutil.move(f.name, fn)

    def clear_cache(self):
        """ Remove cache files and entry from uri open with this store instance """
        if self.cache:
            for uri in self.cache_registry:
                self._clear_cache_item(uri)

    @abstractmethod
    def open_dataset(self, *args, **kwargs):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def read_csv(self):
        raise NotImplementedError("Not implemented")


class filestore(argo_store_proto):
    """Argo local file system

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.local.LocalFileSystem
    """
    protocol = 'file'

    def open_dataset(self, url, *args, **kwargs):
        """ Return a xarray.dataset from an url

            Parameters
            ----------
            Path: str
                Path to resources passed to xarray.open_dataset

            Returns
            -------
            :class:`xarray.DataSet`
        """
        with self.open(url) as of:
            log.debug("Opening dataset: %s" % url)
            ds = xr.open_dataset(of, *args, **kwargs)
            ds.load()
        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url
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
                       max_workers: int = 112,
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
                if max_workers == 112:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor, url,
                                                 preprocess=preprocess, *args, **kwargs): url
                                 for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls))

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

        # elif type(method) == distributed.client.Client:
        #     # Use a dask client:
        #     futures = method.map(self._mfprocessor, urls, preprocess=preprocess, *args, **kwargs)
        #     results = method.gather(futures)

        elif method in ['seq', 'sequential']:
            if progress:
                urls = tqdm(urls, total=len(urls))

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

    def read_csv(self, url, **kwargs):
        """ Return a pandas.dataframe from an url that is a csv resource

            Parameters
            ----------
            Path: str
                Path to csv resources passed to pandas.read_csv

            Returns
            -------
            :class:`pandas.DataFrame`
        """
        log.debug("Reading csv: %s" % url)
        with self.open(url) as of:
            df = pd.read_csv(of, **kwargs)
        return df


class httpstore(argo_store_proto):
    """Argo http file system

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.http.HTTPFileSystem

        This store intends to make argopy: safer to failures from http requests and to provide higher levels methods to
        work with our datasets

        This store is primarily used by the Erddap/Argovis data/index fetchers
    """
    protocol = "http"

    def open_dataset(self, url, *args, **kwargs):
        """ Open and decode a xarray dataset from an url

            Parameters
            ----------
            url: str

            Returns
            -------
            :class:`xarray.Dataset`

        """
        log.debug("Opening dataset: %s" % url)
        # try:
        # with self.fs.open(url) as of:
        #     ds = xr.open_dataset(of, *args, **kwargs)
        data = self.fs.cat_file(url)
        ds = xr.open_dataset(data, *args, **kwargs)
        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url
        self.register(url)
        return ds
        # except Exception as e:
        #     raise e
        # except requests.exceptions.ConnectionError as e:
        #     raise APIServerError("No API response for %s" % url)
        # except requests.HTTPError as e:
        #     self._verbose_requests_exceptions(e)
        #     pass
        # except aiohttp.ClientResponseError as e:
        #     self._verbose_aiohttp_exceptions(e)
        #     pass

    def _mfprocessor_dataset(self, url, preprocess=None, *args, **kwargs):
        # Load data
        ds = self.open_dataset(url, *args, **kwargs)
        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(preprocess, types.MethodType):
            ds = preprocess(ds)
        return ds

    def open_mfdataset(self,  # noqa: C901
                       urls,
                       concat_dim='row',
                       max_workers: int = 112,
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

            Returns
            -------
            :class:`xarray.Dataset`

        """
        strUrl = lambda x: x.replace("https://", "").replace("http://", "")  # noqa: E731

        if not isinstance(urls, list):
            urls = [urls]

        results = []
        failed = []
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                if max_workers == 112:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor_dataset, url,
                                                 preprocess=preprocess, *args, **kwargs): url for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls))

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
        #     futures = method.map(self._mfprocessor_dataset, urls, preprocess=preprocess, *args, **kwargs)
        #     results = method.gather(futures)

        elif method in ['seq', 'sequential']:
            if progress:
                urls = tqdm(urls, total=len(urls))

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_dataset(url, preprocess=preprocess, *args, **kwargs)
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
        log.debug("Opening/reading csv: %s" % url)
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
        log.debug("Opening json: %s" % url)
        # try:
        #     with self.open(url) as of:
        #         js = json.load(of, **kwargs)
        #     return js
        # except ClientResponseError:
        #     raise
        # except json.JSONDecodeError:
        #     raise
        data = self.fs.cat_file(url)
        js = json.loads(data, **kwargs)
        self.register(url)
        return js

    def _mfprocessor_json(self, url, preprocess=None, *args, **kwargs):
        # Load data
        data = self.open_json(url, **kwargs)
        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(preprocess, types.MethodType):
            data = preprocess(data)
        return data

    def open_mfjson(self,  # noqa: C901
                    urls,
                    max_workers=112,
                    method: str = 'thread',
                    progress: bool = False,
                    preprocess=None,
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

            Returns
            -------
            list()
        """
        strUrl = lambda x: x.replace("https://", "").replace("http://", "")  # noqa: E731

        if not isinstance(urls, list):
            urls = [urls]

        results = []
        failed = []
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                if max_workers == 112:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(self._mfprocessor_json, url,
                                                 preprocess=preprocess, *args, **kwargs): url for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls))

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
                urls = tqdm(urls, total=len(urls))

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_json(url, preprocess=preprocess, *args, **kwargs)
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


class memorystore(filestore):
    """ Argo in-memory file system

        Note that this inherits from filestore, not argo_store_proto

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.memory.MemoryFileSystem
    """
    protocol = 'memory'
