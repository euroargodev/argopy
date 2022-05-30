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
import tempfile
import warnings
import logging
from packaging import version

import concurrent.futures
import multiprocessing

from ..options import OPTIONS
from ..errors import FileSystemHasNoCache, CacheFileNotFound, DataNotFound, \
    InvalidMethod
from abc import ABC, abstractmethod
from ..utilities import Registry

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
    elif protocol == 'ftp':
        default_filesystem_kwargs = {**default_filesystem_kwargs,
                                     **{"block_size": 1000 * (2 ** 20)}}
    filesystem_kwargs = {**default_filesystem_kwargs, **kwargs}

    if not cache:
        fs = fsspec.filesystem(protocol, **filesystem_kwargs)
        cache_registry = None
        log_msg = "Opening a fsspec [file] system for '%s' protocol with options: %s" % \
                  (protocol, str(filesystem_kwargs))
    else:
        # https://filesystem-spec.readthedocs.io/en/latest/_modules/fsspec/implementations/cached.html#WholeFileCacheFileSystem
        fs = fsspec.filesystem("filecache",
                               target_protocol=protocol,
                               target_options={**filesystem_kwargs},
                               cache_storage=cachedir,
                               expiry_time=86400, cache_check=10)
        # We use a refresh rate for cache of 1 day,
        # since this is the update frequency of the Ifremer erddap
        cache_registry = Registry(name='Cache')  # Will hold uri cached by this store instance
        log_msg = "Opening a fsspec [filecache, storage='%s'] system for '%s' protocol with options: %s" % \
                  (cachedir, protocol, str(filesystem_kwargs))

    if protocol == 'file' and os.path.sep != fs.sep:
        # For some reasons (see https://github.com/fsspec/filesystem_spec/issues/937), the property fs.sep is
        # not '\' under Windows. So, using this dirty fix to overwrite it:
        fs.sep = os.path.sep
        # fsspec folks recommend to use posix internally. But I don't see how to handle this. So keeping this fix
        # because it solves issues with failing tests under Windows. Enough at this time.
        #todo: Revisit this choice in a while

    # log_msg = "%s\n[sys sep=%s] vs [fs sep=%s]" % (log_msg, os.path.sep, fs.sep)
    # log.warning(log_msg)
    log.debug(log_msg)
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

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.local.LocalFileSystem
    """
    protocol = 'file'

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
        try:
            data = self.fs.cat_file(url)
        except aiohttp.ClientResponseError as e:
            if e.status == 413:
                warnings.warn("Server says payload Too Large ! Try to use 'parallel=True' or a smaller "
                              "chunk parameter with your fetcher")
                log.debug("Error %i (Payload Too Large) raised with %s" % (e.status, url))
            raise

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
                       max_workers: int = 112,
                       method: str = 'thread',
                       progress: bool = False,
                       concat: bool = True,
                       concat_dim='row',
                       preprocess=None,
                       preprocess_opts={},
                       errors: str = 'ignore',
                       *args, **kwargs):
        """ Open multiple urls as a single xarray dataset.

            This is a version of the :class:`argopy.stores.httpstore.open_dataset` method that is able to
            handle a list of urls/paths sequentially or in parallel.

            Use a Threads Pool by default for parallelization.

            Parameters
            ----------
            urls: list(str)
                List of url/path to open
            max_workers: int, default: 112
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

        def drop_variables_not_in_all_datasets(ds_collection):
            """Drop variables that are not in all datasets (Lowest common denominator)"""
            # List all possible data variables:
            vlist = []
            for res in ds_collection:
                [vlist.append(v) for v in list(res.data_vars)]
            vlist = np.unique(vlist)

            # Check if all possible variables are in each datasets:
            ishere = np.zeros((len(vlist), len(ds_collection)))
            for ir, res in enumerate(ds_collection):
                for iv, v in enumerate(res.data_vars):
                    for iu, u in enumerate(vlist):
                        if v == u:
                            ishere[iu, ir] = 1
            # List of dataset with missing variables:
            ir_missing = np.sum(ishere, axis=0) < len(vlist)
            # List of variables missing in some dataset:
            iv_missing = np.sum(ishere, axis=1) < len(ds_collection)

            # List of variables to keep
            iv_tokeep = np.sum(ishere, axis=1) == len(ds_collection)
            for ir, res in enumerate(ds_collection):
                #         print("\n", res.attrs['Fetched_uri'])
                v_to_drop = []
                for iv, v in enumerate(res.data_vars):
                    if v not in vlist[iv_tokeep]:
                        v_to_drop.append(v)
                ds_collection[ir] = ds_collection[ir].drop_vars(v_to_drop)
            return ds_collection

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
                future_to_url = {executor.submit(self._mfprocessor_dataset,
                                                 url,
                                                 preprocess=preprocess,
                                                 preprocess_opts=preprocess_opts, *args, **kwargs): url
                                 for url in urls}
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
                urls = tqdm(urls, total=len(urls))

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor_dataset(url,
                                                     preprocess=preprocess,
                                                     preprocess_opts=preprocess_opts, *args, **kwargs)
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
        log.debug("Opening/reading csv from: %s" % url)
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
        log.debug("Opening json from: %s" % url)
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
                    max_workers=112,
                    method: str = 'thread',
                    progress: bool = False,
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
                                                 preprocess=preprocess, url_follow=url_follow, *args, **kwargs): url for url in urls}
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


class memorystore(filestore):
    """ Argo in-memory file system (global)

        Note that this inherits from filestore, not argo_store_proto

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.memory.MemoryFileSystem
    """
    protocol = 'memory'

    def exists(self, path, *args):
        """ Check if path can be open or not

            Special handling for memory store

            The fsspec.exists() will return False even if the path is in cache.
            Here we bypass this in order to return True if the path is in cache
            This assumes that the goal of fs.exists is to determine if we can load the path or not
            If the path is in cache, it can be loaded
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


class ftpstore(httpstore):
    """ Argo ftp file system

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.ftp.FTPFileSystem
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
                       max_workers: int = 112,
                       method: str = 'thread',
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
            max_workers: int, default: 112
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
        strUrl = lambda x: x.replace("ftps://", "").replace("ftp://", "")  # noqa: E731

        def drop_variables_not_in_all_datasets(ds_collection):
            """Drop variables that are not in all datasets (Lowest common denominator)"""
            # List all possible data variables:
            vlist = []
            for res in ds_collection:
                [vlist.append(v) for v in list(res.data_vars)]
            vlist = np.unique(vlist)

            # Check if all possible variables are in each datasets:
            ishere = np.zeros((len(vlist), len(ds_collection)))
            for ir, res in enumerate(ds_collection):
                for iv, v in enumerate(res.data_vars):
                    for iu, u in enumerate(vlist):
                        if v == u:
                            ishere[iu, ir] = 1
            # List of dataset with missing variables:
            ir_missing = np.sum(ishere, axis=0) < len(vlist)
            # List of variables missing in some dataset:
            iv_missing = np.sum(ishere, axis=1) < len(ds_collection)

            # List of variables to keep
            iv_tokeep = np.sum(ishere, axis=1) == len(ds_collection)
            for ir, res in enumerate(ds_collection):
                #         print("\n", res.attrs['Fetched_uri'])
                v_to_drop = []
                for iv, v in enumerate(res.data_vars):
                    if v not in vlist[iv_tokeep]:
                        v_to_drop.append(v)
                ds_collection[ir] = ds_collection[ir].drop_vars(v_to_drop)
            return ds_collection

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
                                                 preprocess=preprocess,
                                                 preprocess_opts=preprocess_opts, *args, **kwargs): url for url in urls}
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
                urls = tqdm(urls, total=len(urls))

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
