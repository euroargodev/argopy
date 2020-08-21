import os
import io
import types
import xarray as xr
import pandas as pd
import requests
import fsspec
import shutil
import pickle
import json
import tempfile
from IPython.core.display import display, HTML

import concurrent.futures
from tqdm import tqdm
import distributed

from argopy.options import OPTIONS
from argopy.errors import ErddapServerError, FileSystemHasNoCache, CacheFileNotFound, DataNotFound, \
    APIServerError, InvalidMethod
from abc import ABC, abstractmethod


class argo_store_proto(ABC):
    """ Argo Abstract File System

        Provide a prototype for Argo file systems

        Should this class inherits from fsspec.spec.AbstractFileSystem ?
    """
    protocol = ''  # File system name, one in fsspec.registry.known_implementations

    def __init__(self,
                 cache: bool = False,
                 cachedir: str = "",
                 **kw):
        """ Create a file storage system for Argo data

            Parameters
            ----------
            cache : bool (False)
            cachedir : str (from OPTIONS)
            **kwargs:
                Other keywords arguments are passed to :class:`fsspec.filesystem`

        """
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        if not self.cache:
            self.fs = fsspec.filesystem(self.protocol, **kw)
        else:
            self.fs = fsspec.filesystem("filecache",
                                        target_protocol=self.protocol,
                                        target_options= {**{'simple_links': True, "block_size": 0}, **kw},
                                        cache_storage=self.cachedir,
                                        expiry_time=86400, cache_check=10)
            # We use a refresh rate for cache of 1 day,
            # since this is the update frequency of the Ifremer erddap
            self.cache_registry = []  # Will hold uri cached by this store instance

    def open(self, path, *args, **kwargs):
        self.register(path)
        return self.fs.open(path, *args, **kwargs)

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def exists(self, path, *args):
        return self.fs.exists(path, *args)

    def store_path(self, uri):
        if not uri.startswith(self.fs.target_protocol):
            path = self.fs.target_protocol + "://" + uri
        else:
            path = uri
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
            self.fs.load_cache()
            if store_path in self.fs.cached_files[-1]:
                return os.path.sep.join([self.cachedir, self.fs.cached_files[-1][store_path]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    def _clear_cache_item(self, uri):
        """ Open fsspec cache registry (pickle file) and remove entry for uri """
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
        fn2 = tempfile.mktemp()
        with open(fn2, "wb") as f:
            pickle.dump(cache, f)
        shutil.move(fn2, fn)

    def clear_cache(self):
        """ Remove cache files and entry from uri open with this store instance """
        if self.cache:
            for uri in self.cache_registry:
                self._clear_cache_item(uri)

    @abstractmethod
    def open_dataset(self, *args, **kwargs):
        pass

    @abstractmethod
    def read_csv(self):
        pass

    def open_mfdataset(self,
                       urls,
                       concat_dim='row',
                       max_workers = 100,
                       method: str = 'thread',
                       progress: bool = False,
                       concat: bool = True,
                       preprocess = None,
                       *args, **kwargs):
        """ Open multiple urls as a single xarray dataset.

            This is a parallelised version of ``open_dataset``.
            Use a Threads Pool by default for parallelisation.

            Parameters
            ----------
            urls: list(str)
            concat_dim: str
                Name of the dimension to use to concatenate all datasets (passed to :class:`xarray.concat`)
            max_workers: int
                Maximum number of threads or processes.
            method:
                The parallelisation method to execute calls asynchronously:
                    - 'thread' (Default): use a pool of at most ``max_workers`` threads
                    - 'process': use a pool of at most ``max_workers`` processes
                    - Dask client object: use a Dask distributed client object
                Use 'seq' to simply open data sequentially
            progress: bool
                Display a progress bar (True by default)
            preprocess: (callable, optional)
                If provided, call this function on each dataset prior to concatenation.

            Returns
            -------
            :class:`xarray.Dataset`

        """
        def __processor(u, *a, **kw):
            return preprocess(self.open_dataset(u, *a, **kw))
        def __noprocessor(u, *a, **kw):
            return self.open_dataset(u, *a, **kw)
        if not isinstance(preprocess, types.FunctionType) and not isinstance(preprocess, types.MethodType):
            process = __noprocessor
        else:
            process = __processor
        if not isinstance(urls, list):
            urls = [urls]

        results = []
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                # future_to_url = {executor.submit(self.open_dataset, url): url for url in urls}
                # future_to_url = {executor.submit(process, url): url for url in urls}
                future_to_url = {executor.submit(process, url, *args, **kwargs): url for url in urls}
                # future_to_url = (executor.submit(self.open_dataset, url) for url in urls)
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls))

                for future in futures:
                    data = None
                    url = future_to_url[future]
                    try:
                        data = future.result()
                        # if isinstance(preprocess, types.FunctionType):
                        #     data = preprocess(data)
                    except Exception as e:
                        # print("Something went wrong with this url: %s" % url)
                        print(e.args)
                        pass
                    finally:
                        results.append(data)

        elif type(method) == distributed.client.Client:
            # Use a dask client:
            futures = method.map(process, urls, *args, **kwargs)
            results = method.gather(futures)

        elif method in ['seq', 'sequential', 'serie']:
            if progress:
                urls = tqdm(urls, total=len(urls))

            for url in urls:
                data = None
                try:
                    data = process(url, *args, **kwargs)
                except Exception as e:
                    print("Something went wrong with this url: %s" % url)
                    print("Error:", e.args)
                    pass
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

    def open_mfjson(self,
                    urls,
                    max_workers = 100,
                    method: str = 'thread',
                    progress: bool = False,
                    preprocess = None,
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
                The parallelisation method to execute calls asynchronously:
                    - 'thread' (Default): use a pool of at most ``max_workers`` threads
                    - 'process': use a pool of at most ``max_workers`` processes
                    - Dask client object: use a Dask distributed client object
                Use 'seq' to simply open data sequentially
            progress: bool
                Display a progress bar (True by default, not for dask client method)
            preprocess: (callable, optional)
                If provided, call this function on each json set

            Returns
            -------
            list()
        """
        def __processor(u, *a, **kw):
            return preprocess(self.open_json(u, *a, **kw))
        def __noprocessor(u, *a, **kw):
            return self.open_json(u, *a, **kw)
        if not isinstance(preprocess, types.FunctionType) and not isinstance(preprocess, types.MethodType):
            process = __noprocessor
        else:
            process = __processor
        if not isinstance(urls, list):
            urls = [urls]

        results = []
        if method in ['thread', 'process']:
            if method == 'thread':
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            else:
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

            with ConcurrentExecutor as executor:
                future_to_url = {executor.submit(process, url, *args, **kwargs): url for url in urls}
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(futures, total=len(urls))

                for future in futures:
                    data = None
                    url = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as e:
                        print(e.args)
                        pass
                    finally:
                        results.append(data)

        elif type(method) == distributed.client.Client:
            # Use a dask client:
            futures = method.map(process, urls, *args, **kwargs)
            results = method.gather(futures)

        elif method in ['seq', 'sequential', 'serie']:
            if progress:
                urls = tqdm(urls, total=len(urls))

            for url in urls:
                data = None
                try:
                    data = process(url, *args, **kwargs)
                except Exception as e:
                    print("Something went wrong with this url: %s" % url)
                    print("Error:", e.args)
                    pass
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
        with self.fs.open(url) as of:
            ds = xr.open_dataset(of, *args, **kwargs)
        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url
        self.register(url)
        return ds

    def read_csv(self, url, **kwargs):
        """ Return a pandas.dataframe from an url that is a csv ressource

            Parameters
            ----------
            Path: str
                Path to csv resources passed to pandas.read_csv

            Returns
            -------
            :class:`pandas.DataFrame`
        """
        with self.fs.open(url) as of:
            df = pd.read_csv(of, **kwargs)
        self.register(url)
        return df


class httpstore(argo_store_proto):
    """Argo http file system

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.http.HTTPFileSystem

        This store intends to make argopy: safer to failures from http requests, provide more verbose message to users
        if we can identify specific errors in http responses.

        This store is primarily used by the Erddap/Argovis data/index fetchers
    """
    protocol = "http"

    def _verbose_exceptions(self, e):
        r = e.response  # https://requests.readthedocs.io/en/master/api/#requests.Response
        data = io.BytesIO(r.content)
        url = r.url

        # 4XX client error response
        if r.status_code == 404:  # Empty response
            error = ["Error %i " % r.status_code]
            error.append(data.read().decode("utf-8").replace("Error", ""))
            error.append("The URL triggering this error was: \n%s" % url)
            msg = "\n".join(error)
            if "Currently unknown datasetID" in msg:
                raise ErddapServerError("Dataset not found in the Erddap, try again later. "
                                        "The server may be rebooting. \n%s" % msg)
            else:
                raise requests.HTTPError(msg)

        elif r.status_code == 413:  # Too large request
            error = ["Error %i " % r.status_code]
            error.append(data.read().decode("utf-8").replace("Error", ""))
            error.append("The URL triggering this error was: \n%s" % url)
            msg = "\n".join(error)
            if "Payload Too Large" in msg:
                raise ErddapServerError("Your query produced too much data. "
                                        "Try to request less data.\n%s" % msg)
            else:
                raise requests.HTTPError(msg)

        # 5XX server error response
        elif r.status_code == 500:  # 500 Internal Server Error
            if "text/html" in r.headers.get('content-type'):
                display(HTML(data.read().decode("utf-8")))
            error = ["Error %i " % r.status_code]
            error.append(data.read().decode("utf-8"))
            error.append("The URL triggering this error was: \n%s" % url)
            msg = "\n".join(error)
            if "No space left on device" in msg or "java.io.EOFException" in msg:
                raise ErddapServerError("An error occured on the Erddap server side. "
                                        "Please contact assistance@ifremer.fr to ask a "
                                        "reboot of the erddap server. \n%s" % msg)
            else:
                raise requests.HTTPError(msg)

        else:
            error = ["Error %i " % r.status_code]
            error.append(data.read().decode("utf-8"))
            error.append("The URL triggering this error was: \n%s" % url)
            print("\n".join(error))
            r.raise_for_status()

    def open_json(self, url, **kwargs):
        """ Return a json from an url, or verbose errors

            Parameters
            ----------
            url: str

            Returns
            -------
            json

        """
        try:
            with self.fs.open(url) as of:
                js = json.load(of, **kwargs)
            self.register(url)
            return js
        except json.JSONDecodeError as e:
            raise
        except requests.HTTPError as e:
            self._verbose_exceptions(e)

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
            with self.fs.open(url) as of:
                ds = xr.open_dataset(of, *args, **kwargs)
            if "source" not in ds.encoding:
                if isinstance(url, str):
                    ds.encoding["source"] = url
            self.register(url)
            return ds
        except requests.exceptions.ConnectionError as e:
            raise APIServerError("No API response for %s" % url)
        except requests.HTTPError as e:
            self._verbose_exceptions(e)

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
        try:
            with self.fs.open(url) as of:
                df = pd.read_csv(of, **kwargs)
            self.register(url)
            return df
        except requests.HTTPError as e:
            self._verbose_exceptions(e)


class memorystore(filestore):
    """ Argo in-memory file system

        Note that this inherits from filestore, not argo_store_proto

        Relies on:
            https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.memory.MemoryFileSystem
    """
    protocol = 'memory'
