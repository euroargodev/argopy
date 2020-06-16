import os
import io
import xarray as xr
import pandas as pd
import requests
import fsspec
from IPython.core.display import display, HTML

from argopy.options import OPTIONS
from argopy.errors import ErddapServerError, FileSystemHasNoCache, CacheFileNotFound
from abc import ABC, abstractmethod


class argo_store_proto(ABC):
    protocol = '' # One in fsspec.registry.known_implementations

    def __init__(self, cache: bool = False, cachedir: str = "", **kw):
        """ Create a file storage system for http requests

            Parameters
            ----------
            cache : bool (False)
            cachedir : str (from OPTIONS)

        """
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        if not self.cache:
            self.fs = fsspec.filesystem(self.protocol, **kw)
        else:
            self.fs = fsspec.filesystem("filecache",
                                        target_protocol=self.protocol,
                                        target_options={'simple_links': True},
                                        cache_storage=self.cachedir,
                                        expiry_time=86400, cache_check=10, **kw)
            # We use a refresh rate for cache of 1 day,
            # since this is the update frequency of the Ifremer erddap

    def open(self, url, *args, **kwargs):
        return self.fs.open(url, *args, **kwargs)

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        else:
            if not uri.startswith(self.fs.target_protocol):
                store_path = self.fs.target_protocol + "://" + uri
            else:
                store_path = uri
            # return store_path in fs.cached_files[-1]
            self.fs.load_cache()
            if store_path in self.fs.cached_files[-1]:
                # return self.fs.cached_files[-1]
                return os.path.sep.join([self.cachedir, self.fs.cached_files[-1][store_path]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    @abstractmethod
    def open_dataset(self):
        pass

    @abstractmethod
    def open_dataframe(self):
        pass


class filestore(argo_store_proto):
    """Wrapper around fsspec file stores

        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.local.LocalFileSystem

    """
    protocol = 'file'

    def open_dataset(self, url, **kwargs):
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
            ds = xr.open_dataset(of, **kwargs)
        return ds

    def open_dataframe(self, url, **kwargs):
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
        return df


class httpstore(argo_store_proto):
    """Wrapper around fsspec http file store

        This wrapper intend to make argopy safer to failures from http requests
        This wrapper is primarily used by the Erddap data/index fetchers
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

    def open_dataset(self, url, **kwargs):
        """ Return a xarray.dataset from an url, or verbose errors

            Parameters
            ----------
            url: str

            Returns
            -------
            :class:`xarray.DataArray`

        """
        try:
            with self.fs.open(url) as of:
                ds = xr.open_dataset(of, **kwargs)
            return ds
        except requests.HTTPError as e:
            self._verbose_exceptions(e)

    def open_dataframe(self, url, **kwargs):
        """ Return a pandas.dataframe from an url with csv response, or verbose errors

            Parameters
            ----------
            url: str

            Returns
            -------
            :class:`pandas.DataFrame`

        """
        try:
            with self.fs.open(url) as of:
                df = pd.read_csv(of, **kwargs)
            return df
        except requests.HTTPError as e:
            self._verbose_exceptions(e)
