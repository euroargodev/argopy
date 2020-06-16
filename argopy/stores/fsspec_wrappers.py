import os
import io
import xarray as xr
import pandas as pd
import requests
import fsspec
from IPython.core.display import display, HTML

from argopy.options import OPTIONS
from argopy.errors import ErddapServerError, FileSystemHasNoCache, CacheFileNotFound


class filestore():
    """Wrapper around fsspec file stores

        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.local.LocalFileSystem

    """

    def __init__(self, cache: bool = False, cachedir: str = "", **kw):
        """ Create a file storage system for local file requests

            Parameters
            ----------
            cache : bool (False)
            cachedir : str (from OPTIONS)

        """
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        if not self.cache:
            self.fs = fsspec.filesystem("file", **kw)
        else:
            self.fs = fsspec.filesystem("filecache",
                                        target_protocol='file',
                                        cache_storage=self.cachedir,
                                        expiry_time=86400, cache_check=10, **kw)
            # We use a refresh rate for cache of 1 day,
            # since this is the update frequency of the Ifremer erddap

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        else:
            self.fs.load_cache()
            if uri in self.fs.cached_files[-1]:
                return os.path.sep.join([self.fs.storage[-1], self.fs.cached_files[-1][uri]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    def glob(self, path, **kwargs):
        return self.fs.glob(path, **kwargs)

    def open(self, url, *args, **kwargs):
        return self.fs.open(url, *args, **kwargs)

    def open_dataset(self, url, **kwargs):
        """ Return a xarray.dataset from an url

            Parameters
            ----------
            url: str

            Returns
            -------
            :class:`xarray.DataArray`

        """
        with self.fs.open(url) as of:
            ds = xr.open_dataset(of, **kwargs)
        return ds

    def open_dataframe(self, url, **kwargs):
        """ Return a pandas.dataframe from an url that is a csv ressource

            Parameters
            ----------
            url: str

            Returns
            -------
            :class:`pandas.DataFrame`

        """
        with self.fs.open(url) as of:
            df = pd.read_csv(of, **kwargs)
        return df


class ftpstore():
    """Wrapper around fsspec ftp file store

        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.local.LocalFileSystem
        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.ftp.FTPFileSystem
        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.sftp.SFTPFileSystem

        This wrapper is primarily used by the localftp data/index fetchers
    """

    def __init__(self, cache: bool = False, cachedir: str = ""):
        """ Create a file storage system for ftp requests

            Parameters
            ----------
            cache : bool (False)
            cachedir : str (from OPTIONS)

        """
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        if not self.cache:
            self.fs = fsspec.filesystem("ftp")
        else:
            self.fs = fsspec.filesystem("filecache",
                                        target_protocol='ftp',
                                        target_options={'simple_links': True},
                                        cache_storage=self.cachedir,
                                        expiry_time=86400, cache_check=10)
            # We use a refresh rate for cache of 1 day,
            # since this is the update frequency of the Ifremer erddap

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        else:
            self.fs.load_cache()
            if uri in self.fs.cached_files[-1]:
                return os.path.sep.join([self.fs.storage[-1], self.fs.cached_files[-1][uri]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

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

    def open(self, url, **kwargs):
        return self.fs.open(url, **kwargs)

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


class httpstore():
    """Wrapper around fsspec http file store

        This wrapper intend to make argopy safer to failures from http requests
        This wrapper is primarily used by the Erddap data/index fetchers
    """

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
            self.fs = fsspec.filesystem("http", **kw)
        else:
            self.fs = fsspec.filesystem("filecache",
                                        target_protocol='http',
                                        target_options={'simple_links': True},
                                        cache_storage=self.cachedir,
                                        expiry_time=86400, cache_check=10, **kw)
            # We use a refresh rate for cache of 1 day,
            # since this is the update frequency of the Ifremer erddap

    def cachepath(self, uri: str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs))
        else:
            self.fs.load_cache()
            if uri in self.fs.cached_files[-1]:
                return os.path.sep.join([self.fs.storage[-1], self.fs.cached_files[-1][uri]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

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

    def open(self, url, **kwargs):
        return self.fs.open(url, **kwargs)

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
