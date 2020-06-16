#!/bin/env python
# -*coding: UTF-8 -*-
#
# Disclaimer:
# Functions get_sys_info, netcdf_and_hdf5_versions and show_versions are from:
#   xarray/util/print_versions.py
#

import os
import sys
import warnings
import requests
import urllib.request
import io
import json
import xarray as xr
import pandas as pd
from IPython.core.display import display, HTML

import importlib
import locale
import platform
import struct
import subprocess
import fsspec

import pickle
import pkg_resources
path2pkl = pkg_resources.resource_filename('argopy', 'assets/')

from argopy.errors import ErddapServerError, FileSystemHasNoCache, CacheFileNotFound
from argopy.options import OPTIONS


class filestore():
    """Wrapper around fsspec file stores

        https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.local.LocalFileSystem

        This wrapper is primarily used by the localftp data/index fetchers

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

    def cachepath(self, uri : str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs) )
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
        try:
            with self.fs.open(url) as of:
                ds = xr.open_dataset(of, **kwargs)
            return ds
        except:
            raise

    def open_dataframe(self, url, **kwargs):
        """ Return a pandas.dataframe from an url that is a csv ressource

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
        except:
            raise


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

    def cachepath(self, uri : str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs) )
        else:
            self.fs.load_cache()
            if uri in self.fs.cached_files[-1]:
                return os.path.sep.join([self.fs.storage[-1], self.fs.cached_files[-1][uri]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    def _verbose_exceptions(self, e):
        r = e.response # https://requests.readthedocs.io/en/master/api/#requests.Response
        status_code = r.status_code
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
                                        "Please contact assistance@ifremer.fr to ask a reboot of the erddap server. \n%s" % msg)
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

    def cachepath(self, uri : str, errors: str = 'raise'):
        """ Return path to cached file for a given URI """
        if not self.cache:
            if errors == 'raise':
                raise FileSystemHasNoCache("%s has no cache system" % type(self.fs) )
        else:
            self.fs.load_cache()
            if uri in self.fs.cached_files[-1]:
                return os.path.sep.join([self.fs.storage[-1], self.fs.cached_files[-1][uri]['fn']])
            elif errors == 'raise':
                raise CacheFileNotFound("No cached file found in %s for: \n%s" % (self.fs.storage[-1], uri))

    def _verbose_exceptions(self, e):
        r = e.response # https://requests.readthedocs.io/en/master/api/#requests.Response
        status_code = r.status_code
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
                                        "Please contact assistance@ifremer.fr to ask a reboot of the erddap server. \n%s" % msg)
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


def urlopen(url):
    """ Load content from url or raise alarm on status with explicit information on the error

    Parameters
    ----------
    url: str

    Returns
    -------
    io.BytesIO

    """
    # https://github.com/ioos/erddapy/blob/3828a4f479e7f7653fb5fd78cbce8f3b51bd0661/erddapy/utilities.py#L37
    r = requests.get(url)
    data = io.BytesIO(r.content)

    if r.status_code == 200:  # OK
        return data

    # 4XX client error response
    elif r.status_code == 404:  # Empty response
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8").replace("Error", ""))
        error.append("The URL triggering this error was: \n%s" % url)
        msg = "\n".join(error)
        if "Currently unknown datasetID" in msg:
            raise ErddapServerError("Dataset not found in the Erddap, try again later. "
                                    "The server is probably rebooting. \n%s" % msg)
        else:
            raise requests.HTTPError(msg)

    elif r.status_code == 413: # Too large request
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8").replace("Error", ""))
        error.append("The URL triggering this error was: \n%s" % url)
        msg = "\n".join(error)
        if "Payload Too Large" in msg:
            raise ErddapServerError("Your query produced too much data.  "
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
        if "No space left on device" in msg:
            raise ErddapServerError("An error occured on the Erddap server side. "
                                    "Please contact assistance@ifremer.fr to ask a reboot of the erddap server. \n%s" % msg)
        elif "java.io.EOFException" in msg:
            raise ErddapServerError("An error occured on the Erddap server side. "
                                    "Please contact assistance@ifremer.fr to ask a reboot of the erddap server. \n%s" % msg)
        else:
            raise requests.HTTPError(msg)

    else:
        error = ["Error %i " % r.status_code]
        error.append(data.read().decode("utf-8"))
        error.append("The URL triggering this error was: \n%s" % url)
        print("\n".join(error))
        r.raise_for_status()

def load_dict(ptype):
    if ptype=='profilers':        
        with open(os.path.join(path2pkl, 'dict_profilers.pickle'), 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    elif ptype=='institutions':
        with open(os.path.join(path2pkl, 'dict_institutions.pickle'), 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict      
    else:
        raise ValueError("Invalid dictionnary pickle file")

def mapp_dict(Adictionnary, Avalue):
    if Avalue not in Adictionnary:
        return "Unknown"
    else:
        return Adictionnary[Avalue]

def list_available_data_src():
    """ List all available data sources """
    AVAILABLE_SOURCES = {}
    try:
        from .data_fetchers import erddap as Erddap_Fetchers
        AVAILABLE_SOURCES['erddap'] = Erddap_Fetchers
    except:
        warnings.warn("An error occured while loading the ERDDAP data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    try:
        from .data_fetchers import localftp as LocalFTP_Fetchers
        AVAILABLE_SOURCES['localftp'] = LocalFTP_Fetchers
    except:
        warnings.warn("An error occured while loading the local FTP data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    return AVAILABLE_SOURCES

def list_standard_variables():
    """ Return the list of variables for standard users
    """
    return ['DATA_MODE', 'LATITUDE', 'LONGITUDE', 'POSITION_QC', 'DIRECTION', 'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'PRES',
     'TEMP', 'PSAL', 'PRES_QC', 'TEMP_QC', 'PSAL_QC', 'PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED',
     'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC', 'PSAL_ADJUSTED_QC', 'PRES_ADJUSTED_ERROR', 'TEMP_ADJUSTED_ERROR',
     'PSAL_ADJUSTED_ERROR', 'JULD', 'JULD_QC', 'TIME', 'TIME_QC']

def list_multiprofile_file_variables():
    """ Return the list of variables in a netcdf multiprofile file.

        This is for files created by GDAC under <DAC>/<WMO>/<WMO>_prof.nc
    """
    return [ 'CONFIG_MISSION_NUMBER',
             'CYCLE_NUMBER',
             'DATA_CENTRE',
             'DATA_MODE',
             'DATA_STATE_INDICATOR',
             'DATA_TYPE',
             'DATE_CREATION',
             'DATE_UPDATE',
             'DC_REFERENCE',
             'DIRECTION',
             'FIRMWARE_VERSION',
             'FLOAT_SERIAL_NO',
             'FORMAT_VERSION',
             'HANDBOOK_VERSION',
             'HISTORY_ACTION',
             'HISTORY_DATE',
             'HISTORY_INSTITUTION',
             'HISTORY_PARAMETER',
             'HISTORY_PREVIOUS_VALUE',
             'HISTORY_QCTEST',
             'HISTORY_REFERENCE',
             'HISTORY_SOFTWARE',
             'HISTORY_SOFTWARE_RELEASE',
             'HISTORY_START_PRES',
             'HISTORY_STEP',
             'HISTORY_STOP_PRES',
             'JULD',
             'JULD_LOCATION',
             'JULD_QC',
             'LATITUDE',
             'LONGITUDE',
             'PARAMETER',
             'PI_NAME',
             'PLATFORM_NUMBER',
             'PLATFORM_TYPE',
             'POSITIONING_SYSTEM',
             'POSITION_QC',
             'PRES',
             'PRES_ADJUSTED',
             'PRES_ADJUSTED_ERROR',
             'PRES_ADJUSTED_QC',
             'PRES_QC',
             'PROFILE_PRES_QC',
             'PROFILE_PSAL_QC',
             'PROFILE_TEMP_QC',
             'PROJECT_NAME',
             'PSAL',
             'PSAL_ADJUSTED',
             'PSAL_ADJUSTED_ERROR',
             'PSAL_ADJUSTED_QC',
             'PSAL_QC',
             'REFERENCE_DATE_TIME',
             'SCIENTIFIC_CALIB_COEFFICIENT',
             'SCIENTIFIC_CALIB_COMMENT',
             'SCIENTIFIC_CALIB_DATE',
             'SCIENTIFIC_CALIB_EQUATION',
             'STATION_PARAMETERS',
             'TEMP',
             'TEMP_ADJUSTED',
             'TEMP_ADJUSTED_ERROR',
             'TEMP_ADJUSTED_QC',
             'TEMP_QC',
             'VERTICAL_SAMPLING_SCHEME',
             'WMO_INST_TYPE']

def get_sys_info():
    "Returns system information as a dict"

    blob = []

    # get full commit hash
    commit = None
    if os.path.isdir(".git") and os.path.isdir("argopy"):
        try:
            pipe = subprocess.Popen(
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, serr = pipe.communicate()
        except Exception:
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                try:
                    commit = so.decode("utf-8")
                except ValueError:
                    pass
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "%s" % (sysname)),
                ("OS-release", "%s" % (release)),
                ("machine", "%s" % (machine)),
                ("processor", "%s" % (processor)),
                ("byteorder", "%s" % sys.byteorder),
                ("LC_ALL", "%s" % os.environ.get("LC_ALL", "None")),
                ("LANG", "%s" % os.environ.get("LANG", "None")),
                ("LOCALE", "%s.%s" % locale.getlocale()),
            ]
        )
    except Exception:
        pass

    return blob

def netcdf_and_hdf5_versions():
    libhdf5_version = None
    libnetcdf_version = None
    try:
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    except ImportError:
        try:
            import h5py

            libhdf5_version = h5py.version.hdf5_version
        except ImportError:
            pass
    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]

def show_versions(file=sys.stdout):
    """ Print the versions of argopy and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        print(f"Error collecting netcdf / hdf5 version: {e}")

    deps = [
        # (MODULE_NAME, f(mod) -> mod version)
        ("argopy", lambda mod: mod.__version__),
        ("xarray", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        # argopy optionals
        ("netCDF4", lambda mod: mod.__version__),
        ("pydap", lambda mod: mod.__version__),
        ("h5netcdf", lambda mod: mod.__version__),
        ("h5py", lambda mod: mod.__version__),
        ("Nio", lambda mod: mod.__version__),
        ("zarr", lambda mod: mod.__version__),
        ("cftime", lambda mod: mod.__version__),
        ("nc_time_axis", lambda mod: mod.__version__),
        ("PseudoNetCDF", lambda mod: mod.__version__),
        ("rasterio", lambda mod: mod.__version__),
        ("cfgrib", lambda mod: mod.__version__),
        ("iris", lambda mod: mod.__version__),
        ("bottleneck", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
        ("distributed", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        ("cartopy", lambda mod: mod.__version__),
        ("seaborn", lambda mod: mod.__version__),
        ("numbagg", lambda mod: mod.__version__),
        # argopy setup/test
        ("setuptools", lambda mod: mod.__version__),
        ("pip", lambda mod: mod.__version__),
        ("conda", lambda mod: mod.__version__),
        ("pytest", lambda mod: mod.__version__),
        # Misc.
        ("IPython", lambda mod: mod.__version__),
        ("sphinx", lambda mod: mod.__version__),
    ]

    deps_blob = list()
    for (modname, ver_f) in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
        except Exception:
            deps_blob.append((modname, None))
        else:
            try:
                ver = ver_f(mod)
                deps_blob.append((modname, ver))
            except Exception:
                deps_blob.append((modname, "installed"))

    print("\nINSTALLED VERSIONS", file=file)
    print("------------------", file=file)

    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    print("", file=file)
    for k, stat in deps_blob:
        print(f"{k}: {stat}", file=file)

def isconnected(host='http://www.ifremer.fr'):
    """ Determine if we have a live internet connection

        Parameters
        ----------
        host: str
            URL to use, 'http://www.ifremer.fr' by default

        Returns
        -------
        bool
    """
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except:
        return False

def erddap_ds_exists(ds="ArgoFloats"):
    """ Given erddap fetcher, check if a Dataset exists, return a bool"""
    # e = ArgoDataFetcher(src='erddap').float(wmo=0).fetcher
    # erddap_index = json.load(urlopen(e.erddap.server + "/info/index.json"))
    # erddap_index = json.load(urlopen("http://www.ifremer.fr/erddap/info/index.json"))
    with httpstore(timeout=120).open("http://www.ifremer.fr/erddap/info/index.json") as of:
        erddap_index = json.load(of)
    return ds in [row[-1] for row in erddap_index['table']['rows']]

def open_etopo1(box, res='l'):
    """ Download ETOPO for a box

        Parameters
        ----------
        box: [xmin, xmax, ymin, ymax]

        Returns
        -------
        xarray.Dataset
    """
    # This function is in utilities to anticipate usage outside of plotting, eg interpolation, grounding detection
    resx, resy = 0.1, 0.1
    if res == 'h':
        resx, resy = 0.016, 0.016

    uri = ("https://gis.ngdc.noaa.gov/mapviewer-support/wcs-proxy/wcs.groovy?filename=etopo1.nc"
           "&request=getcoverage&version=1.0.0&service=wcs&coverage=etopo1&CRS=EPSG:4326&format=netcdf"
           "&resx={}&resy={}"
           "&bbox={}").format
    thisurl = uri(resx, resy, ",".join([str(b) for b in [box[0], box[2], box[1], box[3]]]))
    ds = httpstore(cache=True).open_dataset(thisurl)
    da = ds['Band1'].rename("topo")
    for a in ds.attrs:
        da.attrs[a] = ds.attrs[a]
    da.attrs['Data source'] = 'https://maps.ngdc.noaa.gov/viewers/wcs-client/'
    da.attrs['URI'] = thisurl
    return da