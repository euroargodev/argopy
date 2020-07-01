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
import urllib.request
import json

import importlib
import locale
import platform
import struct
import subprocess

import xarray as xr
import numpy as np
from scipy import interpolate

import pickle
import pkg_resources
import shutil

from argopy.options import OPTIONS, set_options
from argopy.stores import httpstore
from argopy.errors import FtpPathError, InvalidFetcher

path2pkl = pkg_resources.resource_filename('argopy', 'assets/')


def clear_cache():
    """ Delete argopy cache folder """
    if os.path.exists(OPTIONS['cachedir']):
        shutil.rmtree(OPTIONS['cachedir'])


def load_dict(ptype):
    if ptype == 'profilers':
        with open(os.path.join(path2pkl, 'dict_profilers.pickle'), 'rb') as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    elif ptype == 'institutions':
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
        from .data_fetchers import erddap_data as Erddap_Fetchers
        AVAILABLE_SOURCES['erddap'] = Erddap_Fetchers
    except Exception:
        warnings.warn("An error occured while loading the ERDDAP data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    try:
        from .data_fetchers import localftp_data as LocalFTP_Fetchers
        AVAILABLE_SOURCES['localftp'] = LocalFTP_Fetchers
    except Exception:
        warnings.warn("An error occured while loading the local FTP data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    try:
        from .data_fetchers import argovis_data as ArgoVis_Fetchers
        AVAILABLE_SOURCES['argovis'] = ArgoVis_Fetchers
    except Exception:
        warnings.warn("An error occured while loading the ArgoVis data fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    return AVAILABLE_SOURCES


def list_available_index_src():
    """ List all available index sources """
    AVAILABLE_SOURCES = {}
    try:
        from .data_fetchers import erddap_index as Erddap_Fetchers
        AVAILABLE_SOURCES['erddap'] = Erddap_Fetchers
    except Exception:
        warnings.warn("An error occured while loading the ERDDAP index fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    try:
        from .data_fetchers import localftp_index as LocalFTP_Fetchers
        AVAILABLE_SOURCES['localftp'] = LocalFTP_Fetchers
    except Exception:
        warnings.warn("An error occured while loading the local FTP index fetcher, "
                      "it will not be available !\n%s\n%s" % (sys.exc_info()[0], sys.exc_info()[1]))
        pass

    return AVAILABLE_SOURCES


def list_standard_variables():
    """ Return the list of variables for standard users """
    return ['DATA_MODE', 'LATITUDE', 'LONGITUDE', 'POSITION_QC', 'DIRECTION', 'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'PRES',
     'TEMP', 'PSAL', 'PRES_QC', 'TEMP_QC', 'PSAL_QC', 'PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED',
     'PRES_ADJUSTED_QC', 'TEMP_ADJUSTED_QC', 'PSAL_ADJUSTED_QC', 'PRES_ADJUSTED_ERROR', 'TEMP_ADJUSTED_ERROR',
     'PSAL_ADJUSTED_ERROR', 'JULD', 'JULD_QC', 'TIME', 'TIME_QC']


def list_multiprofile_file_variables():
    """ Return the list of variables in a netcdf multiprofile file.

        This is for files created by GDAC under <DAC>/<WMO>/<WMO>_prof.nc
    """
    return ['CONFIG_MISSION_NUMBER',
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


def check_localftp(path, errors: str = "ignore"):
    """ Check if the path has the expected GDAC ftp structure

        Check if the path is structured like:
        .
        └── dac
            ├── aoml
            ├── ...
            ├── coriolis
            ├── ...
            ├── meds
            └── nmdis

        Parameters
        ----------
        path: str
            Path name to check
        errors: str
            "ignore" or "raise" (or "warn"

        Returms
        -------
        checked: boolean
            True if at least one DAC folder is found under path/dac/<dac_name>
            False otherwise
    """
    dacs = ['aoml', 'bodc', 'coriolis', 'csio', 'csiro', 'incois', 'jma', 'kma', 'kordi', 'meds', 'nmdis']

    # Case 1:
    check1 = os.path.isdir(path) and os.path.isdir(os.path.join(path, "dac")) \
             and np.any([os.path.isdir(os.path.join(path, "dac", dac)) for dac in dacs])

    if check1:
        return True
    elif errors == 'raise':
        # This was possible up to v0.1.3:
        check2 = os.path.isdir(path) and np.any([os.path.isdir(os.path.join(path, dac)) for dac in dacs])
        if check2:
            raise FtpPathError("This path is no longer GDAC compliant for argopy.\n"
                          "Please make sure you point toward a path with a 'dac' folder:\n%s" % path)
        else:
            raise FtpPathError("This path is not GDAC compliant:\n%s" % path)

    elif errors == 'warn':
        # This was possible up to v0.1.3:
        check2 = os.path.isdir(path) and np.any([os.path.isdir(os.path.join(path, dac)) for dac in dacs])
        if check2:
            warnings.warn("This path is no longer GDAC compliant for argopy. This will raise an error in the future.\n"
                          "Please make sure you point toward a path with a 'dac' folder:\n%s" % path)
            return False
        else:
            warnings.warn("This path is not GDAC compliant:\n%s" % path)
            return False
    else:
        return False


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
    """ check if we have a live internet connection

        Parameters
        ----------
        host: str
            URL to use, 'http://www.ifremer.fr' by default

        Returns
        -------
        bool
    """
    try:
        urllib.request.urlopen(host, timeout=2)  # Python 3.x
        return True
    except Exception:
        return False


def isAPIconnected(src='erddap', data=True):
    """ Check if a source web API is alive or not

        Parameters
        ----------
        src: str
            The data or index source name, 'erddap' default
        data: bool
            If True check the data fetcher (default), if False, check the index fetcher

        Returns
        -------
        bool
    """
    if data:
        AVAILABLE_SOURCES = list_available_data_src()
    else:
        AVAILABLE_SOURCES = list_available_index_src()
    if src in AVAILABLE_SOURCES and getattr(AVAILABLE_SOURCES[src], "api_server_check", None):
        with set_options(src=src):
            return isconnected(AVAILABLE_SOURCES[src].api_server_check)
    else:
        raise InvalidFetcher


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

#
#  From xarrayutils : https://github.com/jbusecke/xarrayutils/blob/master/xarrayutils/vertical_coordinates.py
#  Direct integration of those 2 functions to minimize dependencies and possibility of tuning them to our needs
#


def linear_interpolation_remap(z, data, z_regridded, z_dim=None, z_regridded_dim="regridded", output_dim="remapped"):

    # interpolation called in xarray ufunc
    def _regular_interp(x, y, target_values):
        # remove all nans from input x and y
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~idx]
        y = y[~idx]
        # replace nans in target_values with out of bound Values (just in case)
        target_values = np.where(
            ~np.isnan(target_values), target_values, np.nanmax(x) + 1)
        # Interpolate with fill value parameter to extend min pressure toward 0
        interpolated = interpolate.interp1d(
            x, y, bounds_error=False, fill_value=(y[0], y[-1]))(target_values)
        return interpolated

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError(
                "if z_dim is not specified,x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a datset input just apply the function for each appropriate array

    kwargs = dict(
        input_core_dims=[[dim], [dim], [z_regridded_dim]],
        output_core_dims=[[output_dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
        output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
    )
    remapped = xr.apply_ufunc(
        _regular_interp, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped

####################
