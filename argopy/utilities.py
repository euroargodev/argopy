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
import urllib
import json
import collections
import copy
from functools import reduce
from packaging import version

import importlib
import locale
import platform
import struct
import subprocess
import contextlib

import xarray as xr
import pandas as pd
import numpy as np
from scipy import interpolate

import pickle
import pkg_resources
import shutil

import threading

import time

from argopy.options import OPTIONS, set_options
from argopy.stores import httpstore
from argopy.errors import (
    FtpPathError,
    InvalidFetcher,
    InvalidFetcherAccessPoint,
    InvalidOption
)

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


path2pkl = pkg_resources.resource_filename("argopy", "assets/")

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


def clear_cache(fs=None):
    """ Delete argopy cache folder content """
    if os.path.exists(OPTIONS["cachedir"]):
        # shutil.rmtree(OPTIONS["cachedir"])
        for filename in os.listdir(OPTIONS["cachedir"]):
            file_path = os.path.join(OPTIONS["cachedir"], filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
        if fs:
            fs.clear_cache()


def load_dict(ptype):
    if ptype == "profilers":
        with open(os.path.join(path2pkl, "dict_profilers.pickle"), "rb") as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    elif ptype == "institutions":
        with open(os.path.join(path2pkl, "dict_institutions.pickle"), "rb") as f:
            loaded_dict = pickle.load(f)
        return loaded_dict
    else:
        raise ValueError("Invalid dictionary pickle file")


def mapp_dict(Adictionnary, Avalue):
    if Avalue not in Adictionnary:
        return "Unknown"
    else:
        return Adictionnary[Avalue]


def list_available_data_src():
    """ List all available data sources """
    sources = {}
    try:
        from .data_fetchers import erddap_data as Erddap_Fetchers

        sources["erddap"] = Erddap_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ERDDAP data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from .data_fetchers import localftp_data as LocalFTP_Fetchers

        sources["localftp"] = LocalFTP_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the local FTP data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from .data_fetchers import argovis_data as ArgoVis_Fetchers

        sources["argovis"] = ArgoVis_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ArgoVis data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    # return dict(sorted(sources.items()))
    return sources


def list_available_index_src():
    """ List all available index sources """
    AVAILABLE_SOURCES = {}
    try:
        from .data_fetchers import erddap_index as Erddap_Fetchers

        AVAILABLE_SOURCES["erddap"] = Erddap_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ERDDAP index fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from .data_fetchers import localftp_index as LocalFTP_Fetchers

        AVAILABLE_SOURCES["localftp"] = LocalFTP_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the local FTP index fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    return AVAILABLE_SOURCES


def list_standard_variables():
    """ List of variables for standard users """
    return [
        "DATA_MODE",
        "LATITUDE",
        "LONGITUDE",
        "POSITION_QC",
        "DIRECTION",
        "PLATFORM_NUMBER",
        "CYCLE_NUMBER",
        "PRES",
        "TEMP",
        "PSAL",
        "PRES_QC",
        "TEMP_QC",
        "PSAL_QC",
        "PRES_ADJUSTED",
        "TEMP_ADJUSTED",
        "PSAL_ADJUSTED",
        "PRES_ADJUSTED_QC",
        "TEMP_ADJUSTED_QC",
        "PSAL_ADJUSTED_QC",
        "PRES_ADJUSTED_ERROR",
        "TEMP_ADJUSTED_ERROR",
        "PSAL_ADJUSTED_ERROR",
        "JULD",
        "JULD_QC",
        "TIME",
        "TIME_QC",
        "CONFIG_MISSION_NUMBER",
    ]


def list_multiprofile_file_variables():
    """ List of variables in a netcdf multiprofile file.

        This is for files created by GDAC under <DAC>/<WMO>/<WMO>_prof.nc
    """
    return [
        "CONFIG_MISSION_NUMBER",
        "CYCLE_NUMBER",
        "DATA_CENTRE",
        "DATA_MODE",
        "DATA_STATE_INDICATOR",
        "DATA_TYPE",
        "DATE_CREATION",
        "DATE_UPDATE",
        "DC_REFERENCE",
        "DIRECTION",
        "FIRMWARE_VERSION",
        "FLOAT_SERIAL_NO",
        "FORMAT_VERSION",
        "HANDBOOK_VERSION",
        "HISTORY_ACTION",
        "HISTORY_DATE",
        "HISTORY_INSTITUTION",
        "HISTORY_PARAMETER",
        "HISTORY_PREVIOUS_VALUE",
        "HISTORY_QCTEST",
        "HISTORY_REFERENCE",
        "HISTORY_SOFTWARE",
        "HISTORY_SOFTWARE_RELEASE",
        "HISTORY_START_PRES",
        "HISTORY_STEP",
        "HISTORY_STOP_PRES",
        "JULD",
        "JULD_LOCATION",
        "JULD_QC",
        "LATITUDE",
        "LONGITUDE",
        "PARAMETER",
        "PI_NAME",
        "PLATFORM_NUMBER",
        "PLATFORM_TYPE",
        "POSITIONING_SYSTEM",
        "POSITION_QC",
        "PRES",
        "PRES_ADJUSTED",
        "PRES_ADJUSTED_ERROR",
        "PRES_ADJUSTED_QC",
        "PRES_QC",
        "PROFILE_PRES_QC",
        "PROFILE_PSAL_QC",
        "PROFILE_TEMP_QC",
        "PROJECT_NAME",
        "PSAL",
        "PSAL_ADJUSTED",
        "PSAL_ADJUSTED_ERROR",
        "PSAL_ADJUSTED_QC",
        "PSAL_QC",
        "REFERENCE_DATE_TIME",
        "SCIENTIFIC_CALIB_COEFFICIENT",
        "SCIENTIFIC_CALIB_COMMENT",
        "SCIENTIFIC_CALIB_DATE",
        "SCIENTIFIC_CALIB_EQUATION",
        "STATION_PARAMETERS",
        "TEMP",
        "TEMP_ADJUSTED",
        "TEMP_ADJUSTED_ERROR",
        "TEMP_ADJUSTED_QC",
        "TEMP_QC",
        "VERTICAL_SAMPLING_SCHEME",
        "WMO_INST_TYPE",
    ]


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

        Returns
        -------
        checked: boolean
            True if at least one DAC folder is found under path/dac/<dac_name>
            False otherwise
    """
    dacs = [
        "aoml",
        "bodc",
        "coriolis",
        "csio",
        "csiro",
        "incois",
        "jma",
        "kma",
        "kordi",
        "meds",
        "nmdis",
    ]

    # Case 1:
    check1 = (
        os.path.isdir(path)
        and os.path.isdir(os.path.join(path, "dac"))
        and np.any([os.path.isdir(os.path.join(path, "dac", dac)) for dac in dacs])
    )

    if check1:
        return True
    elif errors == "raise":
        # This was possible up to v0.1.3:
        check2 = os.path.isdir(path) and np.any(
            [os.path.isdir(os.path.join(path, dac)) for dac in dacs]
        )
        if check2:
            raise FtpPathError(
                "This path is no longer GDAC compliant for argopy.\n"
                "Please make sure you point toward a path with a 'dac' folder:\n%s"
                % path
            )
        else:
            raise FtpPathError("This path is not GDAC compliant:\n%s" % path)

    elif errors == "warn":
        # This was possible up to v0.1.3:
        check2 = os.path.isdir(path) and np.any(
            [os.path.isdir(os.path.join(path, dac)) for dac in dacs]
        )
        if check2:
            warnings.warn(
                "This path is no longer GDAC compliant for argopy. This will raise an error in the future.\n"
                "Please make sure you point toward a path with a 'dac' folder:\n%s"
                % path
            )
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
        (sysname, nodename, release, version_, machine, processor) = platform.uname()
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


def show_versions(file=sys.stdout):  # noqa: C901
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
        # In REQUIREMENTS:
        ("argopy", lambda mod: mod.__version__),
        ("xarray", lambda mod: mod.__version__),
        ("scipy", lambda mod: mod.__version__),
        ("sklearn", lambda mod: mod.__version__),
        ("netCDF4", lambda mod: mod.__version__),
        ("dask", lambda mod: mod.__version__),
        ("toolz", lambda mod: mod.__version__),
        ("erddapy", lambda mod: mod.__version__),
        ("fsspec", lambda mod: mod.__version__),
        ("gsw", lambda mod: mod.__version__),
        ("aiohttp", lambda mod: mod.__version__),
        #
        ("bottleneck", lambda mod: mod.__version__),
        ("cartopy", lambda mod: mod.__version__),
        ("cftime", lambda mod: mod.__version__),
        ("conda", lambda mod: mod.__version__),
        ("distributed", lambda mod: mod.__version__),
        ("IPython", lambda mod: mod.__version__),
        ("iris", lambda mod: mod.__version__),
        ("matplotlib", lambda mod: mod.__version__),
        ("nc_time_axis", lambda mod: mod.__version__),
        ("numpy", lambda mod: mod.__version__),
        ("pandas", lambda mod: mod.__version__),
        ("packaging", lambda mod: mod.__version__),
        ("pip", lambda mod: mod.__version__),
        ("PseudoNetCDF", lambda mod: mod.__version__),
        ("pytest", lambda mod: mod.__version__),
        ("seaborn", lambda mod: mod.__version__),
        ("setuptools", lambda mod: mod.__version__),
        ("sphinx", lambda mod: mod.__version__),
        ("zarr", lambda mod: mod.__version__),
        ("tqdm", lambda mod: mod.__version__),
        ("ipykernel", lambda mod: mod.__version__),
        ("ipywidgets", lambda mod: mod.__version__),
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


def show_options(file=sys.stdout):  # noqa: C901
    """ Print options of argopy

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    print("\nARGOPY OPTIONS", file=file)
    print("--------------", file=file)
    opts = copy.deepcopy(OPTIONS)
    opts = dict(sorted(opts.items()))
    for k, v in opts.items():
        print(f"{k}: {v}", file=file)


def isconnected(host="https://www.ifremer.fr"):
    """ check if we have a live internet connection

        Parameters
        ----------
        host: str
            URL to use, 'https://www.ifremer.fr' by default

        Returns
        -------
        bool
    """
    if "http" in host or "ftp" in host:
        try:
            urllib.request.urlopen(host, timeout=1)  # Python 3.x
            return True
        except Exception:
            return False
    else:
        return os.path.exists(host)


def isAPIconnected(src="erddap", data=True):
    """ Check if a source API is alive or not

        The API is connected when it has a live URL or valid folder path.

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
        list_src = list_available_data_src()
    else:
        list_src = list_available_index_src()

    if src in list_src and getattr(
        list_src[src], "api_server_check", None
    ):
        if "localftp" in src:
            # This is a special case because the source here is a local folder
            result = check_localftp(OPTIONS["local_ftp"])
        else:
            result = isconnected(list_src[src].api_server_check)
        return result
    else:
        raise InvalidFetcher


def erddap_ds_exists(ds: str = "ArgoFloats", erddap: str = 'https://www.ifremer.fr/erddap') -> bool:
    """ Check if a dataset exists on a remote erddap server
    return a bool

    Parameter
    ---------
    ds: str
        Name of the erddap dataset to check (default: 'ArgoFloats')
    erddap: str
        Url of the erddap server (default: 'https://www.ifremer.fr/erddap')

    Return
    ------
    bool
    """
    with httpstore(timeout=OPTIONS['api_timeout']).open("".join([erddap, "/info/index.json"])) as of:
        erddap_index = json.load(of)
    return ds in [row[-1] for row in erddap_index["table"]["rows"]]


def badge(label="label", message="message", color="green", insert=False):
    """ Return or insert shield.io badge image

        Use the shields.io service to create a badge image

        https://img.shields.io/static/v1?label=<LABEL>&message=<MESSAGE>&color=<COLOR>

    Parameters
    ----------
    label: str
        Left side badge text
    message: str
        Right side badge text
    color: str
        Right side background color
    insert: bool
        Return url to badge image (False, default) or directly insert the image with HTML (True)

    Returns
    -------
    str or IPython.display.Image
    """
    from IPython.display import Image

    url = (
        "https://img.shields.io/static/v1?style=flat-square&label={}&message={}&color={}"
    ).format
    img = url(urllib.parse.quote(label), urllib.parse.quote(message), color)
    if not insert:
        return img
    else:
        return Image(url=img)


def fetch_status(stdout: str = "html", insert: bool = True):
    """ Fetch and report web API status

    Parameters
    ----------
    stdout: str
        Format of the results, default is 'html'. Otherwise a simple string.
    insert: bool
        Print or display results directly in stdout format.

    Returns
    -------
    IPython.display.HTML or str
    """
    results = {}
    list_src = list_available_data_src()
    for api, mod in list_src.items():
        if getattr(mod, "api_server_check", None):
            # status = isconnected(mod.api_server_check)
            status = isAPIconnected(api)
            if api=='localftp' and OPTIONS['local_ftp'] == '-':
                message = "ok" if status else "path undefined !"
            else:
                # message = "up" if status else "down"
                message = "ok" if status else "offline"
            results[api] = {"value": status, "message": message}

    if "IPython" in sys.modules and stdout == "html":
        cols = []
        for api in sorted(results.keys()):
            color = "green" if results[api]["value"] else "orange"
            if isconnected():
                # img = badge("src='%s'" % api, message=results[api]['message'], color=color, insert=False)
                # img = badge(label="argopy src", message="%s is %s" %
                # (api, results[api]['message']), color=color, insert=False)
                img = badge(
                    label="src %s is" % api,
                    message="%s" % results[api]["message"],
                    color=color,
                    insert=False,
                )
                html = ('<td><img src="{}"></td>').format(img)
            else:
                # html = "<th>src %s is:</th><td>%s</td>" % (api, results[api]['message'])
                html = (
                    "<th><div>src %s is:</div></th><td><div style='color:%s;'>%s</div></td>"
                    % (api, color, results[api]["message"])
                )
            cols.append(html)
        this_HTML = ("<table><tr>{}</tr></table>").format("".join(cols))
        if insert:
            from IPython.display import HTML, display

            return display(HTML(this_HTML))
        else:
            return this_HTML
    else:
        rows = []
        for api in sorted(results.keys()):
            # rows.append("argopy src %s: %s" % (api, results[api]['message']))
            rows.append("src %s is: %s" % (api, results[api]["message"]))
        txt = "\n".join(rows)
        if insert:
            print(txt)
        else:
            return txt


class monitor_status:
    """ Monitor data source status with a refresh rate """

    def __init__(self, refresh=1):
        import ipywidgets as widgets

        self.refresh_rate = refresh
        self.text = widgets.HTML(
            value=fetch_status(stdout="html", insert=False),
            placeholder="",
            description="",
        )
        self.start()

    def work(self):
        while True:
            time.sleep(self.refresh_rate)
            self.text.value = fetch_status(stdout="html", insert=False)

    def start(self):
        from IPython.display import display

        thread = threading.Thread(target=self.work)
        display(self.text)
        thread.start()


# def open_etopo1(box, res="l"):
#     """ Download ETOPO for a box
#
#         Parameters
#         ----------
#         box: [xmin, xmax, ymin, ymax]
#
#         Returns
#         -------
#         xarray.Dataset
#     """
#     # This function is in utilities to anticipate usage outside of plotting, eg interpolation, grounding detection
#     resx, resy = 0.1, 0.1
#     if res == "h":
#         resx, resy = 0.016, 0.016
#
#     uri = (
#         "https://gis.ngdc.noaa.gov/mapviewer-support/wcs-proxy/wcs.groovy?filename=etopo1.nc"
#         "&request=getcoverage&version=1.0.0&service=wcs&coverage=etopo1&CRS=EPSG:4326&format=netcdf"
#         "&resx={}&resy={}"
#         "&bbox={}"
#     ).format
#     thisurl = uri(
#         resx, resy, ",".join([str(b) for b in [box[0], box[2], box[1], box[3]]])
#     )
#     ds = httpstore(cache=True).open_dataset(thisurl)
#     da = ds["Band1"].rename("topo")
#     for a in ds.attrs:
#         da.attrs[a] = ds.attrs[a]
#     da.attrs["Data source"] = "https://maps.ngdc.noaa.gov/viewers/wcs-client/"
#     da.attrs["URI"] = thisurl
#     return da


#
#  From xarrayutils : https://github.com/jbusecke/xarrayutils/blob/master/xarrayutils/vertical_coordinates.py
#  Direct integration of those 2 functions to minimize dependencies and possibility of tuning them to our needs
#


def linear_interpolation_remap(
    z, data, z_regridded, z_dim=None, z_regridded_dim="regridded", output_dim="remapped"
):

    # interpolation called in xarray ufunc
    def _regular_interp(x, y, target_values):
        # remove all nans from input x and y
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~idx]
        y = y[~idx]

        # Need at least 5 points in the profile to interpolate, otherwise, return NaNs
        if len(y) < 5:
            interpolated = np.empty(len(target_values))
            interpolated[:] = np.nan
        else:
            # replace nans in target_values with out of bound Values (just in case)
            target_values = np.where(
                ~np.isnan(target_values), target_values, np.nanmax(x) + 1
            )
            # Interpolate with fill value parameter to extend min pressure toward 0
            interpolated = interpolate.interp1d(
                x, y, bounds_error=False, fill_value=(y[0], y[-1])
            )(target_values)
        return interpolated

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a dataset input just apply the function for each appropriate array

    if version.parse(xr.__version__) > version.parse("0.15.0"):
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={'output_sizes': {output_dim: len(z_regridded[z_regridded_dim])}},
        )
    else:
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
        )
    remapped = xr.apply_ufunc(_regular_interp, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped


class Chunker:
    """ To chunk fetcher requests """

    # Default maximum chunks size for all possible request parameters
    default_chunksize = {
        "box": {
            "lon": 20,  # degree
            "lat": 20,  # degree
            "dpt": 500,  # meters/db
            "time": 3 * 30,
        },  # Days
        "wmo": {"wmo": 5, "cyc": 100},  # Nb of floats
    }  # Nb of cycles

    def __init__(self, request: dict, chunks: str = "auto", chunksize: dict = {}):
        """ Create a request Chunker

        Allow to easily split an access point request into chunks

        Parameters
        ----------
        request: dict
            Access point request to be chunked. One of the following:

            - {'box': [lon_min, lon_max, lat_min, lat_max, dpt_min, dpt_max, time_min, time_max]}
            - {'box': [lon_min, lon_max, lat_min, lat_max, dpt_min, dpt_max]}
            - {'wmo': [wmo1, wmo2, ...], 'cyc': [0,1, ...]}
        chunks: 'auto' or dict
            Dictionary with request access point as keys and number of chunks to create as values.

            Eg: {'wmo':10} will create a maximum of 10 chunks along WMOs.
        chunksize: dict, optional
            Dictionary with request access point as keys and chunk size as values (used as maximum values in
            'auto' chunking).

            Eg: {'wmo': 5} will create chunks with as many as 5 WMOs each.

        """
        self.request = request

        if "box" in self.request:
            is_box(self.request["box"])
            if len(self.request["box"]) == 8:
                self.this_chunker = self._chunker_box4d
            elif len(self.request["box"]) == 6:
                self.this_chunker = self._chunker_box3d
        elif "wmo" in self.request:
            self.this_chunker = self._chunker_wmo
        else:
            raise InvalidFetcherAccessPoint(
                "'%s' not valid access point" % ",".join(self.request.keys())
            )

        default = self.default_chunksize[[k for k in self.request.keys()][0]]
        if len(chunksize) == 0:  # chunksize = {}
            chunksize = default
        if not isinstance(chunksize, collectionsAbc.Mapping):
            raise ValueError("chunksize must be mappable")
        else:  # merge with default:
            chunksize = {**default, **chunksize}
        self.chunksize = collections.OrderedDict(sorted(chunksize.items()))

        default = {k: "auto" for k in self.chunksize.keys()}
        if chunks == "auto":  # auto for all
            chunks = default
        elif len(chunks) == 0:  # chunks = {}, i.e. chunk=1 for all
            chunks = {k: 1 for k in self.request}
        if not isinstance(chunks, collectionsAbc.Mapping):
            raise ValueError("chunks must be 'auto' or mappable")
        chunks = {**default, **chunks}
        self.chunks = collections.OrderedDict(sorted(chunks.items()))

    def _split(self, lst, n=1):
        """Yield successive n-sized chunks from lst"""
        for i in range(0, len(lst), n):
            yield lst[i: i + n]

    def _split_list_bychunknb(self, lst, n=1):
        """Split list in n-imposed chunks of similar size
            The last chunk may contain more or less element than the others, depending on the size of the list.
        """
        res = []
        siz = int(np.floor_divide(len(lst), n))
        for i in self._split(lst, siz):
            res.append(i)
        if len(res) > n:
            res[n - 1::] = [reduce(lambda i, j: i + j, res[n - 1::])]
        return res

    def _split_list_bychunksize(self, lst, max_size=1):
        """Split list in chunks of imposed size
            The last chunk may contain more or less element than the others, depending on the size of the list.
        """
        res = []
        for i in self._split(lst, max_size):
            res.append(i)
        return res

    def _split_box(self, large_box, n=1, d="x"):  # noqa: C901
        """Split a box domain in one direction in n-imposed equal chunks """
        if d == "x":
            i_left, i_right = 0, 1
        if d == "y":
            i_left, i_right = 2, 3
        if d == "z":
            i_left, i_right = 4, 5
        if d == "t":
            i_left, i_right = 6, 7
        if n == 1:
            return [large_box]
        boxes = []
        if d in ["x", "y", "z"]:
            n += 1  # Required because we split in linspace
            bins = np.linspace(large_box[i_left], large_box[i_right], n)
            for ii, left in enumerate(bins):
                if ii < len(bins) - 1:
                    right = bins[ii + 1]
                    this_box = large_box.copy()
                    this_box[i_left] = left
                    this_box[i_right] = right
                    boxes.append(this_box)
        elif "t" in d:
            dates = pd.to_datetime(large_box[i_left: i_right + 1])
            date_bounds = [
                d.strftime("%Y%m%d%H%M%S")
                for d in pd.date_range(dates[0], dates[1], periods=n + 1)
            ]
            for i1, i2 in zip(np.arange(0, n), np.arange(1, n + 1)):
                left, right = date_bounds[i1], date_bounds[i2]
                this_box = large_box.copy()
                this_box[i_left] = left
                this_box[i_right] = right
                boxes.append(this_box)
        return boxes

    def _split_this_4Dbox(self, box, nx=1, ny=1, nz=1, nt=1):
        box_list = []
        split_x = self._split_box(box, n=nx, d="x")
        for bx in split_x:
            split_y = self._split_box(bx, n=ny, d="y")
            for bxy in split_y:
                split_z = self._split_box(bxy, n=nz, d="z")
                for bxyz in split_z:
                    split_t = self._split_box(bxyz, n=nt, d="t")
                    for bxyzt in split_t:
                        box_list.append(bxyzt)
        return box_list

    def _split_this_3Dbox(self, box, nx=1, ny=1, nz=1):
        box_list = []
        split_x = self._split_box(box, n=nx, d="x")
        for bx in split_x:
            split_y = self._split_box(bx, n=ny, d="y")
            for bxy in split_y:
                split_z = self._split_box(bxy, n=nz, d="z")
                for bxyz in split_z:
                    box_list.append(bxyz)
        return box_list

    def _chunker_box4d(self, request, chunks, chunks_maxsize):  # noqa: C901
        BOX = request["box"]
        n_chunks = chunks
        for axis, n in n_chunks.items():
            if n == "auto":
                if axis == "lon":
                    Lx = BOX[1] - BOX[0]
                    if Lx > chunks_maxsize["lon"]:  # Max box size in longitude
                        n_chunks["lon"] = int(
                            np.ceil(np.divide(Lx, chunks_maxsize["lon"]))
                        )
                    else:
                        n_chunks["lon"] = 1
                if axis == "lat":
                    Ly = BOX[3] - BOX[2]
                    if Ly > chunks_maxsize["lat"]:  # Max box size in latitude
                        n_chunks["lat"] = int(
                            np.ceil(np.divide(Ly, chunks_maxsize["lat"]))
                        )
                    else:
                        n_chunks["lat"] = 1
                if axis == "dpt":
                    Lz = BOX[5] - BOX[4]
                    if Lz > chunks_maxsize["dpt"]:  # Max box size in depth
                        n_chunks["dpt"] = int(
                            np.ceil(np.divide(Lz, chunks_maxsize["dpt"]))
                        )
                    else:
                        n_chunks["dpt"] = 1
                if axis == "time":
                    Lt = np.timedelta64(
                        pd.to_datetime(BOX[7]) - pd.to_datetime(BOX[6]), "D"
                    )
                    MaxLen = np.timedelta64(chunks_maxsize["time"], "D")
                    if Lt > MaxLen:  # Max box size in time
                        n_chunks["time"] = int(np.ceil(np.divide(Lt, MaxLen)))
                    else:
                        n_chunks["time"] = 1

        boxes = self._split_this_4Dbox(
            BOX,
            nx=n_chunks["lon"],
            ny=n_chunks["lat"],
            nz=n_chunks["dpt"],
            nt=n_chunks["time"],
        )
        return {"chunks": sorted(n_chunks), "values": boxes}

    def _chunker_box3d(self, request, chunks, chunks_maxsize):
        BOX = request["box"]
        n_chunks = chunks
        for axis, n in n_chunks.items():
            if n == "auto":
                if axis == "lon":
                    Lx = BOX[1] - BOX[0]
                    if Lx > chunks_maxsize["lon"]:  # Max box size in longitude
                        n_chunks["lon"] = int(
                            np.floor_divide(Lx, chunks_maxsize["lon"])
                        )
                    else:
                        n_chunks["lon"] = 1
                if axis == "lat":
                    Ly = BOX[3] - BOX[2]
                    if Ly > chunks_maxsize["lat"]:  # Max box size in latitude
                        n_chunks["lat"] = int(
                            np.floor_divide(Ly, chunks_maxsize["lat"])
                        )
                    else:
                        n_chunks["lat"] = 1
                if axis == "dpt":
                    Lz = BOX[5] - BOX[4]
                    if Lz > chunks_maxsize["dpt"]:  # Max box size in depth
                        n_chunks["dpt"] = int(
                            np.floor_divide(Lz, chunks_maxsize["dpt"])
                        )
                    else:
                        n_chunks["dpt"] = 1
                # if axis == 'time':
                #     Lt = np.timedelta64(pd.to_datetime(BOX[5]) - pd.to_datetime(BOX[4]), 'D')
                #     MaxLen = np.timedelta64(chunks_maxsize['time'], 'D')
                #     if Lt > MaxLen:  # Max box size in time
                #         n_chunks['time'] = int(np.floor_divide(Lt, MaxLen))
                #     else:
                #         n_chunks['time'] = 1
        boxes = self._split_this_3Dbox(
            BOX, nx=n_chunks["lon"], ny=n_chunks["lat"], nz=n_chunks["dpt"]
        )
        return {"chunks": sorted(n_chunks), "values": boxes}

    def _chunker_wmo(self, request, chunks, chunks_maxsize):
        WMO = request["wmo"]
        n_chunks = chunks
        if n_chunks["wmo"] == "auto":
            wmo_grps = self._split_list_bychunksize(WMO, max_size=chunks_maxsize["wmo"])
        else:
            n = np.min([n_chunks["wmo"], len(WMO)])
            wmo_grps = self._split_list_bychunknb(WMO, n=n)
        n_chunks["wmo"] = len(wmo_grps)
        return {"chunks": sorted(n_chunks), "values": wmo_grps}

    def fit_transform(self):
        """ Chunk a fetcher request

        Returns
        -------
        list
        """
        self._results = self.this_chunker(self.request, self.chunks, self.chunksize)
        # self.chunks = self._results['chunks']
        return self._results["values"]


def format_oneline(s, max_width=65):
    """ Return a string formatted for a line print """
    if len(s) > max_width:
        padding = " ... "
        n = (max_width - len(padding)) // 2
        q = (max_width - len(padding)) % 2
        if q == 0:
            return "".join([s[0:n], padding, s[-n:]])
        else:
            return "".join([s[0: n + 1], padding, s[-n:]])
    else:
        return s


def is_indexbox(box: list, errors="raise"):
    """ Check if this array matches a 2d or 3d index box definition

        box = [lon_min, lon_max, lat_min, lat_max]
    or:
        box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]

    Parameters
    ----------
    box: list
    errors: 'raise'

    Returns
    -------
    bool
    """

    tests = {}

    # Formats:
    tests["index box must be a list"] = lambda b: isinstance(b, list)
    tests["index box must be a list with 4 or 6 elements"] = lambda b: len(b) in [4, 6]

    error = None
    for msg, test in tests.items():
        if not test(box):
            error = msg
            break

    if error and errors == "raise":
        raise ValueError("%s: %s" % (box, error))
    elif error:
        return False
    else:
        # Insert pressure bounds and use full box validator:
        tmp_box = box.copy()
        tmp_box.insert(4, 0.)
        tmp_box.insert(5, 10000.)
        return is_box(tmp_box, errors=errors)


def is_box(box: list, errors="raise"):
    """ Check if this array matches a 3d or 4d data box definition

        box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
    or:
        box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]

    Parameters
    ----------
    box: list
    errors: 'raise'

    Returns
    -------
    bool
    """
    def is_dateconvertible(d):
        try:
            pd.to_datetime(d)
            isit = True
        except Exception:
            isit = False
        return isit

    tests = {}
    #     print(box)
    # Formats:
    tests["box must be a list"] = lambda b: isinstance(b, list)
    tests["box must be a list with 6 or 8 elements"] = lambda b: len(b) in [6, 8]

    # Types:
    tests["lon_min must be numeric"] = lambda b: (
        isinstance(b[0], int) or isinstance(b[0], (np.floating, float))
    )
    tests["lon_max must be numeric"] = lambda b: (
        isinstance(b[1], int) or isinstance(b[1], (np.floating, float))
    )
    tests["lat_min must be numeric"] = lambda b: (
        isinstance(b[2], int) or isinstance(b[2], (np.floating, float))
    )
    tests["lat_max must be numeric"] = lambda b: (
        isinstance(b[3], int) or isinstance(b[3], (np.floating, float))
    )
    tests["pres_min must be numeric"] = lambda b: (
        isinstance(b[4], int) or isinstance(b[4], (np.floating, float))
    )
    tests["pres_max must be numeric"] = lambda b: (
        isinstance(b[5], int) or isinstance(b[5], (np.floating, float))
    )
    if len(box) == 8:
        tests[
            "datetim_min must be a string convertible to a Pandas datetime"
        ] = lambda b: isinstance(b[-2], str) and is_dateconvertible(b[-2])
        tests[
            "datetim_max must be a string convertible to a Pandas datetime"
        ] = lambda b: isinstance(b[-1], str) and is_dateconvertible(b[-1])

    # Ranges:
    tests["lon_min must be in [-180;180] or [0;360]"] = (
        lambda b: b[0] >= -180.0 and b[0] <= 360.0
    )
    tests["lon_max must be in [-180;180] or [0;360]"] = (
        lambda b: b[1] >= -180.0 and b[1] <= 360.0
    )
    tests["lat_min must be in [-90;90]"] = lambda b: b[2] >= -90.0 and b[2] <= 90
    tests["lat_max must be in [-90;90]"] = lambda b: b[3] >= -90.0 and b[3] <= 90.0
    tests["pres_min must be in [0;10000]"] = lambda b: b[4] >= 0 and b[4] <= 10000
    tests["pres_max must be in [0;10000]"] = lambda b: b[5] >= 0 and b[5] <= 10000

    # Orders:
    tests["lon_max must be larger than lon_min"] = lambda b: b[0] < b[1]
    tests["lat_max must be larger than lat_min"] = lambda b: b[2] < b[3]
    tests["pres_max must be larger than pres_min"] = lambda b: b[4] < b[5]
    if len(box) == 8:
        tests["datetim_max must come after datetim_min"] = lambda b: pd.to_datetime(
            b[-2]
        ) < pd.to_datetime(b[-1])

    error = None
    for msg, test in tests.items():
        if not test(box):
            error = msg
            break

    if error and errors == "raise":
        raise ValueError("%s: %s" % (box, error))
    elif error:
        return False
    else:
        return True


def is_list_of_strings(lst):
    return isinstance(lst, list) and all(isinstance(elem, str) for elem in lst)


def is_list_of_dicts(lst):
    return all(isinstance(x, dict) for x in lst)


def is_list_of_datasets(lst):
    return all(isinstance(x, xr.Dataset) for x in lst)


def is_list_equal(lst1, lst2):
    """ Return true if 2 lists contain same elements"""
    return len(lst1) == len(lst2) and len(lst1) == sum([1 for i, j in zip(lst1, lst2) if i == j])


def check_wmo(lst):
    """ Validate a WMO option and returned it as a list of integers

    Parameters
    ----------
    wmo: int
        WMO must be an integer or an iterable with elements that can be casted as integers
    errors: 'raise'

    Returns
    -------
    list(int)
    """
    is_wmo(lst, errors="raise")

    # Make sure we deal with a list
    if not isinstance(lst, list):
        if isinstance(lst, np.ndarray):
            lst = list(lst)
        else:
            lst = [lst]

    # Then cast list elements as integers
    return [abs(int(x)) for x in lst]


def is_wmo(lst, errors="raise"):  # noqa: C901
    """ Check if a WMO is valid

    Parameters
    ----------
    wmo: int, list(int), array(int)
        WMO must be a single or a list of 5/7 digit positive numbers
    errors: 'raise'
        Possibly raises a ValueError exception, otherwise fails silently.

    Returns
    -------
    bool
        True if wmo is indeed a list of integers
    """

    # Make sure we deal with a list
    if not isinstance(lst, list):
        if isinstance(lst, np.ndarray):
            lst = list(lst)
        else:
            lst = [lst]

    # Error message:
    # msg = "WMO must be an integer or an iterable with elements that can be casted as integers"
    msg = "WMO must be a single or a list of 5/7 digit positive numbers"

    # Then try to cast list elements as integers, return True if ok
    result = True
    try:
        for x in lst:
            if not str(x).isdigit():
                result = False

            if (len(str(x)) != 5) and (len(str(x)) != 7):
                result = False

            if int(x) <= 0:
                result = False

    except Exception:
        result = False
        if errors == "raise":
            raise ValueError(msg)

    if not result and errors == "raise":
        raise ValueError(msg)
    else:
        return result


# def docstring(value):
#     """Replace one function docstring
#
#         To be used as a decorator
#     """
#     def _doc(func):
#         func.__doc__ = value
#         return func
#     return _doc


def warnUnless(ok, txt):
    """ Decorator to raise warning unless condition is True

    This function must be used as a decorator

    Parameters
    ----------
    ok: bool
        Condition to raise the warning or not
    txt: str
        Text to display in the warning
    """
    def inner(fct):
        def wrapper(*args, **kwargs):
            warnings.warn("%s %s" % (fct.__name__, txt))
            return fct(*args, **kwargs)

        return wrapper

    if not ok:
        return inner
    else:
        return lambda f: f


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    # Source: https://github.com/laurent-laporte-pro/stackoverflow-q2059482
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def toYearFraction(this_date: pd._libs.tslibs.timestamps.Timestamp = pd.to_datetime('now', utc=True)):
    """ Compute decimal year, robust to leap years, precision to the second

    Compute the fraction of the year a given timestamp corresponds to.
    The "fraction of the year" goes:
    - from 0 on 01-01T00:00:00.000 of the year
    - to 1 on the 01-01T00:00:00.000 of the following year

    1 second corresponds to the number of days in the year times 86400.
    The faction of the year is rounded to 10-digits in order to have a "second" precision.

    See discussion here: https://github.com/euroargodev/argodmqc_owc/issues/35

    Parameters
    ----------
    pd._libs.tslibs.timestamps.Timestamp

    Returns
    -------
    float
    """
    if "UTC" in [this_date.tzname() if not this_date.tzinfo is None else ""]:
        startOfThisYear = pd.to_datetime("%i-01-01T00:00:00.000" % this_date.year, utc=True)
    else:
        startOfThisYear = pd.to_datetime("%i-01-01T00:00:00.000" % this_date.year)
    yearDuration_sec = (startOfThisYear + pd.offsets.DateOffset(years=1) - startOfThisYear).total_seconds()

    yearElapsed_sec = (this_date - startOfThisYear).total_seconds()
    fraction = yearElapsed_sec / yearDuration_sec
    fraction = np.round(fraction, 10)
    return this_date.year + fraction


def YearFraction_to_datetime(yf: float):
    """ Compute datetime from year fraction

    Inverse the toYearFraction() function

    Parameters
    ----------
    float

    Returns
    -------
    pd._libs.tslibs.timestamps.Timestamp
    """
    year = np.int32(yf)
    fraction = yf - year
    fraction = np.round(fraction, 10)

    startOfThisYear = pd.to_datetime("%i-01-01T00:00:00" % year)
    yearDuration_sec = (startOfThisYear + pd.offsets.DateOffset(years=1) - startOfThisYear).total_seconds()
    yearElapsed_sec = pd.Timedelta(fraction * yearDuration_sec, unit='s')
    return pd.to_datetime(startOfThisYear + yearElapsed_sec, unit='s')


def wrap_longitude(grid_long):
    """ Allows longitude (0-360) to wrap beyond the 360 mark, for mapping purposes.
        Makes sure that, if the longitude is near the boundary (0 or 360) that we
        wrap the values beyond
        360 so it appears nicely on a map
        This is a refactor between get_region_data and get_region_hist_locations to
        avoid duplicate code

        source: https://github.com/euroargodev/argodmqc_owc/blob/e174f4538fdae1534c9740491398972b1ffec3ca/pyowc/utilities.py#L80

        Parameters
        ----------
        grid_long: array of longitude values

        Returns
        -------
        array of longitude values that can extend past 360
    """
    neg_long = np.argwhere(grid_long < 0)
    grid_long[neg_long] = grid_long[neg_long] + 360

    # if we have data close to upper boundary (360), then wrap some of the data round
    # so it appears on the map
    top_long = np.argwhere(grid_long >= 320)
    if top_long.__len__() != 0:
        bottom_long = np.argwhere(grid_long <= 40)
        grid_long[bottom_long] = 360 + grid_long[bottom_long]

    return grid_long


def wmo2box(wmo_id: int):
    """ Convert WMO square box number into a latitude/longitude box

    See:
    https://en.wikipedia.org/wiki/World_Meteorological_Organization_squares
    https://commons.wikimedia.org/wiki/File:WMO-squares-global.gif

    Parameters
    ----------
    wmo_id: int
        WMO square number, must be between 1000 and 7817

    Returns
    -------
    box: list(int)
        [lon_min, lon_max, lat_min, lat_max] bounds to the WMO square number
    """
    if wmo_id < 1000 or wmo_id > 7817:
        raise ValueError("Invalid WMO square number, must be between 1000 and 7817.")
    wmo_id = str(wmo_id)

    # "global quadrant" numbers where 1=NE, 3=SE, 5=SW, 7=NW
    quadrant = int(wmo_id[0])
    if quadrant not in [1, 3, 5 ,7]:
        raise ValueError("Invalid WMO square number, 1st digit must be 1, 3, 5 or 7.")

    # 'minimum' Latitude square boundary, nearest to the Equator
    nearest_to_the_Equator_latitude = int(wmo_id[1])

    # 'minimum' Longitude square boundary, nearest to the Prime Meridian
    nearest_to_the_Prime_Meridian = int(wmo_id[2:4])

    #
    dd = 10
    if quadrant in [1, 3]:
        lon_min = nearest_to_the_Prime_Meridian*dd
        lon_max = nearest_to_the_Prime_Meridian*dd+dd
    elif quadrant in [5, 7]:
        lon_min = -nearest_to_the_Prime_Meridian*dd-dd
        lon_max = -nearest_to_the_Prime_Meridian*dd

    if quadrant in [1, 7]:
        lat_min = nearest_to_the_Equator_latitude*dd
        lat_max = nearest_to_the_Equator_latitude*dd+dd
    elif quadrant in [3, 5]:
        lat_min = -nearest_to_the_Equator_latitude*dd-dd
        lat_max = -nearest_to_the_Equator_latitude*dd

    box = [lon_min, lon_max, lat_min, lat_max]
    return box


def groupby_remap(z, data, z_regridded, z_dim=None, z_regridded_dim="regridded", output_dim="remapped", select='deep', right=False):
    """ todo: Need a docstring here !"""

    # sub-sampling called in xarray ufunc
    def _subsample_bins(x, y, target_values):
        # remove all nans from input x and y
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~idx]
        y = y[~idx]

        ifound = np.digitize(x, target_values, right=right)  # ``bins[i-1] <= x < bins[i]``
        ifound -= 1  # Because digitize returns a 1-based indexing, we need to remove 1
        y_binned = np.ones_like(target_values) * np.nan

        for ib, this_ibin in enumerate(np.unique(ifound)):
            ix = np.where(ifound == this_ibin)
            iselect = ix[-1]

            # Map to y value at specific x index in the bin:
            if select == 'shallow':
                iselect = iselect[0]  # min/shallow
                mapped_value = y[iselect]
            elif select == 'deep':
                iselect = iselect[-1]  # max/deep
                mapped_value = y[iselect]
            elif select == 'middle':
                iselect = iselect[np.where(x[iselect] >= np.median(x[iselect]))[0][0]]  # median/middle
                mapped_value = y[iselect]
            elif select == 'random':
                iselect = iselect[np.random.randint(len(iselect))]
                mapped_value = y[iselect]

            # or Map to y statistics in the bin:
            elif select == 'mean':
                mapped_value = np.nanmean(y[iselect])
            elif select == 'min':
                mapped_value = np.nanmin(y[iselect])
            elif select == 'max':
                mapped_value = np.nanmax(y[iselect])
            elif select == 'median':
                mapped_value = np.median(y[iselect])

            else:
                raise InvalidOption("`select` option has invalid value (%s)" % select)

            y_binned[this_ibin] = mapped_value

        return y_binned

    # infer dim from input
    if z_dim is None:
        if len(z.dims) != 1:
            raise RuntimeError("if z_dim is not specified, x must be a 1D array.")
        dim = z.dims[0]
    else:
        dim = z_dim

    # if dataset is passed drop all data_vars that dont contain dim
    if isinstance(data, xr.Dataset):
        raise ValueError("Dataset input is not supported yet")
        # TODO: for a dataset input just apply the function for each appropriate array

    if version.parse(xr.__version__) > version.parse("0.15.0"):
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            dask_gufunc_kwargs={'output_sizes': {output_dim: len(z_regridded[z_regridded_dim])}},
        )
    else:
        kwargs = dict(
            input_core_dims=[[dim], [dim], [z_regridded_dim]],
            output_core_dims=[[output_dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[data.dtype],
            output_sizes={output_dim: len(z_regridded[z_regridded_dim])},
        )
    remapped = xr.apply_ufunc(_subsample_bins, z, data, z_regridded, **kwargs)

    remapped.coords[output_dim] = z_regridded.rename({z_regridded_dim: output_dim}).coords[output_dim]
    return remapped


class TopoFetcher():
    """ Fetch topographic data through an ERDDAP server for an ocean rectangle

    Example:
        >>> from argopy import TopoFetcher
        >>> box = [-75, -45, 20, 30]  # Lon_min, lon_max, lat_min, lat_max
        >>> ds = TopoFetcher(box).to_xarray()
        >>> ds = TopoFetcher(box, ds='gebco', stride=[10, 10], cache=True).to_xarray()

    """

    class ERDDAP():
        def __init__(self, server: str, protocol: str = 'tabledap'):
            self.server = server
            self.protocol = protocol
            self.response = 'nc'
            self.dataset_id = ''
            self.constraints = ''

    def __init__(
            self,
            box: list,
            ds: str = "gebco",
            cache: bool = False,
            cachedir: str = "",
            api_timeout: int = 0,
            stride: list = [1, 1],
            **kwargs,
    ):
        """ Instantiate an ERDDAP topo data fetcher

        Parameters
        ----------
        ds: str (optional), default: 'gebco'
            Dataset to load:

            - 'gebco' will load the GEBCO_2020 Grid, a continuous terrain model for oceans and land at 15 arc-second intervals
        stride: list, default [1, 1]
            Strides along longitude and latitude. This allows to change the output resolution
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        api_timeout: int (optional)
            Erddap request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout, size_policy='head')
        self.definition = "Erddap topographic data fetcher"

        self.BOX = box
        self.stride = stride
        if ds == "gebco":
            self.definition = "NOAA erddap gebco data fetcher for a space region"
            self.server = 'https://coastwatch.pfeg.noaa.gov/erddap'
            self.server_name = 'NOAA'
            self.dataset_id = 'gebco'

        self._init_erddap()

    def _init_erddap(self):
        # Init erddap
        self.erddap = self.ERDDAP(server=self.server, protocol="griddap")
        self.erddap.response = (
            "nc"
        )

        if self.dataset_id == "gebco":
            self.erddap.dataset_id = "GEBCO_2020"
        else:
            raise ValueError(
                "Invalid database short name for %s erddap" % self.server_name
            )
        return self

    def _cname(self) -> str:
        """ Fetcher one line string definition helper """
        cname = "?"

        if hasattr(self, "BOX"):
            BOX = self.BOX
            cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % (
                BOX[0],
                BOX[1],
                BOX[2],
                BOX[3],
            )
        return cname

    def __repr__(self):
        summary = ["<topofetcher.erddap>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % self.server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return "\n".join(summary)

    def cname(self):
        """ Return a unique string defining the constraints """
        return self._cname()

    @property
    def cachepath(self):
        """ Return path to cached file(s) for this request

        Returns
        -------
        list(str)
        """
        return [self.fs.cachepath(uri) for uri in self.uri]

    def define_constraints(self):
        """ Define request constraints """
        #        Eg: https://coastwatch.pfeg.noaa.gov/erddap/griddap/GEBCO_2020.nc?elevation%5B(34):5:(42)%5D%5B(-21):7:(-12)%5D
        self.erddap.constraints = "%s(%0.2f):%i:(%0.2f)%s%s(%0.2f):%i:(%0.2f)%s" % (
        "%5B", self.BOX[2], self.stride[1], self.BOX[3], "%5D",
        "%5B", self.BOX[0], self.stride[0], self.BOX[1], "%5D")
        return None

    #     @property
    #     def _minimal_vlist(self):
    #         """ Return the minimal list of variables to retrieve """
    #         vlist = list()
    #         vlist.append("latitude")
    #         vlist.append("longitude")
    #         vlist.append("elevation")
    #         return vlist

    def get_url(self):
        """ Return the URL to download data requested

        Returns
        -------
        str
        """
        # First part of the URL:
        protocol = self.erddap.protocol
        dataset_id = self.erddap.dataset_id
        response = self.erddap.response
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"

        # Add variables to retrieve:
        variables = ["elevation"]
        variables = ",".join(variables)
        url += f"{variables}"

        # Add constraints:
        self.define_constraints()  # Define constraint to select this box of data (affect self.erddap.constraints)
        url += f"{self.erddap.constraints}"

        return url

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """
        return [self.get_url()]

    def to_xarray(self, errors: str = 'ignore'):
        """ Load Topographic data and return a xarray.DataSet """

        # Download data
        if len(self.uri) == 1:
            ds = self.fs.open_dataset(self.uri[0])

        return ds

    def load(self, errors: str = 'ignore'):
        """ Load Topographic data and return a xarray.DataSet """
        return self.to_xarray(errors=errors)
