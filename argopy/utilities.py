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
from functools import reduce

import importlib
import locale
import platform
import struct
import subprocess

import xarray as xr
import pandas as pd
import numpy as np
from scipy import interpolate

import pickle
import pkg_resources
import shutil

import threading

# from IPython.display import HTML, display
# import ipywidgets as widgets
import time

from argopy.options import OPTIONS, set_options
from argopy.stores import httpstore
from argopy.errors import (
    FtpPathError,
    InvalidFetcher,
    OptionValueError,
    InvalidFetcherAccessPoint,
)

path2pkl = pkg_resources.resource_filename("argopy", "assets/")


def clear_cache():
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

        AVAILABLE_SOURCES["erddap"] = Erddap_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ERDDAP data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from .data_fetchers import localftp_data as LocalFTP_Fetchers

        AVAILABLE_SOURCES["localftp"] = LocalFTP_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the local FTP data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    try:
        from .data_fetchers import argovis_data as ArgoVis_Fetchers

        AVAILABLE_SOURCES["argovis"] = ArgoVis_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the ArgoVis data fetcher, "
            "it will not be available !\n%s\n%s"
            % (sys.exc_info()[0], sys.exc_info()[1])
        )
        pass

    return AVAILABLE_SOURCES


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
    """ Return the list of variables for standard users """
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
    """ Return the list of variables in a netcdf multiprofile file.

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

        Returms
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
        ("fsspec", lambda mod: mod.__version__),
        ("erddapy", lambda mod: mod.__version__),
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
        ("gsw", lambda mod: mod.__version__),
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


def isconnected(host="http://www.ifremer.fr"):
    """ check if we have a live internet connection

        Parameters
        ----------
        host: str
            URL to use, 'http://www.ifremer.fr' by default

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
        AVAILABLE_SOURCES = list_available_data_src()
    else:
        AVAILABLE_SOURCES = list_available_index_src()
    if src in AVAILABLE_SOURCES and getattr(
        AVAILABLE_SOURCES[src], "api_server_check", None
    ):
        if "localftp" in src:
            # This is a special case because the source here is a local folder, and the folder validity is checked
            # when setting the option value of 'local_ftp'
            # So here, we just need to catch the appropriate error after a call to set_option
            opts = {"src": src, "local_ftp": OPTIONS["local_ftp"]}
            try:
                set_options(**opts)
                return True
            except OptionValueError:
                return False
        else:
            with set_options(src=src):
                return isconnected(AVAILABLE_SOURCES[src].api_server_check)
    else:
        raise InvalidFetcher


def erddap_ds_exists(ds="ArgoFloats"):
    """ Given erddap fetcher, check if a Dataset exists, return a bool"""
    # e = ArgoDataFetcher(src='erddap').float(wmo=0).fetcher
    # erddap_index = json.load(urlopen(e.erddap.server + "/info/index.json"))
    # erddap_index = json.load(urlopen("http://www.ifremer.fr/erddap/info/index.json"))
    with httpstore(timeout=120).open(
        "http://www.ifremer.fr/erddap/info/index.json"
    ) as of:
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


def fetch_status(stdout="html", insert=True):
    """ Fetch and report web API status """
    results = {}
    for api, mod in list_available_data_src().items():
        if getattr(mod, "api_server_check", None):
            # status = isconnected(mod.api_server_check)
            status = isAPIconnected(api)
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
            raise RuntimeError("if z_dim is not specified,x must be a 1D array.")
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
            {'box': [lon_min, lon_max, lat_min, lat_max, dpt_min, dpt_max, time_min, time_max]}
            {'box': [lon_min, lon_max, lat_min, lat_max, dpt_min, dpt_max]}
            {'wmo': [wmo1, wmo2, ...], 'cyc': [0,1, ...]}
        chunks: 'auto' or dict
            Dictionary with request access point as keys and number of chunks to create as values.
            Eg: {'wmo':10} will create a maximum of 10 chunks along WMOs.
        chunksize: dict, optional
            Dictionary with request access point as keys and chunk size as values (used as maximum values in
            'auto' chunking). Eg: {'wmo': 5} will create chunks with as many as 5 WMOs each.

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
        if not isinstance(chunksize, collections.Mapping):
            raise ValueError("chunksize must be mappable")
        else:  # merge with default:
            chunksize = {**default, **chunksize}
        self.chunksize = collections.OrderedDict(sorted(chunksize.items()))

        default = {k: "auto" for k in self.chunksize.keys()}
        if chunks == "auto":  # auto for all
            chunks = default
        elif len(chunks) == 0:  # chunks = {}, i.e. chunk=1 for all
            chunks = {k: 1 for k in self.request}
        if not isinstance(chunks, collections.Mapping):
            raise ValueError("chunks must be 'auto' or mappable")
        chunks = {**default, **chunks}
        self.chunks = collections.OrderedDict(sorted(chunks.items()))

    def _split(self, lst, n=1):
        """Yield successive n-sized chunks from lst"""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def _split_list_bychunknb(self, lst, n=1):
        """Split list in n-imposed chunks of similar size
            The last chunk may contain more or less element than the others, depending on the size of the list.
        """
        res = []
        siz = int(np.floor_divide(len(lst), n))
        for i in self._split(lst, siz):
            res.append(i)
        if len(res) > n:
            res[n - 1 : :] = [reduce(lambda i, j: i + j, res[n - 1 : :])]
        return res

    def _split_list_bychunksize(self, lst, max_size=1):
        """Split list in chunks of imposed size
            The last chunk may contain more or less element than the others, depending on the size of the list.
        """
        res = []
        for i in self._split(lst, max_size):
            res.append(i)
        return res

    def _split_box(self, large_box, n=1, d="x"):
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
            dates = pd.to_datetime(large_box[i_left : i_right + 1])
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

    def _chunker_box4d(self, request, chunks, chunks_maxsize):
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
    """ Return a string formated for a line print """
    if len(s) > max_width:
        padding = " ... "
        n = (max_width - len(padding)) // 2
        q = (max_width - len(padding)) % 2
        if q == 0:
            return "".join([s[0:n], padding, s[-n:]])
        else:
            return "".join([s[0 : n + 1], padding, s[-n:]])
    else:
        return s


def is_box(box: list, errors="raise"):
    """ Check if this array matches a 2d or 3d box definition

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
        isinstance(b[0], int) or isinstance(b[0], float)
    )
    tests["lon_max must be numeric"] = lambda b: (
        isinstance(b[1], int) or isinstance(b[1], float)
    )
    tests["lat_min must be numeric"] = lambda b: (
        isinstance(b[2], int) or isinstance(b[2], float)
    )
    tests["lat_max must be numeric"] = lambda b: (
        isinstance(b[3], int) or isinstance(b[3], float)
    )
    tests["pres_min must be numeric"] = lambda b: (
        isinstance(b[4], int) or isinstance(b[4], float)
    )
    tests["pres_max must be numeric"] = lambda b: (
        isinstance(b[5], int) or isinstance(b[5], float)
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
        raise ValueError(error)
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
