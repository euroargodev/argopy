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
from collections import UserList
import copy
from functools import reduce, wraps
from packaging import version
import logging
from abc import ABC, abstractmethod

import importlib
import locale
import platform
import struct
import subprocess # nosec B404 only used without user inputs
import contextlib
from fsspec.core import split_protocol
import fsspec

import xarray as xr
import pandas as pd
import numpy as np
from scipy import interpolate

import pickle  # nosec B403 only used with internal files/assets
import pkg_resources
import shutil

import threading

import time

from .options import OPTIONS
from .errors import (
    FtpPathError,
    InvalidFetcher,
    InvalidFetcherAccessPoint,
    InvalidOption,
    InvalidDatasetStructure,
    FileSystemHasNoCache,
)

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


path2pkl = pkg_resources.resource_filename("argopy", "assets/")

log = logging.getLogger("argopy.utilities")


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


def lscache(cache_path: str = "", prt=True):
    """ Decode and list cache folder content

        Parameters
        ----------
        cache_path: str
        prt: bool, default=True
            Return a printable string or a :class:`pandas.DataFrame`

        Returns
        -------
        str or :class:`pandas.DataFrame`
    """
    from datetime import datetime
    import math
    summary = []

    cache_path = OPTIONS['cachedir'] if cache_path == '' else cache_path
    apath = os.path.abspath(cache_path)
    log.debug("Listing cache content at: %s" % cache_path)

    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    cached_files = []
    fn = os.path.join(apath, "cache")
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            loaded_cached_files = pickle.load(f)  # nosec B301 because files controlled internally
            for c in loaded_cached_files.values():
                if isinstance(c["blocks"], list):
                    c["blocks"] = set(c["blocks"])
            cached_files.append(loaded_cached_files)
    else:
        raise FileSystemHasNoCache("No fsspec cache system at: %s" % apath)

    cached_files = cached_files or [{}]
    cached_files = cached_files[-1]

    N_FILES = len(cached_files)
    TOTAL_SIZE = 0
    for cfile in cached_files:
        path = os.path.join(apath, cached_files[cfile]['fn'])
        TOTAL_SIZE += os.path.getsize(path)

    summary.append("%s %s" % ("=" * 20, "%i files in fsspec cache folder (%s)" % (N_FILES, convert_size(TOTAL_SIZE))))
    summary.append("lscache %s" % os.path.sep.join([apath, ""]))
    summary.append("=" * 20)

    listing = {'fn': [], 'size': [], 'time': [], 'original': [], 'uid': [], 'blocks': []}
    for cfile in cached_files:
        summary.append("- %s" % cached_files[cfile]['fn'])
        listing['fn'].append(cached_files[cfile]['fn'])

        path = os.path.join(cache_path, cached_files[cfile]['fn'])
        summary.append("\t%8s: %s" % ('SIZE', convert_size(os.path.getsize(path))))
        listing['size'].append(os.path.getsize(path))

        key = 'time'
        ts = cached_files[cfile][key]
        tsf = pd.to_datetime(datetime.fromtimestamp(ts)).strftime("%c")
        summary.append("\t%8s: %s (%s)" % (key, tsf, ts))
        listing['time'].append(pd.to_datetime(datetime.fromtimestamp(ts)))

        if version.parse(fsspec.__version__) > version.parse("0.8.7"):
            key = 'original'
            summary.append("\t%8s: %s" % (key, cached_files[cfile][key]))
            listing[key].append(cached_files[cfile][key])

        key = 'uid'
        summary.append("\t%8s: %s" % (key, cached_files[cfile][key]))
        listing[key].append(cached_files[cfile][key])

        key = 'blocks'
        summary.append("\t%8s: %s" % (key, cached_files[cfile][key]))
        listing[key].append(cached_files[cfile][key])

    summary.append("=" * 20)
    summary = "\n".join(summary)
    if prt:
        # Return string to be printed:
        return summary
    else:
        # Return dataframe listing:
        # log.debug(summary)
        return pd.DataFrame(listing)


def load_dict(ptype):
    if ptype == "profilers":
        with open(os.path.join(path2pkl, "dict_profilers.pickle"), "rb") as f:
            loaded_dict = pickle.load(f)  # nosec B301 because files controlled internally
        return loaded_dict
    elif ptype == "institutions":
        with open(os.path.join(path2pkl, "dict_institutions.pickle"), "rb") as f:
            loaded_dict = pickle.load(f)  # nosec B301 because files controlled internally
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

    try:
        from .data_fetchers import gdacftp_data as GDAC_Fetchers

        sources["gdac"] = GDAC_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the GDAC data fetcher, "
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

    try:
        from .data_fetchers import gdacftp_index as GDAC_Fetchers

        AVAILABLE_SOURCES["gdac"] = GDAC_Fetchers
    except Exception:
        warnings.warn(
            "An error occurred while loading the GDAC index fetcher, "
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
    """Check if the path has the expected GDAC ftp structure.

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
        errors: str, optional
            "ignore" or "raise" (or "warn")

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
            pipe = subprocess.Popen(  # nosec No user provided input to control here
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


def show_versions(file=sys.stdout, conda=False):  # noqa: C901
    """ Print the versions of argopy and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    conda: bool, optional
        format versions to be copy/pasted on a conda environment file (default, False)
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        print(f"Error collecting netcdf / hdf5 version: {e}")

    DEPS = {
        'min': sorted([
            ("argopy", lambda mod: mod.__version__),

            ("xarray", lambda mod: mod.__version__),
            ("scipy", lambda mod: mod.__version__),
            ("netCDF4", lambda mod: mod.__version__),
            ("sklearn", lambda mod: mod.__version__),  # Using 'preprocessing.LabelEncoder()' in xarray accessor, used by filters
            ("erddapy", lambda mod: mod.__version__),  # This could go away from requirements ?
            ("fsspec", lambda mod: mod.__version__),
            ("aiohttp", lambda mod: mod.__version__),
            ("packaging", lambda mod: mod.__version__),  # will come with xarray, Using 'version' to make API compatible with several fsspec releases
            ("toolz", lambda mod: mod.__version__),
        ]),
        'ext.extra': sorted([
            ("dask", lambda mod: mod.__version__),
            ("gsw", lambda mod: mod.__version__),   # Used by xarray accessor to compute new variables
            ("pyarrow", lambda mod: mod.__version__),
            ("tqdm", lambda mod: mod.__version__),
            ("distributed", lambda mod: mod.__version__),
        ]),
        'ext.plotters': sorted([
            ("matplotlib", lambda mod: mod.__version__),
            ("cartopy", lambda mod: mod.__version__),
            ("seaborn", lambda mod: mod.__version__),
            ("IPython", lambda mod: mod.__version__),
            ("ipywidgets", lambda mod: mod.__version__),
            ("ipykernel", lambda mod: mod.__version__),
        ]),
        'dev': sorted([
            ("zarr", lambda mod: mod.__version__),

            ("bottleneck", lambda mod: mod.__version__),
            ("cftime", lambda mod: mod.__version__),
            ("cfgrib", lambda mod: mod.__version__),
            ("conda", lambda mod: mod.__version__),
            ("nc_time_axis", lambda mod: mod.__version__),

            ("numpy", lambda mod: mod.__version__),  # will come with xarray and pandas
            ("pandas", lambda mod: mod.__version__),  # will come with xarray

            ("pip", lambda mod: mod.__version__),
            ("pytest", lambda mod: mod.__version__),  # will come with pandas
            ("setuptools", lambda mod: mod.__version__),  # Provides: pkg_resources
            ("sphinx", lambda mod: mod.__version__),
        ]),
    }

    DEPS_blob = {}
    for level in DEPS.keys():
        deps = DEPS[level]
        deps_blob = list()
        for (modname, ver_f) in deps:
            try:
                if modname in sys.modules:
                    mod = sys.modules[modname]
                else:
                    mod = importlib.import_module(modname)
            except Exception:
                deps_blob.append((modname, '-'))
            else:
                try:
                    ver = ver_f(mod)
                    deps_blob.append((modname, ver))
                except Exception:
                    deps_blob.append((modname, "installed"))
        DEPS_blob[level] = deps_blob

    print("\nSYSTEM", file=file)
    print("------", file=file)
    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    for level in DEPS_blob:
        if conda:
            print("\n# %s:" % level.upper(), file=file)
        else:
            title = "INSTALLED VERSIONS: %s" % level.upper()
            print("\n%s" % title, file=file)
            print("-"*len(title), file=file)
        deps_blob = DEPS_blob[level]
        for k, stat in deps_blob:
            if conda:
                print(f"  - {k} = {stat}", file=file)  # Format like a conda env line, useful to update ci/requirements
            else:
                print("{:<12}: {:<12}".format(k, stat), file=file)



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
    if split_protocol(host)[0] in ["http", "https", "ftp", "sftp"]:
        try:
            urllib.request.urlopen(host, timeout=1)  # nosec B310 because host protocol already checked
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

    if src in list_src and getattr(list_src[src], "api_server_check", None):
        if "localftp" in src:
            # This is a special case because the source here is a local folder
            result = check_localftp(OPTIONS["local_ftp"])
        else:
            result = isconnected(list_src[src].api_server_check)
        return result
    else:
        raise InvalidFetcher


def erddap_ds_exists(
    ds: str = "ArgoFloats", erddap: str = "https://www.ifremer.fr/erddap"
) -> bool:
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
    from .stores import httpstore
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
            if api == "localftp" and OPTIONS["local_ftp"] == "-":
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

    def __init__(self, refresh=60):
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
            dask_gufunc_kwargs={
                "output_sizes": {output_dim: len(z_regridded[z_regridded_dim])}
            },
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
            return "".join([s[0 : n + 1], padding, s[-n:]])
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
    def is_dateconvertible(d):
        try:
            pd.to_datetime(d)
            isit = True
        except Exception:
            isit = False
        return isit

    tests = {}

    # Formats:
    tests["index box must be a list"] = lambda b: isinstance(b, list)
    tests["index box must be a list with 4 or 6 elements"] = lambda b: len(b) in [4, 6]

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
    if len(box) > 4:
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

    # Orders:
    tests["lon_max must be larger than lon_min"] = lambda b: b[0] < b[1]
    tests["lat_max must be larger than lat_min"] = lambda b: b[2] < b[3]
    if len(box) > 4:
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
    return len(lst1) == len(lst2) and len(lst1) == sum(
        [1 for i, j in zip(lst1, lst2) if i == j]
    )


def check_wmo(lst, errors="raise"):
    """ Validate a WMO option and returned it as a list of integers

    Parameters
    ----------
    wmo: int
        WMO must be an integer or an iterable with elements that can be casted as integers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.

    Returns
    -------
    list(int)
    """
    is_wmo(lst, errors=errors)

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
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.

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
    msg = "WMO must be a single or a list of 5/7 digit positive numbers. Invalid: '{}'".format

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
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))

    if not result:
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))
    else:
        return result


def check_cyc(lst, errors="raise"):
    """ Validate a CYC option and returned it as a list of integers

    Parameters
    ----------
    cyc: int
        CYC must be an integer or an iterable with elements that can be casted as positive integers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.

    Returns
    -------
    list(int)
    """
    is_cyc(lst, errors=errors)

    # Make sure we deal with a list
    if not isinstance(lst, list):
        if isinstance(lst, np.ndarray):
            lst = list(lst)
        else:
            lst = [lst]

    # Then cast list elements as integers
    return [abs(int(x)) for x in lst]


def is_cyc(lst, errors="raise"):  # noqa: C901
    """ Check if a CYC is valid
    Parameters
    ----------
    cyc: int, list(int), array(int)
        CYC must be a single or a list of at most 4 digit positive numbers
    errors: {'raise', 'warn', 'ignore'}
        Possibly raises a ValueError exception or UserWarning, otherwise fails silently.
    Returns
    -------
    bool
        True if cyc is indeed a list of integers
    """
    # Make sure we deal with a list
    if not isinstance(lst, list):
        if isinstance(lst, np.ndarray):
            lst = list(lst)
        else:
            lst = [lst]

    # Error message:
    msg = "CYC must be a single or a list of at most 4 digit positive numbers. Invalid: '{}'".format

    # Then try to cast list elements as integers, return True if ok
    result = True
    try:
        for x in lst:
            if not str(x).isdigit():
                result = False

            if (len(str(x)) > 4):
                result = False

            if int(x) <= 0:
                result = False

    except Exception:
        result = False
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))

    if not result:
        if errors == "raise":
            raise ValueError(msg(x))
        elif errors == 'warn':
            warnings.warn(msg(x))
    else:
        return result


def check_index_cols(column_names: list, convention: str = 'ar_index_global_prof'):
    """
        ar_index_global_prof.txt: Index of profile files
        Profile directory file of the Argo Global Data Assembly Center
        file,date,latitude,longitude,ocean,profiler_type,institution,date_update

        argo_bio-profile_index.txt: bgc Argo profiles index file
        The directory file describes all individual bio-profile files of the argo GDAC ftp site.
        file,date,latitude,longitude,ocean,profiler_type,institution,parameters,parameter_data_mode,date_update
    """
    # Default for 'ar_index_global_prof'
    ref = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
           'date_update']
    if convention == 'argo_bio-profile_index' or convention == 'argo_synthetic-profile_index':
        ref = ['file', 'date', 'latitude', 'longitude', 'ocean', 'profiler_type', 'institution',
               'parameters', 'parameter_data_mode', 'date_update']

    if not is_list_equal(column_names, ref):
        # log.debug("Expected: %s, got: %s" % (";".join(ref), ";".join(column_names)))
        raise InvalidDatasetStructure("Unexpected column names in this index !")
    else:
        return column_names


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


def toYearFraction(
    this_date: pd._libs.tslibs.timestamps.Timestamp = pd.to_datetime("now", utc=True)
):
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
    if "UTC" in [this_date.tzname() if this_date.tzinfo is not None else ""]:
        startOfThisYear = pd.to_datetime("%i-01-01T00:00:00.000" % this_date.year, utc=True)
    else:
        startOfThisYear = pd.to_datetime("%i-01-01T00:00:00.000" % this_date.year)
    yearDuration_sec = (
        startOfThisYear + pd.offsets.DateOffset(years=1) - startOfThisYear
    ).total_seconds()

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
    yearDuration_sec = (
        startOfThisYear + pd.offsets.DateOffset(years=1) - startOfThisYear
    ).total_seconds()
    yearElapsed_sec = pd.Timedelta(fraction * yearDuration_sec, unit="s")
    return pd.to_datetime(startOfThisYear + yearElapsed_sec, unit="s")


def wrap_longitude(grid_long):
    """ Allows longitude (0-360) to wrap beyond the 360 mark, for mapping purposes.
        Makes sure that, if the longitude is near the boundary (0 or 360) that we
        wrap the values beyond
        360 so it appears nicely on a map
        This is a refactor between get_region_data and get_region_hist_locations to
        avoid duplicate code

        source:
        https://github.com/euroargodev/argodmqc_owc/blob/e174f4538fdae1534c9740491398972b1ffec3ca/pyowc/utilities.py#L80

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
    if quadrant not in [1, 3, 5, 7]:
        raise ValueError("Invalid WMO square number, 1st digit must be 1, 3, 5 or 7.")

    # 'minimum' Latitude square boundary, nearest to the Equator
    nearest_to_the_Equator_latitude = int(wmo_id[1])

    # 'minimum' Longitude square boundary, nearest to the Prime Meridian
    nearest_to_the_Prime_Meridian = int(wmo_id[2:4])

    #
    dd = 10
    if quadrant in [1, 3]:
        lon_min = nearest_to_the_Prime_Meridian * dd
        lon_max = nearest_to_the_Prime_Meridian * dd + dd
    elif quadrant in [5, 7]:
        lon_min = -nearest_to_the_Prime_Meridian * dd - dd
        lon_max = -nearest_to_the_Prime_Meridian * dd

    if quadrant in [1, 7]:
        lat_min = nearest_to_the_Equator_latitude * dd
        lat_max = nearest_to_the_Equator_latitude * dd + dd
    elif quadrant in [3, 5]:
        lat_min = -nearest_to_the_Equator_latitude * dd - dd
        lat_max = -nearest_to_the_Equator_latitude * dd

    box = [lon_min, lon_max, lat_min, lat_max]
    return box


def groupby_remap(z, data, z_regridded,   # noqa C901
                  z_dim=None,
                  z_regridded_dim="regridded",
                  output_dim="remapped",
                  select='deep',
                  right=False):
    """ todo: Need a docstring here !"""

    # sub-sampling called in xarray ufunc
    def _subsample_bins(x, y, target_values):
        # remove all nans from input x and y
        idx = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~idx]
        y = y[~idx]

        ifound = np.digitize(
            x, target_values, right=right
        )  # ``bins[i-1] <= x < bins[i]``
        ifound -= 1  # Because digitize returns a 1-based indexing, we need to remove 1
        y_binned = np.ones_like(target_values) * np.nan

        for ib, this_ibin in enumerate(np.unique(ifound)):
            ix = np.where(ifound == this_ibin)
            iselect = ix[-1]

            # Map to y value at specific x index in the bin:
            if select == "shallow":
                iselect = iselect[0]  # min/shallow
                mapped_value = y[iselect]
            elif select == "deep":
                iselect = iselect[-1]  # max/deep
                mapped_value = y[iselect]
            elif select == "middle":
                iselect = iselect[
                    np.where(x[iselect] >= np.median(x[iselect]))[0][0]
                ]  # median/middle
                mapped_value = y[iselect]
            elif select == "random":
                iselect = iselect[np.random.randint(len(iselect))]
                mapped_value = y[iselect]

            # or Map to y statistics in the bin:
            elif select == "mean":
                mapped_value = np.nanmean(y[iselect])
            elif select == "min":
                mapped_value = np.nanmin(y[iselect])
            elif select == "max":
                mapped_value = np.nanmax(y[iselect])
            elif select == "median":
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
            dask_gufunc_kwargs={
                "output_sizes": {output_dim: len(z_regridded[z_regridded_dim])}
            },
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

    remapped.coords[output_dim] = z_regridded.rename(
        {z_regridded_dim: output_dim}
    ).coords[output_dim]
    return remapped


class TopoFetcher:
    """ Fetch topographic data through an ERDDAP server for an ocean rectangle

    Example:
        >>> from argopy import TopoFetcher
        >>> box = [-75, -45, 20, 30]  # Lon_min, lon_max, lat_min, lat_max
        >>> ds = TopoFetcher(box).to_xarray()
        >>> ds = TopoFetcher(box, ds='gebco', stride=[10, 10], cache=True).to_xarray()

    """

    class ERDDAP:
        def __init__(self, server: str, protocol: str = "tabledap"):
            self.server = server
            self.protocol = protocol
            self.response = "nc"
            self.dataset_id = ""
            self.constraints = ""

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
        from .stores import httpstore
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(
            cache=cache, cachedir=cachedir, timeout=timeout, size_policy="head"
        )
        self.definition = "Erddap topographic data fetcher"

        self.BOX = box
        self.stride = stride
        if ds == "gebco":
            self.definition = "NOAA erddap gebco data fetcher for a space region"
            self.server = "https://coastwatch.pfeg.noaa.gov/erddap"
            self.server_name = "NOAA"
            self.dataset_id = "gebco"

        self._init_erddap()

    def _init_erddap(self):
        # Init erddap
        self.erddap = self.ERDDAP(server=self.server, protocol="griddap")
        self.erddap.response = "nc"

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
            "%5B",
            self.BOX[2],
            self.stride[1],
            self.BOX[3],
            "%5D",
            "%5B",
            self.BOX[0],
            self.stride[0],
            self.BOX[1],
            "%5D",
        )
        return None

    #     @property
    #     def _minimal_vlist(self):
    #         """ Return the minimal list of variables to retrieve """
    #         vlist = list()
    #         vlist.append("latitude")
    #         vlist.append("longitude")
    #         vlist.append("elevation")
    #         return vlist

    def url_encode(self, url):
        """ Return safely encoded list of urls

            This is necessary because fsspec cannot handle in cache paths/urls with a '[' character
        """

        # return urls
        def safe_for_fsspec_cache(url):
            url = url.replace("[", "%5B")  # This is the one really necessary
            url = url.replace("]", "%5D")  # For consistency
            return url

        return safe_for_fsspec_cache(url)

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

        return self.url_encode(url)

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """
        return [self.get_url()]

    def to_xarray(self, errors: str = "ignore"):
        """ Load Topographic data and return a xarray.DataSet """

        # Download data
        if len(self.uri) == 1:
            ds = self.fs.open_dataset(self.uri[0])

        return ds

    def load(self, errors: str = "ignore"):
        """ Load Topographic data and return a xarray.DataSet """
        return self.to_xarray(errors=errors)


def argo_split_path(this_path):  # noqa C901
    """ Split path from a GDAC ftp style Argo netcdf file and return information

    >>> argo_split_path('coriolis/6901035/profiles/D6901035_001D.nc')
    >>> argo_split_path('https://data-argo.ifremer.fr/dac/csiro/5903939/profiles/D5903939_103.nc')

    Parameters
    ----------
    str

    Returns
    -------
    dict
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
    output = {}

    start_with = lambda f, x: f[0:len(x)] == x if len(x) <= len(f) else False

    def split_path(p, sep='/'):
        """Split a pathname.  Returns tuple "(head, tail)" where "tail" is
        everything after the final slash.  Either part may be empty."""
        # Same as posixpath.py but we get to choose the file separator !
        p = os.fspath(p)
        i = p.rfind(sep) + 1
        head, tail = p[:i], p[i:]
        if head and head != sep * len(head):
            head = head.rstrip(sep)
        return head, tail

    known_origins = ['https://data-argo.ifremer.fr',
                     'ftp://ftp.ifremer.fr/ifremer/argo',
                     'ftp://usgodae.org/pub/outgoing/argo',
                     '']

    output['origin'] = [origin for origin in known_origins if start_with(this_path, origin)][0]
    output['origin'] = '.' if output['origin'] == '' else output['origin'] + '/'
    sep = '/' if output['origin'] != '.' else os.path.sep

    (path, file) = split_path(this_path, sep=sep)

    output['path'] = path.replace(output['origin'], '')
    output['name'] = file

    # Deal with the path:
    # dac/<DAC>/<FloatWmoID>/
    # dac/<DAC>/<FloatWmoID>/profiles
    path_parts = path.split(sep)

    try:
        if path_parts[-1] == 'profiles':
            output['type'] = 'Mono-cycle profile file'
            output['wmo'] = path_parts[-2]
            output['dac'] = path_parts[-3]
        else:
            output['type'] = 'Multi-cycle profile file'
            output['wmo'] = path_parts[-1]
            output['dac'] = path_parts[-2]
    except Exception:
        log.warning(this_path)
        log.warning(path)
        log.warning(sep)
        log.warning(path_parts)
        log.warning(output)
        raise

    if output['dac'] not in dacs:
        log.debug("This is not a Argo GDAC compliant file path: %s" % path)
        log.warning(this_path)
        log.warning(path)
        log.warning(sep)
        log.warning(path_parts)
        log.warning(output)
        raise ValueError("This is not a Argo GDAC compliant file path (invalid DAC name: '%s')" % output['dac'])

    # Deal with the file name:
    filename, file_extension = os.path.splitext(output['name'])
    output['extension'] = file_extension
    if file_extension != '.nc':
        raise ValueError(
            "This is not a Argo GDAC compliant file path (invalid file extension: '%s')" % file_extension)
    filename_parts = output['name'].split("_")

    if "Mono" in output['type']:
        prefix = filename_parts[0].split(output['wmo'])[0]
        if 'R' in prefix:
            output['data_mode'] = 'R, Real-time data'
        if 'D' in prefix:
            output['data_mode'] = 'D, Delayed-time data'

        if 'S' in prefix:
            output['type'] = 'S, Synthetic BGC Mono-cycle profile file'
        if 'M' in prefix:
            output['type'] = 'M, Merged BGC Mono-cycle profile file'
        if 'B' in prefix:
            output['type'] = 'B, BGC Mono-cycle profile file'

        suffix = filename_parts[-1].split(output['wmo'])[-1]
        if 'D' in suffix:
            output['direction'] = 'D, descending profiles'
        elif suffix == "" and "Mono" in output['type']:
            output['direction'] = 'A, ascending profiles (implicit)'

    else:
        typ = filename_parts[-1].split(".nc")[0]
        if typ == 'prof':
            output['type'] = 'Multi-cycle file'
        if typ == 'Sprof':
            output['type'] = 'S, Synthetic BGC Multi-cycle profiles file'
        if typ == 'tech':
            output['type'] = 'Technical data file'
        if typ == 'meta':
            output['type'] = 'Metadata file'
        if 'traj' in typ:
            output['type'] = 'Trajectory file'
            if typ.split("traj")[0] == 'D':
                output['data_mode'] = 'D, Delayed-time data'
            elif typ.split("traj")[0] == 'R':
                output['data_mode'] = 'R, Real-time data'
            else:
                output['data_mode'] = 'R, Real-time data (implicit)'

    # Adjust origin and path for local files:
    # This ensure that output['path'] is agnostic to users and can be reused on any gdac compliant architecture
    parts = path.split(sep)
    i, stop = len(parts) - 1, False
    while not stop:
        if parts[i] == 'profiles' or parts[i] == output['wmo'] or parts[i] == output['dac'] or parts[i] == 'dac':
            i = i - 1
            if i < 0:
                stop = True
        else:
            stop = True
    output['origin'] = sep.join(parts[0:i + 1])
    output['path'] = output['path'].replace(output['origin'], '')

    return dict(sorted(output.items()))


"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass 

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"

src: https://code.activestate.com/recipes/576862/
"""


class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden: break

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents"%self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit


def deprecated(reason):
    """Deprecation warning decorator.

    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    reason: {str, None}
        Text message to send with deprecation warning

    Examples
    --------
    The @deprecated can be used with a 'reason'.

        .. code-block:: python

           @deprecated("please, use another function")
           def old_function(x, y):
             pass

    or without:

        .. code-block:: python

           @deprecated
           def old_function(x, y):
             pass

    References
    ----------
    https://stackoverflow.com/a/40301488
    """
    import inspect

    if isinstance(reason, str):

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))


class RegistryItem(ABC):
    """Prototype for possible custom items in a Registry"""
    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError("Not implemented")

    @property
    @abstractmethod
    def isvalid(self, item):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Not implemented")


class float_wmo(RegistryItem):
    def __init__(self, WMO_number, errors='raise'):
        """Create an Argo float WMO number object

        Parameters
        ----------
        WMO_number: object
            Anything that could be casted as an integer
        errors: {'raise', 'warn', 'ignore'}
            Possibly raises a ValueError exception or UserWarning, otherwise fails silently if WMO_number is not valid

        Returns
        -------
        :class:`argopy.utilities.float_wmo`
        """
        self.errors = errors
        if isinstance(WMO_number, float_wmo):
            item = WMO_number.value
        else:
            item = check_wmo(WMO_number, errors=self.errors)[0]  # This will automatically validate item
        self.item = item

    @property
    def isvalid(self):
        """Check if WMO number is valid"""
        return is_wmo(self.item, errors=self.errors)
        # return True  # Because it was checked at instantiation

    @property
    def value(self):
        """Return WMO number as in integer"""
        return int(self.item)

    def __str__(self):
        # return "%s" % check_wmo(self.item)[0]
        return "%s" % self.item

    def __repr__(self):
        return f"WMO(%s)" % self.item

    def __check_other__(self, other):
        return check_wmo(other)[0] if type(other) is not float_wmo else other.item

    def __eq__(self, other):
        return self.item.__eq__(self.__check_other__(other))

    def __ne__(self, other):
        return self.item.__ne__(self.__check_other__(other))

    def __gt__(self, other):
        return self.item.__gt__(self.__check_other__(other))

    def __lt__(self, other):
        return self.item.__lt__(self.__check_other__(other))

    def __ge__(self, other):
        return self.item.__ge__(self.__check_other__(other))

    def __le__(self, other):
        return self.item.__le__(self.__check_other__(other))

    def __hash__(self):
        return hash(self.item)


class Registry(UserList):
    """A list manager can that validate item type

    Examples
    --------
    You can commit new entry to the registry, one by one:

        >>> R = Registry(name='file')
        >>> R.commit('meds/4901105/profiles/D4901105_017.nc')
        >>> R.commit('aoml/1900046/profiles/D1900046_179.nc')

    Or with a list:

        >>> R = Registry(name='My floats', dtype='wmo')
        >>> R.commit([2901746, 4902252])

    And also at instantiation time (name and dtype are optional):

        >>> R = Registry([2901746, 4902252], name='My floats', dtype=float_wmo)

    Registry can be used like a list.

    It is iterable:

        >>> for wmo in R:
        >>>     print(wmo)

    It has a ``len`` property:

        >>> len(R)

    It can be checked for values:

        >>> 4902252 in R

    You can also remove items from the registry, again one by one or with a list:

        >>> R.remove('2901746')

    """

    def _complain(self, msg):
        if self._invalid == 'raise':
            raise ValueError(msg)
        elif self._invalid == 'warn':
            warnings.warn(msg)
        else:
            log.debug(msg)

    def _str(self, item):
        is_valid = isinstance(item, str)
        if not is_valid:
            self._complain("%s is not a valid %s" % (str(item), self.dtype))
        return is_valid

    def _wmo(self, item):
        return item.isvalid

    def __init__(self, initlist=None, name: str = 'unnamed', dtype='str', invalid='raise'):
        """Create a registry, i.e. a controlled list

        Parameters
        ----------
        initlist: list, optional
            List of values to register
        name: str, default: 'unnamed'
            Name of the Registry
        dtype: :class:`str` or dtype, default: :class:`str`
            Data type of registry content. Supported values are: 'str', 'wmo', float_wmo
        invalid: str, default: 'raise'
            Define what do to when a new item is not valid. Can be 'raise' or 'ignore'
        """
        self.name = name
        self._invalid = invalid
        if repr(dtype) == "<class 'str'>" or dtype == 'str':
            self._validator = self._str
            self.dtype = str
        elif dtype == float_wmo or str(dtype).lower() == 'wmo':
            self._validator = self._wmo
            self.dtype = float_wmo
        else:
            raise ValueError("Unrecognised Registry data type '%s'" % dtype)
        if initlist is not None:
            initlist = self._process_items(initlist)
        super().__init__(initlist)

    def __repr__(self):
        summary = ["<argopy.registry>%s" % str(self.dtype)]
        summary.append("Name: %s" % self.name)
        N = len(self.data)
        msg = "Nitems: %s" % N if N > 1 else "Nitem: %s" % N
        summary.append(msg)
        if N > 0:
            items = [repr(item) for item in self.data]
            # msg = format_oneline("[%s]" % "; ".join(items), max_width=120)
            msg = "[%s]" % "; ".join(items)
            summary.append("Content: %s" % msg)
        return "\n".join(summary)

    def _process_items(self, items):
        if not isinstance(items, list):
            items = [items]
        if self.dtype == float_wmo:
            items = [float_wmo(item, errors=self._invalid) for item in items]
        return items

    def commit(self, values):
        """R.commit(values) -- append values to the end of the registry if not already in"""
        items = self._process_items(values)
        for item in items:
            if item not in self.data and self._validator(item):
                super().append(item)
        return self

    def append(self, value):
        """R.append(value) -- append value to the end of the registry"""
        items = self._process_items(value)
        for item in items:
            if self._validator(item):
                super().append(item)
        return self

    def extend(self, other):
        """R.extend(iterable) -- extend registry by appending elements from the iterable"""
        self.append(other)
        return self

    def remove(self, values):
        """R.remove(valueS) -- remove first occurrence of values."""
        items = self._process_items(values)
        for item in items:
            if item in self.data:
                super().remove(item)
        return self

    def insert(self, index, value):
        """R.insert(index, value) -- insert value before index."""
        item = self._process_items(value)[0]
        if self._validator(item):
            super().insert(index, item)
        return self

    def __copy__(self):
        # Called with copy.copy(R)
        return Registry(copy.copy(self.data), dtype=self.dtype)

    def copy(self):
        """Return a shallow copy of the registry"""
        return self.__copy__()


def get_coriolis_profile_id(WMO, CYC=None):
    """ Return a :class:`pandas.DataFrame` with CORIOLIS ID of WMO/CYC profile pairs

        Parameters
        ----------
        WMO: int, list(int)
            Define the list of Argo floats. This is a list of integers with WMO float identifiers.
            WMO is the World Meteorological Organization.
        CYC: int, list(int)
            Define the list of cycle numbers to load ID for each Argo floats listed in ``WMO``.

        Returns
        -------
        :class:`pandas.DataFrame`
    """
    WMO_list = check_wmo(WMO)
    if CYC is not None:
        CYC_list = check_cyc(CYC)
    URIs = [
        "https://dataselection.euro-argo.eu/api/trajectory/%i" % wmo for wmo in WMO_list
    ]

    def prec(data, url):
        # Transform trajectory json to dataframe
        # See: https://dataselection.euro-argo.eu/swagger-ui.html#!/cycle-controller/getCyclesByPlatformCodeUsingGET
        WMO = check_wmo(url.split("/")[-1])[0]
        rows = []
        for profile in data:
            keys = [x for x in profile.keys() if x not in ["coordinate"]]
            meta_row = dict((key, profile[key]) for key in keys)
            for row in profile["coordinate"]:
                meta_row[row] = profile["coordinate"][row]
            meta_row["WMO"] = WMO
            rows.append(meta_row)
        return pd.DataFrame(rows)

    from .stores import httpstore
    fs = httpstore(cache=True)
    data = fs.open_mfjson(URIs, preprocess=prec, errors="raise", url_follow=True)

    # Merge results (list of dataframe):
    key_map = {
        "id": "ID",
        "lat": "LATITUDE",
        "lon": "LONGITUDE",
        "cvNumber": "CYCLE_NUMBER",
        "level": "level",
        "WMO": "PLATFORM_NUMBER",
    }
    for i, df in enumerate(data):
        df = df.reset_index()
        df = df.rename(columns=key_map)
        df = df[[value for value in key_map.values() if value in df.columns]]
        data[i] = df
    df = pd.concat(data, ignore_index=True)
    df.sort_values(by=["PLATFORM_NUMBER", "CYCLE_NUMBER"], inplace=True)
    df = df.reset_index(drop=True)
    # df = df.set_index(["PLATFORM_NUMBER", "CYCLE_NUMBER"])
    df = df.astype({"ID": int})
    if CYC is not None:
        df = pd.concat([df[df["CYCLE_NUMBER"] == cyc] for cyc in CYC_list]).reset_index(
            drop=True
        )
    return df[
        ["PLATFORM_NUMBER", "CYCLE_NUMBER", "ID", "LATITUDE", "LONGITUDE", "level"]
    ]


def get_ea_profile_page(WMO, CYC=None):
    """ Return a list of URL

        Parameters
        ----------
        WMO: int, list(int)
            WMO must be an integer or an iterable with elements that can be casted as integers
        CYC: int, list(int), default (None)
            CYC must be an integer or an iterable with elements that can be casted as positive integers

        Returns
        -------
        list(str)

        See also
        --------
        get_coriolis_profile_id
    """
    df = get_coriolis_profile_id(WMO, CYC)
    url = "https://dataselection.euro-argo.eu/cycle/{}"
    return [url.format(this_id) for this_id in sorted(df["ID"])]


class ArgoNVSReferenceTables:
    """Argo Reference Tables

    Utility function to retrieve Argo Reference Tables from a NVS server.

    By default, this relies on: https://vocab.nerc.ac.uk/collection

    Examples
    --------
    >>> R = ArgoNVSReferenceTables()
    >>> R.valid_ref
    >>> R.all_tbl_name()
    >>> R.tbl(3)
    >>> R.tbl('R09')
    >>> R.all_tbl()

    """
    valid_ref = [
        "R01",
        "RR2",
        "RD2",
        "RP2",
        "R03",
        "R04",
        "R05",
        "R06",
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R12",
        "R13",
        "R15",
        "RMC",
        "RTV",
        "R16",
        "R19",
        "R20",
        "R21",
        "R22",
        "R23",
        "R24",
        "R25",
        "R26",
        "R27",
    ]
    """List of all available Reference Tables"""

    def __init__(self,
                 nvs="https://vocab.nerc.ac.uk/collection",
                 cache: bool = True,
                 cachedir: str = "",
                 ):
        """Argo Reference Tables from NVS"""
        from .stores import httpstore
        cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.fs = httpstore(cache=cache, cachedir=cachedir)
        self.nvs = nvs

    def _valid_ref(self, rtid):
        if rtid not in self.valid_ref:
            rtid = "R%0.2d" % rtid
            if rtid not in self.valid_ref:
                raise ValueError(
                    "Invalid Argo Reference Table, should be one in: %s"
                    % ", ".join(self.valid_ref)
                )
        return rtid

    def _jsConcept2df(self, data):
        """Return all skos:Concept as class:`pandas.DataFrame`"""
        content = {
            "altLabel": [],
            "prefLabel": [],
            "definition": [],
            "deprecated": [],
            "id": [],
        }
        for k in data["@graph"]:
            if k["@type"] == "skos:Collection":
                Collection_name = k["alternative"]
            elif k["@type"] == "skos:Concept":
                content["altLabel"].append(k["altLabel"])
                content["prefLabel"].append(k["prefLabel"]["@value"])
                content["definition"].append(k["definition"]["@value"])
                content["deprecated"].append(k["deprecated"])
                content["id"].append(k["@id"])
        df = pd.DataFrame.from_dict(content)
        df.name = Collection_name
        return df

    def _jsCollection(self, data):
        """Return last skos:Collection information as data"""
        for k in data["@graph"]:
            if k["@type"] == "skos:Collection":
                name = k["alternative"]
                desc = k["description"]
                rtid = k["@id"]
        return (name, desc, rtid)

    def get_url(self, rtid, fmt="ld+json"):
        """Return URL toward a given reference table for a given format

        Parameters
        ----------
        rtid: {str, int}
            Name or number of the reference table to retrieve. Eg: 'R01', 12
        fmt: str, default: "ld+json"
            Format of the NVS server response. Can be: "ld+json", "rdf+xml" or "text/turtle".

        Returns
        -------
        str
        """
        rtid = self._valid_ref(rtid)
        if fmt == "ld+json":
            fmt_ext = "?_profile=nvs&_mediatype=application/ld+json"
        elif fmt == "rdf+xml":
            fmt_ext = "?_profile=nvs&_mediatype=application/rdf+xml"
        elif fmt == "text/turtle":
            fmt_ext = "?_profile=nvs&_mediatype=text/turtle"
        else:
            raise ValueError("Invalid format. Must be in: 'ld+json', 'rdf+xml' or 'text/turtle'.")
        url = "{}/{}/current/{}".format
        return url(self.nvs, rtid, fmt_ext)

    def tbl(self, rtid):
        """Return an Argo Reference table

        Parameters
        ----------
        rtid: {str, int}
            Name or number of the reference table to retrieve. Eg: 'R01', 12

        Returns
        -------
        class:`pandas.DataFrame`
        """
        rtid = self._valid_ref(rtid)
        js = self.fs.open_json(self.get_url(rtid))
        df = self._jsConcept2df(js)
        return df

    def tbl_name(self, rtid):
        """Return name of an Argo Reference table

        Parameters
        ----------
        rtid: {str, int}
            Name or number of the reference table to retrieve. Eg: 'R01', 12

        Returns
        -------
        tuple('short name', 'description', 'NVS id link')
        """
        rtid = self._valid_ref(rtid)
        js = self.fs.open_json(self.get_url(rtid))
        return self._jsCollection(js)

    def all_tbl(self):
        """Return all Argo Reference tables

        Returns
        -------
        OrderedDict
            Dictionary with all table short names as key and table content as class:`pandas.DataFrame`
        """
        URLs = [self.get_url(rtid) for rtid in self.valid_ref]
        df_list = self.fs.open_mfjson(URLs, preprocess=self._jsConcept2df)
        all_tables = {}
        [all_tables.update({t.name: t}) for t in df_list]
        all_tables = collections.OrderedDict(sorted(all_tables.items()))
        return all_tables

    def all_tbl_name(self):
        """Return names of all Argo Reference tables

        Returns
        -------
        OrderedDict
            Dictionary with all table short names as key and table names as tuple('short name', 'description', 'NVS id link')
        """
        URLs = [self.get_url(rtid) for rtid in self.valid_ref]
        name_list = self.fs.open_mfjson(URLs, preprocess=self._jsCollection)
        all_tables = {}
        [
            all_tables.update({rtid.split("/")[-3]: (name, desc, rtid)})
            for name, desc, rtid in name_list
        ]
        all_tables = collections.OrderedDict(sorted(all_tables.items()))
        return all_tables
