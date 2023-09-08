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
from urllib.parse import urlparse
from typing import Union
import inspect
import pathlib
import importlib
import locale
import platform
import struct
import subprocess  # nosec B404 only used without user inputs
import contextlib
from fsspec.core import split_protocol
import fsspec
from functools import lru_cache

import xarray as xr
import pandas as pd
import numpy as np
from scipy import interpolate

import pickle  # nosec B403 only used with internal files/assets
import shutil

import threading
from socket import gaierror

import time
import setuptools  # noqa: F401

from .options import OPTIONS
from .errors import (
    FtpPathError,
    InvalidFetcher,
    InvalidFetcherAccessPoint,
    InvalidOption,
    InvalidDatasetStructure,
    FileSystemHasNoCache,
    DataNotFound,
)
from .utils import (
    is_box,
    is_list_of_strings,
    is_wmo, check_wmo,
    check_cyc,
)
from .related import (
    ArgoNVSReferenceTables,
)

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

try:
    importlib.import_module('matplotlib')  # noqa: E402
    from matplotlib.colors import to_hex
except ImportError:
    pass

path2assets = importlib.util.find_spec('argopy.static.assets').submodule_search_locations[0]

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
        try:
            nvs = ArgoNVSReferenceTables(cache=True)
            profilers = {}
            for row in nvs.tbl(8).iterrows():
                profilers.update({int(row[1]['altLabel']): row[1]['prefLabel']})
            return profilers
        except Exception:
            with open(os.path.join(path2assets, "profilers.json"), "rb") as f:
                loaded_dict = json.load(f)['data']['profilers']
            return loaded_dict
    elif ptype == "institutions":
        try:
            nvs = ArgoNVSReferenceTables(cache=True)
            institutions = {}
            for row in nvs.tbl(4).iterrows():
                institutions.update({row[1]['altLabel']: row[1]['prefLabel']})
            return institutions
        except Exception:
            with open(os.path.join(path2assets, "institutions.json"), "rb") as f:
                loaded_dict = json.load(f)['data']['institutions']
            return loaded_dict
    else:
        raise ValueError("Invalid dictionary name")


def mapp_dict(Adictionnary, Avalue):
    if Avalue not in Adictionnary:
        return "Unknown"
    else:
        return Adictionnary[Avalue]


def get_sys_info():
    """Returns system information as a dict"""

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
        'core': sorted([
            ("argopy", lambda mod: mod.__version__),

            ("xarray", lambda mod: mod.__version__),
            ("scipy", lambda mod: mod.__version__),
            ("netCDF4", lambda mod: mod.__version__),
            ("erddapy", lambda mod: mod.__version__),  # This could go away from requirements ?
            ("fsspec", lambda mod: mod.__version__),
            ("aiohttp", lambda mod: mod.__version__),
            ("packaging", lambda mod: mod.__version__),  # will come with xarray, Using 'version' to make API compatible with several fsspec releases
            ("requests", lambda mod: mod.__version__),
            ("toolz", lambda mod: mod.__version__),
        ]),
        'ext.util': sorted([
            ("gsw", lambda mod: mod.__version__),   # Used by xarray accessor to compute new variables
            ("tqdm", lambda mod: mod.__version__),
            ("zarr", lambda mod: mod.__version__),
        ]),
        'ext.perf': sorted([
            ("dask", lambda mod: mod.__version__),
            ("distributed", lambda mod: mod.__version__),
            ("pyarrow", lambda mod: mod.__version__),
        ]),
        'ext.plot': sorted([
            ("matplotlib", lambda mod: mod.__version__),
            ("cartopy", lambda mod: mod.__version__),
            ("seaborn", lambda mod: mod.__version__),
            ("IPython", lambda mod: mod.__version__),
            ("ipywidgets", lambda mod: mod.__version__),
            ("ipykernel", lambda mod: mod.__version__),
        ]),
        'dev': sorted([

            ("bottleneck", lambda mod: mod.__version__),
            ("cftime", lambda mod: mod.__version__),
            ("cfgrib", lambda mod: mod.__version__),
            ("conda", lambda mod: mod.__version__),
            ("nc_time_axis", lambda mod: mod.__version__),

            ("numpy", lambda mod: mod.__version__),  # will come with xarray and pandas
            ("pandas", lambda mod: mod.__version__),  # will come with xarray

            ("pip", lambda mod: mod.__version__),
            ("black", lambda mod: mod.__version__),
            ("flake8", lambda mod: mod.__version__),
            ("pytest", lambda mod: mod.__version__),  # will come with pandas
            ("pytest_env", lambda mod: mod.__version__),  # will come with pandas
            ("pytest_cov", lambda mod: mod.__version__),  # will come with pandas
            ("pytest_localftpserver", lambda mod: mod.__version__),  # will come with pandas
            ("pytest_reportlog", lambda mod: mod.__version__),  # will come with pandas
            ("setuptools", lambda mod: mod.__version__),
            ("aiofiles", lambda mod: mod.__version__),
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
            print("-" * len(title), file=file)
        deps_blob = DEPS_blob[level]
        for k, stat in deps_blob:
            if conda:
                if k != 'argopy':
                    kf = k.replace("_", "-")
                    comment = ' ' if stat != '-' else '# '
                    print(f"{comment} - {kf} = {stat}", file=file)  # Format like a conda env line, useful to update ci/requirements
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



def isconnected(host: str = "https://www.ifremer.fr", maxtry: int = 10):
    """Check if an URL is alive

        Parameters
        ----------
        host: str
            URL to use, 'https://www.ifremer.fr' by default
        maxtry: int, default: 10
            Maximum number of host connections to try before

        Returns
        -------
        bool
    """
    # log.debug("isconnected: %s" % host)
    if split_protocol(host)[0] in ["http", "https", "ftp", "sftp"]:
        it = 0
        while it < maxtry:
            try:
                # log.debug("Checking if %s is connected ..." % host)
                urllib.request.urlopen(host, timeout=1)  # nosec B310 because host protocol already checked
                result, it = True, maxtry
            except Exception:
                result, it = False, it + 1
        return result
    else:
        return os.path.exists(host)


def urlhaskeyword(url: str = "", keyword: str = '', maxtry: int = 10):
    """ Check if a keyword is in the content of a URL

        Parameters
        ----------
        url: str
        keyword: str
        maxtry: int, default: 10
            Maximum number of host connections to try before returning False

        Returns
        -------
        bool
    """
    it = 0
    while it < maxtry:
        try:
            with fsspec.open(url) as f:
                data = f.read()
            result = keyword in str(data)
            it = maxtry
        except Exception:
            result, it = False, it + 1
    return result


def isalive(api_server_check: Union[str, dict] = "") -> bool:
    """Check if an API is alive or not

        2 methods are available:

        - URL Ping
        - keyword Check

        Parameters
        ----------
        api_server_check
            Url string or dictionary with [``url``, ``keyword``] keys.

            - For a string, uses: :class:`argopy.utilities.isconnected`
            - For a dictionary,  uses: :class:`argopy.utilities.urlhaskeyword`

        Returns
        -------
        bool
    """
    # log.debug("isalive: %s" % api_server_check)
    if isinstance(api_server_check, dict):
        return urlhaskeyword(url=api_server_check['url'], keyword=api_server_check['keyword'])
    else:
        return isconnected(api_server_check)


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
        return isalive(list_src[src].api_server_check)
    else:
        raise InvalidFetcher


def erddap_ds_exists(
        ds: Union[list, str] = "ArgoFloats",
        erddap: str = None,
        maxtry: int = 2
) -> bool:
    """ Check if a dataset exists on a remote erddap server

    Parameter
    ---------
    ds: str, default='ArgoFloats'
        Name of the erddap dataset to check
    erddap: str, default=OPTIONS['erddap']
        Url of the erddap server
    maxtry: int, default: 2
        Maximum number of host connections to try

    Return
    ------
    bool
    """
    if erddap is None:
        erddap = OPTIONS['erddap']
    # log.debug("from erddap_ds_exists: %s" % erddap)
    from .stores import httpstore
    if isconnected(erddap, maxtry=maxtry):
        with httpstore(timeout=OPTIONS['api_timeout']).open("".join([erddap, "/info/index.json"])) as of:
            erddap_index = json.load(of)
        if is_list_of_strings(ds):
            return [this_ds in [row[-1] for row in erddap_index["table"]["rows"]] for this_ds in ds]
        else:
            return ds in [row[-1] for row in erddap_index["table"]["rows"]]
    else:
        log.debug("Cannot reach erddap server: %s" % erddap)
        warnings.warn("Return False because we cannot reach the erddap server %s" % erddap)
        return False


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


class fetch_status:
    """Fetch and report web API status"""

    def __init__(self, **kwargs):
        if "stdout" in kwargs or "insert" in kwargs:
            warnings.warn("'fetch_status' signature has changed")
        pass

    def fetch(self):
        results = {}
        list_src = list_available_data_src()
        for api, mod in list_src.items():
            if getattr(mod, "api_server_check", None):
                status = isAPIconnected(api)
                message = "ok" if status else "offline"
                results[api] = {"value": status, "message": message}
        return results

    @property
    def text(self):
        results = self.fetch()
        rows = []
        for api in sorted(results.keys()):
            rows.append("src %s is: %s" % (api, results[api]["message"]))
        txt = " | ".join(rows)
        return txt

    def __repr__(self):
        return self.text

    @property
    def html(self):
        results = self.fetch()

        fs = 12

        def td_msg(bgcolor, txtcolor, txt):
            style = "background-color:%s;" % to_hex(bgcolor, keep_alpha=True)
            style += "border-width:0px;"
            style += "padding: 2px 5px 2px 5px;"
            style += "text-align:left;"
            style += "color:%s" % to_hex(txtcolor, keep_alpha=True)
            return "<td style='%s'>%s</td>" % (style, str(txt))

        td_empty = "<td style='border-width:0px;padding: 2px 5px 2px 5px;text-align:left'>&nbsp;</td>"

        html = []
        html.append("<table style='border-collapse:collapse;border-spacing:0;font-size:%ipx'>" % fs)
        html.append("<tbody><tr>")
        cols = []
        for api in sorted(results.keys()):
            color = "yellowgreen" if results[api]["value"] else "darkorange"
            cols.append(td_msg('dimgray', 'w', "src %s is" % api))
            cols.append(td_msg(color, 'w', results[api]["message"]))
            cols.append(td_empty)
        html.append("\n".join(cols))
        html.append("</tr></tbody>")
        html.append("</table>")
        html = "\n".join(html)
        return html

    def _repr_html_(self):
        return self.html


class monitor_status:
    """ Monitor data source status with a refresh rate """

    def __init__(self, refresh=60):
        self.refresh_rate = refresh

        if self.runner == 'notebook':
            import ipywidgets as widgets

            self.text = widgets.HTML(
                value=self.content,
                placeholder="",
                description="",
            )
            self.start()

    def __repr__(self):
        if self.runner != 'notebook':
            return self.content
        else:
            return ""

    @property
    def runner(self) -> str:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return 'notebook'  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return 'terminal'  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return 'standard'  # Probably standard Python interpreter

    @property
    def content(self):
        if self.runner == 'notebook':
            return fetch_status().html
        else:
            return fetch_status().text

    def work(self):
        while True:
            time.sleep(self.refresh_rate)
            self.text.value = self.content

    def start(self):
        from IPython.display import display

        thread = threading.Thread(target=self.work)
        display(self.text)
        thread.start()


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
            yield lst[i: i + n]

    def _split_list_bychunknb(self, lst, n=1):
        """Split list in n-imposed chunks of similar size
            The last chunk may contain less element than the others, depending on the size of the list.
        """
        res = []
        s = int(np.floor_divide(len(lst), n))
        for i in self._split(lst, s):
            res.append(i)
        if len(res) > n:
            res[n - 1::] = [reduce(lambda i, j: i + j, res[n - 1::])]
        return res

    def _split_list_bychunksize(self, lst, max_size=1):
        """Split list in chunks of imposed size
            The last chunk may contain less element than the others, depending on the size of the list.
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
            return "".join([s[0: n], padding, s[-n:]])
        else:
            return "".join([s[0: n + 1], padding, s[-n:]])
    else:
        return s


def warnUnless(ok, txt):
    """Function to raise a warning unless condition is True

    This function IS NOT to be used as a decorator anymore

    Parameters
    ----------
    ok: bool
        Condition to raise the warning or not
    txt: str
        Text to display in the warning
    """
    if not ok:
        msg = "%s %s" % (inspect.stack()[1].function, txt)
        warnings.warn(msg)


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
        try:
            idx = np.logical_or(np.isnan(x), np.isnan(y))
        except TypeError:
            log.debug("Error with this '%s' y data content: %s" % (type(y), str(np.unique(y))))
            raise
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

    # if dataset is passed drop all data_vars that don't contain dim
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

    start_with = lambda f, x: f[0:len(x)] == x if len(x) <= len(f) else False  # noqa: E731

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

    def fix_localhost(host):
        if 'ftp://localhost:' in host:
            return "ftp://%s" % (urlparse(host).netloc)
        if 'http://127.0.0.1:' in host:
            return "http://%s" % (urlparse(host).netloc)
        else:
            return ""

    known_origins = ['https://data-argo.ifremer.fr',
                     'ftp://ftp.ifremer.fr/ifremer/argo',
                     'ftp://usgodae.org/pub/outgoing/argo',
                     fix_localhost(this_path),
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
            # possible typ = [Rtraj, Dtraj, BRtraj, BDtraj]
            output['type'], i = 'Trajectory file', 0
            if typ[0] == 'B':
                output['type'], i = 'BGC Trajectory file', 1
            if typ.split("traj")[0][i] == 'D':
                output['data_mode'] = 'D, Delayed-time data'
            elif typ.split("traj")[0][i] == 'R':
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
    """Argo float WMO number object"""

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
        return f"WMO({self.item})"

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

    def _dict(self, item):
        is_valid = isinstance(item, dict)
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
        elif repr(dtype) == "<class 'dict'>" or dtype == 'dict':
            self._validator = self._dict
            self.dtype = dict
        elif hasattr(dtype, 'isvalid'):
            self._validator = dtype.isvalid
            self.dtype = dtype
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
            items = [str(item) for item in self.data]
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


def log_argopy_callerstack(level='debug'):
    """log the caller’s stack"""
    froot = str(pathlib.Path(__file__).parent.resolve())
    for ideep, frame in enumerate(inspect.stack()[1:]):
        if os.path.join('argopy', 'argopy') in frame.filename:
            # msg = ["└─"]
            # [msg.append("─") for ii in range(ideep)]
            msg = [""]
            [msg.append("  ") for ii in range(ideep)]
            msg.append("└─ %s:%i -> %s" % (frame.filename.replace(froot, ''), frame.lineno, frame.function))
            msg = "".join(msg)
            if level == "info":
                log.info(msg)
            elif level == "debug":
                log.debug(msg)
            elif level == "warning":
                log.warning(msg)


def drop_variables_not_in_all_datasets(ds_collection):
    """Drop variables that are not in all datasets (the lowest common denominator)

    Parameters
    ----------
    list of :class:`xr.DataSet`

    Returns
    -------
    list of :class:`xr.DataSet`
    """

    # List all possible data variables:
    vlist = []
    for res in ds_collection:
        [vlist.append(v) for v in list(res.data_vars)]
    vlist = np.unique(vlist)

    # Check if each variables are in each datasets:
    ishere = np.zeros((len(vlist), len(ds_collection)))
    for ir, res in enumerate(ds_collection):
        for iv, v in enumerate(res.data_vars):
            for iu, u in enumerate(vlist):
                if v == u:
                    ishere[iu, ir] = 1

    # List of dataset with missing variables:
    # ir_missing = np.sum(ishere, axis=0) < len(vlist)
    # List of variables missing in some dataset:
    iv_missing = np.sum(ishere, axis=1) < len(ds_collection)
    if len(iv_missing) > 0:
        log.debug("Dropping these variables that are missing from some dataset in this list: %s" % vlist[iv_missing])

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


def fill_variables_not_in_all_datasets(ds_collection, concat_dim='rows'):
    """Add empty variables to dataset so that all the collection have the same data_vars and coords

    This is to make sure that the collection of dataset can be concatenated

    Parameters
    ----------
    list of :class:`xr.DataSet`

    Returns
    -------
    list of :class:`xr.DataSet`
    """
    def first_variable_with_concat_dim(this_ds, concat_dim='rows'):
        """Return the 1st variable in the collection that have the concat_dim in dims"""
        first = None
        for v in this_ds.data_vars:
            if concat_dim in this_ds[v].dims:
                first = v
                pass
        return first

    def fillvalue(da):
        """ Return fillvalue for a dataarray """
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind
        if da.dtype.kind in ["U"]:
            fillvalue = " "
        elif da.dtype.kind == "i":
            fillvalue = 99999
        elif da.dtype.kind == "M":
            fillvalue = np.datetime64("NaT")
        else:
            fillvalue = np.nan
        return fillvalue

    # List all possible data variables:
    vlist = []
    for res in ds_collection:
        [vlist.append(v) for v in list(res.variables) if concat_dim in res[v].dims]
    vlist = np.unique(vlist)
    # log.debug('variables', vlist)

    # List all possible coordinates:
    clist = []
    for res in ds_collection:
        [clist.append(c) for c in list(res.coords) if concat_dim in res[c].dims]
    clist = np.unique(clist)
    # log.debu('coordinates', clist)

    # Get the first occurrence of each variable, to be used as a template for attributes and dtype
    meta = {}
    for ir, ds in enumerate(ds_collection):
        for v in vlist:
            if v in ds.variables:
                meta[v] = {'attrs': ds[v].attrs, 'dtype': ds[v].dtype, 'fill_value': fillvalue(ds[v])}
    # [log.debug(meta[m]) for m in meta.keys()]

    # Add missing variables to dataset
    datasets = [ds.copy() for ds in ds_collection]
    for ir, ds in enumerate(datasets):
        for v in vlist:
            if v not in ds.variables:
                like = ds[first_variable_with_concat_dim(ds, concat_dim=concat_dim)]
                datasets[ir][v] = xr.full_like(like, fill_value=meta[v]['fill_value'], dtype=meta[v]['dtype'])
                datasets[ir][v].attrs = meta[v]['attrs']

    # Make sure that all datasets have the same set of coordinates
    results = []
    for ir, ds in enumerate(datasets):
        results.append(datasets[ir].set_coords(clist))

    #
    return results
