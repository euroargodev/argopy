# -*- coding: utf-8 -*-

"""
argopy.data_fetchers.erddap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains Argo data fetcher for Ifremer ERDDAP.

This is not intended to be used directly, only by the facade at fetchers.py

"""

import xarray as xr
import pandas as pd
import numpy as np
import copy
import time
from abc import abstractmethod
import getpass
from typing import Union
from aiohttp import ClientResponseError
import logging
from erddapy.erddapy import ERDDAP, parse_dates
from erddapy.erddapy import _quote_string_constraints as quote_string_constraints

from ..options import OPTIONS
from ..utils.format import format_oneline
from ..utils.lists import list_bgc_s_variables, list_core_parameters
from ..stores import httpstore
from ..errors import ErddapServerError, DataNotFound
from ..stores import (
    indexstore_pd as ArgoIndex,
)  # make sure we work with the Pandas index store
from ..utils import is_list_of_strings, to_list, Chunker
from .proto import ArgoDataFetcherProto


log = logging.getLogger("argopy.erddap.data")

access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "ref", "bgc", "bgc-s"]  # First is default
api_server = OPTIONS["erddap"]  # API root url
api_server_check = (
    OPTIONS["erddap"] + "/info/ArgoFloats/index.json"
)  # URL to check if the API is alive


class ErddapArgoDataFetcher(ArgoDataFetcherProto):
    """Manage access to Argo data through Ifremer ERDDAP

    ERDDAP transaction are managed with the https://github.com/ioos/erddapy library

    This class is a prototype not meant to be instantiated directly

    """

    ###
    # Methods to be customised for a specific erddap request
    ###
    @abstractmethod
    def init(self, *args, **kwargs):
        """Initialisation for a specific fetcher"""
        raise NotImplementedError("ErddapArgoDataFetcher.init not implemented")

    @abstractmethod
    def define_constraints(self):
        """Define erddapy constraints"""
        raise NotImplementedError(
            "ErddapArgoDataFetcher.define_constraints not implemented"
        )

    @property
    @abstractmethod
    def uri(self) -> list:
        """Return the list of Unique Resource Identifier (URI) to download data"""
        raise NotImplementedError("ErddapArgoDataFetcher.uri not implemented")

    ###
    # Methods that must not change
    ###
    def __init__(  # noqa: C901
        self,
        ds: str = "",
        cache: bool = False,
        cachedir: str = "",
        parallel: bool = False,
        parallel_method: str = "erddap",  # Alternative to 'thread' with a dashboard
        progress: bool = False,
        chunks: str = "auto",
        chunks_maxsize: dict = {},
        api_timeout: int = 0,
        params: Union[str, list] = "all",
        measured: Union[str, list] = None,
        **kwargs,
    ):
        """Instantiate an ERDDAP Argo data fetcher

        Parameters
        ----------
        ds: str (optional)
            Dataset to load: 'phy' or 'ref' or 'bgc-s'
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        parallel: bool (optional)
            Chunk request to use parallel fetching (default: False)
        parallel_method: str (optional)
            Define the parallelization method: ``thread``, ``process`` or a :class:`dask.distributed.client.Client`.
        progress: bool (optional)
            Show a progress bar or not when ``parallel`` is set to True.
        chunks: 'auto' or dict of integers (optional)
            Dictionary with request access point as keys and number of chunks to create as values.
            Eg: {'wmo': 10} will create a maximum of 10 chunks along WMOs when used with ``Fetch_wmo``.
        chunks_maxsize: dict (optional)
            Dictionary with request access point as keys and chunk size as values (used as maximum values in
            'auto' chunking).
            Eg: {'wmo': 5} will create chunks with as many as 5 WMOs each.
        api_timeout: int (optional)
            Erddap request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        params: Union[str, list] (optional, default='all')
            List of BGC essential variables to retrieve, i.e. that will be in the output :class:`xr.DataSet``.
            By default, this is set to ``all``, i.e. any variable found in at least of the profile in the data
            selection will be included in the output.
        measured: Union[str, list] (optional, default=None)
            List of BGC essential variables that can't be NaN. If set to 'all', this is an easy way to reduce the size of the
            :class:`xr.DataSet`` to points where all variables have been measured. Otherwise, provide a simple list of
            variables.
        """
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.definition = "Ifremer erddap Argo data fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.user_mode =  kwargs["mode"] if "mode" in kwargs else OPTIONS["mode"]
        self.server = kwargs["server"] if "server" in kwargs else OPTIONS["erddap"]
        self.store_opts = {
            "cache": cache,
            "cachedir": cachedir,
            "timeout": timeout,
            "size_policy": "head",
        }
        self.fs = kwargs["fs"] if "fs" in kwargs else httpstore(**self.store_opts)

        if not isinstance(parallel, bool):
            parallel_method = parallel
            parallel = True
        if parallel_method not in ["thread", "seq", "erddap"]:
            raise ValueError(
                "erddap only support multi-threading, use 'thread' or 'erddap' instead of '%s'"
                % parallel_method
            )
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.progress = progress
        self.chunks = chunks
        self.chunks_maxsize = chunks_maxsize

        self.init(**kwargs)
        self._init_erddapy()

        if self.dataset_id in ["bgc", "bgc-s"]:
            # Create an internal ArgoIndex instance:
            # This will be used to:
            # - retrieve the list of BGC variables to ask the erddap server
            # - get <param>_data_mode information because we can't get it from the server
            self.indexfs = (
                kwargs["indexfs"]
                if "indexfs" in kwargs
                else ArgoIndex(
                    index_file="argo_synthetic-profile_index.txt",  # the only available in the erddap
                    cache=kwargs["cache_index"] if "cache_index" in kwargs else cache,
                    cachedir=cachedir,
                    timeout=timeout,
                )
            )
            self.indexfs.fs['src'] = self.fs  # Use only one httpstore instance

            # To handle bugs in the erddap server, we need the list of parameters on the server:
            # todo: Remove this when bug fixed
            self._bgc_vlist_erddap = [v.lower() for v in list_bgc_s_variables()]

            # Handle the 'params' argument:
            self._bgc_params = to_list(params)
            if isinstance(params, str):
                if params == "all":
                    params = self._bgc_vlist_avail
                else:
                    params = to_list(params)
            elif params is None:
                raise ValueError()
            elif params[0] == "all":
                params = self._bgc_vlist_avail
            elif not is_list_of_strings(params):
                raise ValueError("'params' argument must be a list of strings")
                # raise ValueError("'params' argument must be a list of strings (possibly with a * wildcard)")
            self._bgc_vlist_params = [p.upper() for p in params]
            # self._bgc_vlist_params = self._bgc_handle_wildcard(self._bgc_vlist_params)

            for v in self._bgc_vlist_params:
                if v not in self._bgc_vlist_avail:
                    raise ValueError("'%s' not available for this access point. The 'params' argument must have values in [%s]" % (v, ",".join(self._bgc_vlist_avail)))

            for p in list_core_parameters():
                if p not in self._bgc_vlist_params:
                    self._bgc_vlist_params.append(p)

            if self.user_mode in ['standard', 'research'] and 'CDOM' in self._bgc_vlist_params:
                self._bgc_vlist_params.remove('CDOM')
                log.warning("CDOM was requested but was removed from the fetcher because executed in '%s' user mode" % self.user_mode)

            # Handle the 'measured' argument:
            self._bgc_measured = to_list(measured)
            if isinstance(measured, str):
                if measured == "all":
                    measured = self._bgc_vlist_params
                else:
                    measured = to_list(measured)
            elif self._bgc_measured[0] is None:
                measured = []
            elif self._bgc_measured[0] == "all":
                measured = self._bgc_vlist_params
            elif not is_list_of_strings(self._bgc_measured):
                raise ValueError("'measured' argument must be a list of strings")
                # raise ValueError("'measured' argument must be a list of strings (possibly with a * wildcard)")
            self._bgc_vlist_measured = [m.upper() for m in measured]
            # self._bgc_vlist_measured = self._bgc_handle_wildcard(self._bgc_vlist_measured)

            for v in self._bgc_vlist_measured:
                if v not in self._bgc_vlist_avail:
                    raise ValueError("'%s' not available for this access point. The 'measured' argument must have values in [%s]" % (v, ", ".join(self._bgc_vlist_avail)))

    def __repr__(self):
        summary = ["<datafetcher.erddap>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % self.server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        if self.dataset_id in ["bgc", "bgc-s"]:
            summary.append("BGC parameters: %s" % self._bgc_vlist_params)
            summary.append(
                "BGC 'must be measured' parameters: %s" % self._bgc_vlist_measured
            )
        return "\n".join(summary)

    @property
    def server(self):
        """URL of the Erddap server"""
        return self._server

    @server.setter
    def server(self, value):
        self._server = value
        if hasattr(self, "erddap") and self.erddap.server != value:
            log.debug("The erddap server has been modified, updating internal data")
            self._init_erddapy()

    def _add_attributes(self, this):  # noqa: C901
        """Add variables attributes not return by erddap requests (csv)

        This is hard coded, but should be retrieved from an API somewhere
        """

        for v in this.data_vars:
            param = "PRES"
            if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
                this[v].attrs = {
                    "long_name": "Sea Pressure",
                    "standard_name": "sea_water_pressure",
                    "units": "decibar",
                    "valid_min": 0.0,
                    "valid_max": 12000.0,
                    "resolution": 0.1,
                    "axis": "Z",
                    "casted": this[v].attrs["casted"]
                    if "casted" in this[v].attrs
                    else 0,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            param = "TEMP"
            if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
                this[v].attrs = {
                    "long_name": "SEA TEMPERATURE IN SITU ITS-90 SCALE",
                    "standard_name": "sea_water_temperature",
                    "units": "degree_Celsius",
                    "valid_min": -2.0,
                    "valid_max": 40.0,
                    "resolution": 0.001,
                    "casted": this[v].attrs["casted"]
                    if "casted" in this[v].attrs
                    else 0,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            param = "PSAL"
            if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
                this[v].attrs = {
                    "long_name": "PRACTICAL SALINITY",
                    "standard_name": "sea_water_salinity",
                    "units": "psu",
                    "valid_min": 0.0,
                    "valid_max": 43.0,
                    "resolution": 0.001,
                    "casted": this[v].attrs["casted"]
                    if "casted" in this[v].attrs
                    else 0,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            param = "DOXY"
            if v in [param, "%s_ADJUSTED" % param, "%s_ADJUSTED_ERROR" % param]:
                this[v].attrs = {
                    "long_name": "Dissolved oxygen",
                    "standard_name": "moles_of_oxygen_per_unit_mass_in_sea_water",
                    "units": "micromole/kg",
                    "valid_min": -5.0,
                    "valid_max": 600.0,
                    "resolution": 0.001,
                    "casted": this[v].attrs["casted"]
                    if "casted" in this[v].attrs
                    else 0,
                }
                if "ERROR" in v:
                    this[v].attrs["long_name"] = (
                        "ERROR IN %s" % this[v].attrs["long_name"]
                    )

        for v in this.data_vars:
            if "_QC" in v:
                attrs = {
                    "long_name": "Global quality flag of %s profile" % v,
                    "conventions": "Argo reference table 2a",
                    "casted": this[v].attrs["casted"]
                    if "casted" in this[v].attrs
                    else 0,
                }
                this[v].attrs = attrs

        if "CYCLE_NUMBER" in this.data_vars:
            this["CYCLE_NUMBER"].attrs = {
                "long_name": "Float cycle number",
                "conventions": "0..N, 0 : launch cycle (if exists), 1 : first complete cycle",
                "casted": this["CYCLE_NUMBER"].attrs["casted"]
                if "casted" in this["CYCLE_NUMBER"].attrs
                else 0,
            }
        if "DIRECTION" in this.data_vars:
            this["DIRECTION"].attrs = {
                "long_name": "Direction of the station profiles",
                "conventions": "A: ascending profiles, D: descending profiles",
                "casted": this["DIRECTION"].attrs["casted"]
                if "casted" in this["DIRECTION"].attrs
                else 0,
            }

        if "PLATFORM_NUMBER" in this.data_vars:
            this["PLATFORM_NUMBER"].attrs = {
                "long_name": "Float unique identifier",
                "conventions": "WMO float identifier : A9IIIII",
                "casted": this["PLATFORM_NUMBER"].attrs["casted"]
                if "casted" in this["PLATFORM_NUMBER"].attrs
                else 0,
            }

        if "DATA_MODE" in this.data_vars:
            this["DATA_MODE"].attrs = {
                "long_name": "Delayed mode or real time data",
                "conventions": "R : real time; D : delayed mode; A : real time with adjustment",
                "casted": this["DATA_MODE"].attrs["casted"]
                if "casted" in this["DATA_MODE"].attrs
                else 0,
            }

        if self.dataset_id in ["bgc", "bgc-s"]:
            for param in self._bgc_vlist_params:
                if "%s_DATA_MODE" % param in this.data_vars:
                    this["%s_DATA_MODE" % param].attrs = {
                        "long_name": "Delayed mode or real time data",
                        "conventions": "R : real time; D : delayed mode; A : real time with adjustment",
                        "casted": this["%s_DATA_MODE" % param].attrs["casted"]
                        if "casted" in this["%s_DATA_MODE" % param].attrs
                        else 0,
                    }

        return this

    def _init_erddapy(self):
        # Init erddapy
        self.erddap = ERDDAP(server=str(self.server), protocol="tabledap")
        self.erddap.response = (
            "nc"  # This is a major change in v0.4, we used to work with csv files
        )

        if self.dataset_id == "phy":
            self.erddap.dataset_id = "ArgoFloats"
        elif self.dataset_id in ["bgc", "bgc-s"]:
            self.erddap.dataset_id = "ArgoFloats-synthetic-BGC"
        elif self.dataset_id == "ref":
            self.erddap.dataset_id = "ArgoFloats-reference"
        elif self.dataset_id == "ref-ctd":
            self.erddap.dataset_id = "ArgoFloats-reference-CTD"
        elif self.dataset_id == "fail":
            self.erddap.dataset_id = "invalid_db"
        else:
            raise ValueError(
                "Invalid database short name for Ifremer erddap (use: 'phy', 'bgc'/'bgc-s' or 'ref')"
            )
        return self

    @property
    def _bgc_vlist_avail(self):
        """Return the list of the erddap BGC dataset available for this access point

        Apply search criteria in the index, then retrieve the list of parameters
        """
        if hasattr(self, "WMO"):
            if hasattr(self, "CYC") and self.CYC is not None:
                self.indexfs.search_wmo_cyc(self.WMO, self.CYC)
            else:
                self.indexfs.search_wmo(self.WMO)
        elif hasattr(self, "BOX"):
            if len(self.indexBOX) == 4:
                self.indexfs.search_lat_lon(self.indexBOX)
            else:
                self.indexfs.search_lat_lon_tim(self.indexBOX)
        params = self.indexfs.read_params()

        # Temporarily remove from params those missing on the erddap server:
        # params = [p for p in params if p.lower() in self._bgc_vlist_erddap]
        results = []
        for p in params:
            if p.lower() in self._bgc_vlist_erddap:
                results.append(p)
            else:
                log.error(
                    "Removed '%s' because it's not available on the erddap, but it should !"
                    % p
                )

        return results

    # def _bgc_handle_wildcard(self, param_list):
    #     """In a list, replace item with wildcard by available BGC parameter(s)"""
    #     is_valid_param = lambda x: x in list(  # noqa: E731
    #         argopy.ArgoNVSReferenceTables().tbl(3)["altLabel"]
    #     )

    #     results = param_list.copy()
    #     for p in param_list:
    #         if not is_valid_param(p):
    #             if "*" not in p:
    #                 raise ValueError(
    #                     "Invalid BGC parameter '%s' (not listed in Argo reference table 3)"
    #                     % p
    #                 )
    #             else:
    #                 match = fnmatch.filter(self._bgc_vlist_avail, p)
    #                 if len(match) > 0:
    #                     [results.append(m) for m in match]
    #                     results.remove(p)
    #     return results

    @property
    def _minimal_vlist(self):
        """Return the list of variables to retrieve measurements for"""
        vlist = list()
        if self.dataset_id == "phy":
            plist = [
                "data_mode",
                "latitude",
                "longitude",
                "position_qc",
                "time",
                "time_qc",
                "direction",
                "platform_number",
                "cycle_number",
                "config_mission_number",
                "vertical_sampling_scheme",
            ]
            [vlist.append(p) for p in plist]

            # Core/Deep variables:
            plist = [p.lower() for p in list_core_parameters()]
            [vlist.append(p) for p in plist]
            [vlist.append(p + "_qc") for p in plist]
            [vlist.append(p + "_adjusted") for p in plist]
            [vlist.append(p + "_adjusted_qc") for p in plist]
            [vlist.append(p + "_adjusted_error") for p in plist]

        if self.dataset_id in ["bgc", "bgc-s"]:
            plist = [
                # "parameter_data_mode",  # never !!!
                "latitude",
                "longitude",
                "position_qc",
                "time",
                "time_qc",
                "direction",
                "platform_number",
                "cycle_number",
                "config_mission_number",
            ]
            [vlist.append(p) for p in plist]

            # Search in the profile index the list of parameters to load:
            params = self._bgc_vlist_params # rq: include 'core' variables
            # log.debug("erddap-bgc parameters to load: %s" % params)

            for p in params:
                vname = p.lower()
                if self.user_mode in ['expert']:
                    vlist.append("%s" % vname)
                    vlist.append("%s_qc" % vname)
                    vlist.append("%s_adjusted" % vname)
                    vlist.append("%s_adjusted_qc" % vname)
                    vlist.append("%s_adjusted_error" % vname)

                elif self.user_mode in ['standard']:
                    vlist.append("%s" % vname)
                    vlist.append("%s_qc" % vname)
                    vlist.append("%s_adjusted" % vname)
                    vlist.append("%s_adjusted_qc" % vname)
                    vlist.append("%s_adjusted_error" % vname)

                elif self.user_mode in ['research']:
                    vlist.append("%s_adjusted" % vname)
                    vlist.append("%s_adjusted_qc" % vname)
                    vlist.append("%s_adjusted_error" % vname)

                # vlist.append("profile_%s_qc" % vname)  # not in the database

        if self.dataset_id == "ref":
            plist = ["latitude", "longitude", "time", "platform_number", "cycle_number"]
            [vlist.append(p) for p in plist]
            plist = ["pres", "temp", "psal", "ptmp"]
            [vlist.append(p) for p in plist]

        vlist.sort()
        return vlist

    def cname(self):
        """Return a unique string defining the constraints"""
        return self._cname()

    @property
    def cachepath(self):
        """Return path to cached file(s) for this request

        Returns
        -------
        list(str)
        """
        return [self.fs.cachepath(uri) for uri in self.uri]

    def get_url(self):
        """Return the URL to download requested data

        Returns
        -------
        str
        """
        # Replace erddapy get_download_url()
        # We need to replace it to better handle http responses with by-passing the _check_url_response
        # https://github.com/ioos/erddapy/blob/fa1f2c15304938cd0aa132946c22b0427fd61c81/erddapy/erddapy.py#L247

        # First part of the URL:
        protocol = self.erddap.protocol
        dataset_id = self.erddap.dataset_id
        response = self.erddap.response
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"

        # Add variables to retrieve:
        self.erddap.variables = (
            self._minimal_vlist
        )  # Define the list of variables to retrieve
        variables = self.erddap.variables
        variables = ",".join(variables)
        url += f"{variables}"

        # Load constraints to implement the access point:
        self.define_constraints()  # from Fetch_box or Fetch_wmo

        # Possibly add more constraints for the BGC dataset:
        # 2024/07/19: In fact, this is not a good idea. After a while we found that it is slower to ask the
        # erddap to filter parameters (and rather unstable) than to download unfiltered parameters and to
        # apply the filter_measured
        # if self.dataset_id in ["bgc", "bgc-s"]:
        #     params = self._bgc_vlist_measured
        #     # For 'expert' and 'standard' user modes, we cannot filter param and param_adjusted
        #     # because it depends on the unavailable param data mode.
        #     # The only erddap constraints possible is for 'research' user mode because we only request for
        #     # adjusted values.
        #     if self.user_mode == 'research':
        #         for p in params:
        #             self.erddap.constraints.update({"%s_adjusted!=" % p.lower(): "NaN"})

        if self.dataset_id not in ['ref']:
            if self.user_mode == 'research':
                for p in ['pres', 'temp', 'psal']:
                    self.erddap.constraints.update({"%s_adjusted!=" % p.lower(): "NaN"})

        # Possibly add more constraints to make requests even smaller:
        for p in ["latitude", "longitude"]:
            self.erddap.constraints.update({"%s!=" % p.lower(): "NaN"})

        # Post-process all constraints:
        constraints = self.erddap.constraints
        _constraints = copy.copy(constraints)
        for k, v in _constraints.items():
            if k.startswith("time"):
                _constraints.update({k: parse_dates(v)})
        _constraints = quote_string_constraints(_constraints)

        # Remove double-quote around NaN for numerical values:
        for k, v in _constraints.items():
            if v == '"NaN"':
                _constraints.update({k: "NaN"})

        _constraints = "".join([f"&{k}{v}" for k, v in _constraints.items()])
        url += f"{_constraints}"

        # Last part:
        url += "&distinct()"
        if self.user_mode in ['research'] and self.dataset_id not in ['ref']:
            url += '&orderBy("time,pres_adjusted")'
        else:
            url += '&orderBy("time,pres")'
        return url

    @property
    def N_POINTS(self) -> int:
        """Number of measurements expected to be returned by a request

        This is an estimate that could be inaccurate with the synthetic BGC dataset
        """
        def getNfromncHeader(url):
            url = url.replace("." + self.erddap.response, ".ncHeader")
            try:
                ncHeader = str(self.fs.download_url(url))
                lines = [line for line in ncHeader.splitlines() if "row = " in line][0]
                return int(lines.split("=")[1].split(";")[0])
            except Exception:
                raise ErddapServerError(
                    "Erddap server can't return ncHeader for url: %s " % url
                )

        N = 0
        for url in self.uri:
            N += getNfromncHeader(url)
        return N

    def post_process(
        self, this_ds, add_dm: bool = True, URI: list = None
    ):  # noqa: C901
        """Post-process a xarray.DataSet created from a netcdf erddap response

        This method can also be applied on a regular dataset to re-enforce format compliance
        """
        if "row" in this_ds.dims:
            this_ds = this_ds.rename({"row": "N_POINTS"})

        # Set coordinates:
        coords = ("LATITUDE", "LONGITUDE", "TIME", "N_POINTS")
        this_ds = this_ds.reset_coords()
        this_ds["N_POINTS"] = np.arange(0, len(this_ds["N_POINTS"]))

        # Convert all coordinate variable names to upper case
        for v in this_ds.data_vars:
            this_ds = this_ds.rename({v: v.upper()})
        this_ds = this_ds.set_coords(coords)

        if self.dataset_id == "ref":
            this_ds["DIRECTION"] = xr.full_like(this_ds["CYCLE_NUMBER"], "A", dtype=str)

        # Cast data types:
        # log.debug("erddap.post_process WMO=%s" % to_list(np.unique(this_ds['PLATFORM_NUMBER'].values)))
        this_ds = this_ds.argo.cast_types()

        # log.debug("erddap.post_process WMO=%s" % to_list(np.unique(this_ds['PLATFORM_NUMBER'].values)))
        # if '999' in to_list(np.unique(this_ds['PLATFORM_NUMBER'].values)):
        #     log.error(this_ds.attrs)

        # With BGC, some points may not have a PLATFORM_NUMBER !
        # So, we remove these
        if self.dataset_id in ["bgc", "bgc-s"] and "999" in to_list(
            np.unique(this_ds["PLATFORM_NUMBER"].values)
        ):
            log.error("Found points without WMO !")
            this_ds = this_ds.where(this_ds["PLATFORM_NUMBER"] != "999", drop=True)
            this_ds = this_ds.argo.cast_types(overwrite=True)
            log.debug(
                "erddap.post_process WMO=%s"
                % to_list(np.unique(this_ds["PLATFORM_NUMBER"].values))
            )

        # log.debug("erddap.post_process (add_dm=%s): %s" % (add_dm, str(this_ds)))
        if self.dataset_id in ["bgc", "bgc-s"] and add_dm:
            this_ds = self._add_parameters_data_mode_ds(this_ds)
            this_ds = this_ds.argo.cast_types(overwrite=False)
        # log.debug("erddap.post_process (add_dm=%s): %s" % (add_dm, str(this_ds)))

        # Overwrite Erddap variables attributes with those from Argo standards:
        this_ds = self._add_attributes(this_ds)

        # In the case of a parallel download, this is a trick to preserve the chunk uri in the chunk dataset:
        # (otherwise all chunks have the same list of uri)
        Fetched_url = this_ds.attrs.get("Fetched_url", False)
        Fetched_constraints = this_ds.attrs.get("Fetched_constraints", False)

        # Finally overwrite erddap attributes with those from argopy:
        # raw_attrs = this_ds.attrs
        # print(len(this_ds.attrs))
        if 'Processing_history' in this_ds.attrs:
            this_ds.attrs = {'Processing_history': this_ds.attrs['Processing_history']}
        else:
            this_ds.attrs = {}
        # this_ds.attrs.update({'raw_attrs': raw_attrs})
        # print(len(this_ds.attrs))

        raw_attrs = this_ds.attrs.copy()
        this_ds.attrs = {}
        if self.dataset_id == "phy":
            this_ds.attrs["DATA_ID"] = "ARGO"
            this_ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        elif self.dataset_id in ["bgc", "bgc-s"]:
            this_ds.attrs["DATA_ID"] = "ARGO-BGC"
            this_ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        elif self.dataset_id == "ref":
            this_ds.attrs["DATA_ID"] = "ARGO_Reference"
            this_ds.attrs["DOI"] = "-"
            this_ds.attrs["Fetched_version"] = raw_attrs.get('version', '?')
        elif self.dataset_id == "ref-ctd":
            this_ds.attrs["DATA_ID"] = "ARGO_Reference_CTD"
            this_ds.attrs["DOI"] = "-"
            this_ds.attrs["Fetched_version"] = raw_attrs.get('version', '?')

        this_ds.attrs["Fetched_from"] = self.erddap.server
        try:
            this_ds.attrs["Fetched_by"] = getpass.getuser()
        except:  # noqa: E722
            this_ds.attrs["Fetched_by"] = "anonymous"
        this_ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime(
            "%Y/%m/%d"
        )
        this_ds.attrs["Fetched_constraints"] = (
            self.cname() if not Fetched_constraints else Fetched_constraints
        )
        this_ds.attrs["Fetched_uri"] = URI if not Fetched_url else Fetched_url
        this_ds = this_ds[np.sort(this_ds.data_vars)]

        if self.dataset_id in ["bgc", "bgc-s"]:
            n_zero = np.count_nonzero(np.isnan(np.unique(this_ds["PLATFORM_NUMBER"])))
            if n_zero > 0:
                log.error("Some points (%i) have no PLATFORM_NUMBER !" % n_zero)

        # print(len(this_ds.attrs))
        return this_ds

    def to_xarray(  # noqa: C901
        self,
        errors: str = "ignore",
        add_dm: bool = None,
        concat: bool = True,
        max_workers: int = 6,
    ):
        """Load Argo data and return a xarray.DataSet"""

        URI = self.uri  # Call it once

        # Should we compute (from the index) and add DATA_MODE for BGC variables:
        add_dm = self.dataset_id in ["bgc", "bgc-s"] if add_dm is None else bool(add_dm)

        # Download data
        if not self.parallel:
            if len(URI) == 1:
                try:
                    # log_argopy_callerstack()
                    results = self.fs.open_dataset(URI[0])
                    results = self.post_process(results, add_dm=add_dm, URI=URI)
                except ClientResponseError as e:
                    if "Proxy Error" in e.message:
                        raise ErddapServerError(
                            "Your request is probably taking longer than expected to be handled by the server. "
                            "You can try to relaunch you're request or use the 'parallel' option "
                            "to chunk it into small requests."
                        )
                        log.debug(str(e))
                    elif "Payload Too Large" in e.message:
                        raise ErddapServerError("Your request is generating too much data on the server"
                                                "You can try to use the 'parallel' option to chunk it "
                                                "into smaller requests."
                                                )
                    else:
                        raise ErddapServerError(e.message)

            else:
                try:
                    results = self.fs.open_mfdataset(
                        URI,
                        method="erddap",
                        max_workers=1,
                        progress=self.progress,
                        errors=errors,
                        concat=concat,
                        concat_dim="N_POINTS",
                        preprocess=self.post_process,
                        preprocess_opts={"add_dm": False, "URI": URI},
                        final_opts={"data_vars": "all"},
                    )
                    if results is not None:
                        if self.progress:
                            print("Final post-processing of the merged dataset () ...")
                        results = self.post_process(
                            results, **{"add_dm": add_dm, "URI": URI}
                        )
                except ClientResponseError as e:
                    raise ErddapServerError(e.message)
        else:
            try:
                results = self.fs.open_mfdataset(
                    URI,
                    method="erddap",
                    progress=self.progress,
                    max_workers=max_workers,
                    errors=errors,
                    concat=concat,
                    concat_dim="N_POINTS",
                    preprocess=self.post_process,
                    preprocess_opts={"add_dm": False, "URI": URI},
                    final_opts={"data_vars": "all"},
                )
                if concat:
                    if results is not None:
                        if self.progress:
                            print("Final post-processing of the merged dataset () ...")
                        results = self.post_process(
                            results, **{"add_dm": add_dm, "URI": URI}
                        )
            except DataNotFound:
                if self.dataset_id in ["bgc", "bgc-s"] and len(self._bgc_vlist_measured) > 0:
                    msg = (
                        "Your BGC request returned no data. This may be due to the 'measured' "
                        "argument that imposes constraints impossible to fulfill for the "
                        f"access point defined ({self.cname()}), i.e. some "
                        "variables in 'measured' are not available in some floats in this "
                        "access point."
                    )
                    raise DataNotFound(msg)
                else:
                    raise
            except ClientResponseError as e:
                raise ErddapServerError(e.message)

        # Final checks
        if self.dataset_id in ["bgc", "bgc-s"] and concat and len(self._bgc_vlist_measured) > 0:
            if not isinstance(results, list):
                results = self.filter_measured(results)
            else:
                filtered = []
                [filtered.append(self.filter_measured(r)) for r in results]
                results = filtered

            # empty = []
            # for v in self._bgc_vlist_measured:
            #     if v in results and np.count_nonzero(results[v]) != len(results["N_POINTS"]):
            #         empty.append(v)
            # if len(empty) > 0:
            #     msg = (
            #         "After processing, your BGC request returned final data with NaNs (%s). "
            #         "This may be due to the 'measured' argument ('%s') that imposes a no-NaN constraint "
            #         "impossible to fulfill for the access point defined (%s)]. "
            #         "\nUsing the 'measured' argument, you can try to minimize the list of variables to "
            #         "return without NaNs, or set it to 'None' to return all samples."
            #         % (",".join(to_list(v)), ",".join(self._bgc_measured), self.cname())
            #     )
            #     raise ValueError(msg)

        if concat and results is not None:
            results["N_POINTS"] = np.arange(0, len(results["N_POINTS"]))

        return results

    def transform_data_mode(self, ds: xr.Dataset, **kwargs):
        """Apply xarray argo accessor transform_data_mode method"""
        ds = ds.argo.transform_data_mode(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_data_mode(self, ds: xr.Dataset, **kwargs):
        """Apply xarray argo accessor filter_data_mode method"""
        ds = ds.argo.filter_data_mode(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_qc(self, ds: xr.Dataset, **kwargs):
        """Apply xarray argo accessor filter_qc method"""
        ds = ds.argo.filter_qc(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_researchmode(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Filter dataset for research user mode

        This filter will select only QC=1 delayed mode data with pressure errors smaller than 20db

        Use this filter instead of transform_data_mode and filter_qc
        """
        ds = ds.argo.filter_researchmode()
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_measured(self, ds):
        """Re-enforce the 'measured' criteria for BGC requests

        Parameters
        ----------
        ds: :class:`xr.Dataset`

        """
        # Enforce the 'measured' argument for BGC:
        if self.dataset_id in ["bgc", "bgc-s"]:
            if len(self._bgc_vlist_measured) == 0:
                return ds
            elif len(ds["N_POINTS"]) > 0:
                log.debug("Keep only samples without NaN in %s" % self._bgc_vlist_measured)
                for v in self._bgc_vlist_measured:
                    this_mask = None
                    if v in ds and "%s_ADJUSTED" % v in ds:
                        this_mask = np.logical_or.reduce((
                                ds[v].notnull(),
                                ds["%s_ADJUSTED" % v].notnull()
                            ))
                    elif v in ds:
                        this_mask = ds[v].notnull()
                    elif "%s_ADJUSTED" % v in ds:
                        this_mask = ds["%s_ADJUSTED" % v].notnull()
                    else:
                        log.debug("'%s' or '%s_ADJUSTED' not in the dataset to apply the 'filter_measured' method" % (v, v))
                    if this_mask is not None:
                        ds = ds.loc[dict(N_POINTS=this_mask)]

        ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def _add_parameters_data_mode_ds(self, this_ds):  # noqa: C901
        """Compute and add <PARAM>_DATA_MODE variables to a xarray dataset

        This requires an ArgoIndex instance as Pandas Dataframe
        todo: Code this for the pyarrow index backend

        This method consume a collection of points
        """
        # import time

        def list_WMO_CYC(this_ds):
            """Given a dataset, return a list with all possible (PLATFORM_NUMBER, CYCLE_NUMBER) tuple"""
            profiles = []
            for wmo, grp in this_ds.groupby("PLATFORM_NUMBER"):
                [
                    profiles.append((int(wmo), int(cyc)))
                    for cyc in np.unique(grp["CYCLE_NUMBER"])
                ]
            return profiles

        def list_WMO(this_ds):
            """Return all possible WMO as a list"""
            return to_list(np.unique(this_ds["PLATFORM_NUMBER"].values))

        def complete_df(this_df, params):
            """Add 'wmo', 'cyc' and '<param>_data_mode' columns to this dataframe"""
            this_df["wmo"] = this_df["file"].apply(lambda x: int(x.split("/")[1]))
            this_df["cyc"] = this_df["file"].apply(
                lambda x: int(x.split("_")[-1].split(".nc")[0].replace("D", ""))
            )
            this_df["variables"] = this_df["parameters"].apply(lambda x: x.split())
            for param in params:
                this_df["%s_data_mode" % param] = this_df.apply(
                    lambda x: x["parameter_data_mode"][x["variables"].index(param)]
                    if param in x["variables"]
                    else "",
                    axis=1,
                )
            return this_df

        def read_DM(this_df, wmo, cyc, param):
            """Return one parameter data mode for a given wmo/cyc and index dataframe"""
            filt = []
            filt.append(this_df["wmo"].isin([wmo]))
            filt.append(this_df["cyc"].isin([cyc]))
            sub_df = this_df[np.logical_and.reduce(filt)]
            if sub_df.shape[0] == 0:
                log.debug(
                    "Found a profile in the dataset, but not in the index ! wmo=%i, cyc=%i"
                    % (wmo, cyc)
                )
                # This can happen if a Synthetic netcdf file was generated from a non-BGC float.
                # The file exists, but it doesn't have BGC variables. Float is usually not listed in the index.
                return ""
            else:
                return sub_df["%s_data_mode" % param].values[-1]

        def print_etime(txt, t0):
            now = time.process_time()
            print("⏰ %s: %0.2f seconds" % (txt, now - t0))
            return now

        # timer = time.process_time()

        profiles = list_WMO_CYC(this_ds)
        self.indexfs.search_wmo(list_WMO(this_ds))
        params = [
            p
            for p in self.indexfs.read_params()
            if p in this_ds or "%s_ADJUSTED" % p in this_ds
        ]
        # timer = print_etime('Read profiles and params from ds', timer)

        df = self.indexfs.to_dataframe(completed=False)
        df = complete_df(df, params)
        # timer = print_etime('Index search wmo and export to dataframe', timer)

        CYCLE_NUMBER = this_ds["CYCLE_NUMBER"].values
        PLATFORM_NUMBER = this_ds["PLATFORM_NUMBER"].values
        N_POINTS = this_ds["N_POINTS"].values

        for param in params:
            # print("=" * 50)
            # print("Filling DATA MODE for %s ..." % param)
            # tims = {'init': 0, 'read_DM': 0, 'isin': 0, 'where': 0, 'fill': 0}

            for iprof, prof in enumerate(profiles):
                wmo, cyc = prof
                # t0 = time.process_time()

                if "%s_DATA_MODE" % param not in this_ds:
                    this_ds["%s_DATA_MODE" % param] = xr.full_like(
                        this_ds["CYCLE_NUMBER"], dtype=str, fill_value=""
                    )
                # now = time.process_time()
                # tims['init'] += now - t0
                # t0 = now

                param_data_mode = read_DM(df, wmo, cyc, param)
                # log.debug("data mode='%s' for %s/%i/%i" % (param_data_mode, param, wmo, cyc))
                # now = time.process_time()
                # tims['read_DM'] += now - t0
                # t0 = now

                i_cyc = CYCLE_NUMBER == cyc
                i_wmo = PLATFORM_NUMBER == wmo
                # now = time.process_time()
                # tims['isin'] += now - t0
                # t0 = now

                i_points = N_POINTS[np.logical_and(i_cyc, i_wmo)]
                # now = time.process_time()
                # tims['where'] += now - t0
                # t0 = now

                # this_ds["%s_DATA_MODE" % param][i_points] = param_data_mode
                this_ds["%s_DATA_MODE" % param].loc[dict(N_POINTS=i_points)] = param_data_mode
                # now = time.process_time()
                # tims['fill'] += now - t0

            this_ds["%s_DATA_MODE" % param] = this_ds["%s_DATA_MODE" % param].astype(
                "<U1"
            )
            # timer = print_etime('Processed %s (%i profiles)' % (param, len(profiles)), timer)

        return this_ds


class Fetch_wmo(ErddapArgoDataFetcher):
    """Manage access to Argo data through Ifremer ERDDAP for: a list of WMOs

    This class is instantiated when a call is made to these facade access points:
        - `ArgoDataFetcher(src='erddap').float(**)`
        - `ArgoDataFetcher(src='erddap').profile(**)`

    """

    def init(self, WMO=[], CYC=None, **kw):
        """Create Argo data loader for WMOs

        Parameters
        ----------
        WMO : list(int)
            The list of WMOs to load all Argo data for.
        CYC : int, np.array(int), list(int)
            The cycle numbers to load.
        """
        self.WMO = WMO
        self.CYC = CYC

        self.definition = "?"
        if self.dataset_id == "phy":
            self.definition = "Ifremer erddap Argo data fetcher"
        elif self.dataset_id in ["bgc", "bgc-s"]:
            self.definition = "Ifremer erddap Argo BGC data fetcher"
        elif self.dataset_id == "ref":
            self.definition = "Ifremer erddap Argo REFERENCE data fetcher"

        if self.CYC is not None:
            self.definition = "%s for profiles" % self.definition
        else:
            self.definition = "%s for floats" % self.definition

        return self

    def define_constraints(self):
        """Define erddap constraints"""
        self.erddap.constraints = {
            "platform_number=~": "|".join(["%i" % i for i in self.WMO])
        }
        if self.CYC is not None:
            self.erddap.constraints.update(
                {"cycle_number=~": "|".join(["%i" % i for i in self.CYC])}
            )
        return self

    @property
    def uri(self):
        """List of URLs to load for a request

        Returns
        -------
        list(str)
        """
        if not self.parallel:
            chunks = "auto"
            chunks_maxsize = {"wmo": 5}
            if self.dataset_id in ["bgc", "bgc-s"]:
                chunks_maxsize = {"wmo": 1}
        else:
            chunks = self.chunks
            chunks_maxsize = self.chunks_maxsize
            if self.dataset_id in ["bgc", "bgc-s"]:
                chunks_maxsize["wmo"] = 1
        self.Chunker = Chunker(
            {"wmo": self.WMO}, chunks=chunks, chunksize=chunks_maxsize
        )
        wmo_grps = self.Chunker.fit_transform()
        urls = []
        opts = {
            "ds": self.dataset_id,
            "fs": self.fs,
            "server": self.server,
            "parallel": False,
            "CYC": self.CYC,
        }
        if self.dataset_id in ["bgc", "bgc-s"]:
            opts["params"] = self._bgc_params
            opts["measured"] = self._bgc_measured
            opts["indexfs"] = self.indexfs

        for wmos in wmo_grps:
            urls.append(
                Fetch_wmo(
                    WMO=wmos,
                    **opts,
                ).get_url()
            )
        return urls


class Fetch_box(ErddapArgoDataFetcher):
    """Manage access to Argo data through Ifremer ERDDAP for: an ocean rectangle"""

    def init(self, box: list, **kw):
        """Create Argo data loader

        Parameters
        ----------
        box : list(float, float, float, float, float, float, str, str)
            The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
                or:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        self.BOX = box.copy()
        self.indexBOX = [self.BOX[ii] for ii in [0, 1, 2, 3]]
        if len(self.BOX) == 8:
            self.indexBOX = [self.BOX[ii] for ii in [0, 1, 2, 3, 6, 7]]

        if self.dataset_id == "phy":
            self.definition = "Ifremer erddap Argo data fetcher for a space/time region"
        elif self.dataset_id in ["bgc", "bgc-s"]:
            self.definition = (
                "Ifremer erddap Argo BGC data fetcher for a space/time region"
            )
        elif self.dataset_id == "ref":
            self.definition = (
                "Ifremer erddap Argo REFERENCE data fetcher for a space/time region"
            )

        return self

    def define_constraints(self):
        """Define request constraints"""
        self.erddap.constraints = {"longitude>=": self.BOX[0]}
        self.erddap.constraints.update({"longitude<=": self.BOX[1]})
        self.erddap.constraints.update({"latitude>=": self.BOX[2]})
        self.erddap.constraints.update({"latitude<=": self.BOX[3]})
        if self.user_mode in ['research'] and self.dataset_id not in ['ref']:
            self.erddap.constraints.update({"pres_adjusted>=": self.BOX[4]})
            self.erddap.constraints.update({"pres_adjusted<=": self.BOX[5]})
        else:
            self.erddap.constraints.update({"pres>=": self.BOX[4]})
            self.erddap.constraints.update({"pres<=": self.BOX[5]})
        if len(self.BOX) == 8:
            self.erddap.constraints.update({"time>=": self.BOX[6]})
            self.erddap.constraints.update({"time<=": self.BOX[7]})
        return None

    @property
    def uri(self):
        """List of files to load for a request

        Returns
        -------
        list(str)
        """
        if not self.parallel:
            return [self.get_url()]
        else:
            self.Chunker = Chunker(
                {"box": self.BOX}, chunks=self.chunks, chunksize=self.chunks_maxsize
            )
            boxes = self.Chunker.fit_transform()
            urls = []
            opts = {"ds": self.dataset_id, "fs": self.fs, "server": self.server}
            if self.dataset_id in ["bgc", "bgc-s"]:
                opts["params"] = self._bgc_params
                opts["measured"] = self._bgc_measured
                opts["indexfs"] = self.indexfs
            for box in boxes:
                try:
                    fb = Fetch_box(
                        box=box,
                        **opts,
                    )
                    urls.append(fb.get_url())
                except DataNotFound:
                    log.debug("This box fetcher will contain no data")
            return urls
