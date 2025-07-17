# -*- coding: utf-8 -*-

"""
argopy.data_fetchers.erddap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains Argo data fetcher for Ifremer ERDDAP.

This is not intended to be used directly, only by the facade at fetchers.py

"""

import xarray as xr
import numpy as np
import copy
from abc import abstractmethod
from typing import Union
from aiohttp import ClientResponseError
import logging
from erddapy.erddapy import ERDDAP, parse_dates
from erddapy.erddapy import _quote_string_constraints as quote_string_constraints

from ..options import OPTIONS, PARALLEL_SETUP
from ..utils.lists import list_bgc_s_variables, list_core_parameters
from ..errors import ErddapServerError, DataNotFound
from ..stores import httpstore, has_distributed, distributed
from ..stores.index import indexstore_pd as ArgoIndex
from ..utils import is_list_of_strings, to_list, Chunker
from .proto import ArgoDataFetcherProto
from .erddap_data_processors import pre_process


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
    data_source = "erddap"

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
        ds: str, default = OPTIONS['ds']
            Dataset to load: 'phy' or 'ref' or 'bgc-s'
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        parallel: bool, str, :class:`distributed.Client`, default: False
            Set whether to use parallelization or not, and possibly which method to use.

                Possible values:
                    - ``False``: no parallelization is used
                    - ``True``: use default method specified by the ``parallel_default_method`` option
                    - any other values accepted by the ``parallel_default_method`` option
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

        Other parameters
        ----------------
        server: str, default = OPTIONS['erddap']
            URL to erddap server
        mode: str, default = OPTIONS['mode']

        """
        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.definition = "Ifremer erddap Argo data fetcher"
        self.dataset_id = OPTIONS["ds"] if ds == "" else ds
        self.user_mode = kwargs["mode"] if "mode" in kwargs else OPTIONS["mode"]
        self.server = kwargs["server"] if "server" in kwargs else OPTIONS["erddap"]
        self.store_opts = {
            "cache": cache,
            "cachedir": cachedir,
            "timeout": timeout,
            "size_policy": "head",
        }
        self.fs = kwargs["fs"] if "fs" in kwargs else httpstore(**self.store_opts)

        self.parallelize, self.parallel_method = PARALLEL_SETUP(parallel)
        if self.parallelize and self.parallel_method == 'thread':
            self.parallel_method = 'erddap'  # Use our custom filestore
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
            self.indexfs.fs["src"] = self.fs  # Use only one httpstore instance

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
                    raise ValueError(
                        "'%s' not available for this access point. The 'params' argument must have values in [%s]"
                        % (v, ",".join(self._bgc_vlist_avail))
                    )

            for p in list_core_parameters():
                if p not in self._bgc_vlist_params:
                    self._bgc_vlist_params.append(p)

            if (
                self.user_mode in ["standard", "research"]
                and "CDOM" in self._bgc_vlist_params
            ):
                self._bgc_vlist_params.remove("CDOM")
                log.warning(
                    "CDOM was requested but was removed from the fetcher because executed in '%s' user mode"
                    % self.user_mode
                )

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
                    raise ValueError(
                        "'%s' not available for this access point. The 'measured' argument must have values in [%s]"
                        % (v, ", ".join(self._bgc_vlist_avail))
                    )

    def __repr__(self):
        summary = ["<datafetcher.erddap>"]
        summary.append(self._repr_data_source)
        summary.append(self._repr_access_point)
        summary.append(self._repr_server)
        if self.dataset_id in ["bgc", "bgc-s"]:
            summary.append("📗 Parameters: %s" % self._bgc_vlist_params)
            summary.append(
                "📕 BGC 'must be measured' parameters: %s" % self._bgc_vlist_measured
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
                self.indexfs.query.wmo_cyc(self.WMO, self.CYC)
            else:
                self.indexfs.query.wmo(self.WMO)
        elif hasattr(self, "BOX"):
            if len(self.indexBOX) == 4:
                self.indexfs.query.lon_lat(self.indexBOX)
            else:
                self.indexfs.query.box(self.indexBOX)
        params = self.indexfs.read_params()

        # Temporarily remove from params those missing on the erddap server:
        # params = [p for p in params if p.lower() in self._bgc_vlist_erddap]
        results = []
        for p in params:
            if p.lower() in self._bgc_vlist_erddap:
                results.append(p)
            # else:
            #     log.error(
            #         "Removed '%s' because it is not available on the erddap server (%s), but it should !"
            #         % (p, self._server)
            #     )

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
            params = self._bgc_vlist_params  # rq: include 'core' variables
            # log.debug("erddap-bgc parameters to load: %s" % params)

            for p in params:
                vname = p.lower()
                if self.user_mode in ["expert"]:
                    vlist.append("%s" % vname)
                    vlist.append("%s_qc" % vname)
                    vlist.append("%s_adjusted" % vname)
                    vlist.append("%s_adjusted_qc" % vname)
                    vlist.append("%s_adjusted_error" % vname)

                elif self.user_mode in ["standard"]:
                    vlist.append("%s" % vname)
                    vlist.append("%s_qc" % vname)
                    vlist.append("%s_adjusted" % vname)
                    vlist.append("%s_adjusted_qc" % vname)
                    vlist.append("%s_adjusted_error" % vname)

                elif self.user_mode in ["research"]:
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

        if self.dataset_id not in ["ref"]:
            if self.user_mode == "research":
                for p in ["pres", "temp", "psal"]:
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
        if self.user_mode in ["research"] and self.dataset_id not in ["ref"]:
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
                if "Your query produced no matching results. (nRows = 0)" in ncHeader:
                    return 0
                else:
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

    def pre_process(self, this_ds, *args, **kwargs):
        return pre_process(this_ds, *args, **kwargs)

    def to_xarray(  # noqa: C901
        self,
        errors: str = "ignore",
        add_dm: bool = None,
        concat: bool = True,
        max_workers: int = 6,
    ):
        """Load Argo data and return a xarray.DataSet

        Parameters
        ----------
        errors: str, default='ignore'
            Define how to handle errors raised during data URIs fetching:

                - 'ignore' (default): Do not stop processing, simply issue a debug message in logging console
                - 'silent':  Do not stop processing and do not issue log message
                - 'raise': Raise any error encountered

        Returns
        -------
        :class:`xarray.Dataset`
        """
        URI = self.uri  # Call it once

        # Pre-processor options:
        preprocess_opts = {
            "add_dm": False,
            "URI": URI,
            "dataset_id": self.dataset_id,
        }
        if self.dataset_id in ["bgc", "bgc-s"]:
            preprocess_opts = {
                "add_dm": True,
                "URI": URI,
                "dataset_id": self.dataset_id,
                "indexfs": self.indexfs,
            }

        # Download and pre-process data:
        results = []
        if not self.parallelize:
            if len(URI) == 1:
                try:
                    # log_argopy_callerstack()
                    results = self.fs.open_dataset(URI[0])
                    results = pre_process(results, **preprocess_opts)
                except ClientResponseError as e:
                    if "Proxy Error" in e.message:
                        raise ErddapServerError(
                            "Your request is probably taking longer than expected to be handled by the server. "
                            "You can try to relaunch you're request or use the 'parallel' option "
                            "to chunk it into small requests."
                        )
                        log.debug(str(e))
                    elif "Payload Too Large" in e.message:
                        raise ErddapServerError(
                            "Your request is generating too much data on the server"
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
                        preprocess=pre_process,
                        preprocess_opts=preprocess_opts,
                        final_opts={"data_vars": "all"},
                    )
                    if results is not None:
                        if self.progress:
                            print("Final post-processing of the merged dataset () ...")
                        results = pre_process(results, **preprocess_opts)
                except ClientResponseError as e:
                    raise ErddapServerError(e.message)
        else:
            try:
                opts = {
                    "progress": self.progress,
                    "max_workers": max_workers,
                    "errors": errors,
                    "concat": concat,
                    "concat_dim": "N_POINTS",
                    "preprocess": pre_process,
                    "preprocess_opts": preprocess_opts,
                }

                if self.parallel_method in ["erddap"]:
                    opts["method"] = "erddap"
                    opts["final_opts"] = {"data_vars": "all"}

                elif self.parallel_method in ["thread"]:
                    opts["method"] = "thread"

                elif (self.parallel_method in ["process"]) | (
                    has_distributed
                    and isinstance(self.parallel_method, distributed.client.Client)
                ):
                    opts["method"] = self.parallel_method
                    opts["preprocess_opts"] = preprocess_opts
                    opts["open_dataset_opts"] = {
                        "errors": "ignore",
                        "download_url_opts": {"errors": "ignore"},
                    }
                    opts["progress"] = False

                results = self.fs.open_mfdataset(URI, **opts)

                if concat:
                    if results is not None:
                        if self.progress:
                            print("Final post-processing of the merged dataset ...")
                        results = pre_process(results, **preprocess_opts)

            except DataNotFound:
                if (
                    self.dataset_id in ["bgc", "bgc-s"]
                    and len(self._bgc_vlist_measured) > 0
                ):
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
        if (
            self.dataset_id in ["bgc", "bgc-s"]
            and concat
            and len(self._bgc_vlist_measured) > 0
        ):
            if not isinstance(results, list):
                results = self.filter_measured(results)
            else:
                filtered = []
                [filtered.append(self.filter_measured(r)) for r in results]
                results = filtered

        if concat and results is not None:
            results["N_POINTS"] = np.arange(0, len(results["N_POINTS"]))

        return results

    def transform_data_mode(self, ds: xr.Dataset, **kwargs):
        """Apply xarray argo accessor transform_data_mode method"""
        ds = ds.argo.datamode.merge(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_data_mode(self, ds: xr.Dataset, **kwargs):
        """Apply xarray argo accessor filter_data_mode method"""
        ds = ds.argo.datamode.filter(**kwargs)
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
                log.debug(
                    "Keep only samples without NaN in %s" % self._bgc_vlist_measured
                )
                for v in self._bgc_vlist_measured:
                    this_mask = None
                    if v in ds and "%s_ADJUSTED" % v in ds:
                        this_mask = np.logical_or.reduce(
                            (ds[v].notnull(), ds["%s_ADJUSTED" % v].notnull())
                        )
                    elif v in ds:
                        this_mask = ds[v].notnull()
                    elif "%s_ADJUSTED" % v in ds:
                        this_mask = ds["%s_ADJUSTED" % v].notnull()
                    else:
                        log.debug(
                            "'%s' or '%s_ADJUSTED' not in the dataset to apply the 'filter_measured' method"
                            % (v, v)
                        )
                    if this_mask is not None:
                        ds = ds.loc[dict(N_POINTS=this_mask)]

        ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds


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
            self.definition = "Ifremer erddap Argo-based CTD-REFERENCE data fetcher"

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
        if not self.parallelize:
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
            "mode": self.user_mode,
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
        if self.user_mode in ["research"] and self.dataset_id not in ["ref"]:
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
        if not self.parallelize:
            return [self.get_url()]
        else:
            self.Chunker = Chunker(
                {"box": self.BOX}, chunks=self.chunks, chunksize=self.chunks_maxsize
            )
            boxes = self.Chunker.fit_transform()
            urls = []
            opts = {
                "ds": self.dataset_id,
                "mode": self.user_mode,
                "fs": self.fs,
                "server": self.server,
            }
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
                except ValueError as e:
                    if 'not available for this access point' in str(e):
                        log.debug("This box fetcher does not contained required data")
            return urls
