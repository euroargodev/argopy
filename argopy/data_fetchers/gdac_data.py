"""
Argo data fetcher for remote GDAC servers

This is not intended to be used directly, only by the facade at fetchers.py

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import abstractmethod
from typing import Union
import warnings
import getpass
import logging
from typing import Literal

from ..utils.format import argo_split_path
from ..options import OPTIONS, check_gdac_option, PARALLEL_SETUP
from ..utils.lists import list_bgc_s_variables, list_core_parameters
from ..utils import is_list_of_strings, to_list
from ..errors import DataNotFound
from ..stores import ArgoIndex, has_distributed, distributed
from .proto import ArgoDataFetcherProto
from .gdac_data_processors import pre_process_multiprof


log = logging.getLogger("argopy.gdac.data")
access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "bgc", "bgc-s", "bgc-b"]  # First is default
api_server = OPTIONS["gdac"]  # API root url
api_server_check = (
    api_server  # URL to check if the API is alive, used by isAPIconnected
)


class GDACArgoDataFetcher(ArgoDataFetcherProto):
    """Manage access to Argo data from a GDAC server

    Warnings
    --------
    This class is a prototype not meant to be instantiated directly

    """

    data_source = "gdac"

    ###
    # Methods to be customised for a specific request
    ###
    @abstractmethod
    def init(self, *args, **kwargs):
        """Initialisation for a specific fetcher"""
        raise NotImplementedError("Not implemented")

    ###
    # Methods that must not change
    ###
    def __init__(
        self,
        ds: str = "",
        cache: bool = False,
        cachedir: str = "",
        parallel: bool = False,
        progress: bool = False,
        dimension: Literal["point", "profile"] = "point",
        errors: str = "raise",
        api_timeout: int = 0,
        params: Union[str, list] = "all",
        **kwargs
    ):
        """Init fetcher

        Parameters
        ----------
        ds: str, default = OPTIONS['ds']
            Dataset to load: 'phy' or 'bgc'
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        dimension: str, default: 'point'
            Main dimension of the output dataset. This can be "profile" to retrieve a collection of
            profiles, or "point" (default) to have data as a collection of measurements.
            This can be used to optimise performances.
        errors: str (optional)
            If set to 'raise' (default), will raise a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If set to 'ignore', the file not found is skipped when fetching data.
        parallel: bool, str, :class:`distributed.Client`, default: False
            Set whether to use parallelization or not, and possibly which method to use.

                Possible values:
                    - ``False``: no parallelization is used
                    - ``True``: use default method specified by the ``parallel_default_method`` option
                    - any other values accepted by the ``parallel_default_method`` option
        progress: bool (optional)
            Show a progress bar or not when fetching data.
        api_timeout: int (optional)
            Server request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        params: Union[str, list] (optional, default='all')
            List of BGC essential variables to retrieve, i.e. that will be in the output :class:`xr.DataSet``.
            By default, this is set to ``all``, i.e. any variable found in at least of the profile in the data
            selection will be included in the output.

        Other parameters
        ----------------
        gdac: str, default = OPTIONS['gdac']
            Path to the local or remote directory where the 'dac' folder is located
        """
        self.timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.dataset_id = OPTIONS["ds"] if ds == "" else ds
        self.server = kwargs["gdac"] if "gdac" in kwargs else OPTIONS["gdac"]
        self.user_mode = kwargs["mode"] if "mode" in kwargs else OPTIONS["mode"]

        self.errors = errors
        self.dimension = dimension

        # Validate server, raise GdacPathError if not valid.
        check_gdac_option(self.server, errors="raise")

        index_file = "core"
        if self.dataset_id in ["bgc-s", "bgc-b"]:
            index_file = self.dataset_id

        # Validation of self.server is done by the ArgoIndex instance:
        self.indexfs = ArgoIndex(
            host=self.server,
            index_file=index_file,
            cache=cache,
            cachedir=cachedir,
            timeout=self.timeout,
        )
        self.fs = self.indexfs.fs["src"]  # Reuse the appropriate file system

        if self.dataset_id in ["bgc", "bgc-s"]:
            self._bgc_vlist_gdac = [v.lower() for v in list_bgc_s_variables()]
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

        nrows = None
        if "N_RECORDS" in kwargs:
            nrows = kwargs["N_RECORDS"]
        # Number of records in the index, this will force to load the index file:
        self.N_RECORDS = self.indexfs.load(nrows=nrows).N_RECORDS
        self._post_filter_points = False

        # Set method to download data:
        self.parallelize, self.parallel_method = PARALLEL_SETUP(parallel)
        self.progress = progress

        self.init(**kwargs)

    def __repr__(self):
        summary = ["<datafetcher.gdac>"]
        summary.append(self._repr_data_source)
        summary.append(self._repr_access_point)
        summary.append(self._repr_server)
        if self.dataset_id in ["bgc", "bgc-s"]:
            summary.append("📗 Parameters: %s" % self._bgc_vlist_params)
        if hasattr(self.indexfs, "index"):
            summary.append(
                "📗 Index: %s (%i records)" % (self.indexfs.index_file, self.N_RECORDS)
            )
        else:
            summary.append("📕 Index: %s (not loaded)" % self.indexfs.index_file)
        if hasattr(self.indexfs, "search"):
            match = "matches" if self.indexfs.N_MATCH > 1 else "match"
            summary.append(
                "📸 Index searched: True (%i %s, %0.4f%%)"
                % (
                    self.indexfs.N_MATCH,
                    match,
                    self.indexfs.N_MATCH * 100 / self.N_RECORDS,
                )
            )
        else:
            summary.append("📷 Index searched: False")
        return "\n".join(summary)

    def cname(self):
        """Return a unique string defining the constraints"""
        return self._cname()

    @property
    @abstractmethod
    def uri(self):
        """Return the list of files to load

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    def uri_mono2multi(self, URIs: list):
        """Convert mono-profile URI files to multi-profile files

        Multi-profile file name is based on the dataset requested ('phy', 'bgc'/'bgc-s')

        This method does not ensure that multi-profile files exist !

        Parameters
        ----------
        URIs: list(str)
            List of strings with URIs

        Returns
        -------
        list(str)
        """

        def mono2multi(mono_path):
            meta = argo_split_path(mono_path)

            if self.dataset_id == "phy":
                return self.indexfs.fs["src"].fs.sep.join(
                    [
                        meta["origin"],
                        "dac",
                        meta["dac"],
                        meta["wmo"],
                        "%s_prof.nc" % meta["wmo"],
                    ]
                )

            elif self.dataset_id in ["bgc", "bgc-s"]:
                return self.indexfs.fs["src"].fs.sep.join(
                    [
                        meta["origin"],
                        "dac",
                        meta["dac"],
                        meta["wmo"],
                        "%s_Sprof.nc" % meta["wmo"],
                    ]
                )

            else:
                raise ValueError("Dataset '%s' not supported !" % self.dataset_id)

        new_uri = [mono2multi(uri) for uri in URIs]
        new_uri = list(set(new_uri))
        return new_uri
    
    @property
    def _bgc_vlist_avail(self):
        """Return the list of the gdac BGC dataset available for this access point

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
            if p.lower() in self._bgc_vlist_gdac:
                results.append(p)
            # else:
            #     log.error(
            #         "Removed '%s' because it is not available on the erddap server (%s), but it should !"
            #         % (p, self._server)
            #     )

        return results
    
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

            # BGC/Core/Deep variables:
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
        vlist = [p.upper() for p in vlist]
        return vlist

    @property
    def cachepath(self):
        """Return path to cache file(s) for this request

        Returns
        -------
        list(str)
        """
        return [self.fs.cachepath(url) for url in self.uri]

    def clear_cache(self):
        """Remove cached files and entries from resources opened with this fetcher"""
        self.indexfs.clear_cache()
        self.fs.clear_cache()
        return self

    def pre_process(self, ds, *args, **kwargs):
        return pre_process_multiprof(ds, *args, **kwargs)

    def to_xarray(
        self,
        errors: str = "ignore",
        concat: bool = True,
        concat_method: Literal["drop", "fill"] = "fill",
        dimension: Literal["point", "profile"] = "",
    ):
        """Load Argo data and return a :class:`xarray.Dataset`

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
        dimension = self.dimension if dimension == "" else dimension

        if (
            len(URI) > 50
            and not self.parallelize
            and self.parallel_method == "sequential"
        ):
            warnings.warn(
                "Found more than 50 files to load, this may take a while to process sequentially ! "
                "Consider using another data source (eg: 'erddap') or the 'parallel=True' option to improve processing time."
            )
        elif len(URI) == 0:
            raise DataNotFound("No data found for: %s" % self.indexfs.cname)

        # Pre-processor options:
        if hasattr(self, "BOX"):
            access_point = "BOX"
            access_point_opts = {"BOX": self.BOX}
        elif hasattr(self, "CYC"):
            access_point = "CYC"
            access_point_opts = {"CYC": self.CYC}
        elif hasattr(self, "WMO"):
            access_point = "WMO"
            access_point_opts = {"WMO": self.WMO}
        preprocess_opts = {
            "access_point": access_point,
            "access_point_opts": access_point_opts,
            "pre_filter_points": self._post_filter_points,
            "dimension": dimension,
            "params_list": self._minimal_vlist,
        }

        # Download and pre-process data:
        opts = {
            "progress": self.progress,
            "errors": errors,
            "concat": concat,
            "concat_dim": "N_POINTS",
            "preprocess": pre_process_multiprof,
            "preprocess_opts": preprocess_opts,
        }
        # ATTEMPT TO HANDLE BGC PARAMS SELECTION FOR GDAC
        if self.dataset_id in ["bgc", "bgc-s"]:
            opts["data_vars"] = self._bgc_vlist_params

        if self.parallel_method in ["thread"]:
            opts["method"] = "thread"
            opts["open_dataset_opts"] = {"xr_opts": {"engine": "argo"}}

        elif (self.parallel_method in ["process"]) | (
            has_distributed
            and isinstance(self.parallel_method, distributed.client.Client)
        ):
            opts["method"] = self.parallel_method
            opts["open_dataset_opts"] = {
                "errors": "ignore",
                "download_url_opts": {"errors": "ignore"},
            }
            opts["progress"] = False

        results = self.fs.open_mfdataset(URI, **opts)

        if concat and results is not None:
            if self.progress:
                print("Final post-processing of the merged dataset ...")
            # results = pre_process_multiprof(results, **preprocess_opts)
            results = results.argo.cast_types(overwrite=False)

            # Meta-data processing for a single merged dataset:
            results = results.assign_coords(
                {"N_POINTS": np.arange(0, len(results["N_POINTS"]))}
            )
            results = results.sortby("TIME")

            # Remove netcdf file attributes and replace them with simplified argopy ones:
            if "Fetched_from" not in results.attrs:
                raw_attrs = results.attrs

                results.attrs = {}
                if "Processing_history" in raw_attrs:
                    results.attrs.update(
                        {"Processing_history": raw_attrs["Processing_history"]}
                    )
                    raw_attrs.pop("Processing_history")
                results.argo.add_history("URI merged with '%s'" % concat_method)

                results.attrs.update({"raw_attrs": raw_attrs})
                if self.dataset_id == "phy":
                    results.attrs["DATA_ID"] = "ARGO"
                if self.dataset_id in ["bgc", "bgc-s"]:
                    results.attrs["DATA_ID"] = "ARGO-BGC"
                results.attrs["DOI"] = "http://doi.org/10.17882/42182"
                results.attrs["Fetched_from"] = self.server
                try:
                    results.attrs["Fetched_by"] = getpass.getuser()
                except:  # noqa: E722
                    results.attrs["Fetched_by"] = "anonymous"
                results.attrs["Fetched_date"] = pd.to_datetime(
                    "now", utc=True
                ).strftime("%Y/%m/%d")

            results.attrs["Fetched_constraints"] = self.cname()
            if len(self.uri) == 1:
                results.attrs["Fetched_uri"] = self.uri[0]
            else:
                results.attrs["Fetched_uri"] = ";".join(self.uri)

        if concat:
            results.attrs = dict(sorted(results.attrs.items()))
        else:
            for ds in results:
                ds.attrs = dict(sorted(ds.attrs.items()))
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
        ds = ds.argo.filter_qc(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_researchmode(self, ds: xr.Dataset, *args, **kwargs) -> xr.Dataset:
        """Filter dataset for research user mode

        This filter will select only QC=1 delayed mode data with pressure errors smaller than 20db

        Use this filter instead of transform_data_mode and filter_qc
        """
        ds = ds.argo.filter_researchmode(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds


class Fetch_wmo(GDACArgoDataFetcher):
    """Manage access to GDAC Argo data for: a list of WMOs.

    This class is instantiated when a call is made to these facade access points:

    >>> ArgoDataFetcher(src='gdac').float(**)
    >>> ArgoDataFetcher(src='gdac').profile(**)

    """

    def init(self, WMO: list = [], CYC=None, **kwargs):
        """Create Argo data loader for WMOs

        Parameters
        ----------
        WMO: list(int)
            The list of WMOs to load all Argo data for.
        CYC: int, np.array(int), list(int)
            The cycle numbers to load.
        """
        self.WMO = WMO
        self.CYC = CYC
        # self.N_FILES = len(self.uri)  # Trigger search in the index, should we do this at instantiation or later ???
        self.N_FILES = np.nan
        self._nrows = None
        if "MAX_FILES" in kwargs:
            self._nrows = kwargs["MAX_FILES"]

        self.definition = "Ifremer GDAC Argo data fetcher"
        if self.CYC is not None:
            self.definition = "%s for profiles" % self.definition
        else:
            self.definition = "%s for floats" % self.definition
        return self

    @property
    def uri(self):
        """List of files to load for a request

        Returns
        -------
        list(str)
        """
        # Get list of files to load:
        if not hasattr(self, "_list_of_argo_files"):
            if self.CYC is None:
                URIs = self.indexfs.query.wmo(self.WMO, nrows=self._nrows).uri
                self._list_of_argo_files = self.uri_mono2multi(URIs)
            else:
                self._list_of_argo_files = self.indexfs.query.wmo_cyc(
                    self.WMO, self.CYC, nrows=self._nrows
                ).uri

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files


class Fetch_box(GDACArgoDataFetcher):
    """Manage access to GDAC Argo data for: a rectangular space/time domain.

    This class is instantiated when a call is made to these facade access points:

    >>> ArgoDataFetcher(src='gdac').region(**)

    """

    def init(self, box: list, nrows=None, **kwargs):
        """Create Argo data loader

        Parameters
        ----------
        box : list()
            The box domain to load all Argo data for, with one of the following convention:

                - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
                - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        # We use a full domain definition (x, y, z, t) as argument for compatibility with the other fetchers
        # but at this point, we internally work only with x, y and t.
        self.BOX = box.copy()
        self.indexBOX = [self.BOX[ii] for ii in [0, 1, 2, 3]]
        if len(self.BOX) == 8:
            self.indexBOX = [self.BOX[ii] for ii in [0, 1, 2, 3, 6, 7]]
        # self.N_FILES = len(self.uri)  # Trigger search in the index
        self.N_FILES = np.nan
        self._nrows = None
        if "MAX_FILES" in kwargs:
            self._nrows = kwargs["MAX_FILES"]

        self.definition = "Ifremer GDAC Argo data fetcher for a space/time region"
        return self

    @property
    def uri(self):
        """List of files to load for a request

        Returns
        -------
        list(str)
        """
        # Get list of files to load:
        if not hasattr(self, "_list_of_argo_files"):
            if len(self.indexBOX) == 4:
                URIs = self.indexfs.query.lon_lat(self.indexBOX, nrows=self._nrows).uri
            else:
                URIs = self.indexfs.query.box(self.indexBOX, nrows=self._nrows).uri

            if len(URIs) > 25:
                self._list_of_argo_files = self.uri_mono2multi(URIs)
                self._post_filter_points = True
            else:
                self._list_of_argo_files = URIs

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files
