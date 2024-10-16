"""
Argo data fetcher for remote GDAC servers

This is not intended to be used directly, only by the facade at fetchers.py

"""

import numpy as np
import pandas as pd
import xarray as xr
from abc import abstractmethod
import warnings
import getpass
import logging

from ..utils.format import argo_split_path
from ..utils.decorators import deprecated
from ..options import OPTIONS, check_gdac_path, PARALLEL_SETUP
from ..errors import DataNotFound
from ..stores import ArgoIndex
from .proto import ArgoDataFetcherProto
from .gdac_data_processors import pre_process_multiprof, filter_points


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
        gdac: str = "",
        ds: str = "",
        cache: bool = False,
        cachedir: str = "",
        dimension: str = "point",
        errors: str = "raise",
        parallel: bool = False,
        progress: bool = False,
        api_timeout: int = 0,
        **kwargs
    ):
        """Init fetcher

        Parameters
        ----------
        gdac: str (optional)
            Path to the local or remote directory where the 'dac' folder is located
        ds: str (optional)
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
        """
        self.timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.dataset_id = OPTIONS["ds"] if ds == "" else ds
        self.user_mode = kwargs["mode"] if "mode" in kwargs else OPTIONS["mode"]
        self.server = OPTIONS["gdac"] if gdac == "" else gdac
        self.errors = errors

        # Validate server, raise GdacPathError if not valid.
        check_gdac_path(self.server, errors="raise")

        index_file = "core"
        if self.dataset_id in ["bgc-s", "bgc-b"]:
            index_file = self.dataset_id

        # Validation of self.server is done by the ArgoIndex:
        self.indexfs = ArgoIndex(
            host=self.server,
            index_file=index_file,
            cache=cache,
            cachedir=cachedir,
            timeout=self.timeout,
        )
        self.fs = self.indexfs.fs["src"]  # Reuse the appropriate file system

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
        if hasattr(self.indexfs, "index"):
            summary.append(
                "📗 Index: %s (%i records)" % (self.indexfs.index_file, self.N_RECORDS)
            )
        else:
            summary.append("📕 Index: %s (not loaded)" % self.indexfs.index_file)
        if hasattr(self.indexfs, "search"):
            match = "matches" if self.N_FILES > 1 else "match"
            summary.append(
                "📸 Index searched: True (%i %s, %0.4f%%)"
                % (self.N_FILES, match, self.N_FILES * 100 / self.N_RECORDS)
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

    @deprecated(
        "Not serializable, please use 'gdac_data_processors.pre_process_multiprof'",
        version="1.0.0",
    )
    def _preprocess_multiprof(self, ds):
        """Pre-process one Argo multi-profile file as a collection of points

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            Dataset to process

        Returns
        -------
        :class:`xarray.Dataset`

        """
        # Remove raw netcdf file attributes and replace them with argopy ones:
        raw_attrs = ds.attrs
        ds.attrs = {}
        ds.attrs.update({"raw_attrs": raw_attrs})

        # Rename JULD and JULD_QC to TIME and TIME_QC
        ds = ds.rename(
            {"JULD": "TIME", "JULD_QC": "TIME_QC", "JULD_LOCATION": "TIME_LOCATION"}
        )
        ds["TIME"].attrs = {
            "long_name": "Datetime (UTC) of the station",
            "standard_name": "time",
        }

        # Cast data types:
        ds = ds.argo.cast_types()

        # Enforce real pressure resolution : 0.1 db
        for vname in ds.data_vars:
            if "PRES" in vname and "QC" not in vname:
                ds[vname].values = np.round(ds[vname].values, 1)

        # Remove variables without dimensions:
        # todo: We should be able to find a way to keep them somewhere in the data structure
        for v in ds.data_vars:
            if len(list(ds[v].dims)) == 0:
                ds = ds.drop_vars(v)

        ds = (
            ds.argo.profile2point()
        )  # Default output is a collection of points, along N_POINTS

        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        if self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.server
        try:
            ds.attrs["Fetched_by"] = getpass.getuser()
        except:  # noqa: E722
            ds.attrs["Fetched_by"] = "anonymous"
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        ds.attrs["Fetched_uri"] = ds.encoding["source"]
        ds = ds[np.sort(ds.data_vars)]

        if self._post_filter_points:
            ds = self.filter_points(ds)

        return ds

    def pre_process(self, ds, *args, **kwargs):
        return pre_process_multiprof(ds, *args, **kwargs)

    def to_xarray(self, errors: str = "ignore"):
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
        if (
            len(self.uri) > 50
            and not self.parallelize
            and self.parallel_method == "sequential"
        ):
            warnings.warn(
                "Found more than 50 files to load, this may take a while to process sequentially ! "
                "Consider using another data source (eg: 'erddap') or the 'parallel=True' option to improve processing time."
            )
        elif len(self.uri) == 0:
            raise DataNotFound("No data found for: %s" % self.indexfs.cname)

        if hasattr(self, "BOX"):
            access_point = "BOX"
            access_point_opts = {"BOX": self.BOX}
        elif hasattr(self, "CYC"):
            access_point = "CYC"
            access_point_opts = {"CYC": self.CYC}
        elif hasattr(self, "WMO"):
            access_point = "WMO"
            access_point_opts = {"WMO": self.WMO}

        # Download and pre-process data:
        ds = self.fs.open_mfdataset(
            self.uri,
            method=self.parallel_method,
            concat_dim="N_POINTS",
            concat=True,
            preprocess=pre_process_multiprof,
            preprocess_opts={
                "access_point": access_point,
                "access_point_opts": access_point_opts,
                "pre_filter_points": self._post_filter_points,
            },
            progress=self.progress,
            errors=errors,
            open_dataset_opts={
                "xr_opts": {"decode_cf": 1, "use_cftime": 0, "mask_and_scale": 1}
            },
        )

        # Meta-data processing:
        ds["N_POINTS"] = np.arange(
            0, len(ds["N_POINTS"])
        )  # Re-index to avoid duplicate values
        ds = ds.set_coords("N_POINTS")
        ds = ds.sortby("TIME")

        # Remove netcdf file attributes and replace them with simplified argopy ones:
        if "Fetched_from" not in ds.attrs:
            raw_attrs = ds.attrs
            ds.attrs = {}
            ds.attrs.update({"raw_attrs": raw_attrs})
            if self.dataset_id == "phy":
                ds.attrs["DATA_ID"] = "ARGO"
            if self.dataset_id in ["bgc", "bgc-s"]:
                ds.attrs["DATA_ID"] = "ARGO-BGC"
            ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
            ds.attrs["Fetched_from"] = self.server
            try:
                ds.attrs["Fetched_by"] = getpass.getuser()
            except:  # noqa: E722
                ds.attrs["Fetched_by"] = "anonymous"
            ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime(
                "%Y/%m/%d"
            )

        ds.attrs["Fetched_constraints"] = self.cname()
        if len(self.uri) == 1:
            ds.attrs["Fetched_uri"] = self.uri[0]
        else:
            ds.attrs["Fetched_uri"] = ";".join(self.uri)

        return ds

    @deprecated(
        "Not serializable, please use 'gdac_data_processors.filter_points'",
        version="1.0.0",
    )
    def filter_points(self, ds):
        if hasattr(self, "BOX"):
            access_point = "BOX"
            access_point_opts = {"BOX": self.BOX}
        elif hasattr(self, "CYC"):
            access_point = "CYC"
            access_point_opts = {"CYC": self.CYC}
        elif hasattr(self, "WMO"):
            access_point = "WMO"
            access_point_opts = {"WMO": self.WMO}
        return filter_points(ds, access_point=access_point, **access_point_opts)

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
                URIs = self.indexfs.search_wmo(self.WMO, nrows=self._nrows).uri
                self._list_of_argo_files = self.uri_mono2multi(URIs)
            else:
                self._list_of_argo_files = self.indexfs.search_wmo_cyc(
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
                URIs = self.indexfs.search_lat_lon(self.indexBOX, nrows=self._nrows).uri
            else:
                URIs = self.indexfs.search_lat_lon_tim(
                    self.indexBOX, nrows=self._nrows
                ).uri

            if len(URIs) > 25:
                self._list_of_argo_files = self.uri_mono2multi(URIs)
                self._post_filter_points = True
            else:
                self._list_of_argo_files = URIs

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files
