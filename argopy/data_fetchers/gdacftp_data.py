"""
Argo data fetcher for remote GDAC FTP

This is not intended to be used directly, only by the facade at fetchers.py

"""
import numpy as np
import pandas as pd
import xarray as xr
from abc import abstractmethod
import warnings
import getpass
import logging
from fsspec.core import split_protocol
import importlib
from .proto import ArgoDataFetcherProto

from argopy.utilities import (
    list_standard_variables,
    format_oneline,
    argo_split_path
)
from argopy.options import OPTIONS, check_gdac_path
from argopy.plotters import open_dashboard
from argopy.errors import FtpPathError


log = logging.getLogger("argopy.gdacftp.data")

has_pyarrow = importlib.util.find_spec('pyarrow') is not None
if has_pyarrow:
    from argopy.stores.argo_index_pa import indexstore_pyarrow as indexstore
    log.debug("Using pyarrow indexstore")
else:
    from argopy.stores.argo_index_pa import indexstore_pandas as indexstore
    # warnings.warn("Consider installing pyarrow in order to improve performances when fetching GDAC data")
    log.debug("Using pandas indexstore")

access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "bgc"]  # First is default
api_server = OPTIONS['gdac_ftp']  # API root url
api_server_check = api_server  # URL to check if the API is alive, used by isAPIconnected


class FTPArgoDataFetcher(ArgoDataFetcherProto):
    """ Manage access to Argo data from a remote GDAC FTP """

    ###
    # Methods to be customised for a specific request
    ###
    @abstractmethod
    def init(self, *args, **kwargs):
        """ Initialisation for a specific fetcher """
        raise NotImplementedError("Not implemented")

    ###
    # Methods that must not change
    ###
    def __init__(
        self,
        ftp: str = "",
        ds: str = "",
        cache: bool = False,
        cachedir: str = "",
        dimension: str = "point",
        errors: str = "raise",
        parallel: bool = False,
        parallel_method: str = "thread",
        progress: bool = False,
        api_timeout: int = 0,
        **kwargs
    ):
        """ Init fetcher

        Parameters
        ----------
        ftp: str (optional)
            Path to the remote FTP directory where the 'dac' folder is located.
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
        parallel: bool (optional)
            Chunk request to use parallel fetching (default: False)
        parallel_method: str (optional)
            Define the parallelization method: ``thread``, ``process`` or a :class:`dask.distributed.client.Client`.
        progress: bool (optional)
            Show a progress bar or not when fetching data.
        api_timeout: int (optional)
            FTP request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        self.timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.definition = "Ifremer GDAC ftp Argo data fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.server = OPTIONS['gdac_ftp'] if ftp == "" else ftp
        self.errors = errors

        # Validate server, raise FtpPathError if not valid.
        check_gdac_path(self.server, errors='raise')

        if self.dataset_id == 'phy':
            index_file = "ar_index_global_prof.txt"
        elif self.dataset_id == 'bgc':
            index_file = "argo_synthetic-profile_index.txt"

        # Validation of self.server is done by the indexstore:
        self.indexfs = indexstore(host=self.server, index_file=index_file, cache=cache, cachedir=cachedir, timeout=self.timeout)
        self.fs = self.indexfs.fs['index']

        self.N_RECORDS = self.indexfs.load().shape[0]  # Number of records in the index
        self._post_filter_points = False

        if not isinstance(parallel, bool):
            parallel_method = parallel
            parallel = True
        # if parallel_method not in ["thread", "process"]:
        #     raise ValueError(
        #         "'ftp' only support multi-threading, use 'thread' instead of '%s'"
        #         % parallel_method
        #     )
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.progress = progress

        self.init(**kwargs)

    def __repr__(self):
        summary = ["<datafetcher.ftp>"]
        summary.append("Name: %s" % self.definition)
        summary.append("Index: %s" % self.indexfs.index_file)
        summary.append("FTP: %s" % self.server)
        if hasattr(self, 'BOX'):
            summary.append("Domain: %s" % self.cname())
        else:
            summary.append("Domain: %s" % format_oneline(self.cname()))
        if hasattr(self.indexfs, 'index'):
            summary.append("Index loaded: True (%i records)" % self.N_RECORDS)
        else:
            summary.append("Index loaded: False")
        if hasattr(self.indexfs, 'search'):
            match = 'matches' if self.N_FILES > 1 else 'match'
            summary.append("Index searched: True (%i %s, %0.4f%%)" % (self.N_FILES, match,
                                                                      self.N_FILES * 100 / self.N_RECORDS))
        else:
            summary.append("Index searched: False")
        return "\n".join(summary)

    def cname(self):
        """ Return a unique string defining the constraints """
        return self._cname()

    @property
    @abstractmethod
    def uri(self):
        """ Return the list of files to load

        Returns
        -------
        list(str)
        """
        raise NotImplementedError("Not implemented")

    def uri_mono2multi(self, URIs: list):
        """ Convert mono-profile URI files to multi-profile files

        Multi-profile file name is based on the dataset requested ('phy' or 'bgc')

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
            if self.dataset_id == 'phy':
                return "/".join([meta['origin'], 'dac', meta['dac'], meta['wmo'], "%s_prof.nc" % meta['wmo']])
            elif self.dataset_id == 'bgc':
                return "/".join([meta['origin'], 'dac', meta['dac'], meta['wmo'], "%s_Sprof.nc" % meta['wmo']])
        new_uri = [mono2multi(uri) for uri in URIs]
        new_uri = list(set(new_uri))
        return new_uri

    @property
    def cachepath(self):
        """ Return path to cache file(s) for this request

        Returns
        -------
        list(str)
        """
        return [self.fs.cachepath(url) for url in self.uri]

    def _preprocess_multiprof(self, ds):
        """ Pre-process one Argo multi-profile file as a collection of points

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            Dataset to process

        Returns
        -------
        :class:`xarray.Dataset`

        """
        # Replace JULD and JULD_QC by TIME and TIME_QC
        ds = ds.rename(
            {"JULD": "TIME", "JULD_QC": "TIME_QC", "JULD_LOCATION": "TIME_LOCATION"}
        )
        ds["TIME"].attrs = {
            "long_name": "Datetime (UTC) of the station",
            "standard_name": "time",
        }
        # Cast data types:
        ds = ds.argo.cast_types()

        # Enforce real pressure resolution: 0.1 db
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
        )  # Default output is a collection of points along N_POINTS

        # Remove netcdf file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        if self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.server
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        ds.attrs["Fetched_uri"] = ds.encoding["source"]
        ds = ds[np.sort(ds.data_vars)]

        return ds

    def to_xarray(self, errors: str = "ignore"):
        """ Load Argo data and return a :class:`xarray.Dataset`

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
        # Set method to download data:
        if not self.parallel:
            method = "sequential"
            if len(self.uri) > 50:
                warnings.warn("Found more than 50 files to load, this may take a while to process ! "
                              "Consider using another data source (eg: 'erddap') or the 'parallel=True' option to improve processing time.")
        else:
            method = self.parallel_method

        # Download data:
        ds = self.fs.open_mfdataset(
            self.uri,
            method=method,
            concat_dim="N_POINTS",
            concat=True,
            preprocess=self._preprocess_multiprof,
            progress=self.progress,
            errors=errors,
            decode_cf=1,
            use_cftime=0,
            mask_and_scale=1,
        )

        # Data post-processing:
        ds["N_POINTS"] = np.arange(
            0, len(ds["N_POINTS"])
        )  # Re-index to avoid duplicate values
        ds = ds.set_coords("N_POINTS")
        ds = ds.sortby("TIME")

        if self._post_filter_points:
            ds = self.filter_points(ds)

        # Remove netcdf file attributes and replace them with simplified argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        if self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.server
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        if len(self.uri) == 1:
            ds.attrs["Fetched_uri"] = self.uri[0]
        else:
            ds.attrs["Fetched_uri"] = ";".join(self.uri)

        return ds

    def filter_points(self, ds):
        """ Enforce request criteria

        This may be necessary if for download performance improvement we had to work with multi instead of mono profile
        files: we loaded and merged multi-profile files, and then we need to make sure to retain only profiles requested.
        """
        if hasattr(self, 'BOX'):
            # - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
            # - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
            ds = (
                ds
                .where(ds['LONGITUDE'] >= self.BOX[0], drop=True)
                .where(ds['LONGITUDE'] < self.BOX[1], drop=True)
                .where(ds['LATITUDE'] >= self.BOX[2], drop=True)
                .where(ds['LATITUDE'] < self.BOX[3], drop=True)
                .where(ds['PRES'] >= self.BOX[4], drop=True)
                .where(ds['PRES'] < self.BOX[5], drop=True)
            )
            if len(self.BOX) == 8:
                ds = (
                    ds
                    .where(ds['TIME'] >= np.datetime64(self.BOX[6]), drop=True)
                    .where(ds['TIME'] < np.datetime64(self.BOX[7]), drop=True)
                )

        if hasattr(self, 'CYC'):
            this_mask = xr.DataArray(
                np.zeros_like(ds["N_POINTS"]),
                dims=["N_POINTS"],
                coords={"N_POINTS": ds["N_POINTS"]},
            )
            for cyc in self.CYC:
                this_mask += ds['CYCLE_NUMBER'] == cyc
            this_mask = this_mask >= 1  # any
            ds = ds.where(this_mask, drop=True)

        ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))

        return ds

    def filter_data_mode(self, ds, **kwargs):
        ds = ds.argo.filter_data_mode(errors="ignore", **kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_qc(self, ds, **kwargs):
        ds = ds.argo.filter_qc(**kwargs)
        if ds.argo._type == "point":
            ds["N_POINTS"] = np.arange(0, len(ds["N_POINTS"]))
        return ds

    def filter_variables(self, ds, mode="standard"):
        if mode == "standard":
            to_remove = sorted(
                list(set(list(ds.data_vars)) - set(list_standard_variables()))
            )
            return ds.drop_vars(to_remove)
        else:
            return ds


class Fetch_wmo(FTPArgoDataFetcher):
    """ Manage access to GDAC ftp Argo data for: a list of WMOs  """

    def init(self, WMO: list = [], CYC=None, **kwargs):
        """ Create Argo data loader for WMOs

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
        self.N_FILES = np.NaN
        return self

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """
        # Get list of files to load:
        if not hasattr(self, "_list_of_argo_files"):
            if self.CYC is None:
                URIs = self.indexfs.search_wmo(self.WMO).uri
                self._list_of_argo_files = self.uri_mono2multi(URIs)
            else:
                self._list_of_argo_files = self.indexfs.search_wmo_cyc(self.WMO, self.CYC).uri

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files

    def dashboard(self, **kw):
        if len(self.WMO) == 1:
            return open_dashboard(wmo=self.WMO[0], **kw)
        else:
            warnings.warn("Dashboard only available for a single float request")


class Fetch_box(FTPArgoDataFetcher):
    """ Manage access to GDAC ftp Argo data for: a rectangular space/time domain  """

    def init(self, box: list):
        """ Create Argo data loader

        Parameters
        ----------
        box : list()
            The box domain to load all Argo data for, with one of the following convention:

                - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
                - box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        # We use a full domain definition (x, y, z, t) as argument for compatibility with the other fetchers
        # but at this point, we internally work only with x, y and t.
        self.BOX = box
        self.indexBOX = [box[ii] for ii in [0, 1, 2, 3]]
        if len(box) == 8:
            self.indexBOX = [box[ii] for ii in [0, 1, 2, 3, 6, 7]]
        # self.N_FILES = len(self.uri)  # Trigger search in the index
        self.N_FILES = np.NaN
        return self

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """
        # Get list of files to load:
        if not hasattr(self, "_list_of_argo_files"):
            if len(self.indexBOX) == 4:
                URIs = self.indexfs.search_lat_lon(self.indexBOX).uri
            else:
                URIs = self.indexfs.search_lat_lon_tim(self.indexBOX).uri

            if len(URIs) > 25:
                self._list_of_argo_files = self.uri_mono2multi(URIs)
                self._post_filter_points = True
            else:
                self._list_of_argo_files = URIs

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files