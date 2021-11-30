"""
Argo data fetcher for remote GDAC FTP

This is not intended to be used directly, only by the facade at fetchers.py

Since the GDAC ftp is organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.

"""
import os
from glob import glob
import numpy as np
import pandas as pd
from abc import abstractmethod
import warnings
import getpass

from .proto import ArgoDataFetcherProto
from argopy.errors import NetCDF4FileNotFoundError
from argopy.utilities import (
    list_standard_variables,
    check_localftp,
    format_oneline
)
from argopy.options import OPTIONS
from argopy.stores import httpstore
from argopy.plotters import open_dashboard

access_points = ["wmo"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "bgc"]  # First is default
api_server = 'https://data-argo.ifremer.fr/'  # API root url
api_server_check = api_server + 'readme_before_using_the_data.txt'  # URL to check if the API is alive
# api_server_check = OPTIONS["gdac_ftp"]

import pyarrow.csv as csv
import pyarrow as pa
import gzip


class indexstore():
    def __init__(self,
                 host: str = "https://data-argo.ifremer.fr",
                 index_file: str = "ar_index_global_prof.txt",
                 cache: bool = False,
                 cachedir: str = "",
                 api_timeout: int = 0,
                 **kw):
        """ Create a file storage system for Argo index file requests

            Parameters
            ----------
            cache : bool (False)
            cachedir : str (used value in global OPTIONS)
            index_file: str ("ar_index_global_prof.txt")
        """
        self.index_file = index_file
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        self.fs = httpstore(cache, cachedir)  # Only for https://data-argo.ifremer.fr
        self.host = host

    def load(self, force=False):
        """ Load an Argo-index file content

        Try to load the gzipped file first, and if not found, fall back on the raw .txt file.

        Returns
        -------
        pyarrow.table
        """

        def read_csv(input_file):
            this_table = csv.read_csv(
                input_file,
                read_options=csv.ReadOptions(use_threads=True, skip_rows=8),
                convert_options=csv.ConvertOptions(
                    column_types={
                        'date': pa.timestamp('s', tz='utc'),
                        'date_update': pa.timestamp('s', tz='utc')
                    },
                    timestamp_parsers=['%Y%m%d%H%M%S']
                )
            )
            return this_table

        if not hasattr(self, 'index') or force:
            this_path = self.host + "/" + self.index_file
            if self.fs.exists(this_path + '.gz'):
                with self.fs.open(this_path + '.gz', "rb") as fg:
                    with gzip.open(fg) as f:
                        self.index = read_csv(f)
            else:
                with self.fs.open(this_path, "rb") as f:
                    self.index = read_csv(f)

        return self

    def __repr__(self):
        summary = ["<argoindex>"]
        summary.append("Name: %s" % self.index_file)
        summary.append("FTP: %s" % self.host)
        if hasattr(self, 'index'):
            summary.append("Loaded: True (%i records)" % self.shape[0])
        else:
            summary.append("Loaded: False")
        if hasattr(self, 'search'):
            summary.append("Searched: True (%i records, %0.4f%%)" % (self.N_FILES, self.N_FILES * 100 / self.shape[0]))
        else:
            summary.append("Searched: False")
        return "\n".join(summary)

    @property
    def data(self):
        return self.index.to_pandas()

    @property
    def full_uri(self):
        return ["/".join([self.host, "dac", f.as_py()]) for f in self.index['file']]

    @property
    def uri(self):
        return ["/".join([self.host, "dac", f.as_py()]) for f in self.search['file']]

    @property
    def shape(self):
        return self.index.shape

    @property
    def N_FILES(self):
        if hasattr(self, 'search'):
            return self.search.shape[0]
        else:
            return self.index.shape[0]

    def to_dataframe(self):
        return self.search.to_pandas()

    def search_wmo(self, WMOs):
        self.load()
        filt = []
        for wmo in WMOs:
            filt.append(pa.compute.match_substring_regex(self.index['file'], pattern="/%i/" % wmo))
        filt = np.logical_or.reduce(filt)
        self.search = self.index.filter(filt)
        return self

    def search_wmo_cyc(self, WMOs, CYCs):
        self.load()
        filt = []
        for wmo in WMOs:
            for cyc in CYCs:
                if cyc < 1000:
                    pattern = "%i_%0.3d.nc" % (wmo, cyc)
                else:
                    pattern = "%i_%0.4d.nc" % (wmo, cyc)
                filt.append(pa.compute.match_substring_regex(self.index['file'], pattern=pattern))
        filt = np.logical_or.reduce(filt)
        self.search = self.index.filter(filt)
        return self

    def search_latlon(self, BOX):
        self.load()
        filt = []
        filt.append(pa.compute.greater_equal(self.index['longitude'], BOX[0]))
        filt.append(pa.compute.less_equal(self.index['longitude'], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index['latitude'], BOX[2]))
        filt.append(pa.compute.less_equal(self.index['latitude'], BOX[3]))
        filt = np.logical_and.reduce(filt)
        self.search = self.index.filter(filt)
        return self

    def search_tim(self, BOX):
        self.load()
        filt = []
        filt.append(pa.compute.greater_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                             pa.array([pd.to_datetime(BOX[4])], pa.timestamp('ms'))[0]))
        filt.append(pa.compute.less_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                          pa.array([pd.to_datetime(BOX[5])], pa.timestamp('ms'))[0]))
        filt = np.logical_and.reduce(filt)
        self.search = self.index.filter(filt)
        return self

    def search_latlontim(self, BOX):
        self.load()
        filt = []
        filt.append(pa.compute.greater_equal(self.index['longitude'], BOX[0]))
        filt.append(pa.compute.less_equal(self.index['longitude'], BOX[1]))
        filt.append(pa.compute.greater_equal(self.index['latitude'], BOX[2]))
        filt.append(pa.compute.less_equal(self.index['latitude'], BOX[3]))
        filt.append(pa.compute.greater_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                             pa.array([pd.to_datetime(BOX[4])], pa.timestamp('ms'))[0]))
        filt.append(pa.compute.less_equal(pa.compute.cast(self.index['date'], pa.timestamp('ms')),
                                          pa.array([pd.to_datetime(BOX[5])], pa.timestamp('ms'))[0]))
        filt = np.logical_and.reduce(filt)
        self.search = self.index.filter(filt)
        return self


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
        chunks: str = "auto",
        chunks_maxsize: dict = {},
        api_timeout: int = 0,
        **kwargs
    ):
        """ Init fetcher

        Parameters
        ----------
        ftp: str (optional)
            Path to the remote FTP directory where the 'dac' folder is located.
        ds: str (optional)
            Dataset to load: 'phy' or 'ref' or 'bgc'
        errors: str (optional)
            If set to 'raise' (default), will raise a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If set to 'ignore', the file not found is skipped when fetching data.
        cache: bool (optional)
            Cache data or not (default: False)
        cachedir: str (optional)
            Path to cache folder
        dimension: str
            Main dimension of the output dataset. This can be "profile" to retrieve a collection of
            profiles, or "point" (default) to have data as a collection of measurements.
            This can be used to optimise performances.
        parallel: bool (optional)
            Chunk request to use parallel fetching (default: False)
        parallel_method: str (optional)
            Define the parallelization method: ``thread``, ``process`` or a :class:`dask.distributed.client.Client`.
        progress: bool (optional)
            Show a progress bar or not when fetching data.
        chunks: 'auto' or dict of integers (optional)
            Dictionary with request access point as keys and number of chunks to create as values.
            Eg:

                - ``{'wmo': 10}`` will create a maximum of 10 chunks along WMOs when used with ``Fetch_wmo``.
                - ``{'lon': 2}`` will create a maximum of 2 chunks along longitude when used with ``Fetch_box``.

        chunks_maxsize: dict (optional)
            Dictionary with request access point as keys and chunk size as values (used as maximum values in
            'auto' chunking).
            Eg: ``{'wmo': 5}`` will create chunks with as many as 5 WMOs each.
        api_timeout: int (optional)
            FTP request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """

        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout, size_policy='head')
        self.definition = "Ifremer GDAC ftp Argo data fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.server = api_server
        self.errors = errors
        self.ftp = OPTIONS["gdac_ftp"] if ftp == "" else ftp
        # check_gdacftp(self.ftp, errors="raise")  # Validate ftp
        self.indexfs = indexstore(host=self.ftp, cachedir=cachedir, cache=cache, timeout=timeout)

        if not isinstance(parallel, bool):
            parallel_method = parallel
            parallel = True
        if parallel_method not in ["thread"]:
            raise ValueError(
                "erddap only support multi-threading, use 'thread' instead of '%s'"
                % parallel_method
            )
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.progress = progress
        self.chunks = chunks
        self.chunks_maxsize = chunks_maxsize

        self.init(**kwargs)

    def __repr__(self):
        summary = ["<datafetcher.ftp>"]
        summary.append("Name: %s" % self.definition)
        summary.append("FTP: %s" % self.ftp)
        summary.append("Domain: %s" % format_oneline(self.cname()))
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

        # print("DIRECTION", np.unique(ds['DIRECTION']))
        # print("N_PROF", np.unique(ds['N_PROF']))
        ds = (
            ds.argo.profile2point()
        )  # Default output is a collection of points along N_POINTS
        # print("DIRECTION", np.unique(ds['DIRECTION']))

        # Remove netcdf file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        if self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.ftp
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now").strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        ds.attrs["Fetched_uri"] = ds.encoding["source"]
        ds = ds[np.sort(ds.data_vars)]

        return ds

    def to_xarray(self, errors: str = "ignore"):
        """ Load Argo data and return a xarray.Dataset

        Returns
        -------
        :class:`xarray.Dataset`
        """
        # Download data:
        if not self.parallel:
            method = "sequential"
        else:
            method = self.parallel_method
        # ds = self.fs.open_mfdataset(self.uri,
        #                             method=method,
        #                             concat_dim='N_POINTS',
        #                             concat=True,
        #                             preprocess=self._preprocess_multiprof,
        #                             progress=self.progress,
        #                             errors=errors,
        #                             decode_cf=1, use_cftime=0, mask_and_scale=1, engine='h5netcdf')
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

        # Remove netcdf file attributes and replace them with simplified argopy ones:
        ds.attrs = {}
        if self.dataset_id == "phy":
            ds.attrs["DATA_ID"] = "ARGO"
        if self.dataset_id == "bgc":
            ds.attrs["DATA_ID"] = "ARGO-BGC"
        ds.attrs["DOI"] = "http://doi.org/10.17882/42182"
        ds.attrs["Fetched_from"] = self.ftp
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now").strftime("%Y/%m/%d")
        ds.attrs["Fetched_constraints"] = self.cname()
        if len(self.uri) == 1:
            ds.attrs["Fetched_uri"] = self.uri[0]
        else:
            ds.attrs["Fetched_uri"] = ";".join(self.uri)

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
        if isinstance(CYC, int):
            CYC = np.array(
                (CYC,), dtype="int"
            )  # Make sure we deal with an array of integers
        if isinstance(CYC, list):
            CYC = np.array(
                CYC, dtype="int"
            )  # Make sure we deal with an array of integers
        self.WMO = WMO
        self.CYC = CYC

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
                self._list_of_argo_files = self.indexfs.search_wmo(self.WMO).uri
            else:
                self._list_of_argo_files = self.indexfs.search_wmo_cyc(self.WMO, self.CYC).uri

        return self._list_of_argo_files

    def dashboard(self, **kw):
        if len(self.WMO) == 1:
            return open_dashboard(wmo=self.WMO[0], **kw)
        else:
            warnings.warn("Dashboard only available for a single float request")
