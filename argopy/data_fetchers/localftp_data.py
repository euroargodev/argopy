"""Argo data fetcher for a local copy of GDAC ftp.

This is not intended to be used directly, only by the facade at fetchers.py

"""
import os
from glob import glob
import numpy as np
import pandas as pd
from abc import abstractmethod
import getpass
import logging

from .proto import ArgoDataFetcherProto
from argopy.errors import NetCDF4FileNotFoundError
from argopy.utilities import (
    list_standard_variables,
    check_localftp,
    format_oneline,
    deprecated
)
from argopy.options import OPTIONS
from argopy.stores import filestore, indexstore, indexfilter_box


access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "bgc"]  # First is default
api_server_check = OPTIONS["local_ftp"]

log = logging.getLogger("argopy.localftp.data")


class LocalFTPArgoDataFetcher(ArgoDataFetcherProto):
    """Manage access to Argo data from a local copy of GDAC ftp.

    .. warning::

        This fetcher is deprecated. It's been replaced by the ``gdac`` fetcher.

        ================= ======
        Deprecation cycle
        ================= ======
        Warning           0.1.11
        Error             0.1.12
        Delete            0.1.13
        ================= ======
    """

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
    @deprecated("The 'localftp' data source is deprecated. It's been replaced by 'gdac'. It will raise an error after argopy 0.1.12")
    def __init__(
        self,
        local_ftp: str = "",
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
        **kwargs
    ):
        """ Init fetcher

        Parameters
        ----------
        local_ftp: str, optional
            Path to the local directory where the 'dac' folder is located.
        ds: str, optional
            Dataset to load: 'phy' or 'ref' or 'bgc'
        errors: str, optional
            If set to 'raise' (default), will raise a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If set to 'ignore', the file not found is skipped when fetching data.
        cache: bool, default: False
            Cache data or not
        cachedir: str, optional
            Path to cache folder
        dimension: str, default: point
            Main dimension of the output dataset. This can be ``profile`` to retrieve a collection of
            profiles, or ``point`` to have data as a collection of measurements.
            This can be used to optimise performances.
        parallel: bool, default: False
            Chunk request to use parallel fetching
        parallel_method: str, default: thread
            Define the parallelization method: ``thread``, ``process`` or a :class:`dask.distributed.client.Client`.
        progress: bool, default: False
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
        """
        self.cache = cache
        self.cachedir = cachedir
        self.fs = filestore(cache=self.cache, cachedir=self.cachedir)
        self.errors = errors

        if not isinstance(parallel, bool):
            # The parallelization method is passed through the argument 'parallel':
            parallel_method = parallel
            if parallel in ["thread", "process"]:
                parallel = True
        if parallel_method not in ["thread", "process"]:
            raise ValueError(
                "localftp only support multi-threading and processing ('%s' unknown)"
                % parallel_method
            )
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.progress = progress
        self.chunks = chunks
        self.chunks_maxsize = chunks_maxsize

        self.definition = "Local ftp Argo data fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds

        self.local_ftp = OPTIONS["local_ftp"] if local_ftp == "" else local_ftp
        check_localftp(self.local_ftp, errors="raise")  # Validate local_ftp

        self.init(**kwargs)

    def __repr__(self):
        summary = ["<datafetcher.localftp>"]
        summary.append("Name: %s" % self.definition)
        summary.append("FTP: %s" % self.local_ftp)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return "\n".join(summary)

    def cname(self):
        """ Return a unique string defining the constraints """
        return self._cname()

    def get_path(self, wmo: int, cyc: int = None) -> str:  # noqa: C901
        """ Return the absolute path toward the netcdf source file of a given wmo/cyc pair and a dataset

        Based on the dataset, the wmo and the cycle requested, return the absolute path toward the file to load.

        The file is searched using its expected file name pattern (following GDAC conventions).

        If more than one file are found to match the pattern, the first 1 (alphabetically) is returned.

        If no files match the pattern, the function can raise an error or fail silently and return None.

        Parameters
        ----------
        wmo: int
            WMO float code
        cyc: int, optional
            Cycle number (None by default)

        Returns
        -------
        netcdf_file_path : str
        """
        # This function will be used whatever the access point, since we are working with a GDAC like set of files
        def _filepathpattern(wmo, cyc=None):
            """ Return a file path pattern to scan for a given wmo/cyc pair

            Based on the dataset and the cycle number requested, construct the closest file path pattern to be loaded

            This path is absolute, the pattern can contain '*', and it is the file path, so it has '.nc' extension

            Returns
            -------
            file_path_pattern : str
            """
            if cyc is None:
                # Multi-profile file:
                # dac/<DacName>/<FloatWmoID>/<FloatWmoID>_<S>prof.nc
                if self.dataset_id == "phy":
                    return os.path.sep.join(
                        [self.local_ftp, "dac", "*", str(wmo), "%i_prof.nc" % wmo]
                    )
                elif self.dataset_id == "bgc":
                    return os.path.sep.join(
                        [self.local_ftp, "dac", "*", str(wmo), "%i_Sprof.nc" % wmo]
                    )
            else:
                # Single profile file:
                # dac/<DacName>/<FloatWmoID>/profiles/<B/M/S><R/D><FloatWmoID>_<XXX><D>.nc
                if cyc < 1000:
                    return os.path.sep.join(
                        [
                            self.local_ftp,
                            "dac",
                            "*",
                            str(wmo),
                            "profiles",
                            "*%i_%0.3d*.nc" % (wmo, cyc),
                        ]
                    )
                else:
                    return os.path.sep.join(
                        [
                            self.local_ftp,
                            "dac",
                            "*",
                            str(wmo),
                            "profiles",
                            "*%i_%0.4d*.nc" % (wmo, cyc),
                        ]
                    )

        pattern = _filepathpattern(wmo, cyc)
        lst = sorted(glob(pattern))
        # log.debug(lst)
        # lst = sorted(self.fs.glob(pattern))  # Much slower than the regular glob !
        if len(lst) == 1:
            return lst[0]
        elif len(lst) == 0:
            if self.errors == "raise":
                raise NetCDF4FileNotFoundError(pattern)
            else:
                # Otherwise remain silent/ignore
                # todo: should raise a warning instead ?
                return None
        else:
            # warnings.warn("More than one file to load for a single float cycle ! Return the 1st one by default.")
            # The choice of the file to load depends on the user mode and dataset requested.
            # todo: define a robust choice
            if self.dataset_id == "phy":
                if cyc is None:
                    # Use the synthetic profile:
                    lst = [
                        file
                        for file in lst
                        if [
                            file
                            for file in [os.path.split(w)[-1] for w in lst]
                            if file[0] == "S"
                        ][0]
                        in file
                    ]
                else:
                    # Use the ascent profile:
                    lst = [
                        file
                        for file in lst
                        if [
                            file
                            for file in [os.path.split(w)[-1] for w in lst]
                            if file[-1] != "D"
                        ][0]
                        in file
                    ]
            elif self.dataset_id == "bgc":
                lst = [
                    file
                    for file in lst
                    if [
                        file
                        for file in [os.path.split(w)[-1] for w in lst]
                        if file[0] == "M"
                    ][0]
                    in file
                ]
            return lst[0]

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
        ds.attrs["Fetched_from"] = self.local_ftp
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
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
        ds.attrs["Fetched_from"] = self.local_ftp
        ds.attrs["Fetched_by"] = getpass.getuser()
        ds.attrs["Fetched_date"] = pd.to_datetime("now", utc=True).strftime("%Y/%m/%d")
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


class Fetch_wmo(LocalFTPArgoDataFetcher):
    """ Manage access to local ftp Argo data for: a list of WMOs  """

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

        return self

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """

        def list_bunch(wmos, cycs):
            this = []
            for wmo in wmos:
                if cycs is None:
                    this.append(self.get_path(wmo))
                else:
                    for cyc in cycs:
                        this.append(self.get_path(wmo, cyc))
            return this

        # Get list of files to load:
        if not hasattr(self, "_list_of_argo_files"):
            self._list_of_argo_files = list_bunch(self.WMO, self.CYC)

        return self._list_of_argo_files


class Fetch_box(LocalFTPArgoDataFetcher):
    """ Manage access to local ftp Argo data for: a rectangular space/time domain  """

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
        self.BOX = box.copy()
        self.indexBOX = [self.BOX[ii] for ii in [0, 1, 2, 3]]
        if len(self.BOX) == 8:
            self.indexBOX = [self.BOX[ii] for ii in [0, 1, 2, 3, 6, 7]]

        self.fs_index = indexstore(
            self.cache,
            self.cachedir,
            os.path.sep.join([self.local_ftp, "ar_index_global_prof.txt"]),
        )
        # todo we would need to check if the index file is there !

    @property
    def uri(self):
        """ List of files to load for a request

        Returns
        -------
        list(str)
        """
        if not hasattr(self, "_list_of_argo_files"):
            self._list_of_argo_files = []
            # Fetch the index to retrieve the list of files/profiles to load:
            filt = indexfilter_box(self.indexBOX)
            df_index = self.fs_index.read_csv(filt)

            # if isinstance(df_index, pd.core.frame.DataFrame):
            # Ok, we found profiles in the index file,
            # so now we can make sure these files exist:
            lst = list(df_index["file"])
            for file in lst:
                abs_file = os.path.sep.join([self.local_ftp, "dac", file])
                if self.fs.exists(abs_file):
                    self._list_of_argo_files.append(abs_file)
                elif self.errors == "raise":
                    raise NetCDF4FileNotFoundError(abs_file)
                else:
                    # Otherwise remain silent/ignore
                    # todo should raise a warning instead ?
                    return None
        return self._list_of_argo_files
