"""
Argo index fetcher for remote GDAC FTP

This is not intended to be used directly, only by the facade at fetchers.py

"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings
import logging
import importlib

from ..utilities import format_oneline
from ..options import OPTIONS, check_gdac_path
from ..plot import dashboard


log = logging.getLogger("argopy.gdacftp.index")

has_pyarrow = importlib.util.find_spec('pyarrow') is not None
if has_pyarrow:
    from argopy.stores.argo_index_pa import indexstore_pyarrow as indexstore
    log.debug("Using pyarrow indexstore")
else:
    from argopy.stores.argo_index_pd import indexstore_pandas as indexstore
    # warnings.warn("Consider installing pyarrow in order to improve performances when fetching GDAC data")
    log.debug("Using pandas indexstore")

access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "bgc"]  # First is default
api_server = OPTIONS['ftp']  # API root url
api_server_check = api_server  # URL to check if the API is alive, used by isAPIconnected


class FTPArgoIndexFetcher(ABC):
    """ Manage access to Argo index from a remote GDAC FTP """
    # N_FILES = None

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
        errors: str = "raise",
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
        api_timeout: int (optional)
            FTP request time out in seconds. Set to OPTIONS['api_timeout'] by default.
        """
        self.timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.definition = "Ifremer GDAC ftp Argo index fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.server = OPTIONS['ftp'] if ftp == "" else ftp
        self.errors = errors

        # Validate server, raise FtpPathError if not valid.
        check_gdac_path(self.server, errors='raise')

        if self.dataset_id == 'phy':
            index_file = "ar_index_global_prof.txt"
        elif self.dataset_id == 'bgc':
            index_file = "argo_synthetic-profile_index.txt"

        # Validation of self.server is done by the indexstore:
        self.indexfs = indexstore(host=self.server, index_file=index_file, cache=cache, cachedir=cachedir, timeout=self.timeout)
        self.fs = self.indexfs.fs['src']

        nrows = None
        if 'N_RECORDS' in kwargs:
            nrows = kwargs['N_RECORDS']
        self.N_RECORDS = self.indexfs.load(nrows=nrows).N_RECORDS  # Number of records in the index

        self.init(**kwargs)

    def __repr__(self):
        summary = ["<indexfetcher.ftp>"]
        summary.append("Name: %s" % self.definition)
        summary.append("Index: %s" % self.indexfs.index_file)
        summary.append("FTP: %s" % self.server)
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

    def _format(self, x, typ):
        """ string formatting helper """
        if typ == 'lon':
            if x < 0:
                x = 360. + x
            return ("%05d") % (x * 100.)
        if typ == 'lat':
            return ("%05d") % (x * 100.)
        if typ == 'prs':
            return ("%05d") % (np.abs(x)*10.)
        if typ == 'tim':
            return pd.to_datetime(x).strftime('%Y-%m-%d')
        return str(x)

    def cname(self):
        """ Return a unique string defining the constraints """
        return self.indexfs.cname

    @property
    def cachepath(self):
        """ Return path to cache file for this request

        Returns
        -------
        str
        """
        return self.indexfs.cachepath('search')

    def clear_cache(self):
        """ Remove cache files and entries from resources open with this fetcher """
        self.indexfs.clear_cache()
        self.fs.clear_cache()
        return self

    def to_dataframe(self):
        """ Filter index file and return a pandas dataframe """
        df = self.indexfs.run().to_dataframe()

        # Post-processing of the filtered index is done at the indexstore level
        # if 'wmo' not in df:
        #     df['wmo'] = df['file'].apply(lambda x: int(x.split('/')[1]))
        #
        # # institution & profiler mapping for all users
        # # todo: may be we need to separate this for standard and expert users
        # institution_dictionnary = load_dict('institutions')
        # df['tmp1'] = df.institution.apply(lambda x: mapp_dict(institution_dictionnary, x))
        # df = df.rename(columns={"institution": "institution_code", "tmp1": "institution"})
        #
        # profiler_dictionnary = load_dict('profilers')
        # df['profiler'] = df.profiler_type.apply(lambda x: mapp_dict(profiler_dictionnary, int(x)))
        # df = df.rename(columns={"profiler_type": "profiler_code"})

        return df

    def to_xarray(self):
        """ Load Argo index and return a xarray Dataset """
        return self.to_dataframe().to_xarray()

    def dashboard(self, **kw):
        if self.WMO is not None:
            if len(self.WMO) == 1 and self.CYC is not None and len(self.CYC) == 1:
                return dashboard(wmo=self.WMO[0], cyc=self.CYC[0], **kw)
            elif len(self.WMO) == 1:
                return dashboard(wmo=self.WMO[0], **kw)
            else:
                warnings.warn("Dashboard only available for a single float or profile request")


class Fetch_wmo(FTPArgoIndexFetcher):
    """ Manage access to GDAC ftp Argo index for: a list of WMOs, CYCs  """

    def init(self, WMO: list = [], CYC=None, **kwargs):
        """ Create Argo index fetcher for WMOs

        Parameters
        ----------
        WMO: list(int)
            The list of WMOs to load all Argo data for.
        CYC: int, np.array(int), list(int)
            The cycle numbers to load.
        """
        self.WMO = WMO
        self.CYC = CYC
        self._nrows = None
        if 'MAX_FILES' in kwargs:
            self._nrows = kwargs['MAX_FILES']
        self.N_FILES = len(self.uri)  #  Must trigger file index load and search at instantiation
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
                self._list_of_argo_files = self.indexfs.search_wmo(self.WMO, nrows=self._nrows).uri
            else:
                self._list_of_argo_files = self.indexfs.search_wmo_cyc(self.WMO, self.CYC, nrows=self._nrows).uri

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files


class Fetch_box(FTPArgoIndexFetcher):
    """ Manage access to GDAC ftp Argo index for: a rectangular space/time domain  """

    def init(self, box: list, **kwargs):
        """ Create Argo index fetcher

        Parameters
        ----------
        box: list()
            Define the domain to load Argo index for. The box list is made of:
                - lon_min: float, lon_max: float,
                - lat_min: float, lat_max: float,
                - date_min: str (optional), date_max: str (optional)

            Longitude and latitude bounds are required, while the two bounding dates are optional.
            If bounding dates are not specified, the entire time series is fetched.
            Eg: [-60, -55, 40., 45., '2007-08-01', '2007-09-01']

        """
        # We use a full domain definition (x, y, z, t) as argument for compatibility with the other fetchers
        # but at this point, we internally work only with x, y and t.
        # log.debug("Create FTPArgoIndexFetcher.Fetch_box instance with index BOX: %s" % box)
        self.indexBOX = box.copy()

        self._nrows = None
        if 'MAX_FILES' in kwargs:
            self._nrows = kwargs['MAX_FILES']
        self.N_FILES = len(self.uri)  #  Must trigger file index load and search at instantiation

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
                self._list_of_argo_files = self.indexfs.search_lat_lon(self.indexBOX, nrows=self._nrows).uri
            else:
                self._list_of_argo_files = self.indexfs.search_lat_lon_tim(self.indexBOX, nrows=self._nrows).uri

        self.N_FILES = len(self._list_of_argo_files)
        return self._list_of_argo_files