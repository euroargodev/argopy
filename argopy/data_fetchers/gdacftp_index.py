"""
Argo index fetcher for remote GDAC FTP

This is not intended to be used directly, only by the facade at fetchers.py

"""
import os
from glob import glob
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings
import getpass
import logging

from argopy.utilities import (
    list_standard_variables,
    check_localftp,
    format_oneline,
    load_dict,
    mapp_dict
)
from argopy.options import OPTIONS
from argopy.stores import httpstore
from argopy.stores.argo_index_pa import indexstore
from argopy.plotters import open_dashboard

access_points = ["wmo", "box"]
exit_formats = ["xarray"]
dataset_ids = ["phy", "bgc"]  # First is default
api_server = 'https://data-argo.ifremer.fr/'  # API root url
api_server_check = api_server + 'readme_before_using_the_data.txt'  # URL to check if the API is alive
# api_server_check = OPTIONS["gdac_ftp"]

log = logging.getLogger("argopy.gdacftp.index")


class FTPArgoIndexFetcher(ABC):
    """ Manage access to Argo index from a remote GDAC FTP """
    N_FILES = None

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

        timeout = OPTIONS["api_timeout"] if api_timeout == 0 else api_timeout
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout, size_policy='head')
        self.definition = "Ifremer GDAC ftp Argo index fetcher"
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.server = api_server
        self.errors = errors
        self.ftp = OPTIONS["gdac_ftp"] if ftp == "" else ftp
        # check_gdacftp(self.ftp, errors="raise")  # Validate ftp
        self.indexfs = indexstore(host=self.ftp, cachedir=cachedir, cache=cache, timeout=timeout)
        self.init(**kwargs)

    def __repr__(self):
        summary = ["<indexfetcher.ftp>"]
        summary.append("Name: %s" % self.definition)
        summary.append("Index: %s" % self.indexfs.index_file)
        summary.append("FTP: %s" % self.ftp)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        if hasattr(self.indexfs, 'search'):
            summary.append("Records: %i/%i (%0.4f%%)" % (self.indexfs.N_FILES, self.indexfs.shape[0], self.indexfs.N_FILES * 100 / self.indexfs.shape[0]))
        else:
            summary.append("Records: <not loaded yet>")
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

    def to_dataframe(self):
        """ Filter index file and return a pandas dataframe """
        df = self.indexfs.to_dataframe()

        # Post-processing of the filtered index:
        df['wmo'] = df['file'].apply(lambda x: int(x.split('/')[1]))

        # institution & profiler mapping for all users
        # todo: may be we need to separate this for standard and expert users
        institution_dictionnary = load_dict('institutions')
        df['tmp1'] = df.institution.apply(lambda x: mapp_dict(institution_dictionnary, x))
        df = df.rename(columns={"institution": "institution_code", "tmp1": "institution"})

        profiler_dictionnary = load_dict('profilers')
        df['profiler'] = df.profiler_type.apply(lambda x: mapp_dict(profiler_dictionnary, int(x)))
        df = df.rename(columns={"profiler_type": "profiler_code"})

        return df

    def to_xarray(self):
        """ Load Argo index and return a xarray Dataset """
        return self.to_dataframe().to_xarray()


class Fetch_wmo(FTPArgoIndexFetcher):
    """ Manage access to GDAC ftp Argo index for: a list of WMOs, CYCs  """

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
        if len(self.CYC) > 0:
            log.debug("Create FTPArgoIndexFetcher.Fetch_box instance with index with WMOs=[%s] and CYCs=[%s]" % (
                ";".join([str(wmo) for wmo in self.WMO]),
                ";".join([str(cyc) for cyc in self.CYC])))
        else:
            log.debug("Create FTPArgoIndexFetcher.Fetch_box instance with index with WMOs=[%s] and CYCs=[%s]" % (
                ";".join([str(wmo) for wmo in self.WMO])))

        self.N_FILES = len(self.uri)  # Trigger file index load and search
        return self

    def cname(self):
        """ Return a unique string defining the constraints """
        if len(self.WMO) > 1:
            listname = ["WMO%i" % i for i in self.WMO]
            listname = ";".join(listname)
        else:
            listname = "WMO%i" % self.WMO[0]
        return listname

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


class Fetch_box(FTPArgoIndexFetcher):
    """ Manage access to GDAC ftp Argo index for: a rectangular space/time domain  """

    def init(self, box: list):
        """ Create Argo data loader

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
        log.debug("Create FTPArgoIndexFetcher.Fetch_box instance with index BOX: %s" % box)
        self.indexBOX = box
        self.N_FILES = len(self.uri)  # Trigger file index load and search
        return self

    def cname(self):
        """ Return a unique string defining the constraints """
        BOX = self.indexBOX
        if len(BOX) == 6:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; t=%s/%s]") % \
                      (BOX[0], BOX[1], BOX[2], BOX[3], self._format(BOX[4], 'tim'), self._format(BOX[5], 'tim'))
        else:
            boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % \
                      (BOX[0], BOX[1], BOX[2], BOX[3])
        return boxname

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
                self._list_of_argo_files = self.indexfs.search_latlon(self.indexBOX).uri
            else:
                self._list_of_argo_files = self.indexfs.search_latlontim(self.indexBOX).uri

        return self._list_of_argo_files