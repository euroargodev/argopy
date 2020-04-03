#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for a local copy of GDAC ftp.
#
# This is not intended to be used directly, only by the facade at fetchers.py
#
# Since the GDAC ftp is organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.
#

access_points = ['wmo']
exit_formats = ['xarray']
dataset_ids = ['phy', 'bgc'] # First is default

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
import warnings
import getpass
from pathlib import Path

import multiprocessing as mp
import distributed

from .proto import ArgoDataFetcherProto
from argopy.errors import NetCDF4FileNotFoundError
from argopy.utilities import list_multiprofile_file_variables, list_standard_variables
from argopy.options import OPTIONS

class LocalFTPArgoDataFetcher(ArgoDataFetcherProto):
    """ Manage access to Argo data from a local copy of GDAC ftp

    """
    ###
    # Methods to be customised for a specific erddap request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        pass

    @abstractmethod
    def cname(self):
        """ Return a unique string defining the request """
        pass

    @abstractmethod
    def to_xarray(self):
        """ Load Argo data and return a xarray.DataSet """
        pass

    ###
    # Methods that must not change
    ###
    def __init__(self,
                 path_ftp: str = "",
                 ds: str = "",
                 cache: bool = False,
                 cachedir: str = "",
                 **kwargs):
        """ Init fetcher

            Parameters
            ----------
            local_path : str
                Path to the directory with the 'dac' folder and index file
        """
        self.cache = cache
        self.cachedir = OPTIONS['cachedir'] if cachedir == '' else cachedir
        if self.cache:
            #todo check if cachedir is a valid path
            Path(self.cachedir).mkdir(parents=True, exist_ok=True)

        self.definition = 'Local ftp Argo data fetcher'
        self.dataset_id = OPTIONS['dataset'] if ds == '' else ds
        self.path_ftp = OPTIONS['local_ftp'] if path_ftp == '' else path_ftp
        self.init(**kwargs)

    def __repr__(self):
        summary = [ "<datafetcher '%s'>" % self.definition ]
        summary.append( "FTP: %s" % self.path_ftp )
        summary.append( "Domain: %s" % self.cname(cache=0) )
        return '\n'.join(summary)

    def filter_data_mode(self, ds, **kwargs):
        return ds.argo.filter_data_mode(errors='ignore', **kwargs)

    def filter_qc(self, ds, **kwargs):
        return ds.argo.filter_qc(**kwargs)

    def filter_variables(self, ds, mode='standard'):
        if mode == 'standard':
            to_remove = sorted(list(set(list(ds.data_vars)) - set(list_standard_variables())))
            return ds.drop_vars(to_remove)
        else:
            return ds

class Fetch_wmo(LocalFTPArgoDataFetcher):
    """ Manage access to local ftp Argo data for: a list of WMOs

    """
    def init(self, WMO: list = [], CYC=None, **kwargs):
        """ Create Argo data loader for WMOs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to load all Argo data for.
            CYC : int, np.array(int), list(int)
                The cycle numbers to load.
        """
        if isinstance(WMO, int):
            WMO = [WMO] # Make sure we deal with a list
        if isinstance(CYC, int):
            CYC = np.array((CYC,), dtype='int') # Make sure we deal with an array of integers
        if isinstance(CYC, list):
            CYC = np.array(CYC, dtype='int') # Make sure we deal with an array of integers
        self.WMO = WMO
        self.CYC = CYC
        # Build the internal list of files to load:
        # self.argo_files = [] # We do it in a lazy mode, ie only if data are requested with .to_xarray()
        return self

    def cname(self, cache=False):
        """ Return a unique string defining the request """
        if len(self.WMO) > 1:
            if cache:
                listname = ["WMO%i" % i for i in self.WMO]
                if isinstance(self.CYC, (np.ndarray)):
                    [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = "_".join(listname)
                listname = self.dataset_id + "_" + listname
            else:
                listname = ["WMO%i" % i for i in self.WMO]
                if isinstance(self.CYC, (np.ndarray)):
                    [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = ";".join(listname)
                listname = self.dataset_id + ";" + listname
        else:
            listname = "WMO%i" % self.WMO[0]
            if isinstance(self.CYC, (np.ndarray)):
                listname = [listname]
                [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = "_".join(listname)
                listname = self.dataset_id + "_" + listname
        return listname

    def _filepathpattern(self, wmo, cyc=None):
        """ Set netcdf file path pattern to load for a given wmo/cyc pair """
        if cyc is None:
            # Multi-profile file:
            # <FloatWmoID>_prof.nc
            if self.dataset_id == 'phy':
                return os.path.sep.join([self.path_ftp, "*", str(wmo), "%i_prof.nc" % wmo])
            elif self.dataset_id == 'bgc':
                return os.path.sep.join([self.path_ftp, "*", str(wmo), "%i_Sprof.nc" % wmo])
        else:
            # Single profile file:
            # <B/M/S><R/D><FloatWmoID>_<XXX><D>.nc
            if cyc < 1000:
                return os.path.sep.join([self.path_ftp, "*", str(wmo), "profiles", "*%i_%0.3d*.nc" % (wmo, cyc)])
            else:
                return os.path.sep.join([self.path_ftp, "*", str(wmo), "profiles", "*%i_%0.4d*.nc" % (wmo, cyc)])


    def _absfilepath(self, wmo: int, cyc: int = None, errors: str = 'raise') -> str:
        """ Set absolute netcdf file path to load for a given wmo/cyc pair

        Parameters
        ----------
        wmo: int
            WMO float code
        cyc: int, optional
            Cycle number (None by default)
        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if the requested
            file cannot be found. If 'ignore', return None silently.

        Returns
        -------
        netcdf_file_path : str
        """
        p = self._filepathpattern(wmo, cyc)
        l = sorted(glob(p))
        if len(l) == 1:
            return l[0]
        elif len(l) == 0:
            if errors == 'raise':
                raise NetCDF4FileNotFoundError(p)
            else:
                # Otherwise remain silent/ignore
                #todo should raise a warning instead ?
                return None
        else:
            warnings.warn("More than one file to load for a single float cycle ! Return the 1st one by default.")
            # The choice of the file to load depends on the user mode and dataset requested.
            #todo define a robust choice
            if self.dataset_id == 'phy':
                # Use the synthetic profile:
                l = [file for file in l if
                      [file for file in [os.path.split(w)[-1] for w in l] if file[0] == 'S'][0] in file]
                # print('phy', l[0])
            elif self.dataset_id == 'bgc':
                l = [file for file in l if
                     [file for file in [os.path.split(w)[-1] for w in l] if file[0] == 'M'][0] in file]
                # print('bgc:', l)
            return l[0]

    def _list_argo_files(self, errors: str = 'raise'):
        """ Set the internal list of files to load """
        if not hasattr(self, '_list_of_argo_files'):
            self._list_of_argo_files = []
            for wmo in self.WMO:
                if self.CYC is None:
                    self._list_of_argo_files.append(self._absfilepath(wmo, errors=errors))
                else:
                    for cyc in self.CYC:
                        self._list_of_argo_files.append(self._absfilepath(wmo, cyc, errors=errors))
        return self

    @property
    def files(self):
        """ Return files to load """
        if not hasattr(self, '_list_of_argo_files'):
            self._list_argo_files()
        return self._list_of_argo_files

    def _xload_multiprof(self, ncfile: str):
        """Load an Argo multi-profile file as a collection of points

        Parameters
        ----------
        ncfile: str
            Absolute path to a netcdf file to load

        Returns
        -------
        :class:`xarray.Dataset`

        """
        ds = xr.open_dataset(ncfile, decode_cf=1, use_cftime=0, mask_and_scale=1, engine='netcdf4')

        # Replace JULD and JULD_QC by TIME and TIME_QC
        ds = ds.rename({'JULD':'TIME', 'JULD_QC':'TIME_QC'})
        ds['TIME'].attrs = {'long_name': 'Datetime (UTC) of the station',
                            'standard_name':  'time'}
        # Cast data types:
        ds = ds.argo.cast_types()

        # Enforce real pressure resolution: 0.1 db
        for vname in ds.data_vars:
            if 'PRES' in vname and 'QC' not in vname:
                ds[vname].values = np.round(ds[vname].values,1)

        # Remove variables without dimensions:
        # We should be able to find a way to keep them somewhere in the data structure
        for v in ds.data_vars:
            if len(list(ds[v].dims)) == 0:
                ds = ds.drop_vars(v)

        ds = ds.argo.profile2point() # Default output is a collection of points

        # Remove netcdf file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == 'phy':
            ds.attrs['DATA_ID'] = 'ARGO'
        if self.dataset_id == 'bgc':
            ds.attrs['DATA_ID'] = 'ARGO-BGC'
        ds.attrs['DOI'] = 'http://doi.org/10.17882/42182'
        ds.attrs['Fetched_from'] = self.path_ftp
        ds.attrs['Fetched_by'] = getpass.getuser()
        ds.attrs['Fetched_date'] = pd.to_datetime('now').strftime('%Y/%m/%d')
        ds.attrs['Fetched_constraints'] = self.cname()
        ds.attrs['Fetched_url'] = ds.encoding['source']
        ds = ds[np.sort(ds.data_vars)]

        return ds

    def _best_loader(self, **kwargs):
        """ Load data as efficiently as possible

        Returns
        -------
        :class:`xarray.Dataset`

        """
        client = None if 'client' not in kwargs else kwargs['client']

        if len(self.files) == 1:
            return self._xload_multiprof(self.files[0])

        else:
            warnings.warn("Fetching more than one file in a single request is not yet fully reliable. "
                          "If you encounter an error, try to load each float separately.")

            if client is not None:
                if type(client) == distributed.client.Client:
                    # Use dask client:
                    futures = client.map(self._xload_multiprof, self.files)
                    results = client.gather(futures)
                else:
                    # Use multiprocessing Pool
                    with mp.Pool() as pool:
                        results = pool.map(self._xload_multiprof, self.files)
            else:
                results = []
                for f in self.files:
                    results.append(self._xload_multiprof(f))

        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            # ds = xr.concat(results, dim='index', data_vars='all', coords='all', compat='equals')
            ds = xr.concat(results, dim='index', data_vars='all', coords='all', compat='override')
            ds['index'] = np.arange(0, len(ds['index'])) # Re-index to avoid duplicate values
            ds = ds.set_coords('index')
            ds = ds.sortby('TIME')
            return ds
        else:
            raise ValueError("CAN'T FETCH ANY DATA !")

    def to_xarray(self, errors: str = 'raise', client=None):
        """ Load Argo data and return a xarray.Dataset

        Parameters
        ----------
        errors: {'raise','ignore'}, optional
            If 'raise' (default), raises a NetCDF4FileNotFoundError error if any of the requested
            files cannot be found. If 'ignore', ignore this file in fetching data.

        client: None, dask.client or 'mp'

        Returns
        -------
        :class:`xarray.Dataset`
        """
        # Set internal list of files to load:
        self._list_argo_files(errors=errors)

        # Load data (will raise an error if no data found):
        ds = self._best_loader(client=client)

        # Remove netcdf file attributes and replace them with argopy ones:
        ds.attrs = {}
        if self.dataset_id == 'phy':
            ds.attrs['DATA_ID'] = 'ARGO'
        if self.dataset_id == 'bgc':
            ds.attrs['DATA_ID'] = 'ARGO-BGC'
        ds.attrs['DOI'] = 'http://doi.org/10.17882/42182'
        ds.attrs['Fetched_from'] = self.path_ftp
        ds.attrs['Fetched_by'] = getpass.getuser()
        ds.attrs['Fetched_date'] = pd.to_datetime('now').strftime('%Y/%m/%d')
        ds.attrs['Fetched_constraints'] = self.cname()
        if len(self.files) == 1:
            ds.attrs['Fetched_url'] = ds.encoding['source']
        return ds