#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for a local copy of GDAC ftp.
#
# This is not intended to be used directly, only by the facade at fetchers.py
#
# Since the GDAC ftp is organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.
#
# class LocalFTPArgoDataFetcher(ABC)
#   @abstractmethod init(self)
#   @abstractmethod cname(self)
#   @abstractmethod to_xarray(self)
#   __init__
#   __repr__
#   filter_data_mode(self, ds, keep_error=True)
#   filter_qc(self, this, QC_list=[1, 2], drop=True, mode='all', mask=False)
#   filter_variables(self, this, mode='standard')
#
# class ArgoDataFetcher_wmo(LocalFTPArgoDataFetcher)
#   init(self, WMO=[], CYC=None)
#   cname(self, cache=False)
#   to_xarray(self)
#
# Created by gmaze on 18/03/2020
# Building on earlier work from S. Tokunaga (as part of the MOCCA and EARISE H2020 projects)

access_points = ['wmo']
exit_formats = ['xarray']
dataset_ids = ['phy', 'bgc']

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
import warnings

from argopy.xarray import ArgoMultiProfLocalLoader
from argopy.errors import NetCDF4FileNotFoundError
from argopy.utilities import list_multiprofile_file_variables, list_standard_variables


class LocalFTPArgoDataFetcher(ABC):
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
    def __init__(self, local_path='.', ds='phy', cache=False, cachedir=None, **kwargs):
        """ Init fetcher

            Parameters
            ----------
            local_path : str
                Path to the directory with the 'dac' folder and index file
        """
        self.cache = cache or not not cachedir # Yes, this is not not
        self.cachedir = cachedir
        if self.cache:
            #todo check if cachedir is a valid path
            Path(self.cachedir).mkdir(parents=True, exist_ok=True)

        self.definition = 'Local ftp Argo data fetcher'
        self.dataset_id = ds
        self.local_ftp_path = local_path
        self._init_localftp()
        self.init(**kwargs)

    def __repr__(self):
        summary = [ "<datafetcher '%s'>" % self.definition ]
        summary.append( "Domain: %s" % self.cname(cache=0) )
        return '\n'.join(summary)

    def _init_localftp(self):
        """ Internal list of available netcdf files to be processed """
        pattern = os.path.sep.join(["*","*","*_prof.nc"])
        self.argo_files = sorted(glob(os.path.sep.join([self.local_ftp_path, pattern])))
        if self.argo_files is None:
            raise ValueError("Argo root path doesn't contain any netcdf profile files (under %s)" % patterm)
        self.argo_wmos = [int(os.path.basename(x).split("_")[0]) for x in self.argo_files]
        self.argo_dacs = [x.split(os.path.sep)[-3] for x in self.argo_files]

    def filter_data_mode(self, ds, keep_error=True):
        return ds

    def filter_qc(self, ds, QC_list=[1, 2], drop=True, mode='all', mask=False):
        return ds

    def filter_variables(self, ds, mode='standard'):
        if mode == 'standard':
            to_remove = sorted(list(set(list(ds.data_vars)) - set(list_standard_variables())))
            return ds.drop_vars(to_remove)
        else:
            return ds

class ArgoDataFetcher_wmo(LocalFTPArgoDataFetcher):
    """ Manage access to local ftp Argo data for: a list of WMOs

    """
    def init(self, WMO=[], CYC=None):
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
        return self

    def cname(self, cache=False):
        """ Return a unique string defining the request """
        if len(self.WMO) > 1:
            if cache:
                listname = ["WMO%i" % i for i in self.WMO]
                if isinstance(self.CYC, (np.ndarray)):
                    [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = "_".join(listname)
            else:
                listname = ["WMO%i" % i for i in self.WMO]
                if isinstance(self.CYC, (np.ndarray)):
                    [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = ";".join(listname)
        else:
            listname = "WMO%i" % self.WMO[0]
            if isinstance(self.CYC, (np.ndarray)):
                listname = [listname]
                [listname.append("CYC%0.4d" % i) for i in self.CYC]
                listname = "_".join(listname)
        listname = self.dataset_id + "_" + listname
        return listname

    def _xload_multiprof(self, dac_wmo_file):
        """Load an Argo multi-profile file as a collection of points"""
        dac_name, wmo_id = dac_wmo_file
        wmo_id = int(wmo_id)

        # Open file:
        ncfile = os.path.sep.join([self.local_ftp_path, dac_name, str(wmo_id), ("%i_prof.nc" % wmo_id)])
        if not os.path.isfile(ncfile):
            raise NetCDF4FileNotFoundError(ncfile)

        ds = xr.open_dataset(ncfile, decode_cf=1, use_cftime=0)

        # Replace JULD and JULD_QC by TIME and TIME_QC
        ds = ds.rename({'JULD':'TIME', 'JULD_QC':'TIME_QC'})
        ds['TIME'].attrs = {'long_name': 'Datetime (UTC) of the station',
                            'standard_name':  'time'}
        # Cast data types:
        ds = ds.argo.cast_types()

        # Remove variables without dimensions:
        # We should be able to find a way to keep them
        for v in ds.data_vars:
            if len(list(ds[v].dims)) == 0:
                ds = ds.drop(v)

        # Also remove variables with dimensions other than N_PROF or N_LEVELS
        # This is not satisfactory for operators or experts, but that get us started
        for v in ds.data_vars:
            for d in ds[v].dims:
                if d not in ['N_PROF', 'N_LEVELS']:
                    ds = ds.drop(v)
                    break

        ds = ds.argo.profile2point()

        # instantiate the data loader:
        # argo_loader = ArgoMultiProfLocalLoader(self.local_ftp_path)
        # Open the file:
        # ds = argo_loader.load_from_inst(dac_name, wmo_id)
        # Possibly squeeze to correct cycle number:
        return ds

    def to_xarray(self, client=None, n=None):
        """ Load Argo data and return a xarray.DataSet

            Possibly use a dask distributed client for performance
        """
        for wmo in self.WMO:
            if wmo not in self.argo_wmos:
                raise ValueError("This float is not available locally at: %s" % self.local_ftp_path)

        if len(self.WMO) == 1:
            self.argo_dacs[self.argo_wmos.index(self.WMO[0])]
            return self._xload_multiprof([self.argo_dacs[self.argo_wmos.index(self.WMO[0])], self.WMO[0]])

        dac_wmo_files = list(zip(*[self.argo_dacs, self.argo_wmos]))
        if n is not None:  # Sub-sample for test purposes (usefull when fetching the entire dataset)
            dac_wmo_files = list(np.array(dac_wmo_files)[np.random.choice(range(0, len(dac_wmo_files) - 1), n)])
        warnings.warn("NB OF FLOATS TO FETCH: %i" % len(dac_wmo_files))

        if client is not None:
            futures = client.map(self._xload_multiprof, dac_wmo_files)
            results = client.gather(futures, errors='raise')
        else:
            results = []
            for wmo in dac_wmo_files:
                results.append(self._xload_multiprof(wmo))

        results = [r for r in results if r is not None]  # Only keep none empty results
        if len(results) > 0:
            ds = xr.concat(results, dim='index', data_vars='all', compat='equals')
            ds.attrs.pop('DAC')
            ds.attrs.pop('WMO')
            ds = ds.sortby('time')
            ds['index'].values = np.arange(0, len(ds['index']))
            return ds
        else:
            warnings.warn("CAN'T FETCH ANY DATA !")
            return None