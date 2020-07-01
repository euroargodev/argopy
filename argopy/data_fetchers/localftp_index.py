#!/bin/env python
# -*coding: UTF-8 -*-
"""
Argo data fetcher for a local copy of GDAC ftp.

This is not intended to be used directly, only by the facade at fetchers.py

Since the GDAC ftp is organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.



About the index local ftp fetcher:

We have a large index csv file "ar_index_global_prof.txt", that is about ~200Mb
For a given request, we need to load/read it
and then to apply a filter to select lines matching the request
With the current version, a dataframe of the full index is cached
and then another cached file is created for the result of the filter.

df_full = pd.read_csv("index.txt")
df_small = filter(df_full)
write_on_file(df_small)

I think we can avoid this with a virtual file system
When a request is done, we



"""
import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
import warnings
import getpass

import multiprocessing as mp
import distributed

from .proto import ArgoDataFetcherProto
from argopy.errors import NetCDF4FileNotFoundError
from argopy.utilities import list_standard_variables, load_dict, mapp_dict, check_localftp
from argopy.options import OPTIONS
from argopy.stores import filestore, indexstore, indexfilter_wmo, indexfilter_box

access_points = ['wmo', 'box']
exit_formats = ['xarray', 'dataframe']
dataset_ids = ['phy', 'bgc']  # First is default


class LocalFTPArgoIndexFetcher(ABC):
    """ Manage access to Argo index from a local copy of GDAC ftp

    """
    ###
    # Methods to be customised for a specific request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        pass

    @property
    def cachepath(self):
        return self.fs.cachepath(self.fcls.uri())

    def cname(self):
        """ Return a unique string defining the request

        """
        return self.fcls.uri()

    def filter_index(self):
        """ Search for profile in a box in the argo index file

        Parameters
        ----------
        index_file: _io.TextIOWrapper

        Returns
        -------
        csv rows matching the request, as a in-memory string. Or None.
        """
        return self.fcls

    def __init__(self,
                 local_ftp: str = "",
                 index_file: str = "ar_index_global_prof.txt",
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
        self.definition = 'Local ftp Argo index fetcher'
        self.local_ftp = OPTIONS['local_ftp'] if local_ftp == '' else local_ftp
        check_localftp(self.local_ftp, errors='raise')  # Validate local_ftp
        self.index_file = index_file
        self.fs = indexstore(cache, cachedir, os.path.sep.join([self.local_ftp, self.index_file]))
        self.dataset_id = 'index'
        self.init(**kwargs)

    def __repr__(self):
        summary = ["<indexfetcher '%s'>" % self.definition]
        summary.append("FTP: %s" % self.local_ftp)
        summary.append("Domain: %s" % self.cname())
        return '\n'.join(summary)

    def to_dataframe(self):
        """ filter local index file and return a pandas dataframe """
        df = self.fs.open_dataframe(self.filter_index())

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

    def clear_cache(self):
        """ Remove cache files and entries from resources open with this fetcher """
        return self.fs.clear_cache()


class Fetcher_wmo(LocalFTPArgoIndexFetcher):
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
            WMO = [WMO]  # Make sure we deal with a list
        if isinstance(CYC, int):
            CYC = np.array((CYC,), dtype='int')  # Make sure we deal with an array of integers
        if isinstance(CYC, list):
            CYC = np.array(CYC, dtype='int')  # Make sure we deal with an array of integers
        self.WMO = WMO
        self.CYC = CYC
        self.fcls = indexfilter_wmo(self.WMO, self.CYC)


class Fetcher_box(LocalFTPArgoIndexFetcher):
    """ Manage access to local ftp Argo data for: an ocean rectangle

    """
    def init(self, box: list = [-180, 180, -90, 90, '1900-01-01', '2100-12-31']):
        """ Create Argo index loader

            Parameters
            ----------
            box : list(float, float, float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        if len(box) != 4 and len(box) !=6 :
            raise ValueError('Box must be 4 or 6 length')
        self.BOX = box
        self.fcls = indexfilter_box(self.BOX)
