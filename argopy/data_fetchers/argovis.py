#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for Argovis.
# Code borrows heavily from API gathered at:
# https://github.com/earthcube2020/ec20_tucker_etal/blob/master/EC2020_argovis_python_api.ipynb
#
# This is comprised of functions used to query Argovis api
# query functions either return dictionary objects or error messages.
#

import numpy as np
import pandas as pd
import xarray as xr
from .proto import ArgoDataFetcherProto
from abc import abstractmethod
from argopy.stores import httpstore
from argopy.options import OPTIONS
import json
import getpass
from argopy.utilities import list_standard_variables

access_points = ['wmo']
exit_formats = ['xarray']
dataset_ids = ['phy']  # First is default


class ArgovisDataFetcher(ArgoDataFetcherProto):
    ###
    # Methods to be customised for a specific Argovis request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        pass

    @abstractmethod
    def cname(self):
        """ Return a unique string defining the request

            Provide this string to populate meta data and titles
        """
        pass

    @property
    def url(self):
        """ Return the URL used to download data """
        pass

    ###
    # Methods that must not change
    ###
    def __init__(self,
                 ds: str = "",
                 cache: bool = False,
                 cachedir: str = "",
                 **kwargs):
        """ Instantiate an Argovis Argo data loader

            Parameters
            ----------
            ds: 'phy'
            cache : False
            cachedir : None
        """
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=120)
        self.definition = 'Argovis Argo data fetcher'
        self.dataset_id = OPTIONS['dataset'] if ds == '' else ds
        self.server = 'https://argovis.colorado.edu'
        self.init(**kwargs)
        self.key_map = {
            'date': 'TIME',
            'date_qc': 'TIME_QC',
            'lat': 'LATITUDE',
            'lon': 'LONGITUDE',
            'cycle_number': 'CYCLE_NUMBER',
            'DATA_MODE': 'DATA_MODE',
            'DIRECTION': 'DIRECTION',
            'platform_number': 'PLATFORM_NUMBER',
            'position_qc': 'POSITION_QC',
            'pres': 'PRES',
            'temp': 'TEMP',
            'psal': 'PSAL',
            'index': 'N_POINTS'
        }

    def __repr__(self):
        summary = ["<datafetcher '%s'>" % self.definition]
        summary.append("Domain: %s" % self.cname())
        return '\n'.join(summary)

    def json2dataframe(self, profiles):
        """ convert json data to Pandas DataFrame """
        # Make sure we deal with a list
        if isinstance(profiles, list):
            data = profiles
        else:
            data = [profiles]
        # Transform
        rows = []
        for profile in data:
            keys = [x for x in profile.keys() if x not in ['measurements', 'bgcMeas']]
            meta_row = dict((key, profile[key]) for key in keys)
            for row in profile['measurements']:
                row.update(meta_row)
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def filter_data_mode(self, ds, **kwargs):
        # Argovis data already curated !
        # ds = ds.argo.filter_data_mode(errors='ignore', **kwargs)
        if ds.argo._type == 'point':
            ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))
        return ds

    def filter_qc(self, ds, **kwargs):
        # Argovis data already curated !
        # ds = ds.argo.filter_qc(**kwargs)
        if ds.argo._type == 'point':
            ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))
        return ds

    def filter_variables(self, ds, mode='standard'):
        if mode == 'standard':
            to_remove = sorted(list(set(list(ds.data_vars)) - set(list_standard_variables())))
            return ds.drop_vars(to_remove)
        else:
            return ds


class Fetch_wmo(ArgovisDataFetcher):
    def init(self, WMO=[], CYC=None):
        """ Create Argo data loader for WMOs and CYCs

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

        self.definition = "?"
        if self.dataset_id == 'phy':
            self.definition = 'Argovis Argo data fetcher for floats'
        return self

    def cname(self):
        """ Return a unique string defining the constraints """
        if len(self.WMO) > 1:
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

    @property
    def url(self):
        """ Return the URL used to download data """
        urls = []
        if isinstance(self.CYC, (np.ndarray)) and self.CYC.nbytes > 0:
            profIds = [str(wmo) + '_' + str(cyc) for wmo in self.WMO for cyc in self.CYC.tolist()]
            urls.append((self.server + '/catalog/mprofiles/?ids={}').format(profIds).replace(' ', ''))
        # elif self.dataset_id == 'bgc' and isinstance(self.CYC, (np.ndarray)) and self.CYC.nbytes > 0:
        #     profIds = [str(wmo) + '_' + str(cyc) for wmo in self.WMO for cyc in self.CYC.tolist()]
        #     urls.append((self.server + '/catalog/profiles/{}').format(self.CYC))
        else:
            for wmo in self.WMO:
                urls.append((self.server + '/catalog/platforms/{}').format(str(wmo)))
        return urls

    def to_dataframe(self, js):
        df = self.json2dataframe(js)
        df = df.reset_index()
        df = df.rename(columns=self.key_map)
        df = df[[value for value in self.key_map.values()]]
        df = df.set_index(['N_POINTS'])
        return df

    def to_xarray(self):
        """ Download and return data as xarray Datasets """
        results = []
        for url in self.url:
            js = self.fs.open_json(url)
            if isinstance(js, str):
                continue
            ds = self.to_dataframe(js).to_xarray()
            results.append(ds)
        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            ds = xr.concat(results, dim='N_POINTS', data_vars='all', coords='all', compat='override')
            ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))  # Re-index to avoid duplicate values
            ds = ds.sortby('TIME')  # should already be sorted by date in decending order

            # Set coordinates:
            # ds = ds.set_coords('N_POINTS')
            coords = ('LATITUDE', 'LONGITUDE', 'TIME', 'N_POINTS')
            ds = ds.reset_coords()
            ds['N_POINTS'] = ds['N_POINTS']
            # Convert all coordinate variable names to upper case
            for v in ds.data_vars:
                ds = ds.rename({v: v.upper()})
            ds = ds.set_coords(coords)

            # Cast data types and add variable attributes (not available in the csv download):
            ds = ds.argo.cast_types()

            # Add useful attributes to the dataset:
            if self.dataset_id == 'phy':
                ds.attrs['DATA_ID'] = 'ARGO'
            elif self.dataset_id == 'ref':
                ds.attrs['DATA_ID'] = 'ARGO_Reference'
            elif self.dataset_id == 'bgc':
                ds.attrs['DATA_ID'] = 'ARGO-BGC'
            ds.attrs['DOI'] = 'http://doi.org/10.17882/42182'
            ds.attrs['Fetched_from'] = self.server
            ds.attrs['Fetched_by'] = getpass.getuser()
            ds.attrs['Fetched_date'] = pd.to_datetime('now').strftime('%Y/%m/%d')
            ds.attrs['Fetched_constraints'] = self.cname()
            ds.attrs['Fetched_uri'] = self.url
            ds = ds[np.sort(ds.data_vars)]
            return ds
        else:
            raise ValueError("CAN'T FETCH ANY DATA !")
