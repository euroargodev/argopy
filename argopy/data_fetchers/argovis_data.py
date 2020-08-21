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
import getpass
from .proto import ArgoDataFetcherProto
from abc import abstractmethod
import warnings
import distributed

from argopy.stores import httpstore
from argopy.options import OPTIONS
from argopy.utilities import list_standard_variables, format_oneline, is_box, Chunker
from argopy.errors import DataNotFound
from argopy.plotters import open_dashboard

access_points = ['wmo', 'box']
exit_formats = ['xarray']
dataset_ids = ['phy']  # First is default
api_server = 'https://argovis.colorado.edu'  # API root url
api_server_check = api_server + '/selection/overview'  # URL to check if the API is alive


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
    def uri(self):
        """ Return the URL used to download data """
        pass

    @property
    def cachepath(self):
        """ Return path to cache file for this request """
        if isinstance(self.uri, list):
            return [self.fs.cachepath(url) for url in self.uri]
        else:
            return self.fs.cachepath(self.uri)

    ###
    # Methods that must not change
    ###
    def __init__(self,
                 ds: str = "",
                 cache: bool = False,
                 cachedir: str = "",
                 parallel: bool = False,
                 chunks: str = 'auto',
                 chunks_maxsize: dict = {},
                 parallel_method: str = 'thread',
                 api_timeout: int = 0,
                 **kwargs):
        """ Instantiate an Argovis Argo data loader

            Parameters
            ----------
            ds: 'phy'
            cache : False
            cachedir : None
        """
        timeout = OPTIONS['api_timeout'] if api_timeout == 0 else api_timeout
        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=timeout)
        self.definition = 'Argovis Argo data fetcher'
        self.dataset_id = OPTIONS['dataset'] if ds == '' else ds
        self.server = api_server

        if not isinstance(parallel, bool):
            # The parallelisation method is passed through 'parallel':
            if type(parallel) == distributed.client.Client or parallel == 'thread' or parallel == 'process':
                parallel_method = parallel
        self.parallel = parallel
        self.parallel_method = parallel_method
        self.chunks = chunks
        self.chunks_maxsize = chunks_maxsize

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
        summary = ["<datafetcher.argovis>"]
        summary.append("Name: %s" % self.definition)
        summary.append("API: %s" % api_server)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return '\n'.join(summary)

    def _add_history(self, this, txt):
        if 'history' in this.attrs:
            this.attrs['history'] += "; %s" % txt
        else:
            this.attrs['history'] = txt
        return this

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

    def to_dataframe(self):
        """ Load Argo data and return a Pandas dataframe """

        # Download data:
        if not self.parallel:
            method = 'sequential'
        else:
            method = self.parallel_method
        df_list = self.fs.open_mfjson(self.uri,
                                    method=method,
                                    preprocess=self.json2dataframe,
                                    progress=1)

        # Merge results (list of dataframe):
        for i, df in enumerate(df_list):
            df = df.reset_index()
            df = df.rename(columns=self.key_map)
            df = df[[value for value in self.key_map.values() if value in df.columns]]
            df_list[i] = df
        df = pd.concat(df_list, ignore_index=True)
        df.sort_values(by=['TIME', 'PRES'], inplace=True)
        df = df.set_index(['N_POINTS'])
        return df

    def to_xarray(self):
        """ Download and return data as xarray Datasets """
        ds = self.to_dataframe().to_xarray()
        ds = ds.sortby(['TIME', 'PRES'])  # should already be sorted by date in decending order
        ds['N_POINTS'] = np.arange(0, len(ds['N_POINTS']))  # Re-index to avoid duplicate values

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

        # Remove argovis file attributes and replace them with argopy ones:
        ds.attrs = {}
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
        ds.attrs['Fetched_uri'] = self.uri
        ds = ds[np.sort(ds.data_vars)]
        return ds

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
    def uri(self):
        """ Return the list of URLs to download data

        Returns
        -------
        str or list(str)
        """
        def list_bunch(wmos, cycs):
            this = []
            for wmo in wmos:
                if cycs is None:
                    this.append(self.get_url(wmo))
                else:
                    this.append(self.get_url(wmo, cycs))
            return this
        return list_bunch(self.WMO, self.CYC)

    def get_url(self, wmo: int, cyc: int = None) -> str:
        """ Return the path toward the source file of a given wmo/cyc pair """
        if cyc is None:
            return (self.server + '/catalog/platforms/{}').format(str(wmo))
        else:
            profIds = [str(wmo) + '_' + str(c) for c in cyc.tolist()]
            return (self.server + '/catalog/mprofiles/?ids={}').format(profIds).replace(' ', '')

    def dashboard(self, **kw):
        if len(self.WMO) == 1:
            return open_dashboard(wmo=self.WMO[0], **kw)
        else:
            warnings.warn("Plot dashboard only available for one float frequest")


class Fetch_box(ArgovisDataFetcher):

    def init(self, box: list):
        """ Create Argo data loader

            Parameters
            ----------
            box : list(float, float, float, float, float, float, str, str)
                The box domain to load all Argo data for:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
                or:
                box = [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, datim_min, datim_max]
        """
        if len(box) == 6:
            # Select the last months of data:
            end = pd.to_datetime('now')
            start = end - pd.DateOffset(months=1)
            box.append(start.strftime('%Y-%m-%d'))
            box.append(end.strftime('%Y-%m-%d'))
        is_box(box)
        self.BOX = box

        self.definition = '?'
        if self.dataset_id == 'phy':
            self.definition = 'Argovis Argo data fetcher for a space/time region'
        return self

    def cname(self):
        """ Return a unique string defining the constraints """
        BOX = self.BOX
        boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; z=%0.1f/%0.1f; t=%s/%s]") % \
                      (BOX[0], BOX[1], BOX[2], BOX[3], BOX[4], BOX[5],
                       self._format(BOX[6], 'tim'), self._format(BOX[7], 'tim'))
        boxname = self.dataset_id + "_" + boxname
        return boxname

    def get_url(self):
        """ Return the URL used to download data """
        shape = [[[self.BOX[0], self.BOX[2]], [self.BOX[0], self.BOX[3]], [self.BOX[1], self.BOX[3]],
                  [self.BOX[1], self.BOX[2]], [self.BOX[0], self.BOX[2]]]]
        strShape = str(shape).replace(' ', '')
        url = self.server + '/selection/profiles'
        url += '?startDate={}'.format(self.BOX[6])
        url += '&endDate={}'.format(self.BOX[7])
        url += '&shape={}'.format(strShape)
        url += '&presRange=[{},{}]'.format(self.BOX[4], self.BOX[5])
        return url

    @property
    def uri(self):
        """ Return the list of URLs to download data

        Returns
        -------
        str or list(str)
        """
        if not self.parallel:
            # Check if the time range is not larger than allowed (90 days):
            Lt = np.timedelta64(pd.to_datetime(self.BOX[7]) - pd.to_datetime(self.BOX[6]), 'D')
            MaxLen = np.timedelta64(90, 'D')
            print(MaxLen, Lt)
            if Lt > MaxLen:
                C = Chunker({'box': self.BOX}, chunks={'lon': 1, 'lat': 1, 'dpt': 1, 'time': 'auto'},
                            chunksize={'time': 90})
                boxes = C.fit_transform()
                urls = []
                for box in boxes:
                    urls.append(Fetch_box(box=box, ds=self.dataset_id).get_url())
                return urls
            else:
                return self.get_url()
        else:
            C = Chunker({'box': self.BOX}, chunks=self.chunks, chunksize=self.chunks_maxsize)
            boxes = C.fit_transform()
            urls = []
            for box in boxes:
                urls.append(Fetch_box(box=box, ds=self.dataset_id).get_url())
            return urls
