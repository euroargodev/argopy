# -*- coding: utf-8 -*-

"""
argopy.data_fetchers.erddap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains Argo meta-data fetcher for Ifremer ERDDAP.

This is not intended to be used directly, only by the facade at fetchers.py

"""

import pandas as pd
import numpy as np
import copy

from abc import ABC, abstractmethod
import getpass

from .proto import ArgoDataFetcherProto
from argopy.utilities import load_dict, mapp_dict
from argopy.options import OPTIONS
from argopy.utilities import list_standard_variables
from argopy.stores import httpstore

from erddapy import ERDDAP
from erddapy.utilities import parse_dates, quote_string_constraints


access_points = ['wmo', 'box']
exit_formats = ['xarray', 'dataframe']
dataset_ids = ['phy', 'ref', 'bgc']  # First is default
api_server = 'https://www.ifremer.fr/erddap'  # API root url
api_server_check = api_server + '/info/ArgoFloats/index.json'  # URL to check if the API is alive


class ErddapArgoIndexFetcher(ABC):
    """ Manage access to Argo index through Ifremer ERDDAP

        ERDDAP transaction are managed with the erddapy library

        __author__: kevin.balem@ifremer.fr
    """

    ###
    # Methods to be customised for a specific erddap request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        pass

    @abstractmethod
    def define_constraints(self):
        """ Define erddapy constraints """
        pass

    @abstractmethod
    def cname(self):
        """ Return a unique string defining the request """
        pass

    ###
    # Methods that must not changed
    ###
    def __init__(self,
                 cache: bool = False,
                 cachedir: str = "",
                 **kwargs):
        """ Instantiate an ERDDAP Argo index loader with force caching """

        self.fs = httpstore(cache=cache, cachedir=cachedir, timeout=120)
        self.definition = 'Ifremer erddap Argo index fetcher'
        self.dataset_id = 'index'
        self.server = api_server
        self.init(**kwargs)
        self._init_erddapy()

    def __repr__(self):
        if hasattr(self, '_definition'):
            summary = ["<indexfetcher '%s'>" % self.definition]
        else:
            summary = ["<indexfetcher '%s'>" % 'Ifremer erddap Argo Index fetcher']
        summary.append("Domain: %s" % self.cname())
        return '\n'.join(summary)

    def _format(self, x, typ):
        """ string formating helper """
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

    def _init_erddapy(self):
        # Init erddapy
        self.erddap = ERDDAP(
            server=self.server,
            protocol='tabledap'
        )
        self.erddap.response = 'csv'
        self.erddap.dataset_id = 'ArgoFloats-index'
        return self

    @property
    def cachepath(self):
        """ Return path to cache file for this request """
        return self.fs.cachepath(self.url)

    @property
    def url(self, response=None):
        """ Return the URL used to download data

        """
        # Replace erddapy get_download_url
        # We need to replace it to better handle http responses with by-passing the _check_url_response
        # https://github.com/ioos/erddapy/blob/fa1f2c15304938cd0aa132946c22b0427fd61c81/erddapy/erddapy.py#L247

        # Define constraint to select this box of data:
        self.define_constraints()  # This will affect self.erddap.constraints

        # Define the list of variables to retrieve - all for the index
        self.erddap.variables = ['file', 'date', 'longitude', 'latitude',
                                 'ocean', 'profiler_type', 'institution', 'date_update']

        #
        dataset_id = self.erddap.dataset_id
        protocol = self.erddap.protocol
        variables = self.erddap.variables
        if not response:
            response = self.erddap.response
        constraints = self.erddap.constraints
        url = f"{self.erddap.server}/{protocol}/{dataset_id}.{response}?"
        if variables:
            variables = ",".join(variables)
            url += f"{variables}"

        if constraints:
            _constraints = copy.copy(constraints)
            for k, v in _constraints.items():
                if k.startswith("time"):
                    _constraints.update({k: parse_dates(v)})
            _constraints = quote_string_constraints(_constraints)
            _constraints = "".join([f"&{k}{v}" for k, v in _constraints.items()])

            url += f"{_constraints}"

        url += '&distinct()&orderBy("date")'
        # In erddapy:
        # url = _distinct(url, **kwargs)
        # return _check_url_response(url, **self.requests_kwargs)
        return url

    def to_dataframe(self):
        """ Load Argo index and return a pandas dataframe """

        # Download data: get a csv, open it as pandas dataframe, create wmo field
        df = self.fs.open_dataframe(self.url, parse_dates=True, skiprows=[1])

        # erddap date format : 2019-03-21T00:00:35Z
        df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%dT%H:%M:%SZ")
        df['date_update'] = pd.to_datetime(df['date_update'], format="%Y-%m-%dT%H:%M:%SZ")
        df['wmo'] = df.file.apply(lambda x: int(x.split('/')[1]))

        # institution & profiler mapping
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


class Fetcher_wmo(ErddapArgoIndexFetcher):
    """ Manage access to Argo Index through Ifremer ERDDAP for: a list of WMOs

    """

    def init(self, WMO=[]):
        """ Create Argo data loader for WMOs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to load all Argo data for.
        """
        if isinstance(WMO, int):
            WMO = [WMO]  # Make sure we deal with a list
        self.WMO = WMO
        self.definition = 'Ifremer erddap Argo Index fetcher for floats'
        return self

    def define_constraints(self):
        """ Define erddap constraints """
        #  'file=~': "(.*)(R|D)(6902746_|6902747_)(.*)"
        self.erddap.constraints = {'file=~': "(.*)(R|D)("+"|".join(["%i" % i for i in self.WMO])+")(_.*)"}
        return self

    def cname(self):
        """ Return a unique string defining the constraints """
        if len(self.WMO) > 1:
            listname = ["WMO%i" % i for i in self.WMO]
            listname = ";".join(listname)
        else:
            listname = "WMO%i" % self.WMO[0]
        return listname


class Fetcher_box(ErddapArgoIndexFetcher):
    """ Manage access to Argo Index through Ifremer ERDDAP for: an ocean rectangle

        __author__: kevin.balem@ifremer.fr
    """

    def init(self, box=[]):
        """ Create Argo Index loader

            Parameters
            ----------
            box : list(float, float, float, float, str, str)
            The box domain to load all Argo data for:
            box = [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max]
        """
        if len(box) == 4:
            # Use all time line:
            box.append('1900-01-01')
            box.append('2100-12-31')
        elif len(box) != 6:
            raise ValueError('Box must have 4 or 6 elements : [lon_min, lon_max, lat_min, lat_max, datim_min, datim_max] ')
        self.BOX = box
        self.definition = 'Ifremer erddap Argo Index fetcher for a space/time region'

        return self

    def define_constraints(self):
        """ Define request constraints """
        self.erddap.constraints = {'longitude>=': self.BOX[0]}
        self.erddap.constraints.update({'longitude<=': self.BOX[1]})
        self.erddap.constraints.update({'latitude>=': self.BOX[2]})
        self.erddap.constraints.update({'latitude<=': self.BOX[3]})
        self.erddap.constraints.update({'date>=': self.BOX[4]})
        self.erddap.constraints.update({'date<=': self.BOX[5]})
        return None

    def cname(self):
        """ Return a unique string defining the constraints """
        BOX = self.BOX
        boxname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; t=%s/%s]") % \
                  (BOX[0], BOX[1], BOX[2], BOX[3], self._format(BOX[4], 'tim'), self._format(BOX[5], 'tim'))
        return boxname
