#!/bin/env python
# -*coding: UTF-8 -*-
"""

High level helper methods to load Argo data from any source
The facade should be able to work with all available data access point,

Usage for LOCALFTP:

    from argopy import DataFetcher as ArgoDataFetcher

    argo_loader = ArgoDataFetcher(backend='localftp', ds='phy')
or
    argo_loader = ArgoDataFetcher(backend='localftp', ds='bgc')

    argo_loader.float(6902746).to_xarray()
    argo_loader.float([6902746, 6902747, 6902757, 6902766]).to_xarray()


Usage for ERDDAP (default backend):

    from argopy import DataFetcher as ArgoDataFetcher

    argo_loader = ArgoDataFetcher(backend='erddap')
or
    argo_loader = ArgoDataFetcher(backend='erddap', cachedir='tmp', cache=True)
or
    argo_loader = ArgoDataFetcher(backend='erddap', ds='ref')

    argo_loader.profile(6902746, 34).to_xarray()
    argo_loader.profile(6902746, np.arange(12,45)).to_xarray()
    argo_loader.profile(6902746, [1,12]).to_xarray()
or
    argo_loader.float(6902746).to_xarray()
    argo_loader.float([6902746, 6902747, 6902757, 6902766]).to_xarray()
    argo_loader.float([6902746, 6902747, 6902757, 6902766], CYC=1).to_xarray()
or
    argo_loader.region([-85,-45,10.,20.,0,1000.]).to_xarray()
    argo_loader.region([-85,-45,10.,20.,0,1000.,'2012-01','2014-12']).to_xarray()

"""

import os
import sys
import glob
import pandas as pd
import xarray as xr
import numpy as np
import warnings

from argopy.options import OPTIONS
from .errors import InvalidFetcherAccessPoint

# Import data fetchers:
AVAILABLE_BACKENDS = list()
try:
    from .data_fetchers import erddap as Erddap_Fetchers
    AVAILABLE_BACKENDS.append('erddap')
except:
    e = sys.exc_info()[0]
    warnings.warn("An error occured while loading the ERDDAP data fetcher, it will not be available !\n%s" % e)
    pass

try:
    from .data_fetchers import localftp as LocalFTP_Fetchers
    AVAILABLE_BACKENDS.append('localftp')
except:
    e = sys.exc_info()[0]
    warnings.warn("An error occured while loading the local FTP data fetcher, it will not be available !\n%s" % e)
    pass


def backends_check(Cls):
    # warnings.warn( "Fetchers available: %s" % available_backends )
    return Cls

# Highest level API / Facade:
@backends_check
class ArgoDataFetcher(object):
    """ Fetch and process Argo data.

        Can return data selected from:
        - one or more float(s), defined by WMOs
        - one or more profile(s), defined for one WMO and one or more CYCLE NUMBER
        - a space/time rectangular domain, defined by lat/lon/pres/time range

        Can return data from the regular Argo dataset ('phy': temperature, salinity) and the Argo referenced
        dataset used in DMQC ('ref': temperature, salinity).

        This is the main API facade.
        Specify here all options to data_fetchers

    """

    def __init__(self,
                 mode='standard',
                 backend : str = "",
                 ds: str = "",
                 **fetcher_kwargs):

        if mode not in ['standard', 'expert']:
            raise ValueError("Mode must be 'standard' or 'expert'")

        # Facade options:
        self._mode = mode # User mode determining the level of post-processing required
        self._dataset_id = OPTIONS['dataset'] if ds == '' else ds
        self._backend = OPTIONS['data_src'] if backend == '' else backend

        # Load backend access points:
        if self._backend not in AVAILABLE_BACKENDS:
            raise ValueError("Data fetcher '%s' not available" % backend)

        if self._backend == 'erddap' and self._backend in AVAILABLE_BACKENDS:
            Fetchers = Erddap_Fetchers

        if self._backend == 'localftp' and self._backend in AVAILABLE_BACKENDS:
            Fetchers = LocalFTP_Fetchers

        # Auto-discovery of access points for this fetcher:
        # rq: Access point names for the facade are not the same as the access point of fetchers
        self.valid_access_points = ['profile', 'float', 'region']
        self.Fetchers = {}
        for p in Fetchers.access_points:
            if p == 'wmo':  # Required for 'profile' and 'float'
                self.Fetchers['profile'] = Fetchers.Fetch_wmo
                self.Fetchers['float'] = Fetchers.Fetch_wmo
            if p == 'box':  # Required for 'region'
                self.Fetchers['region'] = Fetchers.Fetch_box

        # Init sub-methods:
        self.fetcher = None
        if ds is None:
            ds = Fetchers.dataset_ids[0]
        self.fetcher_options = {**{'ds':ds}, **fetcher_kwargs}
        self.postproccessor = self.__empty_processor

    def __repr__(self):
        if self.fetcher:
            summary = [self.fetcher.__repr__()]
            summary.append("Backend: %s" % self._backend)
            summary.append("User mode: %s" % self._mode)
        else:
            summary = ["<datafetcher 'Not initialised'>"]
            summary.append("Backend: %s" % self._backend)
            summary.append("Fetchers: %s" % ", ".join(self.Fetchers.keys()))
            summary.append("User mode: %s" % self._mode)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """ Do nothing to a dataset """
        return xds

    def __getattr__(self, key):
        """ Validate access points """
        # print("key", key)
        valid_attrs = ['Fetchers', 'fetcher', 'fetcher_options', 'postproccessor']
        if key not in self.valid_access_points and key not in valid_attrs:
            raise InvalidFetcherAccessPoint("'%s' is not a valid access point" % key)
        pass

    def float(self, wmo):
        """ Load data from a float, given one or more WMOs """
        if 'float' in self.Fetchers:
            self.fetcher = self.Fetchers['float'](WMO=wmo, **self.fetcher_options)
        else:
            raise InvalidFetcherAccessPoint("'float' not available with '%s' backend" % self._backend)

        if self._mode == 'standard' and self._dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds
            self.postproccessor = postprocessing
        return self

    def profile(self, wmo, cyc):
        """ Load data from a profile, given one or more WMOs and CYCLE_NUMBER """
        if 'profile' in self.Fetchers:
            self.fetcher = self.Fetchers['profile'](WMO=wmo, CYC=cyc, **self.fetcher_options)
        else:
            raise InvalidFetcherAccessPoint("'profile' not available with '%s' backend" % self._backend)

        if self._mode == 'standard' and self._dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def region(self, box):
        """ Load data from a rectangular region, given latitude, longitude, pressure and possibly time bounds """
        if 'region' in self.Fetchers:
            self.fetcher = self.Fetchers['region'](box=box, **self.fetcher_options)
        else:
            raise InvalidFetcherAccessPoint("'region' not available with '%s' backend" % self._backend)

        if self._mode == 'standard' and self._dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def deployments(self, box):
        """ Retrieve deployment locations in a specific space/time region """
        self.fetcher = ErddapArgoDataFetcher_box_deployments(box=box, **self.fetcher_options)

        if self._mode == 'standard' and self._dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def to_xarray(self, **kwargs):
        """ Fetch and post-process data, return xarray.DataSet """
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)
        return xds

