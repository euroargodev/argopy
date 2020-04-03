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

from argopy.options import OPTIONS, _VALIDATORS
from .errors import InvalidFetcherAccessPoint
from .utilities import list_available_data_backends

AVAILABLE_BACKENDS = list_available_data_backends()

# Highest level API / Facade:
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
                 mode: str = "",
                 backend : str = "",
                 ds: str = "",
                 **fetcher_kwargs):

        # Facade options:
        self._mode = OPTIONS['mode'] if mode == '' else mode
        self._dataset_id = OPTIONS['dataset'] if ds == '' else ds
        self._backend = OPTIONS['datasrc'] if backend == '' else backend

        _VALIDATORS['mode'](self._mode)
        _VALIDATORS['datasrc'](self._backend)
        _VALIDATORS['dataset'](self._dataset_id)

        # Load backend access points:
        if self._backend not in AVAILABLE_BACKENDS:
            raise ValueError("Data fetcher '%s' not available" % self._backend)
        else:
            Fetchers = AVAILABLE_BACKENDS[self._backend]

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

        # Dev warnings
        #Todo Clean-up before each release
        if self._dataset_id == 'bgc' and self._mode == 'standard':
            warnings.warn(" 'BGC' dataset fetching in 'standard' user mode is not reliable. "
                          "Try to switch to 'expert' mode if you encounter errors.")

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

    def float(self, wmo, **kw):
        """ Load data from a float, given one or more WMOs """
        if "CYC" in kw or "cyc" in kw:
            raise TypeError("float() got an unexpected keyword argument 'cyc'. Use 'profile' access "
                            "point to fetch specific profile data.")

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
        warnings.warn("This access point is to be used with an index fetcher")
        pass

    def to_xarray(self, **kwargs):
        """ Fetch and post-process data, return xarray.DataSet """
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)
        return xds

