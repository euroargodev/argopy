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

# Import data fetchers:
available_backends = list()
try:
    from .data_fetchers import erddap as Erddap_Fetcher
    available_backends.append('erddap')
except:
    e = sys.exc_info()[0]
    warnings.warn("An error occured while loading the ERDDAP data fetcher, it will not be available !\n%s" % e)
    pass

try:
    from .data_fetchers import localftp as LocalFTP_Fetcher
    available_backends.append('localftp')
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
    #todo use dynamic loading of all available data fetcher and their access points

    def __init__(self,
                 mode='standard',
                 backend=OPTIONS['data_src'],
                 ds=OPTIONS['dataset'],
                 **fetcher_kwargs):

        if mode not in ['standard', 'expert']:
            raise ValueError("Mode must be 'standard' or 'expert'")

        # Facade options:
        self.mode = mode # User mode determining the level of post-processing required
        self.dataset_id = ds # Database to use
        self.backend = backend # data_fetchers to use

        # Load backend access points:
        if backend not in available_backends:
            raise ValueError("The %s data fetcher is not available" % backend)

        if backend == 'erddap' and backend in available_backends:
            self.Fetchers = Erddap_Fetcher

        if backend == 'localftp' and backend in available_backends:
            self.Fetchers = LocalFTP_Fetcher

        # Auto-discovery of access points for this fetcher:
        self.access_points = []
        for p in self.Fetchers.access_points:
            if p == 'wmo':  # Required for 'profile' and 'float'
                self.Fetcher_wmo = self.Fetchers.ArgoDataFetcher_wmo
                self.access_points.append('profile')
                self.access_points.append('float')
            if p == 'box':  # Required for 'region'
                self.Fetcher_box = self.Fetchers.ArgoDataFetcher_box
                self.access_points.append('region')

        # Init sub-methods:
        self.fetcher = None
        if ds is None:
            ds = self.Fetchers.dataset_ids[0]
        self.fetcher_options = {**{'ds':ds}, **fetcher_kwargs}
        self.postproccessor = self.__empty_processor

    def __repr__(self):
        if self.fetcher:
            summary = [self.fetcher.__repr__()]
            summary.append("Backend: %s" % self.backend)
            summary.append("User mode: %s" % self.mode)
        else:
            summary = ["<datafetcher 'Not initialised'>"]
            summary.append("Backend: %s" % self.backend)
            summary.append("Fetchers: %s" % ", ".join(self.access_points))
            summary.append("User mode: %s" % self.mode)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """ Do nothing to a dataset """
        return xds

    def float(self, wmo):
        """ Load data from a float, given one or more WMOs """
        self.fetcher = self.Fetcher_wmo(WMO=wmo, **self.fetcher_options)

        if self.mode == 'standard' and self.dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self.mode)
                return xds
            self.postproccessor = postprocessing
        return self

    def profile(self, wmo, cyc):
        """ Load data from a profile, given one or more WMOs and CYCLE_NUMBER """
        self.fetcher = self.Fetcher_wmo(WMO=wmo, CYC=cyc, **self.fetcher_options)

        if self.mode == 'standard' and self.dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self.mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def region(self, box):
        """ Load data from a rectangular region, given latitude, longitude, pressure and possibly time bounds """
        self.fetcher = self.Fetcher_box(box=box, **self.fetcher_options)

        if self.mode == 'standard' and self.dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self.mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def deployments(self, box):
        """ Retrieve deployment locations in a specific space/time region """
        self.fetcher = ErddapArgoDataFetcher_box_deployments(box=box, **self.fetcher_options)

        if self.mode == 'standard' and self.dataset_id != 'ref':
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self.mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def to_xarray(self, **kwargs):
        """ Fetch and post-process data, return xarray.DataSet """
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)
        return xds

