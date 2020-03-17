#!/bin/env python
# -*coding: UTF-8 -*-
"""

High level helper methods to load Argo data from any source (but only Ifremer erddap server at this point).
The facade should be able to work with another data access point,
like another webAPI or a local copy of the ftp.

Usage:

    from argopy import DataFetcher as ArgoDataFetcher

    argo_loader = ArgoDataFetcher()
or
    argo_loader = ArgoDataFetcher(cachedir='tmp', cache=True)
or
    argo_loader = ArgoDataFetcher(ds='ref')

    argo_loader.profile(6902746, 34).to_xarray()
    argo_loader.profile(6902746, np.arange(12,45)).to_xarray()
    argo_loader.profile(6902746, [1,12]).to_xarray()

    argo_loader.float(6902746).to_xarray()
    argo_loader.float([6902746, 6902747, 6902757, 6902766]).to_xarray()
    argo_loader.float([6902746, 6902747, 6902757, 6902766], CYC=1).to_xarray()

    argo_loader.region([-85,-45,10.,20.,0,1000.]).to_xarray()
    argo_loader.region([-85,-45,10.,20.,0,1000.,'2012-01','2014-12']).to_xarray()

Created by gmaze on 20/12/2019
"""

import os
import sys
import glob
import pandas as pd
import xarray as xr
import numpy as np

# Import data data_fetchers:
from .data_fetchers import erddap as Erddap_Fetcher

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

    def __init__(self, mode='standard', backend='erddap', ds='phy', **fetcher_kwargs):

        if mode not in ['standard', 'expert']:
            raise ValueError("Mode must be 'standard' or 'expert'")

        # Init sub-methods:
        self.fetcher = None
        self.fetcher_options = {**{'ds':ds}, **fetcher_kwargs}
        self.postproccessor = self.__empty_processor

        # Facade options:
        self.mode = mode # User mode determining the level of post-processing required
        self.dataset_id = ds # Database to use
        self.backend = backend # data_fetchers to use
        if self.backend != 'erddap':
            raise ValueError("Invalid backend, only 'erddap' available at this point")

        # Load backend access points:
        if backend == 'erddap':
            self.Fetcher_wmo = Erddap_Fetcher.ArgoDataFetcher_wmo
            self.Fetcher_box = Erddap_Fetcher.ArgoDataFetcher_box

    def __repr__(self):
        if self.fetcher:
            summary = [self.fetcher.__repr__()]
            summary.append("User mode: %s" % self.mode)
        else:
            summary = ["<datafetcher 'Not initialised'>"]
            summary.append("Fetchers: 'profile', 'float' or 'region'")
            summary.append("User mode: %s" % self.mode)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """ Do nothing to a dataset """
        return xds

    def __drop_vars(self, xds):
        """ Drop Jargon variables for standard users """
        drop_list = ['DATA_MODE', 'DIRECTION']
        xds = xds.drop_vars(drop_list)
        # Also drop all QC variables:
        for v in xds.data_vars:
            if "QC" in v:
                xds = xds.drop_vars(v)
        return xds

    def float(self, wmo):
        """ Load data from a float, given one or more WMOs """
        self.fetcher = self.Fetcher_wmo(WMO=wmo, **self.fetcher_options)

        if self.mode == 'standard' and (self.dataset_id == 'phy' or self.dataset_id == 'bgc'):
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.__drop_vars(xds)
                return xds
            self.postproccessor = postprocessing
        return self

    def profile(self, wmo, cyc):
        """ Load data from a profile, given one ormore WMOs and CYCLE_NUMBER """
        self.fetcher = self.Fetcher_wmo(WMO=wmo, CYC=cyc, **self.fetcher_options)

        if self.mode == 'standard' and (self.dataset_id == 'phy' or self.dataset_id == 'bgc'):
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.__drop_vars(xds)
                return xds
            self.postproccessor = postprocessing

        return self

    def region(self, box):
        """ Load data for a rectangular region, given latitude, longitude, pressure and possibly time bounds """
        self.fetcher = self.Fetcher_box(box=box, **self.fetcher_options)

        if self.mode == 'standard' and (self.dataset_id == 'phy' or self.dataset_id == 'bgc'):
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.__drop_vars(xds)
                return xds
            self.postproccessor = postprocessing

        return self

    def deployments(self, box):
        """ Retrieve deployment locations in a specific space/time region """
        self.fetcher = ErddapArgoDataFetcher_box_deployments(box=box, **self.fetcher_options)

        if self.mode == 'standard' and (self.dataset_id == 'phy' or self.dataset_id == 'bgc'):
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.__drop_vars(xds)
                return xds
            self.postproccessor = postprocessing

        return self

    def to_xarray(self, **kwargs):
        """ Fetch and post-process data, return xarray.DataSet """
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)
        return xds

