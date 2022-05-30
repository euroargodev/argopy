#!/bin/env python
# -*coding: UTF-8 -*-
"""

High level helper methods to load Argo data from any source
The facade should be able to work with all available data access point,

Validity of access points parameters (eg: wmo) is made here, not at the data/index source fetcher level

"""

import warnings
import xarray as xr
import pandas as pd
import numpy as np
import logging

from argopy.options import OPTIONS, _VALIDATORS
from .errors import InvalidFetcherAccessPoint, InvalidFetcher

from .utilities import (
    list_available_data_src, list_available_index_src,
    is_box, is_indexbox,
    check_wmo, check_cyc,
    get_coriolis_profile_id
)
from argopy.stores import filestore
from .plot import plot_trajectory, bar_plot, open_sat_altim_report


AVAILABLE_DATA_SOURCES = list_available_data_src()
AVAILABLE_INDEX_SOURCES = list_available_index_src()

log = logging.getLogger("argopy.fetchers.facade")


def checkAccessPoint(AccessPoint):
    """Decorator to validate fetcher access points of a given data source.

    This decorator will check if an access point (eg: 'profile') is available for the
    data source (eg: 'erddap') used to initiate the checker. If not, an error is raised.

    #todo Make sure this decorator preserves the doc string !
    """
    def wrapper(*args):
        if AccessPoint.__name__ not in args[0].valid_access_points:
            raise InvalidFetcherAccessPoint(
                            "'%s' not available with '%s' src. Available access point(s): %s" %
                            (AccessPoint.__name__, args[0]._src, ", ".join(args[0].Fetchers.keys()))
                        )
        return AccessPoint(*args)
    wrapper.__name__ = AccessPoint.__name__
    wrapper.__doc__ = AccessPoint.__doc__
    return wrapper


class ArgoDataFetcher:
    """ Fetcher and post-processor of Argo data (API facade)

    Parameters
    ----------
    mode: str, optional
        User mode. Eg: ``standard`` or ``expert``. Set to OPTIONS['mode'] by default if empty.
    src: str, optional
         Source of the data to use. Eg: ``erddap``. Set to OPTIONS['src'] by default if empty.
    ds: str, optional
        Name of the dataset to load. Eg: ``phy``. Set to OPTIONS['dataset'] by default if empty.
    **fetcher_kwargs: optional
        Additional arguments passed on data source fetcher creation of each access points.

    Examples
    --------
    >>> from argopy import DataFetcher
    >>> adf = DataFetcher().region([-75, -65, 10, 20]).load()
    >>> idx.plot()
    >>> idx.data

    """

    def __init__(self, mode: str = "", src: str = "", ds: str = "", **fetcher_kwargs):

        """ Create a fetcher instance


        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher`
        """

        # Facade options:
        self._mode = OPTIONS["mode"] if mode == "" else mode
        self._dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self._src = OPTIONS["src"] if src == "" else src

        _VALIDATORS["mode"](self._mode)
        _VALIDATORS["src"](self._src)
        _VALIDATORS["dataset"](self._dataset_id)

        # Load data source access points:
        if self._src not in AVAILABLE_DATA_SOURCES:
            raise InvalidFetcher(
                "Requested data fetcher '%s' not available ! Please try again with any of: %s"
                % (self._src, "\n".join(AVAILABLE_DATA_SOURCES))
            )
        else:
            Fetchers = AVAILABLE_DATA_SOURCES[self._src]

        # Auto-discovery of access points for this fetcher:
        # rq: Access point names for the facade are not the same as the access point of fetchers
        self.Fetchers = {}
        self.valid_access_points = []
        for p in Fetchers.access_points:
            if p == "box":  # Required for 'region'
                self.Fetchers["region"] = Fetchers.Fetch_box
                self.valid_access_points.append("region")
            if p == "wmo":  # Required for 'profile' and 'float'
                self.Fetchers["float"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("float")
                self.Fetchers["profile"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("profile")

        # Init sub-methods:
        self.fetcher = None
        if self._dataset_id not in Fetchers.dataset_ids:
            raise ValueError(
                "%s dataset is not available for this data source (%s)"
                % (self._dataset_id, self._src)
            )
        self.fetcher_kwargs = {**fetcher_kwargs}
        self.fetcher_options = {**{"ds": self._dataset_id}, **fetcher_kwargs}
        self.postproccessor = self.__empty_processor
        self._AccessPoint = None

        # Init data structure holders:
        self._index = None
        self._data = None

        # Init file system for local storage
        # self.cache = True if 'cache' not in fetcher_kwargs else fetcher_kwargs['cache']
        # self.cachedir = OPTIONS['cachedir'] if 'cachedir' not in fetcher_kwargs else fetcher_kwargs['cachedir']
        # self.fs = filestore(cache=self.cache, cachedir=self.cachedir)

        # More init:
        self._loaded = False
        self._request = ""

        # Dev warnings
        # Todo Clean-up before each release
        if self._dataset_id == "bgc" and self._mode == "standard":
            warnings.warn(
                "'BGC' dataset fetching in 'standard' user mode is not yet reliable. "
                "Try to switch to 'expert' mode if you encounter errors."
            )

    def __repr__(self):

        para = self.fetcher_options['parallel'] if "parallel" in self.fetcher_options else False
        cach = self.fetcher_options['cache'] if "cache" in self.fetcher_options else False

        if self.fetcher:
            summary = [self.fetcher.__repr__()]
        else:
            summary = ["<datafetcher.%s> 'No access point initialised'" % self._src]
            summary.append("Available access points: %s" % ", ".join(self.Fetchers.keys()))

        summary.append("Performances: cache=%s, parallel=%s" % (str(cach), str(para)))
        summary.append("User mode: %s" % self._mode)
        summary.append("Dataset: %s" % self._dataset_id)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """ Do nothing to a dataset """
        return xds

    def __getattr__(self, key):
        """ Validate access points """
        valid_attrs = [
            "Fetchers",
            "fetcher",
            "fetcher_options",
            "postproccessor",
            "data",
            "index",
            "domain",
            "_loaded",
            "_request",
            "cache", "cachedir"
        ]
        if key not in self.valid_access_points and key not in valid_attrs:
            raise InvalidFetcherAccessPoint("'%s' is not a valid access point" % key)
        pass

    # def _write(self, path, obj, format='zarr'):
    #     """ Write internal array object to file store
    #
    #         Parameters
    #         ----------
    #         obj: :class:`xarray.DataSet` or :class:`pandas.DataFrame`
    #     """
    #     with self.fs.open(path, "wb") as handle:
    #         if format in ['zarr']:
    #             obj.to_zarr(handle)
    #         elif format in ['pk']:
    #             obj.to_pickle(handle)  # obj is a :class:`pandas.DataFrame`
    #     return self
    #
    # def _read(self, path, format='zarr'):
    #     """ Read internal array object from file store
    #
    #         Returns
    #         -------
    #         obj: :class:`xarray.DataSet` or :class:`pandas.DataFrame`
    #     """
    #     with self.fs.open(path, "rb") as handle:
    #         if format in ['zarr']:
    #             obj = xr.open_zarr(handle)
    #         elif format in ['pk']:
    #             obj = pd.read_pickle(handle)
    #     return obj

    @property
    def uri(self):
        """ List of resources to load for a request

        This can be a list of paths or urls, depending on the data source selected.

        Returns
        -------
        list(str)
            List of resources used to fetch data
        """
        if self.fetcher:
            return self.fetcher.uri
        else:
            raise InvalidFetcherAccessPoint(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )

    @property
    def data(self):
        """ Data structure

            Returns
            --------
            :class:`xarray.DataArray`
                Fetched data
        """
        if not isinstance(self._data, xr.Dataset):
            self.load()
        return self._data

    @property
    def index(self):
        """ Index structure, as returned by the to_index method

            Returns
            --------
            :class:`pandas.DataFrame`
                Argo-like index of fetched data
        """
        if not isinstance(self._index, pd.core.frame.DataFrame):
            if "gdac" in self._src or "localftp" in self._src:
                self.to_index(full=True)
            else:
                self.load()
        return self._index

    @property
    def domain(self):
        """" Domain of the dataset

            This is different from a usual ``box`` because dates are already in numpy.datetime64 format.
        """
        this_ds = self.data
        if 'PRES_ADJUSTED' in this_ds.data_vars:
            Pmin = np.nanmin((np.min(this_ds['PRES'].values), np.min(this_ds['PRES_ADJUSTED'].values)))
            Pmax = np.nanmax((np.max(this_ds['PRES'].values), np.max(this_ds['PRES_ADJUSTED'].values)))
        else:
            Pmin = np.min(this_ds['PRES'].values)
            Pmax = np.max(this_ds['PRES'].values)

        return [np.min(this_ds['LONGITUDE'].values), np.max(this_ds['LONGITUDE'].values),
                np.min(this_ds['LATITUDE'].values), np.max(this_ds['LATITUDE'].values),
                Pmin, Pmax,
                np.min(this_ds['TIME'].values), np.max(this_ds['TIME'].values)]

    def dashboard(self, **kw):
        """Open access point dashboard.

            See Also
            --------
            :class:`argopy.dashboard`
        """
        try:
            return self.fetcher.dashboard(**kw)
        except Exception:
            warnings.warn(
                "dashboard not available for this fetcher access point (%s/%s)"
                % (self._src, self._AccessPoint)
            )

    @checkAccessPoint
    def float(self, wmo, **kw):
        """ Float data fetcher

        Parameters
        ----------
        wmo: int, list(int)
            Define the list of Argo floats to load data for. This is a list of integers with WMO float identifiers.
            WMO is the World Meteorological Organization.

        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher.float`
            A data source fetcher for all float profiles
        """
        wmo = check_wmo(wmo)  # Check and return a valid list of WMOs
        if "CYC" in kw or "cyc" in kw:
            raise TypeError(
                "float() got an unexpected keyword argument 'cyc'. Use 'profile' access "
                "point to fetch specific profile data."
            )

        self.fetcher = self.Fetchers["float"](WMO=wmo, **self.fetcher_options)
        self._AccessPoint = "float"  # Register the requested access point
        self._AccessPoint_data = {'wmo': wmo}  # Register the requested access point data

        if self._mode == "standard" and self._dataset_id != "ref":
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        return self

    @checkAccessPoint
    def profile(self, wmo, cyc):
        """  Profile data fetcher

        Parameters
        ----------
        wmo: int, list(int)
            Define the list of Argo floats to load data for. This is a list of integers with WMO float identifiers.
            WMO is the World Meteorological Organization.
        cyc: list(int)
            Define the list of cycle numbers to load for each Argo floats listed in ``wmo``.

        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher.profile`
            A data source fetcher for specific float profiles
        """
        wmo = check_wmo(wmo)  # Check and return a valid list of WMOs
        cyc = check_cyc(cyc)  # Check and return a valid list of CYCs
        self.fetcher = self.Fetchers["profile"](WMO=wmo, CYC=cyc, **self.fetcher_options)
        self._AccessPoint = "profile"  # Register the requested access point
        self._AccessPoint_data = {'wmo': wmo, 'cyc': cyc}  # Register the requested access point data

        if self._mode == "standard" and self._dataset_id != "ref":
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds
            self.postproccessor = postprocessing

        return self

    @checkAccessPoint
    def region(self, box: list):
        """ Space/time domain data fetcher

            Parameters
            ----------
            box: list()
                Define the domain to load Argo data for. The box list is made of:
                    - lon_min: float, lon_max: float,
                    - lat_min: float, lat_max: float,
                    - dpt_min: float, dpt_max: float,
                    - date_min: str (optional), date_max: str (optional)

                Longitude, latitude and pressure bounds are required, while the two bounding dates are optional.
                If bounding dates are not specified, the entire time series is fetched.
                Eg: [-60, -55, 40., 45., 0., 10., '2007-08-01', '2007-09-01']

            Returns
            -------
            :class:`argopy.fetchers.ArgoDataFetcher`
                A data source fetcher for a space/time domain
        """
        is_box(box, errors="raise")  # Validate the box definition

        self.fetcher = self.Fetchers["region"](box=box, **self.fetcher_options)
        self._AccessPoint = "region"  # Register the requested access point
        self._AccessPoint_data = {'box': box}  # Register the requested access point data

        if self._mode == "standard" and self._dataset_id != "ref":
            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds
            self.postproccessor = postprocessing

        return self

    def to_xarray(self, **kwargs):
        """ Fetch and return data as xarray.DataSet

            Trigger a fetch of data by the specified source and access point.

            Returns
            -------
            :class:`xarray.DataSet`
                Fetched data
        """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)

        # data_path = self.fetcher.cname() + self._mode + ".zarr"
        # log.debug(data_path)
        # if self.cache and self.fs.exists(data_path):
        #     xds = self._read(data_path)
        # else:
        #     xds = self.fetcher.to_xarray(**kwargs)
        #     xds = self.postproccessor(xds)
        #     xds = self._write(data_path, xds)._read(data_path)
        return xds

    def to_dataframe(self, **kwargs):
        """ Fetch and return data as pandas.Dataframe

            Trigger a fetch of data by the specified source and access point.

            Returns
            -------
            :class:`pandas.DataFrame`
                Fetched data
        """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.load().data.to_dataframe(**kwargs)

    def to_index(self, full: bool = False, coriolis_id = False):
        """ Create an index of Argo data, fetch data if necessary

            Build an Argo-like index of profiles from fetched data.

            Parameters
            ----------
            full: bool, default: False
                Should extract a reduced index (only a space/time) from fetched profiles, or a full index,
                as returned by an IndexFetcher.

            Returns
            -------
            :class:`pandas.DataFrame`
                Argo-like index of fetched data
        """
        if not full:
            self.load()
            ds = self.data.argo.point2profile()
            df = ds[["PLATFORM_NUMBER", "CYCLE_NUMBER", "LONGITUDE", "LATITUDE", "TIME"]].to_dataframe()
            df = (
                df.reset_index()
                .rename(
                    columns={
                        "PLATFORM_NUMBER": "wmo",
                        "CYCLE_NUMBER": "cyc",
                        "LONGITUDE": "longitude",
                        "LATITUDE": "latitude",
                        "TIME": "date",
                    }
                )
                .drop(columns="N_PROF")
            )

            df = df[["date", "latitude", "longitude", "wmo", "cyc"]]
            if coriolis_id:
                df['id'] = None
                def fc(row):
                    row['id'] = get_coriolis_profile_id(row['wmo'], row['cyc'])['ID'].values[0]
                    return row
                df = df.apply(fc, axis=1)
        else:
            # Instantiate and load an IndexFetcher:
            index_loader = ArgoIndexFetcher(mode=self._mode,
                                            src=self._src,
                                            ds=self._dataset_id,
                                            **self.fetcher_kwargs)
            if self._AccessPoint == 'float':
                index_loader.float(self._AccessPoint_data['wmo']).load()
            if self._AccessPoint == 'profile':
                index_loader.profile(self._AccessPoint_data['wmo'], self._AccessPoint_data['cyc']).load()
            if self._AccessPoint == 'region':
                # Convert data box to index box (remove depth info):
                index_box = self._AccessPoint_data['box'].copy()
                del index_box[4:6]
                index_loader.region(index_box).load()
            df = index_loader.index

            # if self._loaded and self._mode == 'standard' and len(self._index) != len(df):
            #     warnings.warn("Loading a full index in 'standard' user mode may lead to more profiles in the "
            #                   "index than reported in data.")

            # Possibly replace the light index with the full version:
            if not self._loaded or self._request == self.__repr__():
                self._index = df

        return df

    def load(self, force: bool = False, **kwargs):
        """ Fetch data (and compute an index) if not already in memory

            Apply the default to_xarray() and to_index() methods and store results in memory.
            You can access loaded measurements structure with the `data` and `index` properties.

            Parameters
            ----------
            force: bool
                Force fetching data if not already in memory, default is False.

            Returns
            -------
            :class:`argopy.fetchers.ArgoDataFetcher.float`
                Data fetcher with `data` and `index` properties in memory

            Examples
            --------
            >>> ds = ArgoDataFetcher().profile(6902746, 34).load().data
            >>> df = ArgoDataFetcher().float(6902746).load().index
        """
        # Force to load data if the fetcher definition has changed
        if self._loaded and self._request != self.__repr__():
            force = True

        if not self._loaded or force:
            # Fetch measurements:
            self._data = self.to_xarray(**kwargs)
            # Next 2 lines must come before ._index because to_index(full=False) calls back on .load() to read .data
            self._request = self.__repr__()  # Save definition of loaded data
            self._loaded = True
            # Extract measurements index from data:
            self._index = self.to_index(full=False)
        return self

    def clear_cache(self):
        """ Clear data cached by fetcher """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.fetcher.clear_cache()

    def plot(self, ptype="trajectory", **kwargs):
        """ Create custom plots from data

            Parameters
            ----------
            ptype: str, optional, default: 'trajectory'
                Plot type, one of the following: 'trajectory',' profiler', 'dac', 'qc_altimetry'.

            Returns
            -------
            fig: :class:`matplotlib.figure.Figure`
            ax: :class:`matplotlib.axes.Axes`
        """
        self.load()
        if ptype in ["dac", "institution"]:
            if "institution" not in self.index:
                self.to_index(full=True)
            return bar_plot(self.index, by="institution", **kwargs)
        elif ptype == "profiler":
            if "profiler" not in self.index:
                self.to_index(full=True)
            return bar_plot(self.index, by="profiler", **kwargs)
        elif ptype == "trajectory":
            return plot_trajectory(self.index, **kwargs)
        elif ptype == "qc_altimetry":
            WMOs = np.unique(self.data['PLATFORM_NUMBER'])
            return open_sat_altim_report(WMOs, **kwargs)
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'trajectory', 'dac', 'profiler', 'qc_altimetry'"
            )


class ArgoIndexFetcher:
    """ Fetcher and post-processor of Argo index data (API facade)

    Parameters
    ----------
    mode: str, optional
        User mode. Eg: ``standard`` or ``expert``. Set to OPTIONS['mode'] by default if empty.
    src: str, optional
         Source of the data to use. Eg: ``erddap``. Set to OPTIONS['src'] by default if empty.
    ds: str, optional
        Name of the dataset to load. Eg: ``phy``. Set to OPTIONS['dataset'] by default if empty.
    **fetcher_kwargs: optional
        Additional arguments passed on data source fetcher of each access points.

    Notes
    -----
    Spec discussions can be found here:
        https://github.com/euroargodev/argopy/issues/8

        https://github.com/euroargodev/argopy/pull/6

    Examples
    --------
    >>> from argopy import IndexFetcher
    >>> adf = IndexFetcher.region([-75, -65, 10, 20]).load()
    >>> idx.plot()
    >>> idx.index
    """

    def __init__(self, mode: str = "", src: str = "", ds: str = "", **fetcher_kwargs):

        # Facade options:
        self._mode = OPTIONS["mode"] if mode == "" else mode
        self._dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self._src = OPTIONS["src"] if src == "" else src

        _VALIDATORS["mode"](self._mode)
        _VALIDATORS["src"](self._src)

        # Load data source access points:
        if self._src not in AVAILABLE_INDEX_SOURCES:
            raise InvalidFetcher(
                "Requested index fetcher '%s' not available ! "
                "Please try again with any of: %s"
                % (self._src, "\n".join(AVAILABLE_INDEX_SOURCES))
            )
        else:
            Fetchers = AVAILABLE_INDEX_SOURCES[self._src]

        # Auto-discovery of access points for this fetcher:
        # rq: Access point names for the facade are not the same as the access point of fetchers
        self.Fetchers = {}
        self.valid_access_points = []
        for p in Fetchers.access_points:
            if p == "box":  # Required for 'region'
                self.Fetchers["region"] = Fetchers.Fetch_box
                self.valid_access_points.append("region")
            if p == "wmo":  # Required for 'profile' and 'float'
                self.Fetchers["float"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("float")
                self.Fetchers["profile"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("profile")

        # Init sub-methods:
        self.fetcher = None
        if self._dataset_id not in Fetchers.dataset_ids:
            raise ValueError(
                "%s dataset is not available for this index source (%s)"
                % (self._dataset_id, self._src)
            )
        # self.fetcher_kwargs = {**fetcher_kwargs}
        self.fetcher_options = {**{"ds": self._dataset_id}, **fetcher_kwargs}
        self.postproccessor = self.__empty_processor
        self._AccessPoint = None

        # Init data structure holders:
        self._index = None

        # More init:
        self._loaded = False
        self._request = ""

    def __repr__(self):
        if self.fetcher:
            summary = [self.fetcher.__repr__(),
                       "Backend: %s" % self._src]
        else:
            summary = ["<indexfetcher.%s> 'No access point initialised'" % self._src,
                       "Available access points: %s" % ", ".join(self.Fetchers.keys()),
                       "Backend: %s" % self._src]

        summary.append("User mode: %s" % self._mode)
        summary.append("Dataset: %s" % self._dataset_id)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """ Do nothing to a dataset """
        return xds

    def __getattr__(self, key):
        """ Validate access points """
        valid_attrs = [
            "Fetchers",
            "fetcher",
            "fetcher_options",
            "postproccessor",
            "index",
            "_loaded",
        ]
        if key not in self.valid_access_points and key not in valid_attrs:
            raise InvalidFetcherAccessPoint("'%s' is not a valid access point" % key)
        pass

    @property
    def index(self):
        """ Index structure

            Returns
            --------
            :class:`pandas.DataFrame`
                Argo-like index of fetched data
        """
        if not isinstance(self._index, pd.core.frame.DataFrame):
            self.load()
        return self._index

    @checkAccessPoint
    def float(self, wmo):
        """ Float index fetcher

        Parameters
        ----------
        wmo: list(int)
            Define the list of Argo floats to load data for. This is a list of integers with WMO numbers.

        Returns
        -------
        :class:`argopy.fetchers.ArgoIndexFetcher`
            An index fetcher initialised for specific floats
        """
        wmo = check_wmo(wmo)  # Check and return a valid list of WMOs
        self.fetcher = self.Fetchers["float"](WMO=wmo, **self.fetcher_options)
        self._AccessPoint = "float"  # Register the requested access point
        return self

    @checkAccessPoint
    def profile(self, wmo, cyc):
        """ Profile index fetcher

            Parameters
            ----------
            wmo: int, list(int)
                Define the list of Argo floats to load index for. This is a list of integers with WMO float identifiers.
                WMO is the World Meteorological Organization.
            cyc: list(int)
                Define the list of cycle numbers to load for each Argo floats listed in ``wmo``.

            Returns
            -------
            :class:`argopy.fetchers.ArgoIndexFetcher`
                An index fetcher initialised for specific float profiles
        """
        wmo = check_wmo(wmo)  # Check and return a valid list of WMOs
        cyc = check_cyc(cyc)  # Check and return a valid list of CYCs
        self.fetcher = self.Fetchers["profile"](WMO=wmo, CYC=cyc, **self.fetcher_options)
        self._AccessPoint = "profile"  # Register the requested access point
        return self

    @checkAccessPoint
    def region(self, box):
        """ Space/time domain index fetcher

            Parameters
            ----------
            box: list()
                Define the domain to load Argo index for. The box list is made of:
                    - lon_min: float, lon_max: float,
                    - lat_min: float, lat_max: float,
                    - date_min: str (optional), date_max: str (optional)

                Longitude and latitude bounds are required, while the two bounding dates are optional.
                If bounding dates are not specified, the entire time series is fetched.
                Eg: [-60, -55, 40., 45., '2007-08-01', '2007-09-01']

            Returns
            -------
            :class:`argopy.fetchers.ArgoIndexFetcher`
                An index fetcher initialised for a space/time domain

            Warnings
            --------
            Note that the box option for an index fetcher does not have pressure bounds, contrary to the data fetcher.
        """
        is_indexbox(box, errors="raise")  # Validate the box definition
        self.fetcher = self.Fetchers["region"](box=box, **self.fetcher_options)
        self._AccessPoint = "region"  # Register the requested access point
        return self

    def to_dataframe(self, **kwargs):
        """ Fetch and return index data as pandas Dataframe

            Returns
            -------
            :class:`pandas.DataFrame`
        """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.fetcher.to_dataframe(**kwargs)

    def to_xarray(self, **kwargs):
        """ Fetch and return index data as xarray DataSet

            This is a shortcut to .load().index.to_xarray()

            Returns
            -------
            :class:`xarray.DataSet`
        """
        if self._AccessPoint not in self.valid_access_points:
            raise InvalidFetcherAccessPoint(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.load().index.to_xarray(**kwargs)

    def to_csv(self, file: str = "output_file.csv"):
        """ Fetch and save index data as csv in a file

            Notes
            -----
            >>> idx.to_csv()
            is a shortcut to:
            >>> idx.load().index.to_csv()

            Since the ``index`` property is a :class:`pandas.DataFrame`, this is currently a short
            cut to :meth:`pandas.DataFrame.to_index`

            Returns
            -------
            None
        """
        if self._AccessPoint not in self.valid_access_points:
            raise InvalidFetcherAccessPoint(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.load().index.to_csv(file)

    def load(self, force: bool = False):
        """ Load index in memory

        Apply the default to_dataframe() method and store results in memory.
        You can access the index array with the `index` property::

        >>> df = ArgoIndexFetcher().float(6902746).load().index

        Parameters
        ----------
        force: bool
            Force loading, default is False.

        Returns
        -------
        :class:`argopy.fetchers.ArgoIndexFetcher.float`
            Index fetcher with `index` property in memory
        """
        # Force to load data if the fetcher definition has changed
        if self._loaded and self._request != self.__repr__():
            force = True

        if not self._loaded or force:
            self._index = self.to_dataframe()
            self._request = self.__repr__()  # Save definition of loaded data
            self._loaded = True
        return self

    def plot(self, ptype="trajectory", **kwargs):
        """ Create custom plots from index

            Parameters
            ----------
            ptype: {'trajectory',' profiler', 'dac', 'qc_altimetry}, default: 'trajectory'

            Returns
            -------
            fig: :class:`matplotlib.figure.Figure`
            ax: :class:`matplotlib.axes.Axes`
        """
        self.load()
        if ptype in ["dac", "institution"]:
            return bar_plot(self.index, by="institution", **kwargs)
        elif ptype == "profiler":
            return bar_plot(self.index, by="profiler", **kwargs)
        elif ptype == "trajectory":
            return plot_trajectory(self.index.sort_values(["file"]), **kwargs)
        elif ptype == "qc_altimetry":
            WMOs = np.unique(self.index['wmo'])
            return open_sat_altim_report(WMOs, **kwargs)
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'trajectory', 'dac', 'profiler', 'qc_altimetry'"
            )

    def clear_cache(self):
        """ Clear fetcher cached data """
        return self.fetcher.clear_cache()
