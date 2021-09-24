#!/bin/env python
# -*coding: UTF-8 -*-
"""

High level helper methods to load Argo data from any source
The facade should be able to work with all available data access point,

"""

import warnings
import xarray as xr
import pandas as pd
import logging

from argopy.options import OPTIONS, _VALIDATORS
from .errors import InvalidFetcherAccessPoint, InvalidFetcher
from .utilities import list_available_data_src, list_available_index_src, is_box
from .plotters import plot_trajectory, bar_plot

AVAILABLE_DATA_SOURCES = list_available_data_src()
AVAILABLE_INDEX_SOURCES = list_available_index_src()

log = logging.getLogger("argopy.fetchers.facade")

def checkAccessPoint(AccessPoint):
    """ Decorator to validate fetcher access points of a given data source

        This decorator will check if an access point (eg: 'profile') is available for the data source (eg: 'erddap')
        used to initiate the checker. If not, an error is raised.
    """
    def wrapper(*args):
        if AccessPoint.__name__ not in args[0].valid_access_points:
            raise InvalidFetcherAccessPoint(
                            "'%s' not available with '%s' src. Available access point(s): %s" %
                            (AccessPoint.__name__, args[0]._src, ", ".join(args[0].Fetchers.keys()))
                        )
        return AccessPoint(*args)
    return wrapper


class ArgoDataFetcher:
    """ Fetcher and post-processor of Argo data (API facade) """

    def __init__(self, mode: str = "", src: str = "", ds: str = "", **fetcher_kwargs):

        """ Create a fetcher instance

        Parameters
        ----------
        mode: str, optional
            User mode. Eg: ``standard`` or ``expert``. Set to OPTIONS['mode'] by default if empty.
        src: str, optional
             Source of the data to use. Eg: ``erddap``. Set to OPTIONS['src'] by default if empty.
        ds: str, optional
            Name of the dataset to load. Eg: ``phy``. Set to OPTIONS['dataset'] by default if empty.
        **fetcher_kwargs: optional
            Additional arguments passed on data source instance creation of each access points.

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
            if p == "wmo":  # Required for 'profile' and 'float'
                self.Fetchers["profile"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("profile")
                self.Fetchers["float"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("float")
            if p == "box":  # Required for 'region'
                self.Fetchers["region"] = Fetchers.Fetch_box
                self.valid_access_points.append("region")

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

        # Dev warnings
        # Todo Clean-up before each release
        if self._dataset_id == "bgc" and self._mode == "standard":
            warnings.warn(
                "'BGC' dataset fetching in 'standard' user mode is not reliable. "
                "Try to switch to 'expert' mode if you encounter errors."
            )

    def __repr__(self):
        if self.fetcher:
            summary = [self.fetcher.__repr__()]
            if "parallel" in self.fetcher_options:
                summary.append(
                    "Backend: %s (parallel=%s)"
                    % (self._src, str(self.fetcher_options["parallel"]))
                )
            else:
                summary.append("Backend: %s" % self._src)
            summary.append("User mode: %s" % self._mode)
        else:
            summary = ["<datafetcher> 'Not initialised'"]
            summary.append("Current backend: %s" % self._src)
            summary.append("Available fetchers: %s" % ", ".join(self.Fetchers.keys()))
            summary.append("User mode: %s" % self._mode)
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
            "_loaded",
            "_request"
        ]
        if key not in self.valid_access_points and key not in valid_attrs:
            raise InvalidFetcherAccessPoint("'%s' is not a valid access point" % key)
        pass

    @property
    def uri(self):
        """ List of resources to load for a request

        This can be a list of paths or urls, depending on the data source selected.

        Returns
        -------
        list(str)
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
        """
        if not isinstance(self._data, xr.Dataset):
            self.load()
        return self._data

    @property
    def index(self):
        """ Index structure, as returned by the to_index method

            Returns
            --------
            :class:`pandas.Dataframe`

        """
        if not isinstance(self._index, pd.core.frame.DataFrame):
            self.load()
        return self._index

    def dashboard(self, **kw):
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
        wmo: list(int)
            Define the list of Argo floats to load data for. This is a list of integers with WMO numbers.

        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher.float`
            A data source fetcher for all float profiles
        """
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
        wmo: list(int)
            Define the list of Argo floats to load data for. This is a list of integers with WMO numbers.
        cyc: list(int)
            Define the list of cycle numbers to load for each Argo floats listed in ``wmo``.

        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher.profile`
            A data source fetcher for specific float profiles
        """
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

            Returns
            -------
            :class:`xarray.DataSet`
        """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)
        return xds

    def to_dataframe(self, **kwargs):
        """  Fetch and return data as pandas.Dataframe

            Returns
            -------
            :class:`pandas.Dataframe`
        """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.load().data.to_dataframe(**kwargs)

    def to_index(self, full: bool = False):
        """ Create an index of Argo data

            Parameters
            ----------
            full: bool
                Should extract a full index, as returned by an IndexFetcher or only a space/time
                index of fetched profiles (this is the default choice, i.e. full=False).

            Returns
            -------
            :class:`pandas.Dataframe`
        """
        if not full:
            self.load()
            ds = self.data.argo.point2profile()
            df = (
                ds.drop_vars(set(ds.data_vars) - set(["PLATFORM_NUMBER"]))
                .drop_dims("N_LEVELS")
                .to_dataframe()
            )
            df = (
                df.reset_index()
                .rename(
                    columns={
                        "PLATFORM_NUMBER": "wmo",
                        "LONGITUDE": "longitude",
                        "LATITUDE": "latitude",
                        "TIME": "date",
                    }
                )
                .drop(columns="N_PROF")
            )
            df = df[["date", "latitude", "longitude", "wmo"]]

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
                index_box = self._AccessPoint_data['box']
                del index_box[4:6]
                index_loader.region(index_box).load()
            df = index_loader.index

            if self._loaded and self._mode == 'standard' and len(self._index) != len(df):
                warnings.warn("Loading a full index in 'standard' user mode may lead to more profiles in the "
                              "index than reported in data.")

            # Possibly replace the light index with the full version:
            if not self._loaded or self._request == self.__repr__():
                self._index = df

        return df

    def load(self, force: bool = False, **kwargs):
        """ Load data in memory

            Apply the default to_xarray() and to_index() methods and store results in memory.
            Access loaded measurements structure with the `data` and `index` properties::

                ds = ArgoDataFetcher().profile(6902746, 34).load().data
                # or
                df = ArgoDataFetcher().float(6902746).load().index

            Parameters
            ----------
            force: bool
                Force loading, default is False.

            Returns
            -------
            :class:`argopy.fetchers.ArgoDataFetcher.float`
                Data fetcher with `data` and `index` properties in memory
        """
        # Force to load data if the fetcher definition has changed
        if self._loaded and self._request != self.__repr__():
            force = True

        if not self._loaded or force:
            # Fetch measurements:
            self._data = self.to_xarray(**kwargs)
            # Next 2 lines must come before ._index because to_index() calls back on .load() to read .data
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
            ptype: str
                Type of plot to generate. This can be: 'trajectory',' profiler', 'dac'.

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
            return plot_trajectory(self.index, **kwargs)
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'dac', 'profiler' or 'trajectory' (default)"
            )


class ArgoIndexFetcher:
    """
    Specs discussion :
    https://github.com/euroargodev/argopy/issues/8
    https://github.com/euroargodev/argopy/pull/6)

    Usage:

    from argopy import ArgoIndexFetcher
    idx = ArgoIndexFetcher.region([-75, -65, 10, 20])
    idx.plot.trajectories()
    idx.load().to_dataframe()

    Fetch and process Argo index.

    Can return metadata from index of :
        - one or more float(s), defined by WMOs
        - one or more profile(s), defined for one WMO and one or more CYCLE NUMBER
        - a space/time rectangular domain, defined by lat/lon/pres/time range

    idx object can also be used as an input :
     argo_loader = ArgoDataFetcher(index=idx)

    Specify here all options to data_fetchers

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
            if p == "wmo":  # Required for 'profile' and 'float'
                self.Fetchers["profile"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("profile")
                self.Fetchers["float"] = Fetchers.Fetch_wmo
                self.valid_access_points.append("float")
            if p == "box":  # Required for 'region'
                self.Fetchers["region"] = Fetchers.Fetch_box
                self.valid_access_points.append("region")

        # Init sub-methods:
        self.fetcher = None
        if self._dataset_id not in Fetchers.dataset_ids:
            raise ValueError(
                "%s dataset is not available for this index source (%s)"
                % (self._dataset_id, self._src)
            )
        self.fetcher_options = {**fetcher_kwargs}
        self.postproccessor = self.__empty_processor
        self._AccessPoint = None

        # Init data structure holders:
        self._index = None

    def __repr__(self):
        if self.fetcher:
            summary = [self.fetcher.__repr__()]
            summary.append("Backend: %s" % self._src)
            summary.append("User mode: %s" % self._mode)
        else:
            summary = ["<indexfetcher> 'Not initialised'"]
            summary.append("Current backend: %s" % self._src)
            summary.append("Available fetchers: %s" % ", ".join(self.Fetchers.keys()))
            summary.append("User mode: %s" % self._mode)
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
            :class:`pandas.Dataframe`
        """
        if not isinstance(self._index, pd.core.frame.DataFrame):
            self.load()
        return self._index

    @checkAccessPoint
    def profile(self, wmo, cyc):
        """ Profile index fetcher

        Parameters
        ----------
        wmo: list(int)
            Define the list of Argo floats to load data for. This is a list of integers with WMO numbers.
        cyc: list(int)
            Define the list of cycle numbers to load for each Argo floats listed in ``wmo``.

        Returns
        -------
        :class:`argopy.fetchers.ArgoIndexFetcher`
            A index fetcher initialised for specific float profiles
        """
        self.fetcher = self.Fetchers["profile"](WMO=wmo, CYC=cyc, **self.fetcher_options)
        self._AccessPoint = "profile"  # Register the requested access point
        return self

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
            A index fetcher initialised for all float profiles
        """
        self.fetcher = self.Fetchers["float"](WMO=wmo, **self.fetcher_options)
        self._AccessPoint = "float"  # Register the requested access point
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
            Eg: [-60, -55, 40., 45., 0., 10., '2007-08-01', '2007-09-01']

        Returns
        -------
        :class:`argopy.fetchers.ArgoIndexFetcher`
            A index fetcher initialised for a space/time domain

        Warning
        -------
        Note that the box option for an index fetcher does not have pressure bounds, contrary to the data fetcher.

        """
        self.fetcher = self.Fetchers["region"](box=box, **self.fetcher_options)
        self._AccessPoint = "region"  # Register the requested access point
        return self

    def to_dataframe(self, **kwargs):
        """ Fetch and return index data as pandas Dataframe

            Returns
            -------
            :class:`pandas.Dataframe`
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

            This is a shortcut to .load().index.to_csv()

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
            Access loaded index structure with the `index` property::

                df = ArgoIndexFetcher().float(6902746).load().index

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
            ptype: str
                Type of plot to generate. This can be: 'trajectory',' profiler', 'dac'.

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
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'dac', 'profiler' or 'trajectory' (default)"
            )

    def clear_cache(self):
        """ Clear fetcher cached data """
        return self.fetcher.clear_cache()
