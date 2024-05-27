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

from .options import OPTIONS, _VALIDATORS
from .errors import InvalidFetcherAccessPoint, InvalidFetcher, OptionValueError
from .related import (
    get_coriolis_profile_id,
)
from .utils.checkers import (
    is_box,
    is_indexbox,
    check_wmo,
    check_cyc
)
from .utils.lists import (
    list_available_data_src,
    list_available_index_src,
)
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
                "'%s' not available with '%s' src. Available access point(s): %s"
                % (
                    AccessPoint.__name__,
                    args[0]._src,
                    ", ".join(args[0].Fetchers.keys()),
                )
            )
        return AccessPoint(*args)

    wrapper.__name__ = AccessPoint.__name__
    wrapper.__doc__ = AccessPoint.__doc__
    return wrapper


class ArgoDataFetcher:
    """Fetcher and post-processor of Argo data (API facade)

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
        """Create a fetcher instance


        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher`
        """

        # Facade options :
        self._mode = OPTIONS["mode"] if mode == "" else mode
        self._dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self._src = OPTIONS["src"] if src == "" else src

        if not _VALIDATORS["mode"](self._mode):
            raise OptionValueError(
                f"option 'mode' given an invalid value: {self._mode}"
            )
        if not _VALIDATORS["dataset"](self._dataset_id):
            raise OptionValueError(
                f"option 'dataset' given an invalid value: {self._dataset_id}"
            )
        if not _VALIDATORS["src"](self._src):
            raise OptionValueError(f"option 'src' given an invalid value: {self._src}")

        # Load data source access points:
        if self._src == "localftp":
            raise ValueError(
                "The 'localftp' data source is deprecated. It's been replaced by 'gdac'."
            )

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
                "The '%s' dataset is not available for the '%s' data source"
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
        if self._dataset_id == "bgc" and (
            self._mode == "standard" or self._mode == "research"
        ):
            warnings.warn(
                "The 'bgc' dataset fetching is only available in 'expert' mode at this point."
            )

    def __repr__(self):
        para = (
            self.fetcher_options["parallel"]
            if "parallel" in self.fetcher_options
            else False
        )
        cache = (
            self.fetcher_options["cache"] if "cache" in self.fetcher_options else False
        )

        if self.fetcher:
            summary = [self.fetcher.__repr__()]
        else:
            summary = ["<datafetcher.%s> 'No access point initialised'" % self._src]
            summary.append(
                "Available access points: %s" % ", ".join(self.Fetchers.keys())
            )

        summary.append("Performances: cache=%s, parallel=%s" % (str(cache), str(para)))
        summary.append("User mode: %s" % self._mode)
        summary.append("Dataset: %s" % self._dataset_id)
        # summary.append("Loaded: %s" % self._loaded)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """Do nothing to a dataset"""
        return xds

    def __getattr__(self, key):
        """Validate access points"""
        valid_attrs = [
            "Fetchers",
            "fetcher",
            "fetcher_options",
            "postproccessor",
            "data",
            "index",
            "domain",
            "mission",
            "_loaded",
            "_request",
            "cache",
            "cachedir",
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
        """List of resources to load for a request

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
        """Data structure

        Returns
        --------
        :class:`xarray.DataArray`
            Fetched data
        """
        if not isinstance(self._data, xr.Dataset) or self._request != self.__repr__():
            self.load()
        return self._data

    @property
    def index(self):
        """Index structure, as returned by the to_index method

        Returns
        --------
        :class:`pandas.DataFrame`
            Argo-like index of fetched data
        """
        if (
            not isinstance(self._index, pd.core.frame.DataFrame)
            or self._request != self.__repr__()
        ):
            self.load()
        return self._index

    @property
    def domain(self):
        """Space/time domain of the dataset

        This is different from a usual ``box`` because dates are in :class:`numpy.datetime64` format.
        """
        return self.data.argo.domain

    @property
    def mission(self):
        if self._dataset_id == 'bgc':
            return "BGC"
        else:
            return "core+deep"

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
        """Float data fetcher

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
        self._AccessPoint_data = {
            "wmo": wmo
        }  # Register the requested access point data

        if self._mode == "standard" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        elif self._mode == "research" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_researchmode(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        return self

    @checkAccessPoint
    def profile(self, wmo, cyc):
        """Profile data fetcher

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
        self.fetcher = self.Fetchers["profile"](
            WMO=wmo, CYC=cyc, **self.fetcher_options
        )
        self._AccessPoint = "profile"  # Register the requested access point
        self._AccessPoint_data = {
            "wmo": wmo,
            "cyc": cyc,
        }  # Register the requested access point data

        if self._mode == "standard" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        elif self._mode == "research" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_researchmode(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        return self

    @checkAccessPoint
    def region(self, box: list):
        """Space/time domain data fetcher

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
        self._AccessPoint_data = {
            "box": box
        }  # Register the requested access point data

        if self._mode == "standard" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        elif self._mode == "research" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_researchmode(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        return self

    def to_xarray(self, **kwargs):
        """Fetch and return data as xarray.DataSet

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

        return xds

    def to_dataframe(self, **kwargs):
        """Fetch and return data as pandas.Dataframe

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

    def to_index(self, full: bool = False, coriolis_id: bool = False):
        """Create a profile index of Argo data, fetch data if necessary

        Build an Argo-like index of profiles from fetched data.

        Parameters
        ----------
        full: bool, default: False
            If possible, should extract a reduced index (only space/time/wmo/cyc) from fetched profiles, otherwise a full index,
            as returned by an IndexFetcher.
        coriolis_id: bool, default: False
            Add a column to the index with the Coriolis ID of profiles

        Returns
        -------
        :class:`pandas.DataFrame`
            Argo-like index of fetched data
        """
        def prt(txt):
            msg = [txt]
            if self._request != self.__repr__():
                msg.append(self._request)
            log.debug("\n".join(msg))

        def add_coriolis(this_df):
            if 'id' not in this_df:
                this_df["id"] = None

                def fc(row):
                    row["id"] = get_coriolis_profile_id(row["wmo"], row["cyc"])["ID"].values[0]
                    return row

                this_df = this_df.apply(fc, axis=1)
            return this_df

        # With the gdac and erddap+bgc,
        # we rely on the fetcher ArgoIndex:
        # (hence we always return a full index)
        if (self._src == 'erddap' and self._dataset_id == 'bgc') or (self._src == 'gdac'):
            prt("to_index working with fetcher ArgoIndex instance")
            idx = self.fetcher.indexfs
            if self._AccessPoint == "region":
                # Convert data box to index box (remove depth info):
                index_box = self._AccessPoint_data["box"].copy()
                del index_box[4:6]
                if len(index_box) == 4:
                    idx.search_lat_lon(index_box)
                else:
                    idx.search_lat_lon_tim(index_box)
            if self._AccessPoint == "float":
                idx.search_wmo(self._AccessPoint_data["wmo"])
            if self._AccessPoint == "profile":
                idx.search_wmo_cyc(self._AccessPoint_data["wmo"], self._AccessPoint_data["cyc"])

            # Then export search result to Index dataframe:
            df = idx.to_dataframe()

            # Add Coriolis ID if requested:
            df = add_coriolis(df) if coriolis_id else df

        # For all other data source and dataset, we need to compute the index:
        else:

            if not full:
                prt("to_index working with argo accessor attribute for a light index")
                # Get a small index from the argo accessor attribute
                self.load()
                df = self.data.argo.index

                # Add Coriolis ID if requested:
                df = add_coriolis(df) if coriolis_id else df

            else:
                prt("to_index working with IndexFetcher for a full index")
                # Instantiate and load an IndexFetcher:
                index_loader = ArgoIndexFetcher(
                    mode=self._mode,
                    src=self._src,
                    ds=self._dataset_id,
                    **self.fetcher_kwargs,
                )
                if self._AccessPoint == "float":
                    index_loader.float(self._AccessPoint_data["wmo"]).load()
                if self._AccessPoint == "profile":
                    index_loader.profile(
                        self._AccessPoint_data["wmo"], self._AccessPoint_data["cyc"]
                    ).load()
                if self._AccessPoint == "region":
                    # Convert data box to index box (remove depth info):
                    index_box = self._AccessPoint_data["box"].copy()
                    del index_box[4:6]
                    index_loader.region(index_box).load()
                df = index_loader.index

                # Add Coriolis ID if requested:
                df = add_coriolis(df) if coriolis_id else df

                # Possibly replace the light index with the full version:
                if 'profiler_code' not in df or self._request == self.__repr__():
                    prt("to_index replaced the light index with the full version")
                    self._index = df

        return df

    def load(self, force: bool = False, **kwargs):
        """ Fetch data (and compute a profile index) if not already in memory

        Apply the default to_xarray() and to_index() methods and store results in memory.
        You can access loaded measurements structure with the `data` and `index` properties.

        Parameters
        ----------
        force: bool
            Force fetching data even if not already in memory, default is False.

        Returns
        -------
        :class:`argopy.fetchers.ArgoDataFetcher`
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
        """Clear data cached by fetcher"""
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.fetcher.clear_cache()

    def plot(self, ptype: str = "trajectory", **kwargs):
        """Create custom plots from this fetcher data or index.

        This is basically shortcuts to some plotting submodules:

        - **trajectory** calls :class:`argopy.plot.plot_trajectory` with index DataFrame
        - **profiler** or **dac** calls :class:`argopy.plot.bar_plot` with index DataFrame
        - **qc_altimetry** calls :class:`argopy.plot.open_sat_altim_report` with data unique list of ``PLATFORM_NUMBER``

        Parameters
        ----------
        ptype: str, default: 'trajectory'
            Plot type, one of the following: ``trajectory``, ``profiler``, ``dac`` or ``qc_altimetry``.
        kwargs:
            Other arguments passed to the plotting submodule.

        Returns
        -------
        fig: :class:`matplotlib.figure.Figure`
        ax: :class:`matplotlib.axes.Axes`

        Warnings
        --------
        Calling this method will automatically trigger a call to the :class:`argopy.DataFetcher.load` method.

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
            defaults = {"style": "white"}
            return plot_trajectory(self.index, **{**defaults, **kwargs})
        elif ptype == "qc_altimetry":
            WMOs = np.unique(self.data["PLATFORM_NUMBER"])
            return open_sat_altim_report(WMOs, **kwargs)
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'trajectory', 'dac', 'profiler', 'qc_altimetry'"
            )


class ArgoIndexFetcher:
    """Fetcher and post-processor of Argo index data (API facade)

    An index dataset gather space/time information, and possibly more meta-data, of Argo profiles.

    Examples
    --------
    >>> from argopy import IndexFetcher
    >>> adf = IndexFetcher.region([-75, -65, 10, 20]).load()
    >>> idx.index
    >>> idx.plot()
    """

    def __init__(
        self,
        mode: str = OPTIONS["mode"],
        src: str = OPTIONS["src"],
        ds: str = OPTIONS["dataset"],
        **fetcher_kwargs,
    ):
        """Facade for Argo index fetchers

        Parameters
        ----------
        mode: str, optional
            User mode. Eg: ``standard`` or ``expert``.

        src: str, optional
             Source of the data to use. Eg: ``erddap``.

        ds: str, optional
            Name of the dataset to load. Eg: ``phy``.

        **fetcher_kwargs: optional
            Additional arguments passed on data source fetcher of each access points.
        """
        self._mode = mode
        self._dataset_id = ds
        self._src = src

        _VALIDATORS["mode"](self._mode)
        _VALIDATORS["src"](self._src)

        # Load data source access points:
        if self._src == "localftp":
            raise ValueError(
                "The 'localftp' data source is deprecated. It's been replaced by 'gdac'."
            )

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
            summary = [self.fetcher.__repr__(), "Backend: %s" % self._src]
        else:
            summary = [
                "<indexfetcher.%s> 'No access point initialised'" % self._src,
                "Available access points: %s" % ", ".join(self.Fetchers.keys()),
                "Backend: %s" % self._src,
            ]

        summary.append("User mode: %s" % self._mode)
        summary.append("Dataset: %s" % self._dataset_id)
        summary.append("Loaded: %s" % self._loaded)
        return "\n".join(summary)

    def __empty_processor(self, xds):
        """Do nothing to a dataset"""
        return xds

    def __getattr__(self, key):
        """Validate access points"""
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
        """Index structure

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
        """Float index fetcher

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
        """Profile index fetcher

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
        self.fetcher = self.Fetchers["profile"](
            WMO=wmo, CYC=cyc, **self.fetcher_options
        )
        self._AccessPoint = "profile"  # Register the requested access point
        return self

    @checkAccessPoint
    def region(self, box):
        """Space/time domain index fetcher

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
        """Fetch and return index data as pandas Dataframe

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
        """Fetch and return index data as xarray DataSet

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
        """Fetch and save index data as csv in a file

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
        """Load index in memory

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

    def plot(self, ptype: str = "trajectory", **kwargs):
        """Create custom plots from this fetcher index.

        This is basically shortcuts to some plotting submodules:

        - **trajectory** calls :class:`argopy.plot.plot_trajectory` with index DataFrame
        - **profiler** or **dac** calls :class:`argopy.plot.bar_plot` with index DataFrame
        - **qc_altimetry** calls :class:`argopy.plot.open_sat_altim_report` with index unique list of ``wmo``

        Parameters
        ----------
        ptype: str, default: 'trajectory'
            Plot type, one of the following: ``trajectory``, ``profiler``, ``dac`` or ``qc_altimetry``.
        kwargs:
            Other arguments passed to the plotting submodule.

        Returns
        -------
        fig: :class:`matplotlib.figure.Figure`
        ax: :class:`matplotlib.axes.Axes`

        Warnings
        --------
        Calling this method will automatically trigger a call to the :class:`argopy.IndexFetcher.load` method.

        """
        self.load()
        if ptype in ["dac", "institution"]:
            return bar_plot(self.index, by="institution", **kwargs)
        elif ptype == "profiler":
            return bar_plot(self.index, by="profiler", **kwargs)
        elif ptype == "trajectory":
            defaults = {"style": "white"}
            return plot_trajectory(
                self.index.sort_values(["file"]), **{**defaults, **kwargs}
            )
        elif ptype == "qc_altimetry":
            WMOs = np.unique(self.index["wmo"])
            return open_sat_altim_report(WMOs, **kwargs)
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'trajectory', 'dac', 'profiler', 'qc_altimetry'"
            )

    def clear_cache(self):
        """Clear fetcher cached data"""
        return self.fetcher.clear_cache()
