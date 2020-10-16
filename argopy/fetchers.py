#!/bin/env python
# -*coding: UTF-8 -*-
"""

High level helper methods to load Argo data from any source
The facade should be able to work with all available data access point,

"""

import warnings

from argopy.options import OPTIONS, _VALIDATORS
from .errors import InvalidFetcherAccessPoint, InvalidFetcher
from .utilities import list_available_data_src, list_available_index_src, is_box
from .plotters import plot_trajectory, plot_dac, plot_profilerType

AVAILABLE_DATA_SOURCES = list_available_data_src()
AVAILABLE_INDEX_SOURCES = list_available_index_src()


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
            raise ValueError("%s dataset is not available for this data source (%s)" % (self._dataset_id, self._src))
        self.fetcher_options = {**{"ds": self._dataset_id}, **fetcher_kwargs}
        self.postproccessor = self.__empty_processor
        self._AccessPoint = None

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
            if 'parallel' in self.fetcher_options:
                summary.append("Backend: %s (parallel=%s)" % (self._src, str(self.fetcher_options['parallel'])))
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
        valid_attrs = ["Fetchers", "fetcher", "fetcher_options", "postproccessor", "dashboard"]
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

    def dashboard(self, **kw):
        try:
            return self.fetcher.dashboard(**kw)
        except Exception:
            warnings.warn(
                "dashboard not available for this fetcher access point (%s/%s)"
                % (self._src, self._AccessPoint)
            )

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

        if "float" in self.Fetchers:
            self.fetcher = self.Fetchers["float"](WMO=wmo, **self.fetcher_options)
            self._AccessPoint = "float"  # Register the requested access point
        else:
            raise InvalidFetcherAccessPoint(
                "'float' not available with '%s' src" % self._src
            )

        if self._mode == "standard" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing
        return self

    def profile(self, wmo, cyc):
        """ Specific profile data fetcher

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
        if "profile" in self.Fetchers:
            self.fetcher = self.Fetchers["profile"](
                WMO=wmo, CYC=cyc, **self.fetcher_options
            )
            self._AccessPoint = "profile"  # Register the requested access point
        else:
            raise InvalidFetcherAccessPoint(
                "'profile' not available with '%s' src" % self._src
            )

        if self._mode == "standard" and self._dataset_id != "ref":

            def postprocessing(xds):
                xds = self.fetcher.filter_data_mode(xds)
                xds = self.fetcher.filter_qc(xds)
                xds = self.fetcher.filter_variables(xds, self._mode)
                return xds

            self.postproccessor = postprocessing

        return self

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
        if "region" in self.Fetchers:
            self.fetcher = self.Fetchers["region"](box=box, **self.fetcher_options)
            self._AccessPoint = "region"  # Register the requested access point
        else:
            raise InvalidFetcherAccessPoint(
                "'region' not available with '%s' src" % self._src
            )

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
            :class:`xarray.DataArray`
        """
        if not self.fetcher:
            raise InvalidFetcher(" Initialize an access point (%s) first." %
                                 ",".join(self.Fetchers.keys()))
        xds = self.fetcher.to_xarray(**kwargs)
        xds = self.postproccessor(xds)
        return xds

    def to_dataframe(self, **kwargs):
        """  Fetch and return data as pandas.Dataframe """
        if not self.fetcher:
            raise InvalidFetcher(" Initialize an access point (%s) first." %
                                 ",".join(self.Fetchers.keys()))
        return self.to_xarray(**kwargs).to_dataframe()

    def clear_cache(self):
        """ Clear data cached by fetcher """
        if not self.fetcher:
            raise InvalidFetcher(" Initialize an access point (%s) first." %
                                 ",".join(self.Fetchers.keys()))
        return self.fetcher.clear_cache()


class ArgoIndexFetcher:
    """
    Specs discussion :
    https://github.com/euroargodev/argopy/issues/8
    https://github.com/euroargodev/argopy/pull/6)

    Usage:

    from argopy import ArgoIndexFetcher
    idx = ArgoIndexFetcher.region([-75, -65, 10, 20])
    idx.plot.trajectories()
    idx.to_dataframe()

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
            raise ValueError("%s dataset is not available for this index source (%s)" % (self._dataset_id, self._src))
        self.fetcher_options = {**fetcher_kwargs}
        self.postproccessor = self.__empty_processor
        self._AccessPoint = None

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
        valid_attrs = ["Fetchers", "fetcher", "fetcher_options", "postproccessor"]
        if key not in self.valid_access_points and key not in valid_attrs:
            raise InvalidFetcherAccessPoint("'%s' is not a valid access point" % key)
        pass

    def profile(self, wmo, cyc):
        """ Fetch index for a profile

            given one or more WMOs and CYCLE_NUMBER
        """
        if "profile" in self.Fetchers:
            self.fetcher = self.Fetchers["profile"](
                WMO=wmo, CYC=cyc, **self.fetcher_options
            )
            self._AccessPoint = "profile"  # Register the requested access point
        else:
            raise InvalidFetcherAccessPoint(
                "'profile' not available with '%s' src" % self._src
            )
        return self

    def float(self, wmo):
        """ Load index for one or more WMOs """
        if "float" in self.Fetchers:
            self.fetcher = self.Fetchers["float"](WMO=wmo, **self.fetcher_options)
            self._AccessPoint = "float"  # Register the requested access point
        else:
            raise InvalidFetcherAccessPoint(
                "'float' not available with '%s' src" % self._src
            )
        return self

    def region(self, box):
        """ Load index for a rectangular space/time domain region """
        if "region" in self.Fetchers:
            self.fetcher = self.Fetchers["region"](box=box, **self.fetcher_options)
            self._AccessPoint = "region"  # Register the requested access point
        else:
            raise InvalidFetcherAccessPoint(
                "'region' not available with '%s' src" % self._src
            )
        return self

    def to_dataframe(self, **kwargs):
        """ Fetch index and return pandas.Dataframe """
        if not self.fetcher:
            raise InvalidFetcher(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.fetcher.to_dataframe(**kwargs)

    def to_xarray(self, **kwargs):
        """ Fetch index and return xr.dataset """
        if self._AccessPoint not in self.valid_access_points:
            raise InvalidFetcherAccessPoint(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.fetcher.to_xarray(**kwargs)

    def to_csv(self, file: str = "output_file.csv"):
        """ Fetch index and return csv """
        if self._AccessPoint not in self.valid_access_points:
            raise InvalidFetcherAccessPoint(
                " Initialize an access point (%s) first."
                % ",".join(self.Fetchers.keys())
            )
        return self.to_dataframe().to_csv(file)

    def plot(self, ptype="trajectory"):
        """ Create custom plots from index

            Parameters
            ----------
            ptype: str
                Type of plot to generate. This can be: 'trajectory',' profiler', 'dac'.

            Returns
            -------
            fig : :class:`matplotlib.pyplot.figure.Figure`
                Figure instance
        """
        idx = self.to_dataframe()
        if ptype == "dac":
            return plot_dac(idx)
        elif ptype == "profiler":
            return plot_profilerType(idx)
        elif ptype == "trajectory":
            return plot_trajectory(idx.sort_values(["file"]))
        else:
            raise ValueError(
                "Type of plot unavailable. Use: 'dac', 'profiler' or 'trajectory' (default)"
            )

    def clear_cache(self):
        """ Clear fetcher cached data """
        return self.fetcher.clear_cache()
