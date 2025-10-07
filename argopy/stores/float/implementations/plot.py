import numpy as np
from typing import Any

from ....plot import scatter_plot, scatter_map
from ....utils.lists import list_multiprofile_file_variables, list_bgc_s_variables
from ..extensions import ArgoFloatPlotProto


class ArgoFloatPlot(ArgoFloatPlotProto):
    """Extension providing plot methods

    Examples
    --------
    .. code-block:: python
        :caption: Examples of :class:`ArgoFloat` plotting methods

        from argopy import ArgoFloat

        af = ArgoFloat(wmo)

        af.plot.trajectory()

        af.plot.trajectory(figsize=(18,18), padding=[1, 5])

        af.plot.map('TEMP', pres=450, cmap='Spectral_r')

        af.plot.map('DATA_MODE', cbar=False, legend=True)

        af.plot.scatter('PSAL')

        af.plot.scatter('DOXY', ds='Sprof')

        af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

    See Also
    --------
    :meth:`ArgoFloat.plot.trajectory`, :meth:`ArgoFloat.plot.map`, :meth:`ArgoFloat.plot.scatter`

    Notes
    -----
    This extension works with both the offline and online ArgoFloat implementations. It is based on data downloaded using the :meth:`ArgoFloat.open_dataset` method.

    """

    @property
    def _default_title(self):
        return "Argo float WMO: %s" % self._obj.WMO

    def trajectory(self, **kwargs) -> Any:
        """Quick map of the float trajectory

        This is a pre-defined call to the :meth:`ArgoFloat.plot.map` method like:

        .. code-block:: python

            Argofloat(WMO).plot.trajectory(**kwargs)
            # is similar to:
            Argofloat(WMO).plot.map('CYCLE_NUMBER', cmap='Spectral_r', **kwargs)

        Parameters
        ----------
        **kwargs
            All arguments are passed to :meth:`ArgoFloat.plot.map`.

        Returns
        -------
        tuple
            Output from :class:`argopy.plot.scatter_map`, typically:
                - fig: :class:`matplotlib.figure.Figure`
                - ax: :class:`matplotlib.axes.Axes`
                - patches: Dict with ax collections

        Notes
        -----
        You can adjust the map aspect ratio with the ``figsize`` argument, e.g.: (18,18)

        You can also adjust space around the trajectory with the ``padding`` argument, e.g.: [1, 5]

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat

            af = ArgoFloat(wmo)
            af.plot.trajectory()
            af.plot.trajectory(figsize=(18,18), padding=[1, 5])
        """
        # todo: Load (cyc,lat,lon) from an API call or index, much faster than a netcdf dataset

        fig, ax, hdl = self.map("CYCLE_NUMBER", cmap="Spectral_r", **kwargs)
        ax.set_title(self._default_title)
        return fig, ax, hdl

    def map(
        self,
        param,
        ds="prof",
        pres=0.0,
        pres_bin_size=100.0,
        select="shallow",
        **kwargs
    ) -> Any:
        """Plot a map of one dataset parameter, possibly sliced at a given pressure value.

        This method creates a 2D geo-projection-based plot with the :meth:`argopy.plot.scatter_map` method.

        Parameters
        ----------
        param : str
            Name of the dataset parameter to map.
        ds: str, default='prof'
            Argo dataset name to load the parameter to plot. Must be valid key from :meth:`ArgoFloat.ls_dataset`.
        pres: float, default=0.
            If the parameter has a N_LEVELS dimension, this is the pressure value to slice the vertical dimension of the parameter to plot.

            We use the :meth:`xarray.Dataset.argo.groupby_pressure_bins` method to select the parameter value the closest to the pressure target value.

        Returns
        -------
        tuple
            Output from :class:`argopy.plot.scatter_map`, typically:
                - fig: :class:`matplotlib.figure.Figure`
                - ax: :class:`matplotlib.axes.Axes`
                - patches: Dict with ax collections

        Other Parameters
        ----------------
        pres_bin_size: float, default=100.
            Pressure bin size deeper than the `pres` argument to consider in slicing the parameter to plot.
        select: str, default='shallow'
            Pressure bin parameter value selection method. This is directly passed to :meth:`xarray.Dataset.argo.groupby_pressure_bins`.

        Notes
        -----
        You can adjust the map aspect ratio with the ``figsize`` argument, e.g.: (18,18)

        You can also adjust space around the trajectory with the ``padding`` argument, e.g.: [1, 5]

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            af = ArgoFloat(wmo)
            af.plot.map('TEMP', pres=450, cmap='Spectral_r')
            af.plot.map('DATA_MODE', cbar=False, legend=True)
        """

        if ds == "prof" and param not in list_multiprofile_file_variables():
            raise ValueError(
                "'%s' variables is not available in the 'prof' dataset file"
            )
        if ds == "Sprof" and param not in list_bgc_s_variables():
            raise ValueError(
                "'%s' variables is not available in the 'Sprof' dataset file"
            )
        this_ds = self._obj.dataset(ds)
        if param not in this_ds:
            raise ValueError(
                "'%s' parameter is not available in the '%s' dataset (%s)"
                % (param, ds, self._obj.ls_dataset()[ds])
            )

        # Slice dataset to the appropriate level:
        if pres == 0:
            bins = [0.0, pres + pres_bin_size, 10000.0]
        else:
            bins = [
                np.max([0, pres - pres_bin_size / 2]),
                pres + pres_bin_size / 2,
                10000.0,
            ]
        this_ds = this_ds.argo.groupby_pressure_bins(bins=bins, select=select).isel(
            N_LEVELS=0
        )

        default_kwargs = {
            "x": "LONGITUDE",
            "y": "LATITUDE",
            "hue": param,
            "legend": False,
            "cbar": True,
        }
        this_kwargs = {**default_kwargs, **kwargs}

        fig, ax, hdl = scatter_map(this_ds, **this_kwargs)
        ax.set_title(self._default_title)
        return fig, ax, hdl

    def scatter(self, param, ds="prof", **kwargs) -> Any:
        """Scatter plot for one dataset parameter

        This method creates a 2D scatter plot with the :meth:`argopy.plot.scatter_plot` method.

        Parameters
        ----------
        param : str
            Name of the dataset parameter to map.
        ds: str, default='prof'
            Argo dataset name to load the parameter to plot. Must be valid key from :meth:`ArgoFloat.ls_dataset`.

        Returns
        -------
        tuple
            Output from :meth:`argopy.plot.scatter_plot`, typically:
                - :class:`matplotlib.figure.Figure`
                - :class:`matplotlib.axes.Axes`
                - list of patches

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            af = ArgoFloat(wmo)
            af.plot.scatter('PSAL')
            af.plot.scatter('DOXY', ds='Sprof')
            af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

        """

        if ds == "prof" and param not in list_multiprofile_file_variables():
            raise ValueError(
                "'%s' variables is not available in the 'prof' dataset file"
            )
        if ds == "Sprof" and param not in list_bgc_s_variables():
            raise ValueError(
                "'%s' variables is not available in the 'Sprof' dataset file"
            )
        this_ds = self._obj.dataset(ds)
        if param not in this_ds:
            raise ValueError(
                "'%s' parameter is not available in the '%s' dataset (%s)"
                % (param, ds, self._obj.ls_dataset()[ds])
            )

        default_kwargs = {"this_x": "JULD", "cbar": True}
        this_kwargs = {**default_kwargs, **kwargs}

        if this_kwargs["cbar"]:
            fig, ax, m, cbar = scatter_plot(this_ds, param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m, cbar
        else:
            fig, ax, m = scatter_plot(this_ds, param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m
