import numpy as np
from typing import Any

from ....plot import scatter_plot, scatter_map, ArgoColors
from ....utils import list_multiprofile_file_variables, list_bgc_s_variables, to_list
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

        af.plot.map('DATA_MODE')

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
        param : str,
        ds : str ="prof",
        pres : float = 0.0,
        pres_axis : str = 'PRES',
        pres_bin_size : float = 100.0,
        select : str = "middle",
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
        pres: float, optional, default=0
            If the parameter has a N_LEVELS dimension, this is the pressure value to slice the vertical dimension of the parameter to plot.

            - If `pres=0`, we plot parameter value at the shallowest pressure level.
            - If `pres=-1`, we plot parameter value at the deepest pressure level.
            - If `pres` is anything positive, we plot parameter value at the nearest pressure level (approximately, see below)

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
        pres_axis: str, default='PRES'
            If the parameter has N_LEVELS dimension, this is the pressure variable to use for slicing.
        pres_bin_size: float, default=100.
            Pressure bin size deeper than the `pres` argument to consider in slicing the parameter to plot.
        select: str, default='middle'
            Pressure bin parameter value selection method. This is directly passed to :meth:`xarray.Dataset.argo.groupby_pressure_bins`.

        Notes
        -----
        You can adjust the map aspect ratio with the ``figsize`` argument, e.g.: (18,18)

        You can also adjust space around the trajectory with the ``padding`` argument, e.g.: [1, 5]

        If a target pressure value is provided with the `pres` argument, we plot the parameter value at the 'middle' pressure axis value for a bin centered around the target. Bin size is controlled with the `pres_bin_size` argument.

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            af = ArgoFloat(wmo)

            af.plot.map('TEMP')  # Plot pressure level closest to 0 by default
            af.plot.map('PSAL', pres=450.)
            af.plot.map('PROFILE_TEMP_QC')
            af.plot.map('DATA_MODE')
        """

        if ds == "prof" and param not in list_multiprofile_file_variables():
            raise ValueError(
                f"'{param}' variables is not available in the 'prof' dataset file"
            )
        if ds == "Sprof" and param not in list_bgc_s_variables():
            raise ValueError(
                f"'{param}' variables is not available in the 'Sprof' dataset file"
            )
        this_ds = self._obj.dataset(ds)
        if param not in this_ds:
            raise ValueError(
                "'%s' parameter is not available in the '%s' dataset (%s)"
                % (param, ds, self._obj.ls_dataset()[ds])
            )
        param_toplot : str = param

        # Slice dataset to the appropriate level, if necessary:
        if 'N_LEVELS' in this_ds[param].dims:
            if pres == 0 or pres == -1:
                if pres == 0:
                    def reducer(p, y):
                        """shallowest_value"""
                        for data in zip(p, y):
                            if ~np.isnan(data[0]):
                                return data[1]
                elif pres == -1:
                    def reducer(p, y):
                        """deepest_value"""
                        for data in zip(p[::-1], y[::-1]):
                            if ~np.isnan(data[0]):
                                return data[1]
                this_ds[f"_{param}_slice"] = this_ds.argo.reduce_profile(reducer, params=[pres_axis, param])
                param_toplot = f"_{param}_slice"

            else:
                bins = [
                    max([0, pres - pres_bin_size / 2]),
                    pres + pres_bin_size / 2,
                    10000.0,
                ]
                this_ds = this_ds.argo.groupby_pressure_bins(bins=bins, select=select, axis=pres_axis).isel(
                    N_LEVELS=0
                )

        # Check if param will be plotted using a discrete and known Argo colormap
        discrete, cmap = False, "Spectral_r"
        if "qc" in param.lower() or "mode" in param.lower():
            discrete, cmap = True, None  # Let scatter_map guess cmap

        if "N_LEVELS" in self._obj.dataset(ds)[param].dims:
            if pres == 0:
                legend_title = f"{param} @ shallowest {pres_axis} level"
            elif pres == -1:
                legend_title = f"{param} @ deepest {pres_axis} level"
            else:
                legend_title = "%s @ %s %s level in [%0.1f-%0.1f] db" % (
                    param,
                    select,
                    pres_axis,
                    bins[0],
                    bins[1],
                )
        else:
            legend_title = param

        default_kwargs = {
            "x": "LONGITUDE",
            "y": "LATITUDE",
            "hue": param_toplot,
            "cmap": cmap,
            "legend": True if discrete else False,
            "cbar": False if discrete else True,
            "legend_title": legend_title,
        }
        this_kwargs = {**default_kwargs, **kwargs}

        fig, ax, hdl = scatter_map(this_ds, **this_kwargs)
        ax.set_title(self._default_title)

        # Clean-up the dataset if we added a special slice of variable
        if param_toplot.startswith("_") and "slice" in param:
            this_ds.drop_vars(param_toplot)

        return fig, ax, hdl

    def scatter(self, param, ds="prof", **kwargs) -> Any:
        """Scatter plot for a 2-dimensional dataset parameter

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
                - :class:`matplotlib.colorbar.Colorbar`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoFloat
            af = ArgoFloat(wmo)
            af.plot.scatter('TEMP')
            af.plot.scatter('PSAL_QC')  # Appropriate colormap automatically selected
            af.plot.scatter('PRES', x='PSAL', y='TEMP')
            af.plot.scatter('DOXY', ds='Sprof')
            af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

        """

        if ds == "prof" and param not in list_multiprofile_file_variables():
            raise ValueError(
                f"'{param}' variables is not available in the 'prof' dataset file"
            )
        if ds == "Sprof" and param not in list_bgc_s_variables():
            raise ValueError(
                f"'{param}' variables is not available in the 'Sprof' dataset file"
            )
        this_ds = self._obj.dataset(ds)
        if param not in this_ds:
            raise ValueError(
                "'%s' parameter is not available in the '%s' dataset (%s)"
                % (param, ds, self._obj.ls_dataset()[ds])
            )

        default_kwargs = {"x": "JULD", "cbar": True}
        this_kwargs = {**default_kwargs, **kwargs}

        if this_kwargs["cbar"]:
            fig, ax, m, cbar = scatter_plot(this_ds, param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m, cbar
        else:
            fig, ax, m = scatter_plot(this_ds, param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m
