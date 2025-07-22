import numpy as np
from typing import NoReturn, Any

from ...plot import scatter_plot, scatter_map
from ...utils.lists import list_multiprofile_file_variables, list_bgc_s_variables


class ArgoFloatPlotAccessor:
    """
    Enables use of :module:`argopy.plot` functions as attributes on a ArgoFloat.

    Examples
    --------
    .. code-block:: python
        :caption: Examples of plotting methods

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
    :class:`ArgoFloat.plot.trajectory`, :class:`ArgoFloat.plot.map`, :class:`ArgoFloat.plot.scatter`

    """

    _af: None
    __slots__ = "_af"

    def __init__(self, af) -> None:
        self._af = af

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ArgoFloat.plot.trajectory(...)"
        )

    @property
    def _default_title(self):
        return "Argo float WMO: %s" % self._af.WMO

    def trajectory(self, **kwargs) -> Any:
        """Quick map of the float trajectory

        This is a pre-defined call to the :meth:`ArgoFloat.plot.map` method like:

        .. code-block:: python

            Argofloat(WMO).plot.trajectory(**kwargs)
            # is similar to:
            Argofloat(WMO).plot.map('CYCLE_NUMBER', cmap='Spectral_r', **kwargs)

        Parameters
        ----------
        All arguments are passed to :meth:`ArgoFloat.plot.map`.

        Returns
        -------
        Output from :meth:`ArgoFloat.plot.map`, typically:

            fig: :class:`matplotlib.figure.Figure`
            ax: :class:`matplotlib.axes.Axes`
            patches: Dict with ax collections

        Notes
        -----
        You can adjust the map aspect ratio with the `figsize` argument, e.g.: (18,18)

        You can also adjust space around the trajectory with the `padding` argument, e.g.: [1, 5]
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
            Argo dataset (netcdf file) to use to load the parameter to plot.
        pres: float, default=0.
            If the parameter has a N_LEVELS dimension, this is the pressure value to slice the vertical dimension of the parameter to plot.

            We use the :class:`Dataset.argo.groupby_pressure_bins` method to select the parameter value the closest to the pressure target value.

        Returns
        -------
        Output from :class:`argopy.plot.scatter_map`, typically:

            fig: :class:`matplotlib.figure.Figure`
            ax: :class:`matplotlib.axes.Axes`
            patches: Dict with ax collections

        Other Parameters
        ----------------
        pres_bin_size: float, default=100.
            Pressure bin size deeper than the `pres` argument to consider in slicing the parameter to plot.
        select: str, default='shallow'
            Pressure bin parameter value selection method. This is directly passed to :class:`Dataset.argo.groupby_pressure_bins`.

        Notes
        -----
        You can adjust the map aspect ratio with the `figsize` argument, e.g.: (18,18)

        You can also adjust space around the trajectory with the `padding` argument, e.g.: [1, 5]

        """

        if ds == "prof" and param not in list_multiprofile_file_variables():
            raise ValueError(
                "'%s' variables is not available in the 'prof' dataset file"
            )
        if ds == "Sprof" and param not in list_bgc_s_variables():
            raise ValueError(
                "'%s' variables is not available in the 'Sprof' dataset file"
            )
        if ds not in self._af._dataset:
            self._af._dataset[ds] = self._af.open_dataset(ds)
        if param not in self._af._dataset[ds]:
            raise ValueError(
                "'%s' parameter is not available in the '%s' dataset (%s)"
                % (param, ds, self._af.ls_dataset()[ds])
            )

        this_ds = self._af._dataset[ds]
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
            Argo dataset (netcdf file) to use to load the parameter to plot.
        pres: float, default=0.
            If the parameter has a N_LEVELS dimension, this is the pressure value to slice the vertical dimension of the parameter to plot.

            We use the :class:`Dataset.argo.groupby_pressure_bins` method to select the parameter value the closest to the pressure target value.

        Returns
        -------
        Output from :meth:`argopy.plot.scatter_plot`, typically:

            fig: :class:`matplotlib.figure.Figure`
            ax: :class:`matplotlib.axes.Axes`
            patches
        """

        if ds == "prof" and param not in list_multiprofile_file_variables():
            raise ValueError(
                "'%s' variables is not available in the 'prof' dataset file"
            )
        if ds == "Sprof" and param not in list_bgc_s_variables():
            raise ValueError(
                "'%s' variables is not available in the 'Sprof' dataset file"
            )
        if ds not in self._af._dataset:
            self._af._dataset[ds] = self._af.open_dataset(ds)
        if param not in self._af._dataset[ds]:
            raise ValueError(
                "'%s' parameter is not available in the '%s' dataset (%s)"
                % (param, ds, self._af.ls_dataset()[ds])
            )

        default_kwargs = {"this_x": "JULD", "cbar": True}
        this_kwargs = {**default_kwargs, **kwargs}

        if this_kwargs["cbar"]:
            fig, ax, m, cbar = scatter_plot(self._af._dataset[ds], param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m, cbar
        else:
            fig, ax, m = scatter_plot(self._af._dataset[ds], param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m
