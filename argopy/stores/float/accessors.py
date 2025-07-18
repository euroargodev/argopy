import numpy as np
from typing import NoReturn, Any

from ...plot import scatter_plot, scatter_map
from ...utils.lists import list_multiprofile_file_variables, list_bgc_s_variables


class ArgoFloatPlotAccessor:
    """
    Enables use of argopy.plot functions as attributes on a ArgoFloat.

    Examples
    --------
    .. code-block:: python

        from argopy import ArgoFloat

        af = ArgoFloat(wmo)

        af.plot.trajectory()

        af.plot.map('TEMP', pres=450, cmap='Spectral_r')

        af.plot.map('DATA_MODE', cbar=False, legend=True)

        af.plot.scatter('PSAL')

        af.plot.scatter('MEASUREMENT_CODE', ds='Rtraj')

    """

    _af: None
    __slots__ = ("_af")

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
        """Quick map of the float trajectory"""

        fig, ax, hdl = self.map('CYCLE_NUMBER', cmap='Spectral_r', **kwargs)
        ax.set_title(self._default_title)
        return fig, ax, hdl

    def map(self, param, ds='prof', pres=0, pres_bin_size=100., select='shallow', **kwargs) -> Any:
        """Scatter map for one dataset parameter sliced at a given pressure"""

        if ds == 'prof' and param not in list_multiprofile_file_variables():
            raise ValueError("'%s' variables is not available in the 'prof' dataset file")
        if ds == 'Sprof' and param not in list_bgc_s_variables():
            raise ValueError("'%s' variables is not available in the 'Sprof' dataset file")
        if ds not in self._af._dataset:
            self._af._dataset[ds] = self._af.open_dataset(ds)

        this_ds = self._af._dataset[ds]
        # Slice dataset to the appropriate level:
        if pres == 0:
            bins = [0., pres + pres_bin_size, 10000.]
        else:
            bins = [np.max([0, pres - pres_bin_size / 2]), pres + pres_bin_size / 2, 10000.]
        this_ds = this_ds.argo.groupby_pressure_bins(bins=bins, select=select).isel(N_LEVELS=0)

        default_kwargs = {'x': 'LONGITUDE',
                          'y': 'LATITUDE',
                          'hue': param,
                          'legend': False,
                          'cbar': True}
        this_kwargs = {**default_kwargs, **kwargs}

        if this_kwargs['hue'] not in this_ds:
            raise ValueError("The parameter to map must be a variable in %s" % str([c for c in this_ds.data_vars]))

        fig, ax, hdl = scatter_map(this_ds, **this_kwargs)
        ax.set_title(self._default_title)
        return fig, ax, hdl

    def scatter(self, param, ds='prof', **kwargs) -> Any:
        """Scatter plot for one dataset parameter"""

        if ds == 'prof' and param not in list_multiprofile_file_variables():
            raise ValueError("'%s' variables is not available in the 'prof' dataset file")
        if ds == 'Sprof' and param not in list_bgc_s_variables():
            raise ValueError("'%s' variables is not available in the 'Sprof' dataset file")
        if ds not in self._af._dataset:
            self._af._dataset[ds] = self._af.open_dataset(ds)

        default_kwargs = {'this_x': 'JULD', 'cbar': True}
        this_kwargs = {**default_kwargs, **kwargs}

        if this_kwargs['cbar']:
            fig, ax, m, cbar = scatter_plot(self._af._dataset[ds], param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m, cbar
        else:
            fig, ax, m = scatter_plot(self._af._dataset[ds], param, **this_kwargs)
            ax.set_title(self._default_title)
            return fig, ax, m
