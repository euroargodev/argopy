"""
Plot Submodule

See Also
--------
:func:`argopy.plot.scatter_map`, :func:`argopy.plot.scatter_plot`, :func:`argopy.plot.plot_trajectory`, :func:`argopy.plot.dashboard`, :class:`argopy.plot.ArgoColors`, :class:`argopy.ArgoColors`, :func:`argopy.plot.latlongrid`


"""
from .plot import plot_trajectory, bar_plot, open_sat_altim_report, scatter_map, scatter_plot
from .argo_colors import ArgoColors
from .dashboards import open_dashboard as dashboard
from .utils import latlongrid, ARGOPY_COLORS


__all__ = (
    # Also available on the argopy module level:
    "dashboard",
    "ArgoColors",

    # Plot:
    "plot_trajectory",
    "bar_plot",
    "scatter_map",
    "scatter_plot",
    "open_sat_altim_report",

    # Utils:
    "latlongrid",
    "ARGOPY_COLORS",
)
