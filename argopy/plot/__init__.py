from .plot import plot_trajectory, bar_plot, open_sat_altim_report, scatter_map
from .argo_colors import ArgoColors
from .dashboards import open_dashboard as dashboard
from .utils import discrete_coloring, latlongrid

__all__ = (
    # Also available on the argopy module level:
    "dashboard",
    "ArgoColors",

    # Plot:
    "plot_trajectory",
    "bar_plot",
    "scatter_map",
    "open_sat_altim_report",

    # Utils:
    "latlongrid",
    "discrete_coloring",
)