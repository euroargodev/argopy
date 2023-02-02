import importlib
from .plot import plot_trajectory, bar_plot, open_sat_altim_report
from .argo_colors import ArgoColors
from .dashboards import open_dashboard as dashboard


__all__ = (
    "dashboard",
    "ArgoColors",
    "plot_trajectory",
    "bar_plot",
    "open_sat_altim_report",
)
