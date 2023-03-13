"""
This file covers the plotters module
We test plotting functions from IndexFetcher and DataFetcher
"""
import pytest
import logging
from typing import Callable

import argopy
from argopy.errors import InvalidDashboard
from utils import (
    requires_gdac,
    requires_localftp,
    requires_connection,
    requires_matplotlib,
    requires_ipython,
    requires_cartopy,
    has_matplotlib,
    has_seaborn,
    has_cartopy,
    has_ipython,
    has_ipywidgets,
)
from ..plot import bar_plot, plot_trajectory, open_sat_altim_report, scatter_map
from argopy import DataFetcher as ArgoDataFetcher

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

if has_ipython:
    import IPython

log = logging.getLogger("argopy.tests.plot")


