import numpy as np
from contextlib import contextmanager
import importlib


def _importorskip(modname):
    try:
        importlib.import_module(modname)  # noqa: E402
        has = True
    except ImportError:
        has = False
    return has


has_mpl = _importorskip("matplotlib")
has_cartopy = _importorskip("cartopy")
has_seaborn = _importorskip("seaborn")
has_ipython = _importorskip("IPython")
has_ipywidgets = _importorskip("ipywidgets")

STYLE = {"axes": "whitegrid", "palette": "Set1"}  # Default styles


if has_mpl:
    import matplotlib as mpl  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401
    import matplotlib.ticker as mticker
    import matplotlib.cm as cm  # noqa: F401
    import matplotlib.colors as mcolors  # noqa: F401


if has_cartopy:
    import cartopy
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    land_feature = cfeature.NaturalEarthFeature(
        category="physical", name="land", scale="50m", facecolor=[0.4, 0.6, 0.7]
    )
else:
    land_feature = ()

if has_seaborn:
    STYLE["axes"] = "dark"
    import seaborn as sns


@contextmanager
def axes_style(style: str = STYLE["axes"]):
    """ Provide a context for plots

        The point is to handle the availability of :mod:`seaborn` or not and to be able to use::

            with axes_style(style):
                fig, ax = plt.subplots()

        in all situations.
    """
    if has_seaborn:  # Execute within a seaborn context:
        with sns.axes_style(style):
            yield
    else:  # Otherwise do nothing
        yield


def latlongrid(ax, dx="auto", dy="auto", fontsize="auto", label_style_arg={}, **kwargs):
    """ Add latitude/longitude grid line and labels to a cartopy geoaxes

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxesSubplot
        Cartopy axes to add the lat/lon grid to
    dx: 'auto' or float
        Grid spacing along longitude
    dy: 'auto' or float
        Grid spacing along latitude
    fontsize: 'auto' or int
        Grid label font size

    Returns
    -------
    class:`cartopy.mpl.geoaxes.GeoAxesSubplot.gridlines`
    """
    if not isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
        raise ValueError("Please provide a cartopy.mpl.geoaxes.GeoAxesSubplot instance")
    defaults = {"linewidth": 0.5, "color": "gray", "alpha": 0.5, "linestyle": ":"}
    gl = ax.gridlines(crs=ax.projection, draw_labels=True, **{**defaults, **kwargs})
    if dx != "auto":
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180 + 1, dx))
    if dy != "auto":
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90 + 1, dy))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # Cartopy <= 0.18:
    # gl.xlabels_top = False
    # gl.ylabels_right = False
    # Cartopy >= 0.18:
    gl.top_labels = False
    gl.right_labels = False
    label_style_arg_defaults = {"fontsize": None}
    if fontsize != "auto":
        label_style_arg_defaults = {"fontsize": fontsize}

    gl.xlabel_style = {**label_style_arg_defaults, **label_style_arg}
    gl.ylabel_style = {**label_style_arg_defaults, **label_style_arg}
    return gl
