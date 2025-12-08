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

STYLE = {"axes": "argopy", "palette": "Set1"}  # Default styles
ARGOPY_COLORS = {
    "CYAN": (18 / 256, 235 / 256, 229 / 256),
    "BLUE": (16 / 256, 137 / 256, 182 / 256),
    "DARKBLUE": (10 / 256, 89 / 256, 162 / 256),
    "YELLOW": (229 / 256, 174 / 256, 41 / 256),
    "DARKYELLOW": (224 / 256, 158 / 256, 37 / 256),
}
ARGOPY_STYLE = {
    "axes.facecolor": "white",
    "axes.edgecolor": ARGOPY_COLORS["DARKBLUE"],
    "axes.grid": True,
    "axes.axisbelow": "line",
    "axes.labelcolor": ARGOPY_COLORS["BLUE"],
    "figure.facecolor": "white",
    "grid.color": ARGOPY_COLORS["DARKBLUE"],
    "grid.linestyle": ":",
    "text.color": ARGOPY_COLORS["DARKBLUE"],
    "xtick.color": ARGOPY_COLORS["BLUE"],
    "ytick.color": ARGOPY_COLORS["BLUE"],
    # 'lines.solid_capstyle': <CapStyle.projecting: 'projecting'>,
    "patch.edgecolor": "black",
    "patch.force_edgecolor": False,
    "image.cmap": "viridis",
    "font.family": ["sans-serif"],
    "font.sans-serif": [
        "DejaVu Sans",
        "Bitstream Vera Sans",
        "Computer Modern Sans Serif",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Arial",
        "Helvetica",
        "Avant Garde",
        "sans-serif",
    ],
    "xtick.bottom": True,
    "xtick.top": False,
    "ytick.left": True,
    "ytick.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
}

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
    # STYLE["axes"] = "dark"
    import seaborn as sns


@contextmanager
def axes_style(style: str = STYLE["axes"]):
    """Provide a context for plots

    The point is to handle the availability of :mod:`seaborn` or not and to be able to use::

        with axes_style(style):
            fig, ax = plt.subplots()

    in all situations.
    """
    if has_seaborn:  # Execute within a seaborn context:
        if style == "argopy":
            with sns.axes_style('whitegrid', rc=ARGOPY_STYLE):
                yield
        else:
            with sns.axes_style(style):
                yield
    else:  # Otherwise do nothing
        yield


def latlongrid(
    ax,
    dx="auto",
    dy="auto",
    fontsize="auto",
    label_style_arg={},
    **kwargs
):
    """Add latitude/longitude grid line and labels to a cartopy geoaxes

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
    defaults = {"linewidth": 0.5, "color": ARGOPY_COLORS['BLUE'], "alpha": 0.5, "linestyle": ":"}
    gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=True, **{**defaults, **kwargs})
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
