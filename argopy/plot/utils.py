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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors


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


class discrete_coloring:
    """ Handy class to manage discrete coloring and the associated colorbar

    Example
    -------
    This class can be used like this::

        year_range = np.arange(2002,2010)
        dc = discrete_coloring(name='Spectral', N=len(year_range) )
        plt.scatter(this['LONGITUDE'], this['LATITUDE'], c=this['TIME.year'],
                    cmap=dc.cmap, vmin=year_range[0], vmax=year_range[-1])
        dc.cbar(ticklabels=yr_range, fraction=0.03, label='Years')

    """

    def __init__(self, name="Set1", N=12):
        """

        Parameters
        ----------
        name: str
            Name if the colormap to use. Default: 'Set1'
        N: int
            Number of colors to reduce the colormap to. Default: 12
        """
        self.name = name
        self.Ncolors = N

    @property
    def cmap(self):
        """Return a discrete colormap from a quantitative or continuous colormap name

        Returns
        -------
        :class:`matplotlib.colors.LinearSegmentedColormap`
        """
        name = self.name
        K = self.Ncolors
        if name in [
            "Set1",
            "Set2",
            "Set3",
            "Pastel1",
            "Pastel2",
            "Paired",
            "Dark2",
            "Accent",
        ]:
            # Segmented (or quantitative) colormap:
            N_ref = {
                "Set1": 9,
                "Set2": 8,
                "Set3": 12,
                "Pastel1": 9,
                "Pastel2": 8,
                "Paired": 12,
                "Dark2": 8,
                "Accent": 8,
            }
            N = N_ref[name]
            cmap = plt.get_cmap(name=name)
            colors_i = np.concatenate(
                (np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)), axis=0
            )
            cmap = cmap(colors_i)  # N x 4
            n = np.arange(0, N)
            new_n = n.copy()
            if K > N:
                for k in range(N, K):
                    r = np.roll(n, -k)[0][np.newaxis]
                    new_n = np.concatenate((new_n, r), axis=0)
            new_cmap = cmap.copy()
            new_cmap = cmap[new_n, :]
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                name + "_%d" % K, colors=new_cmap, N=K
            )
        elif name == "Month":
            clist = [
                "darkslateblue",
                "skyblue",
                "powderblue",
                "honeydew",
                "lemonchiffon",
                "pink",
                "salmon",
                "deeppink",
                "gold",
                "chocolate",
                "darkolivegreen",
                "cadetblue",
            ]
            cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", clist)
            N = 12
            colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
            colors_rgba = cmap(colors_i)
            indices = np.linspace(0, 1.0, N + 1)
            cdict = {}
            for ki, key in enumerate(("red", "green", "blue")):
                cdict[key] = [
                    (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                    for i in np.arange(N + 1)
                ]
            new_cmap = mcolors.LinearSegmentedColormap("month_%d" % N, cdict, N)
        else:
            # Continuous colormap:
            N = K
            cmap = plt.get_cmap(name=name)
            colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
            colors_rgba = cmap(colors_i)  # N x 4
            indices = np.linspace(0, 1.0, N + 1)
            cdict = {}
            for ki, key in enumerate(("red", "green", "blue")):
                cdict[key] = [
                    (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                    for i in np.arange(N + 1)
                ]
            # Return colormap object.
            new_cmap = mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, N)
        self._colormap = new_cmap
        return new_cmap

    def cbar(self, ticklabels=None, **kwargs):
        """Return a colorbar with adjusted tick labels

        Returns
        -------
        :class:`matplotlib.pyplot.colorbar`
        """
        cmap = self.cmap
        ncolors = self.Ncolors
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors + 0.5)
        colorbar = plt.colorbar(mappable, **kwargs)
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels(ticklabels)
        self._colorbar = colorbar
        return colorbar

    def to_rgba(self, range, value):
        """ Return the RGBA color for a given value of the colormap and a range """
        norm = mpl.colors.Normalize(vmin=range[0], vmax=range[-1])
        scalarMap = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return scalarMap.to_rgba(value)


def latlongrid(ax, dx="auto", dy="auto", fontsize="auto", **kwargs):
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
    if fontsize != "auto":
        gl.xlabel_style = {"fontsize": fontsize}
        gl.ylabel_style = {"fontsize": fontsize}
    return gl
