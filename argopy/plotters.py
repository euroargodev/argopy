#!/bin/env python
# -*coding: UTF-8 -*-
#
# We try to import dependencies and catch missing module errors in order to avoid to load argopy just because
# Matplotlib is not installed.
#
# Decorator warnUnless is mandatory
#

import numpy as np
import pandas as pd
import warnings
from contextlib import contextmanager
from argopy.errors import InvalidDashboard
from argopy.utilities import warnUnless, check_wmo


try:
    with_matplotlib = True
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

except ModuleNotFoundError:
    warnings.warn("argopy requires matplotlib installed for any plotting functionality")
    with_matplotlib = False

try:
    with_cartopy = True
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    land_feature = cfeature.NaturalEarthFeature(
        category="physical", name="land", scale="50m", facecolor=[0.4, 0.6, 0.7]
    )
except ModuleNotFoundError:
    with_cartopy = False

# Default styles:
STYLE = {"axes": "whitegrid", "palette": "Set1"}

try:
    import seaborn as sns

    STYLE["axes"] = "dark"
    with_seaborn = True
except ModuleNotFoundError:
    with_seaborn = False


@contextmanager
def axes_style(style: str = STYLE['axes']):
    """ Provide a context for plots

        The point is to handle the availability of :mod:`seaborn` or not and to be able to use::

            with axes_style(style):
                fig, ax = plt.subplots()

        in all situations.
    """
    if with_seaborn:  # Execute within a seaborn context:
        with sns.axes_style(style):
            yield
    else:  # Otherwise do nothing
        yield


def open_dashboard(wmo=None, cyc=None, width="100%", height=1000, url=None, type="ea"):
    """ Insert the Euro-Argo dashboard page in a notebook cell

        Parameters
        ----------
        wmo: int
            The float WMO to display. By default, this is set to None and will insert the general dashboard.

        Returns
        -------
        :class:`IPython.lib.display.IFrame`
    """
    if type not in ["ea", "eric", "coriolis"]:
        raise InvalidDashboard("Invalid dashboard type")

    from IPython.display import IFrame

    if url is None:
        if type == "ea" or type == "eric":  # Open Euro-Argo dashboard
            if wmo is None:
                url = "https://fleetmonitoring.euro-argo.eu"
            else:
                wmo = check_wmo(wmo)
                url = "https://fleetmonitoring.euro-argo.eu/float/{}".format(str(wmo[0]))
        elif type == 'coriolis':  # Open Coriolis dashboard
            if wmo is not None:
                wmo = check_wmo(wmo)
                url = ("https://co-insitucharts.ifremer.fr/platform/{}/charts").format(
                    str(wmo[0])
                )

        # return open_dashboard(url=("https://co-insitucharts.ifremer.fr/platform/{}/charts").format(str(self.WMO[0])), **kw)

        # # Note that argovis doesn't allow X-Frame insertion !
        # elif type == 'argovis':
        #     if cyc is None:
        #         url = "https://argovis.colorado.edu/catalog/platforms/{}/page".format(str(wmo))
        #     else:
        #         url = "https://argovis.colorado.edu/catalog/profiles/{}_{}/page".format(str(wmo),str(cyc))

    return IFrame(url, width=width, height=height)


def open_sat_altim_report(WMO=None, embed='dropdown'):
    """ Insert the CLS Satellite Altimeter Report figure in notebook cell

        This is the method called when using the facade fetcher methods ``plot``:

        >>> DataFetcher().float(6902745).plot('qc_altimetry')

        Parameters
        ----------
        WMO: int or list
            The float WMO to display. By default, this is set to None and will insert the general dashboard.
        embed: {'list', 'slide', 'dropdown'}, default: 'dropdown'
            Set the embedding method. If set to None, simply return the list of urls to figures.
    """
    if embed in ['list', 'slide', 'dropdown']:
        from IPython.display import Image
    if embed in ['list']:
        from IPython.display import display
    if embed in ['slide', 'dropdown']:
        import ipywidgets as wg

    WMOs = check_wmo(WMO)
    urls = []
    urls_dict = {}
    for this_wmo in WMOs:
        url = "https://data-argo.ifremer.fr/etc/argo-ast9-item13-AltimeterComparison/figures/%i.png" % this_wmo
        if embed == 'list':
            urls.append(Image(url, embed=True))
        else:
            urls.append(url)
            urls_dict[this_wmo] = url

    if embed == 'list':
        return display(*urls)
    elif embed == 'slide':
        def f(Float):
            return Image(url=urls[Float])
        return wg.interact(f, Float=wg.IntSlider(min=0, max=len(urls) - 1, step=1))
    elif embed == 'dropdown':
        def f(Float):
            return Image(url=urls_dict[int(Float)])
        return wg.interact(f, Float=[str(wmo) for wmo in WMOs])
    else:
        return urls_dict


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


@warnUnless(with_matplotlib, "requires matplotlib installed")
def plot_trajectory(
    df: pd.core.frame.DataFrame,
    style: str = STYLE["axes"],
    add_legend: bool = True,
    palette: str = STYLE["palette"],
    set_global: bool = False,
    with_cartopy: bool = with_cartopy,
    with_seaborn: bool = with_seaborn,
    **kwargs
):
    """ Plot trajectories for an Argo index dataframe

    This function is called by the Data and Index fetchers method 'plot' with the 'trajectory' option::

        from argopy import IndexFetcher as ArgoIndexFetcher
        from argopy import DataFetcher as ArgoDataFetcher
        obj = ArgoIndexFetcher().float([6902766, 6902772, 6902914, 6902746])
        # OR
        obj = ArgoDataFetcher().float([6902766, 6902772, 6902914, 6902746])

        fig, ax = obj.plot('trajectory')

    Parameters
    ----------
    df: Pandas DataFrame
        Input data with columns: 'wmo','longitude','latitude'.
    style: str
        Define the axes style: 'white', 'darkgrid', 'whitegrid', 'dark', 'ticks'. Only used if Seaborn is available.
    add_legend: bool
        Add a box legend with list of floats. True by default for a maximum of 15 floats, otherwise no legend.
    palette: str
        Define colors to be used for floats: 'Set1' (default) or any other matplotlib colormap or name of
        a Seaborn palette (deep, muted, bright, pastel, dark, colorblind).
    set_global: bool
        Plot trajectories on a global world map or not. False by default.

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`
    """
    with axes_style(style):
        # Set-up the figure and axis:
        defaults = {"figsize": (10, 6), "dpi": 90}
        if with_cartopy:
            subplot_kw = {"projection": ccrs.PlateCarree()}
            fig, ax = plt.subplots(**{**defaults, **kwargs}, subplot_kw=subplot_kw)
            ax.add_feature(land_feature, edgecolor="black")
        else:
            fig, ax = plt.subplots(**{**defaults, **kwargs})

        # How many float in this dataset ?
        nfloat = len(df.groupby("wmo").first())

        # Let's do the plot:
        if with_seaborn:
            mypal = sns.color_palette(palette, nfloat)
            sns.lineplot(
                x="longitude",
                y="latitude",
                hue="wmo",
                data=df,
                sort=False,
                palette=mypal,
                legend=False,
            )
            sns.scatterplot(
                x="longitude", y="latitude", hue="wmo", data=df, palette=mypal
            )
        else:
            mypal = discrete_coloring(palette, N=nfloat).cmap
            for k, [name, group] in enumerate(df.groupby("wmo")):
                group.plot.line(
                    x="longitude",
                    y="latitude",
                    ax=ax,
                    color=mypal(k),
                    legend=False,
                    label="_nolegend_",
                )
                group.plot.scatter(
                    x="longitude", y="latitude", ax=ax, color=mypal(k), label=name
                )

        if with_cartopy:
            if set_global:
                ax.set_global()
            latlongrid(ax, dx="auto", dy="auto", fontsize="auto")
            if not with_seaborn:
                ax.get_yaxis().set_visible(False)
        else:
            if set_global:
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
            ax.grid(visible=True, linewidth=1, color="gray", alpha=0.7, linestyle=":")

        if add_legend and nfloat <= 15:
            handles, labels = ax.get_legend_handles_labels()
            # if with_seaborn:
            # handles, labels = handles[1:], labels[1:]
            plt.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(1.25, 1),
                title="Floats WMO",
            )
        else:
            ax.get_legend().remove()

    return fig, ax


def plot_dac(idx):
    """ Histogram of DAC for an index dataframe """
    warnings.warn(
        "plot_dac(idx) is deprecated; use bar_plot(idx, by='institution') instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )


def plot_profilerType(idx):
    """ Histogram of profile types for an index dataframe """
    warnings.warn(
        "plot_profilerType(idx) is deprecated; use bar_plot(idx, by='profiler') instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )


@warnUnless(with_matplotlib, "requires matplotlib installed")
def bar_plot(
    df: pd.core.frame.DataFrame,
    by: str = "institution",
    style: str = STYLE["axes"],
    with_seaborn: bool = with_seaborn,
    **kwargs
):
    """ Create a bar plot for an Argo index dataframe

    Parameters
    ----------
    df: Pandas DataFrame
        As returned by a fetcher index property
    by: str
        The profile property to plot. Default is 'institution'
    style: str
        Define the axes style: 'white', 'darkgrid', 'whitegrid', 'dark', 'ticks'. Only used if Seaborn is available.

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`
    """
    if by not in df:
        raise ValueError("'%s' is not a valid field for a bar plot" % by)
    with axes_style(style):
        defaults = {"figsize": (10, 6), "dpi": 90}
        fig, ax = plt.subplots(**{**defaults, **kwargs})
        if with_seaborn:
            mind = df.groupby(by).size().sort_values(ascending=False).index
            sns.countplot(y=by, data=df, order=mind)
        else:
            df.groupby(by).size().sort_values(ascending=True).plot.barh(ax)
        ax.set_xlabel("Number of profiles")
        ax.set_ylabel("")
    return fig, ax
