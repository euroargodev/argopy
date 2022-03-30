#!/bin/env python
# -*coding: UTF-8 -*-
#
# We try to import dependencies and catch missing module errors in order to avoid to load argopy just because
# Matplotlib is not installed.
#
# Decorator warnUnless is mandatory
#
import pandas as pd
from .utils import STYLE, has_seaborn, has_mpl, has_cartopy, has_ipython, has_ipywidgets
from .utils import axes_style, discrete_coloring, latlongrid, land_feature
from ..utilities import warnUnless, check_wmo


if has_mpl:
    import matplotlib.pyplot as plt

if has_seaborn:
    STYLE["axes"] = "dark"
    import seaborn as sns

if has_cartopy:
    import cartopy.crs as ccrs

if has_ipython:
    from IPython.display import Image, display

if has_ipywidgets:
    import ipywidgets


@warnUnless(has_ipython, "requires IPython to work as expected, only URLs are returned otherwise")
def open_sat_altim_report(WMO=None, embed="dropdown"):
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
    WMOs = check_wmo(WMO)
    urls = []
    urls_dict = {}
    for this_wmo in WMOs:
        url = (
            "https://data-argo.ifremer.fr/etc/argo-ast9-item13-AltimeterComparison/figures/%i.png"
            % this_wmo
        )
        if embed == "list":
            urls.append(Image(url, embed=True))
        else:
            urls.append(url)
            urls_dict[this_wmo] = url

    if embed == "list" and has_ipython:
        return display(*urls)

    elif embed == "slide" and has_ipython and has_ipywidgets:
        def f(Float):
            return Image(url=urls[Float])
        return ipywidgets.interact(
            f, Float=ipywidgets.IntSlider(min=0, max=len(urls) - 1, step=1)
        )

    elif embed == "dropdown" and has_ipython and has_ipywidgets:
        def f(Float):
            return Image(url=urls_dict[int(Float)])
        return ipywidgets.interact(f, Float=[str(wmo) for wmo in WMOs])
    else:
        return urls_dict


@warnUnless(has_mpl, "requires matplotlib installed")
def plot_trajectory(
    df: pd.core.frame.DataFrame,
    style: str = STYLE["axes"],
    add_legend: bool = True,
    palette: str = STYLE["palette"],
    set_global: bool = False,
    with_cartopy: bool = has_cartopy,
    with_seaborn: bool = has_seaborn,
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
            if not has_seaborn:
                ax.get_yaxis().set_visible(False)
        else:
            if set_global:
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
            ax.grid(visible=True, linewidth=1, color="gray", alpha=0.7, linestyle=":")

        if add_legend and nfloat <= 15:
            handles, labels = ax.get_legend_handles_labels()
            # if has_seaborn:
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


@warnUnless(has_mpl, "requires matplotlib installed")
def bar_plot(
    df: pd.core.frame.DataFrame,
    by: str = "institution",
    style: str = STYLE["axes"],
    with_seaborn: bool = has_seaborn,
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
