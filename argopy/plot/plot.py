#!/bin/env python
# -*coding: UTF-8 -*-
#
# We try to import dependencies and catch missing module errors in order to avoid to load argopy just because
# Matplotlib is not installed.
#
# Decorator warnUnless is mandatory
#
import warnings
import logging

import xarray as xr
import pandas as pd
import numpy as np
from typing import Union

from .utils import STYLE, has_seaborn, has_mpl, has_cartopy, has_ipython, has_ipywidgets
from .utils import axes_style, latlongrid, land_feature
from .argo_colors import ArgoColors

from ..utils.loggers import warnUnless
from ..utils.checkers import check_wmo
from ..errors import InvalidDatasetStructure

if has_mpl:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

if has_seaborn:
    STYLE["axes"] = "dark"
    import seaborn as sns

if has_cartopy:
    import cartopy.crs as ccrs

if has_ipython:
    from IPython.display import Image, display

if has_ipywidgets:
    import ipywidgets


log = logging.getLogger("argopy.plot.plot")


def open_sat_altim_report(WMO: Union[str, list] = None, embed: Union[str, None] = "dropdown", **kwargs):
    """ Insert the CLS Satellite Altimeter Report figure in notebook cell

        This is the method called when using the facade fetcher methods ``plot``::

            DataFetcher().float(6902745).plot('qc_altimetry')

        Parameters
        ----------
        WMO: int or list
            The float WMO to display. By default, this is set to None and will insert the general dashboard.
        embed: str, default='dropdown'
            Set the embedding method. If set to None, simply return the list of urls to figures.
            Possible values are: ``dropdown``, ``slide`` and ``list``.

        Returns
        -------
        list of Image with ``list`` embed or a dict with URLs

        Notes
        -----
        Requires IPython to work as expected. If IPython is not available only URLs are returned.

    """
    warnUnless(has_ipython, "requires IPython to work as expected, only URLs are returned otherwise")

    if 'api_server' in kwargs:
        api_server = kwargs['api_server']
    else:
        api_server = "https://data-argo.ifremer.fr"

    # Create the list of URLs and put them in a dictionary with WMO as keys:
    WMOs = check_wmo(WMO)
    urls = []
    urls_dict = {}
    for this_wmo in WMOs:
        url = "%s/etc/argo-ast9-item13-AltimeterComparison/figures/%i.png" % (api_server, this_wmo)
        log.debug(url)
        if has_ipython and embed == "list":
            urls.append(Image(url, embed=True))
        else:
            urls.append(url)
            urls_dict[this_wmo] = url

    # Prepare rendering:
    if has_ipython and embed is not None:
        if has_ipywidgets and embed == "dropdown":
            def f(Float):
                return Image(url=urls_dict[int(Float)])
            return ipywidgets.interact(f, Float=[str(wmo) for wmo in WMOs])
        elif has_ipywidgets and embed == "slide":
            def f(Float):
                return Image(url=urls[Float])
            return ipywidgets.interact(
                f, Float=ipywidgets.IntSlider(min=0, max=len(urls) - 1, step=1)
            )
        elif embed == "list":
            return display(*urls)
        else:
            raise ValueError("Invalid value for 'embed' argument. Must be: 'dropdown', 'slide', 'list' or None")
    else:
        return urls_dict


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

        from argopy import DataFetcher, IndexFetcher

        obj = IndexFetcher().float([6902766, 6902772, 6902914, 6902746])
        # OR
        obj = DataFetcher().float([6902766, 6902772, 6902914, 6902746])

        fig, ax = obj.plot('trajectory')

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Input data with columns: 'wmo', 'longitude', 'latitude'.
    style: str
        Define the Seaborn axes style: 'white', 'darkgrid', 'whitegrid', 'dark', 'ticks'.
    add_legend: bool, default=True
        Add a box legend with list of floats. True by default for a maximum of 15 floats, otherwise no legend.
    palette: str
        Define colors to be used for floats: 'Set1' (default) or any other matplotlib colormap or name of
        a Seaborn palette (deep, muted, bright, pastel, dark, colorblind).
    set_global: bool, default=False
        Plot trajectories on a global world map or not.

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`

    Warnings
    --------
    This function will produce a plot even if `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_ is not installed.
    If `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_ is found, then this function will call
    :class:`argopy.plot.scatter_map`.

    """
    warnUnless(has_mpl, "requires matplotlib installed")

    with axes_style(style):
        # Set up the figure and axis:
        defaults = {"figsize": (10, 6), "dpi": 90}
        if with_cartopy:
            opts = {**defaults, **{'x': 'longitude', 'y': 'latitude', 'hue': 'wmo',
                                   'traj': True, 'legend': add_legend, 'set_global': set_global,
                                   'cmap': palette}}
            opts = {**opts, **kwargs}
            return scatter_map(df, **opts)
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
            mypal = ArgoColors(palette, N=nfloat).cmap
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


def bar_plot(
    df: pd.core.frame.DataFrame,
    by: str = "institution",
    style: str = STYLE["axes"],
    with_seaborn: bool = has_seaborn,
    **kwargs
):
    """ Create a bar plot for an Argo index dataframe


    This is the method called when using the facade fetcher methods ``plot`` with the ``dac`` or ``profiler`` arguments::

        IndexFetcher(src='gdac').region([-80,-30,20,50,'2021-01','2021-08']).plot('dac')

    To use it directly, you must pass a :class:`pandas.DataFrame` as returned by a :class:`argopy.DataFetcher.index` or :class:`argopy.IndexFetcher.index` property::

        from argopy import IndexFetcher
        df = IndexFetcher(src='gdac').region([-80,-30,20,50,'2021-01','2021-08']).index
        bar_plot(df, by='profiler')

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        As returned by a fetcher index property
    by: str, default='institution'
        The profile property to plot
    style: str, optional
        Define the Seaborn axes style: 'white', 'darkgrid', 'whitegrid', 'dark', 'ticks'

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`
    """

    warnUnless(has_mpl, "requires matplotlib installed")

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


def scatter_map(  # noqa: C901
        data: Union[xr.Dataset, pd.core.frame.DataFrame],
        x: Union[str] = None,
        y: Union[str] = None,
        hue: Union[str] = None,
        markersize: int = 36,
        markeredgesize: float = 0.5,
        markeredgecolor: str = 'default',
        cmap: Union[str] = None,
        traj: bool = True,
        traj_axis: Union[str] = None,
        traj_color: str = 'default',
        legend: bool = True,
        legend_title: str = 'default',
        legend_location: Union[str, int] = 0,
        cbar: bool = False,
        cbarlabels: Union[str, list] = 'auto',
        set_global: bool = False,
        **kwargs
):
    """Try-to-be generic function to create a scatter plot on a map from **argopy** :class:`xarray.Dataset` or :class:`pandas.DataFrame` data

    Each point is an Argo profile location, colored with a user defined variable and colormap. Floats trajectory can be plotted or not.

    Note that all parameters have default values.

    Warnings
    --------
    This function requires `Cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_.

    Examples
    --------
    ::

        from argopy.plot import scatter_map
        from argopy import DataFetcher

        ArgoSet = DataFetcher(mode='expert').float([6902771, 4903348]).load()
        ds = ArgoSet.data.argo.point2profile()
        df = ArgoSet.index

        scatter_map(df)
        scatter_map(ds)
        scatter_map(ds, hue='DATA_MODE')
        scatter_map(ds, hue='PSAL_QC')

    ::

        from argopy import OceanOPSDeployments
        df = OceanOPSDeployments([-90, 0, 0, 90]).to_dataframe()
        scatter_map(df, hue='status_code', traj=False)
        scatter_map(df, x='lon', y='lat', hue='status_code', traj=False, cmap='deployment_status')

    Parameters
    ----------
    data: :class:`xarray.Dataset` or :class:`pandas.DataFrame`
        Input data structure
    x: str, default=None
        Name of the data variable to use as longitude.
        If x is set to None, we'll try to guess which variable to use among standard names.
    y: str, default=None
        Name of the data variable to use as latitude.
        If y is set to None, we'll try to guess which variable to use among standard names.
    hue: str, default=None
        Name of the data variable to use for points coloring.
        If hue is set to None, we'll try to guess which variable to use to color points according to WMO.

    Returns
    -------
    fig: :class:`matplotlib.figure.Figure`
    ax: :class:`matplotlib.axes.Axes`

    Other Parameters
    ----------------
    markersize: int, default=36
        Size of the marker used for profiles location.
    markeredgesize: float, default=0.5
        Size of the marker edge used for profiles location.
    markeredgecolor: str, default='default'
        Color to use for the markers edge. The default color is 'DARKBLUE' from :class:`argopy.plot.ArgoColors.COLORS`

    cmap: str, default=None
        Colormap to use for points coloring. If set to None, we'll try to guess the most appropriate colormap for the
        ``hue`` argument by matching it to values in :class:`argopy.plot.ArgoColors.list_valid_known_colormaps`.

    traj: bool, default=True
        Set to True in order to plot each float trajectories, i.e. join with a line all profiles from a single platform.
    traj_axis: str, default='wmo'
        Name of the data variable to use in order to determine profiles group making a single trajectory.
    traj_color: str, default='default'
        The unique color to use for all trajectories. The default color is the ``markeredgecolor`` value.

    legend: bool, default=True
        Display or not a legend for hue colors meaning. If the legend is too large, it can be removed with ``ax.get_legend().remove()``
    legend_title: str, default='default'
        String title of the legend box. By default, it is set to the ``hue`` value.
    legend_location: str, default='upper right'
        Location of the legend box. This is passed to the ``loc`` argument of :class:`~matplotlib:matplotlib.legend.Legend`.

    set_global: bool, default=False
        Force the map to be global.

    kwargs
        All other arguments are passed to :class:`matplotlib.figure.Figure.subplots`

    """
    warnUnless(has_mpl and has_cartopy, "requires matplotlib AND cartopy installed")

    if isinstance(data, xr.Dataset) and data.argo._type == "point":
        # data = data.argo.point2profile(drop=True)
        raise InvalidDatasetStructure('Function only available to a collection of profiles')

    # Try to guess the default hue, i.e. name for WMO:
    def guess_trajvar(data):
        for v in ['WMO', 'PLATFORM_NUMBER']:
            if v.lower() in data:
                return v.lower()
            if v.upper() in data:
                return v.upper()
        raise ValueError("Can't guess the variable name for default hue/trajectory grouping (WMO)")
    hue = guess_trajvar(data) if hue is None else hue

    if isinstance(data, xr.Dataset) and data.argo.N_LEVELS > 1:
        warnings.warn("More than one N_LEVELS found in this dataset, scatter_map will use the first level only")
        data = data.isel(N_LEVELS=0)

    # Try to guess the colormap to use as a function of the 'hue' variable:
    def guess_cmap(hue):
        if hue.lower() in ArgoColors().list_valid_known_colormaps:
            cmap = hue.lower()
        elif 'qc' in hue.lower():
            cmap = 'qc'
        elif 'mode' in hue.lower():
            cmap = 'data_mode'
        elif 'status_code' in hue.lower():
            cmap = 'deployment_status'
        else:
            cmap = STYLE['palette']
        return cmap
    cmap = guess_cmap(hue) if cmap is None else cmap

    # Try to guess the x and y variables:
    def guess_xvar(data):
        for v in ['lon', 'long', 'longitude', 'x']:
            if v.lower() in data:
                return v.lower()
            if v.upper() in data:
                return v.upper()

        if isinstance(data, xr.Dataset):
            for v in data.coords:
                if '_CoordinateAxisType' in data[v].attrs and data[v].attrs['_CoordinateAxisType'] == 'Lon':
                    return v
                if 'axis' in data[v].attrs and data[v].attrs['axis'] == 'X':
                    return v

        raise ValueError("Can't guess the variable name for longitudes")

    def guess_yvar(data):
        for v in ['lat', 'lati', 'latitude', 'y']:
            if v.lower() in data:
                return v.lower()
            if v.upper() in data:
                return v.upper()

        if isinstance(data, xr.Dataset):
            for v in data.coords:
                if '_CoordinateAxisType' in data[v].attrs and data[v].attrs['_CoordinateAxisType'] == 'Lat':
                    return v
                if 'axis' in data[v].attrs and data[v].attrs['axis'] == 'Y':
                    return v

        raise ValueError("Can't guess the variable name for latitudes")
    x = guess_xvar(data) if x is None else x
    y = guess_yvar(data) if y is None else y

    # Adjust legend title:
    if legend_title == 'default':
        legend_title = str(hue)

    # Load Argo colors:
    nHue = len(data.groupby(hue).first()) if isinstance(data, pd.DataFrame) else len(data.groupby(hue))
    mycolors = ArgoColors(cmap, nHue)

    COLORS = mycolors.COLORS
    if markeredgecolor == 'default':
        markeredgecolor = COLORS['DARKBLUE']

    if traj_color == 'default':
        traj_color = markeredgecolor

    # Try to guess the trajectory grouping variable, i.e. name for WMO
    traj_axis = guess_trajvar(data) if traj and traj_axis is None else traj_axis

    # Set up the figure and axis:
    defaults = {"figsize": (10, 6), "dpi": 90}

    subplot_kw = {"projection": ccrs.PlateCarree()}
    fig, ax = plt.subplots(**{**defaults, **kwargs}, subplot_kw=subplot_kw)
    ax.add_feature(land_feature, color=COLORS['BLUE'], edgecolor=COLORS['CYAN'], linewidth=.1, alpha=0.3)

    # vmin = data[hue].min() if vmin == 'auto' else vmin
    # vmax = data[hue].max() if vmax == 'auto' else vmax

    patches = []
    for k, [name, group] in enumerate(data.groupby(hue)):
        if mycolors.registered and name not in mycolors.lookup:
            log.info("Found '%s' values not available in the '%s' colormap" % (name, mycolors.definition['name']))
        else:
            scatter_opts = {
                'color': mycolors.lookup[name] if mycolors.registered else mycolors.cmap(k),
                'label': "%s: %s" % (name, mycolors.ticklabels[name]) if mycolors.registered else name,
                'zorder': 10,
                'sizes': [markersize],
                'edgecolor': markeredgecolor,
                'linewidths': markeredgesize,
            }
            if isinstance(data, pd.DataFrame) and not legend:
                scatter_opts['legend'] = False  # otherwise Pandas will add a legend even if we set legend=False
            sc = group.plot.scatter(
                x=x, y=y,
                ax=ax,
                **scatter_opts
            )
            patches.append(sc)

    if cbar:
        if cbarlabels == 'auto':
            cbarlabels = None
        mycolors.cbar(ticklabels=cbarlabels,
                      ax=ax,
                      cax=sc,
                      fraction=0.03, label=legend_title)

    if traj:
        for k, [name, group] in enumerate(data.groupby(traj_axis)):
            ax.plot(group[x], group[y],
                    color=traj_color,
                    linewidth=0.5,
                    label="_nolegend_",
                    zorder=2,
                    )

    if set_global:
        ax.set_global()

    latlongrid(ax, dx="auto", dy="auto",
               label_style_arg={'color': COLORS['BLUE'], 'fontsize': 10},
               **{"color": COLORS['BLUE'], "alpha": 0.7})
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles,
            labels,
            loc=legend_location,
            bbox_to_anchor=(1.26, 1),
            title=legend_title,
        )

    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['DARKBLUE'])

    ax.set_title('')

    return fig, ax


def scatter_plot(ds: xr.Dataset,
                  this_param,
                  this_x='TIME',
                  this_y='PRES',
                  figsize=(18, 6),
                  cmap=None,
                  vmin=None,
                  vmax=None,
                  s=4,
                  bgcolor='lightgrey',
                  ):
    """A quick-and-dirty parameter scatter plot for one variable"""
    warnUnless(has_mpl, "requires matplotlib installed")

    if cmap is None:
        cmap = mpl.colormaps['gist_ncar']

    def get_vlabel(this_ds, this_v):
        attrs = this_ds[this_v].attrs
        if 'standard_name' in attrs:
            name = attrs['standard_name']
        elif 'long_name' in attrs:
            name = attrs['long_name']
        else:
            name = this_v
        units = attrs['units'] if 'units' in attrs else None
        return "%s\n[%s]" % (name, units) if units else name

    # Read variables for the plot:
    x, y = ds[this_x], ds[this_y]
    if "INTERPOLATED" in this_y:
        x_bounds, y_bounds = np.meshgrid(x, y, indexing='ij')
    c = ds[this_param]

    #
    fig, ax = plt.subplots(dpi=90, figsize=figsize)

    if vmin == 'attrs':
        vmin = c.attrs['valid_min'] if 'valid_min' in c.attrs else None
    if vmax == 'attrs':
        vmax = c.attrs['valid_max'] if 'valid_max' in c.attrs else None
    if vmin is None:
        vmin = np.percentile(c, 10)
    if vmax is None:
        vmax = np.percentile(c, 90)

    if "INTERPOLATED" in this_y:
        m = ax.pcolormesh(x_bounds, y_bounds, c, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        m = ax.scatter(x, y, c=c, cmap=cmap, s=s, vmin=vmin, vmax=vmax)
        ax.set_facecolor(bgcolor)

    cbar = fig.colorbar(m, shrink=0.9, extend='both', ax=ax)
    cbar.ax.set_ylabel(get_vlabel(ds, this_param), rotation=90)

    ylim = ax.get_ylim()
    if 'PRES' in this_y:
        ax.invert_yaxis()
        y_bottom, y_top = np.max(ylim), np.min(ylim)
    else:
        y_bottom, y_top = ylim

    if this_x == 'CYCLE_NUMBER':
        ax.set_xlim([np.min(ds[this_x]) - 1, np.max(ds[this_x]) + 1])
    elif this_x == 'TIME':
        ax.set_xlim([np.min(ds[this_x]), np.max(ds[this_x])])
    if 'PRES' in this_y:
        ax.set_ylim([y_bottom, 0])

    #
    ax.set_xlabel(get_vlabel(ds, this_x))
    ax.set_ylabel(get_vlabel(ds, this_y))

    return fig, ax
