#!/bin/env python
# -*coding: UTF-8 -*-
#
# We try to import depedencies and catch missing module errors in order to avoid to load argopy just because
# Matplotlib is not installed.
#
# Decorator warnUnless is mandatory
#

import numpy as np
import warnings
from argopy.errors import InvalidDashboard


try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    with_matplotlib = True
except ModuleNotFoundError:
    warnings.warn("argopy requires matplotlib installed for any plotting functionality")
    with_matplotlib = False

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    with_cartopy = True
except ModuleNotFoundError:
    warnings.warn("argopy requires cartopy installed for full map plotting functionality")
    with_cartopy = False

try:
    import seaborn as sns
    sns.set_style("dark")
    with_seaborn = True
except ModuleNotFoundError:
    warnings.warn("argopy requires seaborn installed for full plotting functionality")
    with_seaborn = False

if with_cartopy:
    land_feature = cfeature.NaturalEarthFeature(category='physical', name='land',
                                                scale='50m', facecolor=[0.4, 0.6, 0.7])


def open_dashboard(wmo=None, cyc=None, width="100%", height=1000, url=None, type='ea'):
    """ Insert in a notebook the Euro-Argo dashboard page

        Parameters
        ----------
        wmo: int
            The float WMO to display. By default, this is set to None and will insert the general dashboard.

        Returns
        -------
        IFrame: IPython.lib.display.IFrame
    """
    if type not in ['ea', 'eric', 'coriolis']:
        raise InvalidDashboard("Invalid dashboard type")

    from IPython.display import IFrame
    if url is None:
        if type == 'ea' or type == 'eric':  # Open Euro-Argo dashboard
            if wmo is None:
                url = "https://fleetmonitoring.euro-argo.eu"
            else:
                url = "https://fleetmonitoring.euro-argo.eu/float/{}".format(str(wmo))
        elif type == 'coriolis': # Open Coriolis dashboard
            if wmo is not None:
                url = ("https://co-insitucharts.ifremer.fr/platform/{}/charts").format(str(wmo))

        # return open_dashboard(url=("https://co-insitucharts.ifremer.fr/platform/{}/charts").format(str(self.WMO[0])), **kw)

        # # Note that argovis doesn't allow X-Frame insertion !
        # elif type == 'argovis':
        #     if cyc is None:
        #         url = "https://argovis.colorado.edu/catalog/platforms/{}/page".format(str(wmo))
        #     else:
        #         url = "https://argovis.colorado.edu/catalog/profiles/{}_{}/page".format(str(wmo),str(cyc))

    return IFrame(url, width=width, height=height)


class discrete_coloring():
    """ Handy class to manage discrete coloring and the associated colorbar

    Example:
        year_range = np.arange(2002,2010)
        dc = discrete_coloring(name='Spectral', N=len(year_range) )
        plt.scatter(this['LONGITUDE'], this['LATITUDE'], c=this['TIME.year'],
                    cmap=dc.cmap, vmin=year_range[0], vmax=year_range[-1])
        dc.cbar(ticklabels=yr_range, fraction=0.03, label='Years')

    """
    def __init__(self, name='Set1', N=12):
        self.name = name
        self.Ncolors = N

    @property
    def cmap(self):
        """Return a discrete colormap from a quantitative or continuous colormap name

        name: name of the colormap, eg 'Paired' or 'jet'
        K: number of colors in the final discrete colormap
        """
        name = self.name
        K = self.Ncolors
        if name in ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'Dark2', 'Accent']:
            # Segmented (or quantitative) colormap:
            N_ref = {'Set1': 9, 'Set2': 8, 'Set3': 12, 'Pastel1': 9, 'Pastel2': 8, 'Paired': 12, 'Dark2': 8,
                     'Accent': 8}
            N = N_ref[name]
            cmap = plt.get_cmap(name=name)
            colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)), axis=0)
            cmap = cmap(colors_i)  # N x 4
            n = np.arange(0, N)
            new_n = n.copy()
            if K > N:
                for k in range(N, K):
                    r = np.roll(n, -k)[0][np.newaxis]
                    new_n = np.concatenate((new_n, r), axis=0)
            new_cmap = cmap.copy()
            new_cmap = cmap[new_n, :]
            new_cmap = mcolors.LinearSegmentedColormap.from_list(name + "_%d" % K, colors=new_cmap, N=K)
        elif name == 'Month':
            clist = ['darkslateblue', 'skyblue', 'powderblue',
                     'honeydew', 'lemonchiffon', 'pink',
                     'salmon', 'deeppink', 'gold',
                     'chocolate', 'darkolivegreen', 'cadetblue']
            cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', clist)
            N = 12
            colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
            colors_rgba = cmap(colors_i)
            indices = np.linspace(0, 1., N + 1)
            cdict = {}
            for ki, key in enumerate(('red', 'green', 'blue')):
                cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                              for i in np.arange(N + 1)]
            new_cmap = mcolors.LinearSegmentedColormap("month_%d" % N, cdict, N)
        else:
            # Continuous colormap:
            N = K
            cmap = plt.get_cmap(name=name)
            colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
            colors_rgba = cmap(colors_i)  # N x 4
            indices = np.linspace(0, 1., N + 1)
            cdict = {}
            for ki, key in enumerate(('red', 'green', 'blue')):
                cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                              for i in np.arange(N + 1)]
            # Return colormap object.
            new_cmap = mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, N)
        self._colormap = new_cmap
        return new_cmap

    def cbar(self, ticklabels=None, **kwargs):
        """Return a colorbar with adjusted tick labels"""
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


def latlongrid(ax, dx=5., dy=5., fontsize=6, **kwargs):
    """ Add latitude/longitude grid line and labels to a cartopy geoaxes """
    if not isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
        raise ValueError("Please provide a cartopy.mpl.geoaxes.GeoAxesSubplot instance")
    defaults = {'linewidth': .5, 'color': 'gray', 'alpha': 0.5, 'linestyle': '--'}
    gl = ax.gridlines(crs=ax.projection, draw_labels=True, **{**defaults, **kwargs})
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180+1, dx))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90+1, dy))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_top = False
    gl.xlabel_style = {'fontsize': fontsize}
    gl.ylabels_right = False
    gl.ylabel_style = {'fontsize': fontsize}
    return gl


def warnUnless(ok, txt):
    def inner(fct):
        def wrapper(*args, **kwargs):
            warnings.warn("%s %s" % (fct.__name__, txt))
            return fct(*args, **kwargs)
        return wrapper
    if not ok:
        return inner
    else:
        return lambda f: f


@warnUnless(with_matplotlib and with_cartopy and with_seaborn, "requires matplotlib, cartopy and seaborn installed")
def plot_trajectory(idx):
    """ Plot trajectories for an index dataframe """
    if not with_seaborn:
        raise BaseException("This function requires seaborn")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(land_feature, edgecolor='black')
    nfloat = len(idx.groupby('wmo').first())
    mypal = sns.color_palette("bright", nfloat)

    sns.lineplot(x="longitude", y="latitude", hue="wmo", data=idx, sort=False, palette=mypal, legend=False)
    sns.scatterplot(x="longitude", y="latitude", hue='wmo', data=idx, palette=mypal)
    # width = np.abs(idx['longitude'].max()-idx['longitude'].min())
    # height = np.abs(idx['latitude'].max()-idx['latitude'].min())
    # extent = (idx['longitude'].min()-width/4,
    #          idx['longitude'].max()+width/4,
    #          idx['latitude'].min()-height/4,
    #          idx['latitude'].max()+height/4)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.7, linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # ax.set_extent(extent)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    if (nfloat > 15):
        ax.get_legend().remove()
    return fig, ax


@warnUnless(with_matplotlib and with_cartopy and with_seaborn, "requires matplotlib, cartopy and seaborn installed")
def plot_dac(idx):
    """ Histogram of DAC for an index dataframe """
    if not with_seaborn:
        raise BaseException("This function requires seaborn")
    fig = plt.figure(figsize=(10, 5))
    mind = idx.groupby('institution').size().sort_values(ascending=False).index
    sns.countplot(y='institution', data=idx, order=mind)
    plt.ylabel('number of profiles')
    return fig


@warnUnless(with_matplotlib and with_cartopy and with_seaborn, "requires matplotlib, cartopy and seaborn installed")
def plot_profilerType(idx):
    """ Histogram of profile types for an index dataframe """
    if not with_seaborn:
        raise BaseException("This function requires seaborn")
    fig = plt.figure(figsize=(10, 5))
    mind = idx.groupby('profiler').size().sort_values(ascending=False).index
    sns.countplot(y='profiler', data=idx, order=mind)
    plt.xlabel('number of profiles')
    plt.ylabel('')
    return fig
