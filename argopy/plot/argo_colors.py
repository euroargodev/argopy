import numpy as np
from .utils import has_mpl, has_seaborn
from ..utilities import warnUnless

if has_mpl:
    from .utils import mpl, cm, mcolors, plt

if has_seaborn:
    from .utils import sns

@warnUnless(has_mpl, "requires matplotlib to be installed")
class ArgoColors:
    """ Handy class to manage discrete coloring for Argo related variables

    """
    quantitative = {
        "Set1": 9,
        "Set2": 8,
        "Set3": 12,
        "Pastel1": 9,
        "Pastel2": 8,
        "Paired": 12,
        "Dark2": 8,
        "Accent": 8,
    }
    """Number of colors in quantitative maps"""

    COLORS = {'CYAN': (18 / 256, 235 / 256, 229 / 256),
              'BLUE': (16 / 256, 137 / 256, 182 / 256),
              'DARKBLUE': (10 / 256, 89 / 256, 162 / 256),
              'YELLOW': (229 / 256, 174 / 256, 41 / 256),
              'DARKYELLOW': (224 / 256, 158 / 256, 37 / 256),
              }
    """Set of Argo colors derived from the logo"""

    def __init__(self, name="Set1", N=12):
        """

        Parameters
        ----------
        name: str
            Name of the colormap to use. Default: 'Set1'
        N: int
            Number of colors to reduce the colormap to. Default: 12
        """
        self.Ncolors = N
        self.name = name
        self.known_colormaps = {
            "data_mode": {
                "aka": ["datamode", "dm"],
                "constructor": self._colormap_datamode,
                "ticks": ["R", "A", "D"],
                "ticklabels": ["Real-time", "Adjusted", "Delayed"],
            },
            "deployment_status": {
                "aka": ["deployment_code", "deployment_id", "ptfstatus.id", "ptfstatus"],
                "constructor": self._colormap_deployment_status,
                "ticks": [0, 1, 2, 6, 4, 5],
                "ticklabels": ['PROBABLE', 'CONFIRMED', 'REGISTERED', 'OPERATIONAL', 'INACTIVE', 'CLOSED'],
            },
            "month": {
                "aka": None,
                "constructor": self._colormap_month,
                "ticks": np.arange(0, 12) + 1,
                "ticklabels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            },
            "qc": {
                "aka": ["qc_flag", "quality_control", "quality_control_flag", "Quality control flag scale"],
                "constructor": self._colormap_quality_control_flag,
                "ticks": np.arange(0, 9 + 1),
                "ticklabels": ["No QC performed",
                               "Good data",
                               "Probably good data",
                               "Probably bad data that are potentially adjustable",
                               "Bad data",
                               "Value changed",
                               "Not used",
                               "Not used",
                               "Estimated value",
                               "Missing value"]
            },
        }
        self.registered = self.name in self._list_valid_known_colormaps

    @property
    def _list_valid_known_colormaps(self):
        """Return the list of all known colormaps, including a.k.a. names"""
        defs = self.known_colormaps
        l = []
        for cmap in defs.keys():
            l.append(cmap)
            cmap_others = defs[cmap]["aka"]
            if cmap_others is not None:
                [l.append(s) for s in cmap_others]
        return l

    @property
    def _get_known_colormap_constructor(self):
        """Return the method constructor of a known colormap"""
        constructor = None
        if self.name in self.known_colormaps:
            constructor = self.known_colormaps[self.name]['constructor']
        else:
            for cmap in self.known_colormaps:
                if self.name in self.known_colormaps[cmap]['aka']:
                    constructor = self.known_colormaps[cmap]['constructor']
        return constructor

    @property
    def definition(self):
        defs = None
        if self.registered:
            if self.name in self.known_colormaps:
                defs = self.known_colormaps[self.name]
            else:
                for cmap in self.known_colormaps:
                    if self.name in self.known_colormaps[cmap]['aka']:
                        defs = self.known_colormaps[cmap]
            return defs

    def _argo2rgba(self, x):
        return tuple([int(v * 255) for v in mpl.colors.to_rgba(self.COLORS[x])])

    def _colormap_constant(self):
        """Colormap for a single color"""
        clist = [self.name, self.name]
        cmap = mcolors.LinearSegmentedColormap.from_list("my_colormap", clist)
        N = 1
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
        new_cmap.name = self.name
        return new_cmap

    def _colormap_segmented(self):
        """Segmented (or quantitative) colormap"""
        N = self.quantitative[self.name]
        K = self.Ncolors
        cmap = plt.get_cmap(name=self.name)
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
        new_cmap = cmap[new_n, :]
        new_cmap = mcolors.LinearSegmentedColormap.from_list(
            self.name + "_%d" % K, colors=new_cmap, N=K
        )
        return new_cmap

    def _colormap_month(self):
        """Return colormap with one value per month"""
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
        return new_cmap

    def _colormap_continuous(self):
        """Return a continuous colormap"""
        N = self.Ncolors
        cmap = plt.get_cmap(name=self.name)
        colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
        colors_rgba = cmap(colors_i)  # N x 4
        indices = np.linspace(0, 1.0, N + 1)
        cdict = {}
        for ki, key in enumerate(("red", "green", "blue")):
            cdict[key] = [
                (indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                for i in np.arange(N + 1)
            ]
        new_cmap = mcolors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, N)
        return new_cmap

    def _colormap_deployment_status(self):
        """Return a colormap with 6 colors for Float's deployment status"""
        # cmap = ArgoColors('Spectral', 6).cmap
        # List colors consistent with ticks order:
        # [0, 1, 2, 6, 4, 5]:
        # ['PROBABLE', 'CONFIRMED', 'REGISTERED', 'OPERATIONAL', 'INACTIVE', 'CLOSED']
        clist = ["white", "yellow", "orange", "limegreen", "red", "black"]
        cmap = mcolors.LinearSegmentedColormap.from_list("?", clist, 6)
        cmap.name = "Deployment status"
        return cmap

    def _colormap_datamode(self):
        """Return a colormap with 3 colors for variable's data mode (R, A, D)"""
        clist = [
            "orangered",
            "orange",
            "limegreen",
        ]
        return mcolors.LinearSegmentedColormap.from_list("Argo Data-Mode", clist, 3)

    def _colormap_quality_control_flag(self):
        """Return a colormap for QC flag"""
        clist = ['#000000',
                 '#31FC03',
                 '#ADFC03',
                 '#FCBA03',
                 '#FC1C03',
                 '#324CA8',
                 '#000000',
                 '#000000',
                 '#B22CC9',
                 '#000000'
                 ]
        return mcolors.LinearSegmentedColormap.from_list("Quality control flag", clist, 10)

    @property
    def cmap(self):
        """Return a discrete colormap

        Returns
        -------
        :class:`matplotlib.colors.LinearSegmentedColormap`
        """
        if self.name in self._list_valid_known_colormaps:
            cmap = self._get_known_colormap_constructor()
        elif self.Ncolors == 1:
            cmap = self._colormap_constant()
        else:
            cmap = self._colormap_continuous()

        self._colormap = cmap
        return cmap

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
        if ticklabels is not None:
            colorbar.set_ticklabels(ticklabels)
        self._colorbar = colorbar
        return colorbar

    def to_rgba(self, range, value):
        """ Return the RGBA color for a given value of the colormap and a range """
        norm = mpl.colors.Normalize(vmin=range[0], vmax=range[-1])
        scalarMap = cm.ScalarMappable(norm=norm, cmap=self.cmap)
        return scalarMap.to_rgba(value)

    @property
    def lookup(self):
        """Return a dictionary with ticks as keys and colors as values"""
        cmap = self.cmap
        defs = self.definition
        if defs is not None:
            lookup = {}
            for i, k in enumerate(defs['ticks']):
                lookup[k] = cmap(i)
            return lookup
        else:
            raise ValueError("Can't get a color lookup table for Argo-unknown colormap")

    @property
    def ticklabels(self):
        """Return a dictionary with ticks as keys and ticklabels as values"""
        defs = self.definition
        if defs is not None:
            lookup = {}
            for i, k in enumerate(defs['ticks']):
                lookup[k] = defs['ticklabels'][i]
            return lookup
        else:
            raise ValueError("Can't get a ticklabel lookup table for Argo-unknown colormap")

    @property
    def palette(self):
        try:
            if self.Ncolors == 1:
                return sns.color_palette("light:%s" % self.name, self.Ncolors)
            else:
                return sns.color_palette(self.name, self.Ncolors)
        except ValueError:
            return self.cmap