"""
This file covers the plotters module
We test plotting functions from IndexFetcher and DataFetcher
"""
import pytest
import logging

from utils import (
    requires_matplotlib,
    requires_seaborn,
    has_matplotlib,
    has_seaborn,
)
from argopy.plot import ArgoColors

if has_matplotlib:
    import matplotlib as mpl
    known_colormaps = ArgoColors().known_colormaps
    list_valid_known_colormaps = ArgoColors().list_valid_known_colormaps
    quantitative = ArgoColors().quantitative
else:
    known_colormaps = []
    list_valid_known_colormaps = []
    quantitative = []

if has_seaborn:
    import seaborn as sns

log = logging.getLogger("argopy.tests.plot")


@requires_matplotlib
class Test_ArgoColors:

    @pytest.mark.parametrize("cname", known_colormaps, indirect=False)
    def test_Argo_colormaps(self, cname):
        ac = ArgoColors(name=cname)
        assert ac.registered
        assert isinstance(ac.cmap, mpl.colors.LinearSegmentedColormap)

    @pytest.mark.parametrize("cname", quantitative, indirect=False)
    def test_quantitative_colormaps(self, cname):
        ac = ArgoColors(name=cname)
        assert ac.Ncolors == quantitative[cname]
        assert isinstance(ac.cmap, mpl.colors.LinearSegmentedColormap)

    @pytest.mark.parametrize("opts", [('Spectral', None),
                                      ('Blues', 13),
                                      ('k', 1)],
                             ids = ["name='Spectral', N=None",
                                    "name='Blues', N=13",
                                    "name='k', N=1"],
                             indirect=False)
    def test_other_colormaps(self, opts):
        name, N = opts
        ac = ArgoColors(name, N)
        assert isinstance(ac.cmap, mpl.colors.LinearSegmentedColormap)

    @pytest.mark.parametrize("N", [12.35, '12'],
                             ids=['N is a float', 'N is a str'],
                             indirect=False)
    def test_invalid_Ncolors(self, N):
        with pytest.raises(ValueError):
            ArgoColors(N=N)

    @pytest.mark.parametrize("cname", ['data_mode', 'dm'], ids=['key', 'aka'], indirect=False)
    def test_definition(self, cname):
        assert isinstance( ArgoColors(cname).definition, dict)

    def test_colors_lookup_dict(self):
        ac = ArgoColors(list_valid_known_colormaps[0])
        assert isinstance(ac.lookup, dict)

        with pytest.raises(ValueError):
            ArgoColors('Blues').lookup

    def test_ticklabels_dict(self):
        ac = ArgoColors(list_valid_known_colormaps[0])
        assert isinstance(ac.ticklabels, dict)

        with pytest.raises(ValueError):
            ArgoColors('Blues').ticklabels

    @requires_seaborn
    def test_seaborn_palette(self):
        assert isinstance(ArgoColors('Set1').palette, sns.palettes._ColorPalette)
        assert isinstance(ArgoColors('k', N=1).palette, sns.palettes._ColorPalette)

    @pytest.mark.parametrize("cname", ['data_mode', 'Blues'],
                             ids=['known', 'other'],
                             indirect=False)
    def test_repr_html_(self, cname):
        ac = ArgoColors(cname)
        assert isinstance(ac._repr_html_(), str)

    @pytest.mark.parametrize("cname", ['data_mode', 'Blues'],
                             ids=['known', 'other'],
                             indirect=False)
    def test_show_COLORS(self, cname):
        ac = ArgoColors(cname)
        assert isinstance(ac.show_COLORS(), str)
