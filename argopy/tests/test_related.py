import pytest
import xarray as xr
import pandas as pd
from collections import ChainMap, OrderedDict
import shutil

from mocked_http import mocked_httpserver, mocked_server_address

from utils import (
    requires_matplotlib,
    requires_cartopy,
    requires_oops,
    has_matplotlib,
    has_cartopy,
    has_ipython,
    create_temp_folder,
)
import argopy
from argopy.related import (
    TopoFetcher,
    ArgoNVSReferenceTables,
    OceanOPSDeployments,
    ArgoDocs,
    load_dict, mapp_dict,
    get_coriolis_profile_id, get_ea_profile_page
)
from argopy.utils.checkers import (
    is_list_of_strings,
)

if has_matplotlib:
    import matplotlib as mpl

if has_cartopy:
    import cartopy

if has_ipython:
    import IPython


class Test_TopoFetcher():
    box = [81, 123, -67, -54]

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = create_temp_folder().folder

    def teardown_class(self):
        """Cleanup once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.cachedir)
        remove_test_dir()

    def make_a_fetcher(self, cached=False):
        opts = {'ds': 'gebco', 'stride': [10, 10], 'server': mocked_server_address}
        if cached:
            opts = ChainMap(opts, {'cache': True, 'cachedir': self.cachedir})
        return TopoFetcher(self.box, **opts)

    def assert_fetcher(self, f):
        ds = f.to_xarray()
        assert isinstance(ds, xr.Dataset)
        assert 'elevation' in ds.data_vars

    def test_load_mocked_server(self, mocked_httpserver):
        """This will easily ensure that the module scope fixture is available to all methods !"""
        assert True

    params = [True, False]
    ids_params = ["cached=%s" % p for p in params]
    @pytest.mark.parametrize("params", params, indirect=False, ids=ids_params)
    def test_fetching(self, params):
        fetcher = self.make_a_fetcher(cached=params)
        self.assert_fetcher(fetcher)


class Test_ArgoNVSReferenceTables:

    def setup_class(self):
        """setup any state specific to the execution of the given class"""
        # Create the cache folder here, so that it's not the same for the pandas and pyarrow tests
        self.cachedir = create_temp_folder().folder
        self.nvs = ArgoNVSReferenceTables(cache=True, cachedir=self.cachedir, nvs=mocked_server_address)

    def teardown_class(self):
        """Cleanup once we are finished."""
        def remove_test_dir():
            shutil.rmtree(self.cachedir)
        remove_test_dir()

    def test_load_mocked_server(self, mocked_httpserver):
        """This will easily ensure that the module scope fixture is available to all methods !"""
        assert True

    def test_valid_ref(self):
        assert is_list_of_strings(self.nvs.valid_ref)

    opts = [3, 'R09']
    opts_ids = ["rtid is a %s" % type(o) for o in opts]
    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_tbl(self, opts):
        assert isinstance(self.nvs.tbl(opts), pd.DataFrame)

    opts = [3, 'R09']
    opts_ids = ["rtid is a %s" % type(o) for o in opts]
    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_tbl_name(self, opts):
        names = self.nvs.tbl_name(opts)
        assert isinstance(names, tuple)
        assert isinstance(names[0], str)
        assert isinstance(names[1], str)
        assert isinstance(names[2], str)

    def test_all_tbl(self):
        all = self.nvs.all_tbl
        assert isinstance(all, OrderedDict)
        assert isinstance(all[list(all.keys())[0]], pd.DataFrame)

    def test_all_tbl_name(self):
        all = self.nvs.all_tbl_name
        assert isinstance(all, OrderedDict)
        assert isinstance(all[list(all.keys())[0]], tuple)

    opts = ["ld+json", "rdf+xml", "text/turtle", "invalid"]
    opts_ids = ["fmt=%s" % o for o in opts]
    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_get_url(self, opts):
        if opts != 'invalid':
            url = self.nvs.get_url(3, fmt=opts)
            assert isinstance(url, str)
            if "json" in opts:
                data = self.nvs.fs.open_json(url)
                assert isinstance(data, dict)
            elif "xml" in opts:
                data = self.nvs.fs.download_url(url)
                assert data[0:5] == b'<?xml'
            else:
                # log.debug(self.nvs.fs.fs.info(url))
                data = self.nvs.fs.download_url(url)
                assert data[0:6] == b'PREFIX'
        else:
            with pytest.raises(ValueError):
                self.nvs.get_url(3, fmt=opts)


@requires_oops
class Test_OceanOPSDeployments:

    # scenarios generation can't be automated because of the None/True combination of arguments.
    # If box=None and deployed_only=True, OceanOPSDeployments will seek for OPERATING floats deployed today ! which is
    # impossible and if it happens it's due to an error in the database...
    scenarios = [
        # (None, True),  # This often lead to an empty dataframe !
        # (None, False),  # Can't be handled by the mocked server (test date is surely different from the test data date)
        # ([-90, 0, 0, 90], True),
        # ([-90, 0, 0, 90], False),  # Can't be handled by the mocked server (test date is surely different from the test data date)
        ([-90, 0, 0, 90, '2022-01'], True),
        ([-90, 0, 0, 90, '2022-01'], False),
        ([None, 0, 0, 90, '2022-01-01', '2023-01-01'], True),
        ([None, 0, 0, 90, '2022-01-01', '2023-01-01'], False),
        ([-90, None, 0, 90, '2022-01-01', '2023-01-01'], True),
        ([-90, None, 0, 90, '2022-01-01', '2023-01-01'], False),
        ([-90, 0, None, 90, '2022-01-01', '2023-01-01'], True),
        ([-90, 0, None, 90, '2022-01-01', '2023-01-01'], False),
        ([-90, 0, 0, None, '2022-01-01', '2023-01-01'], True),
        ([-90, 0, 0, None, '2022-01-01', '2023-01-01'], False),
        ([-90, 0, 0, 90, None, '2023-01-01'], True),
        ([-90, 0, 0, 90, None, '2023-01-01'], False),
        ([-90, 0, 0, 90, '2022-01-01', None], True),
        ([-90, 0, 0, 90, '2022-01-01', None], False)]
    scenarios_ids = ["%s, %s" % (opt[0], opt[1]) for opt in scenarios]

    @pytest.fixture
    def an_instance(self, request):
        """ Fixture to create a OceanOPS_Deployments instance for a given set of arguments """
        if isinstance(request.param, tuple):
            box = request.param[0]
            deployed_only = request.param[1]
        else:
            box = request.param
            deployed_only = None

        args = {"box": box, "deployed_only": deployed_only}
        oops = OceanOPSDeployments(**args)

        # Adjust server info to use the mocked HTTP server:
        oops.api = mocked_server_address
        oops.model = 'data/platform'

        return oops

    def test_load_mocked_server(self, mocked_httpserver):
        """This will easily ensure that the module scope fixture is available to all methods !"""
        assert True

    @pytest.mark.parametrize("an_instance", scenarios, indirect=True, ids=scenarios_ids)
    def test_init(self, an_instance):
        assert isinstance(an_instance, OceanOPSDeployments)

    @pytest.mark.parametrize("an_instance", scenarios, indirect=True, ids=scenarios_ids)
    def test_attributes(self, an_instance):
        dep = an_instance
        assert isinstance(dep.uri, str)
        assert isinstance(dep.uri_decoded, str)
        assert isinstance(dep.status_code, pd.DataFrame)
        assert isinstance(dep.box_name, str)
        assert len(dep.box) == 6

    @pytest.mark.parametrize("an_instance", scenarios, indirect=True, ids=scenarios_ids)
    def test_to_dataframe(self, an_instance):
        assert isinstance(an_instance.to_dataframe(), pd.DataFrame)

    @pytest.mark.parametrize("an_instance", scenarios, indirect=True, ids=scenarios_ids)
    @requires_matplotlib
    @requires_cartopy
    def test_plot_status(self, an_instance):
        fig, ax, hdl = an_instance.plot_status()
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot)


@pytest.mark.skipif(True, reason="Skipped temporarily, see http://github.com/euroargodev/argopy/issues/488")
class Test_ArgoDocs:

    @pytest.fixture
    def an_instance(self, request):
        """ Fixture to create a ArgoDocs instance for a given set of arguments """
        docid = request.param

        Ad = ArgoDocs(docid=docid, cache=False)

        # Adjust server info to use the mocked HTTP server:
        Ad._doiserver = mocked_server_address
        Ad._archimer = mocked_server_address

        return Ad

    def test_load_mocked_server(self, mocked_httpserver):
        """This will easily ensure that the module scope fixture is available to all methods !"""
        assert True

    @pytest.mark.parametrize("an_instance", [None], indirect=True, ids=["docid=%s" % t for t in [None]])
    def test_list(self, an_instance):
        assert isinstance(an_instance.list, pd.DataFrame)

    @pytest.mark.parametrize("an_instance", [None, 35385, '10.13155/46202'], indirect=True,
                             ids=["docid=%s" % t for t in [None, 35385, '10.13155/46202']])
    def test_init(self, an_instance):
        assert isinstance(an_instance, ArgoDocs)
        assert isinstance(an_instance.__repr__(), str)

    @pytest.mark.parametrize("docid", [12, 'dummy'], indirect=False, ids=["docid=%s" % t for t in [12, 'dummy']])
    def test_init_with_error(self, docid):
        with pytest.raises(ValueError):
            ArgoDocs(docid)

    @pytest.mark.parametrize("where", ['title', 'abstract'], indirect=False,
                             ids=["where=%s" % t for t in ['title', 'abstract']])
    @pytest.mark.parametrize("an_instance", [None], indirect=True, ids=["docid=%s" % t for t in [None]])
    def test_search(self, where, an_instance):
        txt = "CDOM"
        results = an_instance.search(txt, where=where)
        assert isinstance(results, list)

    @pytest.mark.parametrize("an_instance", [None, 35385], indirect=True, ids=["docid=%s" % t for t in [None, 35385]])
    def test_js(self, an_instance):
        if an_instance.docid is not None:
            assert isinstance(an_instance.js, dict)
        else:
            with pytest.raises(ValueError):
                an_instance.js

    @pytest.mark.parametrize("an_instance", [None, 35385], indirect=True, ids=["docid=%s" % t for t in [None, 35385]])
    def test_properties(self, an_instance):
        if an_instance.docid is not None:
            ris = an_instance.ris  # Fetch RIS metadata for this document
            abstract = an_instance.abstract
            assert isinstance(ris, dict)
            assert 'AB' in ris  # must have an abstract
            assert 'UR' in ris  # must have an url
            assert isinstance(abstract, str)
        else:
            with pytest.raises(ValueError):
                an_instance.ris
            with pytest.raises(ValueError):
                an_instance.abstract

    @pytest.mark.parametrize("an_instance", [None, 35385], indirect=True, ids=["docid=%s" % t for t in [None, 35385]])
    def test_show(self, an_instance):
        if an_instance.docid is not None:
            if has_ipython:
                assert isinstance(an_instance.show(), IPython.core.display.HTML)
                assert isinstance(an_instance.show(height=120), IPython.core.display.HTML)
            else:
                pytest.skip("Requires IPython")
        else:
            with pytest.raises(ValueError):
                an_instance.show()

    @pytest.mark.parametrize("page", [None, 12], indirect=False, ids=["page=%s" % t for t in [None, 12]])
    @pytest.mark.parametrize("an_instance", [None, 35385], indirect=True, ids=["docid=%s" % t for t in [None, 35385]])
    def test_open_pdf(self, page, an_instance):
        if an_instance.docid is not None:
            assert isinstance(an_instance.open_pdf(url_only=True, page=page), str)
        else:
            with pytest.raises(ValueError):
                an_instance.show()


def test_invalid_dictionnary():
    with pytest.raises(ValueError):
        load_dict("invalid_dictionnary")


def test_invalid_dictionnary_key():
    d = load_dict("profilers")
    assert mapp_dict(d, "invalid_key") == "Unknown"


@pytest.mark.parametrize("params", [[6901929, None], [6901929, 12]], indirect=False, ids=['float', 'profile'])
def test_get_coriolis_profile_id(params, mocked_httpserver):
    with create_temp_folder() as temp_folder:
        with argopy.set_options(cachedir=temp_folder, server=mocked_server_address):
            assert isinstance(get_coriolis_profile_id(params[0], params[1]), pd.core.frame.DataFrame)

@pytest.mark.parametrize("params", [[6901929, None], [6901929, 12]], indirect=False, ids=['float', 'profile'])
def test_get_ea_profile_page(params, mocked_httpserver):
    with create_temp_folder() as temp_folder:
        with argopy.set_options(cachedir=temp_folder):
            assert is_list_of_strings(get_ea_profile_page(params[0], params[1], api_server=mocked_server_address))
