import json
import pytest
import pandas as pd
from collections import OrderedDict
import shutil
import tempfile
import logging
from collections.abc import Iterable

from mocked_http import mocked_httpserver, mocked_server_address
from utils import (
    create_temp_folder,
)
from argopy.errors import OptionValueError
from argopy.related.vocabulary.reference_tables import ArgoNVSReferenceTables
from argopy.related.vocabulary.concept import ArgoReferenceValue
from argopy.related.vocabulary.vocabulary import ArgoReferenceTable
from argopy.stores.nvs.implementations.offline.nvs import NVS
from argopy.utils.checkers import (
    is_list_of_strings,
)
from argopy.utils.checkers import is_list_of_strings

log = logging.getLogger("argopy.tests.related.reference")


# @pytest.mark.skip
class Test_ArgoNVSReferenceTables_Deprecated:

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

    opts = [3, '12', 'R09']
    opts_ids = ["rtid is a %s" % type(o) for o in opts]
    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_tbl(self, opts):
        assert isinstance(self.nvs.tbl(opts), pd.DataFrame)

    opts = [3, '12', 'R09']
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


class MiscPath:

    @classmethod
    def none(cls, *args, **kwargs):
        class MyContext:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self, *args, **kwargs):
                return None

            def __exit__(self, *args, **kwargs):
                pass

        return MyContext(*args, **kwargs)

    @classmethod
    def named(cls, *args, **kwargs):
        class MyContext:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self, *args, **kwargs):
                self.tf = tempfile.NamedTemporaryFile(delete=True, mode='w')
                return self.tf.name

            def __exit__(self, *args, **kwargs):
                self.tf.close()

        return MyContext(*args, **kwargs)

    @classmethod
    def obj(cls, *args, **kwargs):
        class MyContext:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self, *args, **kwargs):
                self.tf = tempfile.NamedTemporaryFile(delete=True, mode='w')
                return self.tf

            def __exit__(self, *args, **kwargs):
                self.tf.close()

        return MyContext(*args, **kwargs)

default_name = ['AANDERAA_OPTODE_3835']

# @pytest.mark.skip
class Test_ArgoReferenceValue:

    def test_init_implicit(self):
        arv = ArgoReferenceValue(default_name[0])
        assert isinstance(arv, ArgoReferenceValue)

        arv = ArgoReferenceValue('4', 'RR2')
        # arv = ArgoReferenceValue('4', 'RT_QC_FLAG')
        assert isinstance(arv, ArgoReferenceValue)

        with pytest.raises(ValueError):
            ArgoReferenceValue('dummy')

        with pytest.raises(ValueError):
            ArgoReferenceValue('AANDERAA_OPTODE_3835', 'R01')

        with pytest.raises(ValueError):
            ArgoReferenceValue('4')

    def test_init_explicit(self):
        data = NVS().load_concept('4', 'RR2')
        arv = ArgoReferenceValue(None, data=data)
        assert isinstance(arv, ArgoReferenceValue)

    def test_init_with_extra(self):
        arv = ArgoReferenceValue('FLUORESCENCE_CHLA', 'R03')
        assert isinstance(arv._extra, dict)
        assert 'local_attributes' in arv._extra
        assert 'properties' in arv._extra

    @pytest.mark.parametrize("name", default_name,
                             indirect=False,
                             ids=[f"name='{x}'" for x in default_name])
    def test_readonly_instance(self, name):
        arv = ArgoReferenceValue(name)
        with pytest.raises(AttributeError):
            arv.definition = 'new value'

    @pytest.mark.parametrize("name", default_name,
                             indirect=False,
                             ids=[f"name='{x}'" for x in default_name])
    def test_getitem(self, name):
        arv = ArgoReferenceValue(name)
        assert isinstance(arv['name'], str)

        with pytest.raises(ValueError):
            arv['dummy']

    @pytest.mark.parametrize("name", default_name,
                             indirect=False,
                             ids=[f"name='{x}'" for x in default_name])
    @pytest.mark.parametrize("attr", ['nvs', 'context', 'extra'], indirect=False)
    def test_props(self, name, attr):
        arv = ArgoReferenceValue(name)
        arv.__getattribute__(attr)

    def test_from_urn(self):
        arv = ArgoReferenceValue.from_urn('SDN:R27::AANDERAA_OPTODE_3835')
        assert isinstance(arv, ArgoReferenceValue)

    def test_from_dict(self):
        data = NVS().load_concept('4', 'RR2')
        arv = ArgoReferenceValue.from_dict(data)
        assert isinstance(arv, ArgoReferenceValue)

    @pytest.mark.parametrize("keys", [None, ['name', 'deprecated']],
                             indirect=False,
                             ids=[f"keys='{x}'" for x in [None, ['name', 'deprecated']]])
    @pytest.mark.parametrize("name", default_name,
                             indirect=False,
                             ids=[f"name='{x}'" for x in default_name])
    def test_to_dict(self, name, keys):
        arv = ArgoReferenceValue(name)
        assert isinstance(arv.to_dict(keys), dict)

    @pytest.mark.parametrize("keys", ['dummy'], indirect=False, ids=[f"keys='{x}'" for x in ['dummy']])
    @pytest.mark.parametrize("name", default_name,
                             indirect=False,
                             ids=[f"name='{x}'" for x in default_name])
    def test_to_dict_invalidkeys(self, name, keys):
        arv = ArgoReferenceValue(name)
        with pytest.raises(OptionValueError):
            arv.to_dict(keys)

    @pytest.mark.parametrize("keys", [None, ['name', 'deprecated']],
                             indirect=False,
                             ids=[f"keys='{x}'" for x in [None, ['name', 'deprecated']]])
    @pytest.mark.parametrize("path", [MiscPath.none(), MiscPath.named(), MiscPath.obj()],
                             indirect=False,
                             ids=[f"path='{x}'" for x in [None, 'A file name', 'A file obj']])
    @pytest.mark.parametrize("name", default_name,
                             indirect=False,
                             ids=[f"name='{x}'" for x in default_name])
    def test_to_json(self, name, path, keys):
        arv = ArgoReferenceValue(name)
        with path as p:
            arv.to_json(path=p, keys=keys)


default_identifier = ['R01']

class Test_ArgoReferenceTable:

    @pytest.mark.parametrize("identifier", ['R01', 'DATA_TYPE'],
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in ['R01', 'DATA_TYPE']])
    def test_init(self, identifier):
        art = ArgoReferenceTable(identifier)
        assert isinstance(art, ArgoReferenceTable)

    def test_from_urn(self):
        art = ArgoReferenceTable.from_urn('SDN:R27::AANDERAA_OPTODE_3835')
        assert isinstance(art, ArgoReferenceTable)

    @pytest.mark.parametrize("identifier", ['dummy'],
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in ['dummy']])
    def test_init_error(self, identifier):
        with pytest.raises(ValueError):
            ArgoReferenceTable(identifier)

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_readonly_instance(self, identifier):
        art = ArgoReferenceTable(identifier)
        with pytest.raises(AttributeError):
            art.parameter = 'dummy'

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_keys(self, identifier):
        art = ArgoReferenceTable(identifier)
        assert is_list_of_strings(art.keys())

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_values(self, identifier):
        art = ArgoReferenceTable(identifier)
        values = art.values()
        isinstance(values, list) and all(isinstance(elem, ArgoReferenceValue) for elem in values)

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_contains(self, identifier):
        art = ArgoReferenceTable(identifier)
        assert art.keys()[0] in art

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_len(self, identifier):
        art = ArgoReferenceTable(identifier)
        assert isinstance(len(art), int)

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_iter(self, identifier):
        art = ArgoReferenceTable(identifier)
        assert isinstance(art, Iterable)

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    def test_getitem(self, identifier):
        art = ArgoReferenceTable(identifier)
        key = art.keys()[0]
        assert isinstance(art[key], ArgoReferenceValue)

        with pytest.raises(ValueError):
            art['dummy']

    @pytest.mark.parametrize("identifier", default_identifier,
                             indirect=False,
                             ids=[f"identifier='{x}'" for x in default_identifier])
    @pytest.mark.parametrize("columns", [None, ArgoReferenceValue.keys[0:2]],
                             indirect=False,
                             ids=[f"columns='{x}'" for x in [None, ArgoReferenceValue.keys[0:2]]])
    def test_to_dataframe(self, identifier, columns):
        art = ArgoReferenceTable(identifier)
        assert isinstance(art.to_dataframe(columns), pd.DataFrame)
