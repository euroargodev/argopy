import pytest
import logging
import numpy as np

from argopy.stores.nvs.implementations.offline.nvs import NVS as NVS_offline
from argopy.stores.nvs.implementations.online.nvs import NVS as NVS_online
from argopy.stores.nvs.spec import NVSProto as NVS
from argopy.stores.nvs.utils import (
    concept2vocabulary,
    check_vocabulary,
    id2urn,
    extract_local_attributes,
    extract_properties_section,
    curate_r03definition,
    curate_r14definition,
    curate_r18definition,
    LocalAttributes,
    Properties,
    known_mappings,
    sparql_mapping_request,
)

from mocked_http import mocked_httpserver, mocked_server_address

log = logging.getLogger("argopy.tests.nvsstores")

skip_offline = pytest.mark.skipif(0, reason="Skipped tests for offline implementation")
skip_online = pytest.mark.skipif(0, reason="Skipped tests for online implementation")


USE_MOCKED_SERVER = True
RTID_TESTED = ["R27"]
CONCEPTID_TESTED = ["AANDERAA_OPTODE_3835"]
FMT_TESTED = ["json", "xml", "turtle"]
MAPPING_TESTED = [("R08", "R23")]
DATA_VALIDATOR = {
    "json": lambda x: isinstance(x, dict),
    "xml": lambda x: x.startswith("<?xml"),
    "turtle": lambda x: x.startswith("PREFIX"),
}


def is_offline_urivalid(url: str) -> bool:
    """Use to validate an offline asset uri to be loaded with :class:`Asset`

    This will depend on the :class:`Asset` implementation.
    """
    return isinstance(url, str) and url.startswith("vocabulary:offline:")


def is_online_urivalid(url: str) -> bool:
    """Use to validate an online asset uri to be loaded with :class:`Asset`

    This will depend on the :class:`Asset` implementation.
    """
    return isinstance(url, str) and url.startswith("http") and "current" in url


@skip_offline
class Test_NVS_Offline:
    #############
    # UTILITIES #
    #############
    @pytest.fixture
    def nvs(self) -> NVS:
        defaults_args = {}
        return NVS_offline(**defaults_args)

    #########
    # TESTS #
    #########
    def test_init(self, nvs):
        assert nvs.online == False

    def test_uniqueinstance(self):
        nvs1: NVS = NVS_offline()
        nvs2: NVS = NVS_offline()
        assert nvs1.uid == nvs2.uid

    def test_readonlyinstance(self, nvs):
        with pytest.raises(AttributeError):
            nvs.nvs = "dummy"

    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    def test_vocabulary2uri(self, nvs, rtid):
        assert is_offline_urivalid(nvs._vocabulary2uri(rtid))

    @pytest.mark.parametrize(
        "fmt",
        ["xml", "turtle"],
        indirect=False,
        ids=[f"fmt='{x}'" for x in ["xml", "turtle"]],
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    def test_vocabulary2uri_invalidformat(self, nvs, rtid, fmt):
        with pytest.raises(ValueError):
            nvs._vocabulary2uri(rtid, fmt=fmt)

    @pytest.mark.parametrize(
        "fmt", ["json"], indirect=False, ids=[f"fmt='{x}'" for x in ["json"]]
    )
    @pytest.mark.parametrize(
        "rtid",
        ["R27", None],
        indirect=False,
        ids=[f"rtid='{x}'" for x in ["R27", None]],
    )
    @pytest.mark.parametrize(
        "conceptid",
        ["AANDERAA_OPTODE_3835"],
        indirect=False,
        ids=[f"conceptid='{x}'" for x in ["AANDERAA_OPTODE_3835"]],
    )
    def test_concept2uri(self, nvs, conceptid, rtid, fmt):
        assert is_offline_urivalid(nvs._concept2uri(conceptid, rtid, fmt))

    def test_concept2uri_errors(self, nvs):
        conceptid = "invalid"
        with pytest.raises(ValueError):
            nvs._concept2uri(conceptid)

        conceptid = "1"
        with pytest.raises(ValueError) as e:
            nvs._concept2uri(conceptid)
            assert "This Concept appears in more than one Vocabulary" in str(e)

    @pytest.mark.parametrize(
        "fmt", ["json"], indirect=False, ids=[f"fmt='{x}'" for x in ["json"]]
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    def test_load_vocabulary(self, nvs, rtid, fmt):
        assert DATA_VALIDATOR[fmt](nvs.load_vocabulary(rtid, fmt))

    @pytest.mark.parametrize(
        "fmt", ["json"], indirect=False, ids=[f"fmt='{x}'" for x in ["json"]]
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    @pytest.mark.parametrize(
        "conceptid",
        CONCEPTID_TESTED,
        indirect=False,
        ids=[f"conceptid='{x}'" for x in CONCEPTID_TESTED],
    )
    def test_load_concept(self, nvs, conceptid, rtid, fmt):
        assert DATA_VALIDATOR[fmt](nvs.load_concept(conceptid, rtid, fmt))

    @pytest.mark.parametrize(
        "sub_obj",
        MAPPING_TESTED,
        indirect=False,
        ids=[f"subject_id='{x}', object_id='{y}'" for x, y in MAPPING_TESTED],
    )
    def test_load_mapping(self, nvs, sub_obj):
        assert DATA_VALIDATOR['json'](nvs.load_mapping(sub_obj[0], sub_obj[1]))


@skip_online
class Test_NVS_Online:

    #############
    # UTILITIES #
    #############
    @pytest.fixture
    def nvs(self):
        defaults_args = {}
        if USE_MOCKED_SERVER:
            defaults_args["nvs"] = mocked_server_address

        return NVS_online(**defaults_args)

    #########
    # TESTS #
    #########
    def test_init(self, nvs):
        assert nvs.online == True

    def test_uniqueinstance(self, mocked_httpserver):
        nvs1: NVS = NVS_online(nvs=mocked_server_address)
        nvs2: NVS = NVS_online(nvs=mocked_server_address)
        assert nvs1.uid == nvs2.uid

    def test_readonlyinstance(self, nvs):
        with pytest.raises(AttributeError):
            nvs.nvs = "dummy"

    @pytest.mark.parametrize(
        "fmt", FMT_TESTED, indirect=False, ids=[f"fmt='{x}'" for x in FMT_TESTED]
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    def test_vocabulary2uri(self, nvs, rtid, fmt):
        assert is_online_urivalid(nvs._vocabulary2uri(rtid, fmt))

    @pytest.mark.parametrize(
        "fmt", ["invalid"], indirect=False, ids=[f"fmt='{x}'" for x in ["invalid"]]
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    def test_vocabulary2uri_invalidformat(self, nvs, rtid, fmt):
        with pytest.raises(ValueError):
            nvs._vocabulary2uri(rtid, fmt=fmt)

    @pytest.mark.parametrize(
        "fmt", FMT_TESTED, indirect=False, ids=[f"fmt='{x}'" for x in FMT_TESTED]
    )
    @pytest.mark.parametrize(
        "rtid",
        ["R27", None],
        indirect=False,
        ids=[f"rtid='{x}'" for x in ["R27", None]],
    )
    @pytest.mark.parametrize(
        "conceptid",
        ["AANDERAA_OPTODE_3835"],
        indirect=False,
        ids=[f"conceptid='{x}'" for x in ["AANDERAA_OPTODE_3835"]],
    )
    def test_concept2uri(self, nvs, conceptid, rtid, fmt):
        assert is_online_urivalid(nvs._concept2uri(conceptid, rtid, fmt))

    def test_concept2uri_errors(self, nvs):
        conceptid = "invalid"
        with pytest.raises(ValueError):
            nvs._concept2uri(conceptid)

        conceptid = "1"
        with pytest.raises(ValueError) as e:
            nvs._concept2uri(conceptid)
            assert "This Concept appears in more than one Vocabulary" in str(e)

    @pytest.mark.parametrize(
        "fmt", FMT_TESTED, indirect=False, ids=[f"fmt='{x}'" for x in FMT_TESTED]
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    def test_load_vocabulary(self, nvs, rtid, fmt):
        assert DATA_VALIDATOR[fmt](nvs.load_vocabulary(rtid, fmt))

    @pytest.mark.parametrize(
        "fmt", FMT_TESTED, indirect=False, ids=[f"fmt='{x}'" for x in FMT_TESTED]
    )
    @pytest.mark.parametrize(
        "rtid", RTID_TESTED, indirect=False, ids=[f"rtid='{x}'" for x in RTID_TESTED]
    )
    @pytest.mark.parametrize(
        "conceptid",
        CONCEPTID_TESTED,
        indirect=False,
        ids=[f"conceptid='{x}'" for x in CONCEPTID_TESTED],
    )
    def test_load_concept(self, nvs, conceptid, rtid, fmt):
        assert DATA_VALIDATOR[fmt](nvs.load_concept(conceptid, rtid, fmt))

    # @pytest.mark.parametrize(
    #     "sub_obj",
    #     MAPPING_TESTED,
    #     indirect=False,
    #     ids=[f"subject_id='{x}', object_id='{y}'" for x, y in MAPPING_TESTED],
    # )
    # def test_load_mapping(self, nvs, sub_obj):
    #     assert DATA_VALIDATOR['json'](nvs.load_mapping(sub_obj[0], sub_obj[1]))


def test_NVS_Spec():
    class DummyClass(NVS):
        """Some dummy implementation"""

    with pytest.raises(TypeError):
        DummyClass()


class Test_Utils:

    def test_concept2vocabulary(self):
        assert concept2vocabulary('FLOAT_COASTAL') == ['R22']
        assert concept2vocabulary('dummy') is None

    def test_check_vocabulary(self):
        assert check_vocabulary('R22') == 'R22'
        assert check_vocabulary('PLATFORM_FAMILY') == 'R22'
        assert check_vocabulary('dummy') is None

    def test_id2urn(self):
        assert id2urn("http://vocab.nerc.ac.uk/collection/R27/current/PAL_UW/") == "SDN:R27::PAL_UW"
        assert id2urn("http://vocab.nerc.ac.uk/collection/R27/dummy/PAL_UW/") == 'SDN:R27::'
        with pytest.raises(ValueError):
            id2urn("http://vocab.nerc.ac.uk/collection/Q27/current/PAL_UW/")

    def test_extract_local_attributes(self):
        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors. Local_Attributes:{long_name:Chlorophyll-A signal from fluorescence sensor; standard_name:-; units:count; valid_min:-; valid_max:-; fill_value:99999.f}. Properties:{category:ib; data_type:float}'
        assert isinstance(extract_local_attributes(definition), dict)
        assert 'long_name' in extract_local_attributes(definition)

        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors.'
        assert extract_local_attributes(definition) is None

    def test_extract_properties_section(self):
        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors. Local_Attributes:{long_name:Chlorophyll-A signal from fluorescence sensor; standard_name:-; units:count; valid_min:-; valid_max:-; fill_value:99999.f}. Properties:{category:ib; data_type:float}'
        assert isinstance(extract_properties_section(definition), dict)
        assert 'data_type' in extract_properties_section(definition)

        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors.'
        assert extract_properties_section(definition) is None

    def test_curate_r03definition(self):
        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors. Local_Attributes:{long_name:Chlorophyll-A signal from fluorescence sensor; standard_name:-; units:count; valid_min:-; valid_max:-; fill_value:99999.f}. Properties:{category:ib; data_type:float}'
        data = curate_r03definition(definition)
        assert isinstance(data, dict)
        assert 'Local_Attributes' in data
        assert isinstance(data['Local_Attributes'], LocalAttributes)
        assert 'Properties' in data
        assert isinstance(data['Properties'], Properties)

        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors. Properties:{category:ib; data_type:float}'
        data = curate_r03definition(definition)
        assert isinstance(data, dict)
        assert data['Local_Attributes'] is None
        assert 'Properties' in data
        assert isinstance(data['Properties'], Properties)

        definition = 'Raw fluorescence signal from chlorophyll-a fluorometer, reported by ECO3, FLNTU and FLBB sensors. Local_Attributes:{long_name:Chlorophyll-A signal from fluorescence sensor; standard_name:-; units:count; valid_min:-; valid_max:-; fill_value:99999.f}.'
        data = curate_r03definition(definition)
        assert isinstance(data, dict)
        assert 'Local_Attributes' in data
        assert isinstance(data['Local_Attributes'], LocalAttributes)
        assert 'Properties' in data
        assert data['Properties'] is None

    def test_curate_r14definition(self):
        definition = 'Median of the mixed layer samples taken. Template_Values:{unit:[degC, mdegC]}.'
        data = curate_r14definition(definition)
        assert isinstance(data, dict)
        assert 'Template_Values' in data

    def test_curate_r18definition(self):
        definition = 'Threshold between depth zone #<N> and depth zone #<N+1> for <short_sensor_name>. Template_Values:{short_sensor_name:[Crover, Ctd, Eco, Flbb, Flntu, Ocr, Optode, Sfet, Suna];N:[1..4];unit:[bar, dbar, cbar, mbar, inHg]}.'
        data = curate_r18definition(definition)
        assert isinstance(data, dict)
        assert 'Template_Values' in data

    def test_known_mappings(self):
        maps = known_mappings()
        assert isinstance(maps, list)
        assert all([isinstance(m, tuple) for m in maps])

    def test_sparql_mapping_request(self):
        assert 'R26' in sparql_mapping_request('R26', 'R27')
