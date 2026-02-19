import pytest
import logging

from argopy import ArgoFloat
from mocked_http import mocked_server_address
from mocked_http import mocked_httpserver as mocked_erddapserver


log = logging.getLogger("argopy.tests.extensions.optic")
USE_MOCKED_SERVER = False


@pytest.fixture
def Sprof():
    defaults_args = {
        "wmo": 6901864,
        "host": "http",
        "cache": True,
    }
    if USE_MOCKED_SERVER:
        defaults_args["server"] = mocked_server_address

    return ArgoFloat(**defaults_args).open_dataset("Sprof")


@pytest.fixture
def Sprof_CHLA():
    defaults_args = {
        "wmo": 1902303,
        "host": "http",
        "cache": True,
    }
    if USE_MOCKED_SERVER:
        defaults_args["server"] = mocked_server_address

    return ArgoFloat(**defaults_args).open_dataset("Sprof")


@pytest.mark.parametrize(
    "method",
    [
        "percentage",
        "KdPAR",
    ],
    indirect=False,
    ids=["method=%s" % m for m in ["percentage", "KdPAR"]],
)
@pytest.mark.parametrize(
    "inplace",
    [True, False],
    ids=["inplace=%s" % ii for ii in [True, False]],
    indirect=False,
)
def test_compute_Zeu(Sprof, method, inplace, mocked_erddapserver):
    obj = Sprof.argo.optic.Zeu(method=method, inplace=inplace)
    if inplace:
        assert "Zeu" in obj
    else:
        assert obj.name == "Zeu"


@pytest.mark.parametrize(
    "inplace",
    [True, False],
    ids=["inplace=%s" % ii for ii in [True, False]],
    indirect=False,
)
def test_compute_Zpd(Sprof, inplace, mocked_erddapserver):
    obj = Sprof.argo.optic.Zpd(inplace=inplace)
    if inplace:
        assert "Zpd" in obj
    else:
        assert obj.name == "Zpd"


@pytest.mark.parametrize(
    "inplace",
    [True, False],
    ids=["inplace=%s" % ii for ii in [True, False]],
    indirect=False,
)
def test_compute_Z_iPAR_threshold(Sprof, inplace, mocked_erddapserver):
    obj = Sprof.argo.optic.Z_iPAR_threshold(inplace=inplace)
    if inplace:
        assert "Z_iPAR" in obj
    else:
        assert obj.name == "Z_iPAR"


@pytest.mark.parametrize(
    "inplace",
    [True, False],
    ids=["inplace=%s" % ii for ii in [True, False]],
    indirect=False,
)
def test_compute_DCM(Sprof_CHLA, inplace, mocked_erddapserver):
    obj = Sprof_CHLA.argo.optic.DCM(inplace=inplace)
    if inplace:
        assert "DCM" in obj
    else:
        assert obj.name == "DCM"

