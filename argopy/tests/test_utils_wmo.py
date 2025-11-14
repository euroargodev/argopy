import pytest
import numpy as np
from argopy.utils.wmo import float_wmo, is_wmo, check_wmo


def test_is_wmo():
    assert is_wmo(12345)
    assert is_wmo([12345])
    assert is_wmo([12345, 1234567])

    with pytest.raises(ValueError):
        is_wmo(1234, errors="raise")
    with pytest.raises(ValueError):
        is_wmo(-1234, errors="raise")
    with pytest.raises(ValueError):
        is_wmo(1234.12, errors="raise")
    with pytest.raises(ValueError):
        is_wmo(12345.7, errors="raise")

    with pytest.warns(UserWarning):
        is_wmo(1234, errors="warn")
    with pytest.warns(UserWarning):
        is_wmo(-1234, errors="warn")
    with pytest.warns(UserWarning):
        is_wmo(1234.12, errors="warn")
    with pytest.warns(UserWarning):
        is_wmo(12345.7, errors="warn")

    assert not is_wmo(12, errors="ignore")
    assert not is_wmo(-12, errors="ignore")
    assert not is_wmo(1234.12, errors="ignore")
    assert not is_wmo(12345.7, errors="ignore")


def test_check_wmo():
    assert check_wmo(12345) == [12345]
    assert check_wmo([1234567]) == [1234567]
    assert check_wmo([12345, 1234567]) == [12345, 1234567]
    assert check_wmo(np.array((12345, 1234567), dtype='int')) == [12345, 1234567]


class Test_float_wmo():

    def test_init(self):
        assert isinstance(float_wmo(2901746), float_wmo)
        assert isinstance(float_wmo(float_wmo(2901746)), float_wmo)

    def test_isvalid(self):
        assert float_wmo(2901746).isvalid
        assert not float_wmo(12, errors='ignore').isvalid

    def test_ppt(self):
        assert isinstance(str(float_wmo(2901746)), str)
        assert isinstance(repr(float_wmo(2901746)), str)

    def test_comparisons(self):
        assert float_wmo(2901746) == float_wmo(2901746)
        assert float_wmo(2901746) != float_wmo(2901745)
        assert float_wmo(2901746) >= float_wmo(2901746)
        assert float_wmo(2901746) > float_wmo(2901745)
        assert float_wmo(2901746) <= float_wmo(2901746)
        assert float_wmo(2901746) < float_wmo(2901747)

    def test_hashable(self):
        assert isinstance(hash(float_wmo(2901746)), int)
