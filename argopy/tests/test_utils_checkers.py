import pytest
import numpy as np

from argopy.utils import (
    is_box, is_indexbox,
    check_wmo, is_wmo,
    check_cyc, is_cyc,
)


class Test_is_box:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.BOX3d = [0, 20, 40, 60, 0, 1000]
        self.BOX4d = [0, 20, 40, 60, 0, 1000, "2001-01", "2001-6"]

    def test_box_ok(self):
        assert is_box(self.BOX3d)
        assert is_box(self.BOX4d)

    def test_box_notok(self):
        for box in [[], list(range(0, 12))]:
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")

    def test_box_invalid_num(self):
        for i in [0, 1, 2, 3, 4, 5]:
            box = self.BOX3d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")

    def test_box_invalid_range(self):
        for i in [0, 1, 2, 3, 4, 5]:
            box = self.BOX3d
            box[i] = -1000
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")

    def test_box_invalid_str(self):
        for i in [6, 7]:
            box = self.BOX4d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_box(box)
            with pytest.raises(ValueError):
                is_box(box, errors="raise")
            assert not is_box(box, errors="ignore")


class Test_is_indexbox:
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.BOX2d = [0, 20, 40, 60]
        self.BOX3d = [0, 20, 40, 60, "2001-01", "2001-6"]

    def test_box_ok(self):
        assert is_indexbox(self.BOX2d)
        assert is_indexbox(self.BOX3d)

    def test_box_notok(self):
        for box in [[], list(range(0, 12))]:
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")

    def test_box_invalid_num(self):
        for i in [0, 1, 2, 3]:
            box = self.BOX2d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")

    def test_box_invalid_range(self):
        for i in [0, 1, 2, 3]:
            box = self.BOX2d
            box[i] = -1000
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")

    def test_box_invalid_str(self):
        for i in [4, 5]:
            box = self.BOX3d
            box[i] = "str"
            with pytest.raises(ValueError):
                is_indexbox(box)
            with pytest.raises(ValueError):
                is_indexbox(box, errors="raise")
            assert not is_indexbox(box, errors="ignore")


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


def test_is_cyc():
    assert is_cyc(123)
    assert is_cyc([123])
    assert is_cyc([12, 123, 1234])

    with pytest.raises(ValueError):
        is_cyc(12345, errors="raise")
    with pytest.raises(ValueError):
        is_cyc(-1234, errors="raise")
    with pytest.raises(ValueError):
        is_cyc(1234.12, errors="raise")
    with pytest.raises(ValueError):
        is_cyc(12345.7, errors="raise")

    with pytest.warns(UserWarning):
        is_cyc(12345, errors="warn")
    with pytest.warns(UserWarning):
        is_cyc(-1234, errors="warn")
    with pytest.warns(UserWarning):
        is_cyc(1234.12, errors="warn")
    with pytest.warns(UserWarning):
        is_cyc(12345.7, errors="warn")

    assert not is_cyc(12345, errors="ignore")
    assert not is_cyc(-12, errors="ignore")
    assert not is_cyc(1234.12, errors="ignore")
    assert not is_cyc(12345.7, errors="ignore")


def test_check_cyc():
    assert check_cyc(123) == [123]
    assert check_cyc([12]) == [12]
    assert check_cyc([12, 123]) == [12, 123]
    assert check_cyc(np.array((123, 1234), dtype='int')) == [123, 1234]

