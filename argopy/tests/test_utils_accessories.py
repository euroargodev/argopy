import pytest
from argopy.utils.accessories import float_wmo, Registry


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


class Test_Registry():

    opts = [(None, 'str'), (['hello', 'world'], str), (None, float_wmo), ([2901746, 4902252], float_wmo)]
    opts_ids = ["%s, %s" % ((lambda x: 'iterlist' if x is not None else x)(opt[0]), repr(opt[1])) for opt in opts]

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_init(self, opts):
        assert isinstance(Registry(opts[0], dtype=opts[1]), Registry)

    opts = [(['hello', 'world'], str), ([2901746, 4902252], float_wmo)]
    opts_ids = ["%s, %s" % ((lambda x: 'iterlist' if x is not None else x)(opt[0]), repr(opt[1])) for opt in opts]

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_commit(self, opts):
        R = Registry(dtype=opts[1])
        R.commit(opts[0])

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_append(self, opts):
        R = Registry(dtype=opts[1])
        R.append(opts[0][0])

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_extend(self, opts):
        R = Registry(dtype=opts[1])
        R.append(opts[0])

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_insert(self, opts):
        R = Registry(opts[0][0], dtype=opts[1])
        R.insert(0, opts[0][-1])
        assert R[0] == opts[0][-1]

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_remove(self, opts):
        R = Registry(opts[0], dtype=opts[1])
        R.remove(opts[0][0])
        assert opts[0][0] not in R

    @pytest.mark.parametrize("opts", opts, indirect=False, ids=opts_ids)
    def test_copy(self, opts):
        R = Registry(opts[0], dtype=opts[1])
        assert R == R.copy()

    bad_opts = [(['hello', 12], str), ([2901746, 1], float_wmo)]
    bad_opts_ids = ["%s, %s" % ((lambda x: 'iterlist' if x is not None else x)(opt[0]), repr(opt[1])) for opt in opts]

    @pytest.mark.parametrize("opts", bad_opts, indirect=False, ids=bad_opts_ids)
    def test_invalid_dtype(self, opts):
        with pytest.raises(ValueError):
            Registry(opts[0][0], dtype=opts[1], invalid='raise').commit(opts[0][-1])
        with pytest.warns(UserWarning):
            Registry(opts[0][0], dtype=opts[1], invalid='warn').commit(opts[0][-1])
        # Raise nothing:
        Registry(opts[0][0], dtype=opts[1], invalid='ignore').commit(opts[0][-1])

