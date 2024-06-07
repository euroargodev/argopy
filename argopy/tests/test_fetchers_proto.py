import pytest
import xarray
from argopy.data_fetchers.proto import ArgoDataFetcherProto
from argopy.utils import to_list


class Fetcher(ArgoDataFetcherProto):
    dataset_id = 'phy'

    def to_xarray(self, *args, **kwargs):
        super(Fetcher, self).to_xarray(*args, **kwargs)

    def transform_data_mode(self, *args, **kwargs):
        super(Fetcher, self).transform_data_mode(*args, **kwargs)

    def filter_qc(self, *args, **kwargs):
        super(Fetcher, self).filter_qc(*args, **kwargs)

    def filter_researchmode(self, *args, **kwargs):
        super(Fetcher, self).filter_researchmode(*args, **kwargs)


def test_required_methods():
    f = Fetcher()
    with pytest.raises(NotImplementedError):
        f.to_xarray()

    with pytest.raises(NotImplementedError):
        f.transform_data_mode(xarray.Dataset, str)

    with pytest.raises(NotImplementedError):
        f.filter_qc(xarray.Dataset)

    with pytest.raises(NotImplementedError):
        f.filter_researchmode(xarray.Dataset)


@pytest.mark.parametrize("profile", [[13857, None], [13857, 90]],
                         indirect=False,
                         ids=["%s" % p for p in [[13857, None], [13857, 90]]])
def test_dashboard(profile):

    f = Fetcher()
    f.WMO, f.CYC = profile
    f.WMO = to_list(f.WMO)
    f.CYC = to_list(f.CYC)
    assert isinstance(f.dashboard(url_only=True), str)

    with pytest.warns(UserWarning):
        f = Fetcher()
        f.WMO = [1901393, 6902746]
        f.CYC = None
        f.dashboard(url_only=True)
