import pytest
import xarray
from argopy.data_fetchers.proto import ArgoDataFetcherProto


class Fetcher(ArgoDataFetcherProto):

    def to_xarray(self, *args, **kwargs):
        super(Fetcher, self).to_xarray(*args, **kwargs)

    def filter_variables(self, *args, **kwargs):
        super(Fetcher, self).filter_variables(*args, **kwargs)

    def filter_qc(self, *args, **kwargs):
        super(Fetcher, self).filter_qc(*args, **kwargs)

    def filter_data_mode(self, *args, **kwargs):
        super(Fetcher, self).filter_data_mode(*args, **kwargs)


def test_abstracts():
    f = Fetcher()
    with pytest.raises(NotImplementedError):
        f.to_xarray()
    with pytest.raises(NotImplementedError):
        f.filter_variables(xarray.Dataset, str)
    with pytest.raises(NotImplementedError):
        f.filter_qc(xarray.Dataset)
    with pytest.raises(NotImplementedError):
        f.filter_data_mode(xarray.Dataset)
