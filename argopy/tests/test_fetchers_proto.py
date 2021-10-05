import pytest
from argopy.data_fetchers.proto import ArgoDataFetcherProto


class Fetcher(ArgoDataFetcherProto):

    def to_xarray(self):
        super(Fetcher, self).to_xarray()

    def filter_variables(self):
        super(Fetcher, self).filter_variables()

    def filter_qc(self):
        super(Fetcher, self).filter_qc()

    def filter_data_mode(self):
        super(Fetcher, self).filter_data_mode()


def test_abstracts():
    f = Fetcher()
    with pytest.raises(NotImplementedError):
        f.to_xarray()
    with pytest.raises(NotImplementedError):
        f.filter_variables()
    with pytest.raises(NotImplementedError):
        f.filter_qc()
    with pytest.raises(NotImplementedError):
        f.filter_data_mode()
