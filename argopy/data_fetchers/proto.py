from abc import ABC, abstractmethod


class ArgoDataFetcherProto(ABC):
    @abstractmethod
    def to_xarray(self):
        pass

    @abstractmethod
    def filter_data_mode(self):
        pass

    @abstractmethod
    def filter_qc(self):
        pass

    @abstractmethod
    def filter_variables(self):
        pass

    def clear_cache(self):
        """ Remove cache files and entries from resources open with this fetcher """
        return self.fs.clear_cache()
