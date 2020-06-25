from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

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

    def _format(self, x, typ):
        """ string formating helper """
        if typ == 'lon':
            if x < 0:
                x = 360. + x
            return ("%05d") % (x * 100.)
        if typ == 'lat':
            return ("%05d") % (x * 100.)
        if typ == 'prs':
            return ("%05d") % (np.abs(x) * 10.)
        if typ == 'tim':
            return pd.to_datetime(x).strftime('%Y-%m-%d')
        return str(x)
