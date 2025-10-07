from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import xarray
import hashlib
import warnings
import logging
from ..plot import dashboard
from ..utils.lists import list_standard_variables
from ..utils.format import UriCName, format_oneline


log = logging.getLogger("argopy.fetcher.proto")


class ArgoDataFetcherProto(ABC):
    @abstractmethod
    def to_xarray(self, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def transform_data_mode(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def filter_data_mode(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def filter_qc(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def filter_researchmode(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    def filter_variables(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        """Filter variables according to dataset and user mode"""
        if self.dataset_id in ['phy']:

            if self.user_mode == "standard":
                to_remove = sorted(
                    list(set(list(ds.data_vars)) - set(list_standard_variables()))
                )
                return ds.drop_vars(to_remove)

            elif self.user_mode == "research":
                to_remove = sorted(
                    list(set(list(ds.data_vars)) - set(list_standard_variables()))
                )
                to_remove.append('DATA_MODE')
                [to_remove.append(v) for v in ds.data_vars if "QC" in v]
                return ds.drop_vars(to_remove)

            else:
                return ds

        elif self.dataset_id in ['bgc', 'bgc-s']:

            if self.user_mode == "standard":
                to_remove = sorted(
                    list(set(list(ds.data_vars)) - set(list_standard_variables(ds=self.dataset_id)))
                )
                [to_remove.append(v) for v in ds.data_vars if 'CDOM' in v]
                return ds.drop_vars(to_remove)

            elif self.user_mode == "research":
                to_remove = sorted(
                    list(set(list(ds.data_vars)) - set(list_standard_variables(ds=self.dataset_id)))
                )
                [to_remove.append(v) for v in ds.data_vars if 'CDOM' in v]
                [to_remove.append(v) for v in ds.data_vars if "DATA_MODE" in v]
                [to_remove.append(v) for v in ds.data_vars if "QC" in v]
                return ds.drop_vars(to_remove)

            else:
                return ds

        else:
            raise ValueError("No filter_variables support for ds='%s'" % self.dataset_id)

    def clear_cache(self):
        """ Remove cache files and entries from resources opened with this fetcher """
        return self.fs.clear_cache()

    def _format(self, x, typ: str) -> str:
        """ string formatting helper """
        if typ == "lon":
            if x < 0:
                x = 360.0 + x
            return ("%05d") % (x * 100.0)
        if typ == "lat":
            return ("%05d") % (x * 100.0)
        if typ == "prs":
            return ("%05d") % (np.abs(x) * 10.0)
        if typ == "tim":
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        return str(x)

    def _cname(self) -> str:
        """ Fetcher one line string definition helper """
        return UriCName(self).cname

    @property
    def sha(self) -> str:
        """ Returns a unique SHA for a specific cname / fetcher implementation"""
        path = "%s-%s-%s" % (self.definition, self.cname(), self.user_mode)
        if self.dataset_id in ['bgc', 'bgc-s']:
            path = "%s-%s-%s" % (path, self._bgc_params, self._bgc_measured)
        return hashlib.sha256(path.encode()).hexdigest()

    def dashboard(self, **kw):
        """Return 3rd party dashboard for the access point"""
        if 'type' not in kw and self.dataset_id in ['bgc', 'bgc-s']:
            kw['type'] = 'bgc'
        if self.WMO is not None:
            if len(self.WMO) == 1 and self.CYC is not None and len(self.CYC) == 1:
                return dashboard(wmo=self.WMO[0], cyc=self.CYC[0], **kw)
            elif len(self.WMO) == 1:
                return dashboard(wmo=self.WMO[0], **kw)
            else:
                warnings.warn("Dashboard only available for a single float or profile request")

    @property
    def _icon_access_point(self):
        if hasattr(self, "WMO"):
            if self.CYC is not None:
                return "⚓"
            else:
                return "🤖"
        else:
            return "🗺 "

    @property
    def _icon_data_source(self):
        if self.data_source == 'erddap':
            return "⭐"
        if self.data_source == 'gdac':
            return "🌐"
        if self.data_source == 'argovis':
            return "👁"

    @property
    def _repr_access_point(self):
        if hasattr(self, "BOX"):
            return "%s Domain: %s" % (self._icon_access_point, self.cname())
        else:
            return "%s Domain: %s" % (self._icon_access_point, format_oneline(self.cname()))

    @property
    def _repr_server(self):
        return "🔗 API: %s" % self.server

    @property
    def _repr_data_source(self):
        return "%s Name: %s" % (self._icon_data_source, self.definition)
