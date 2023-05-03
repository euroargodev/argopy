from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import xarray
import hashlib
import warnings
from ..plot import dashboard
from ..utilities import list_standard_variables

class ArgoDataFetcherProto(ABC):
    @abstractmethod
    def to_xarray(self, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def filter_data_mode(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def filter_qc(self, ds: xarray.Dataset, *args, **kwargs) -> xarray.Dataset:
        raise NotImplementedError("Not implemented")

    def filter_variables(self, ds: xarray.Dataset, mode: str, *args, **kwargs) -> xarray.Dataset:
        """Filter variables according to user mode"""
        if mode == "standard":
            to_remove = sorted(
                list(set(list(ds.data_vars)) - set(list_standard_variables()))
            )
            return ds.drop_vars(to_remove)
        elif mode == "research":
            to_remove = sorted(
                list(set(list(ds.data_vars)) - set(list_standard_variables()))
            )
            to_remove.append('DATA_MODE')
            [to_remove.append(v) for v in ds.data_vars if "QC" in v]
            return ds.drop_vars(to_remove)
        else:
            return ds

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
        cname = "?"

        if hasattr(self, "BOX"):
            BOX = self.BOX
            cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f]") % (
                BOX[0],
                BOX[1],
                BOX[2],
                BOX[3],
            )
            if len(BOX) == 6:
                cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; z=%0.1f/%0.1f]") % (
                    BOX[0],
                    BOX[1],
                    BOX[2],
                    BOX[3],
                    BOX[4],
                    BOX[5],
                )
            if len(BOX) == 8:
                cname = ("[x=%0.2f/%0.2f; y=%0.2f/%0.2f; z=%0.1f/%0.1f; t=%s/%s]") % (
                    BOX[0],
                    BOX[1],
                    BOX[2],
                    BOX[3],
                    BOX[4],
                    BOX[5],
                    self._format(BOX[6], "tim"),
                    self._format(BOX[7], "tim"),
                )

        elif hasattr(self, "WMO"):
            prtcyc = lambda f, wmo: "WMO%i_%s" % (  # noqa: E731
                wmo,
                "_".join(["CYC%i" % (cyc) for cyc in sorted(f.CYC)]),
            )
            if len(self.WMO) == 1:
                if hasattr(self, "CYC") and self.CYC is not None:
                    cname = prtcyc(self, self.WMO[0])
                else:
                    cname = "WMO%i" % (self.WMO[0])
            else:
                cname = ";".join(["WMO%i" % wmo for wmo in sorted(self.WMO)])
                if hasattr(self, "CYC") and self.CYC is not None:
                    cname = ";".join([prtcyc(self, wmo) for wmo in self.WMO])
            if hasattr(self, "dataset_id"):
                cname = self.dataset_id + ";" + cname

        return cname

    @property
    def sha(self) -> str:
        """ Returns a unique SHA for a specific cname / fetcher implementation"""
        path = "%s-%s" % (self.definition, self.cname())
        return hashlib.sha256(path.encode()).hexdigest()

    def dashboard(self, **kw):
        """Return 3rd party dashboard for the access point"""
        if self.WMO is not None:
            if len(self.WMO) == 1 and self.CYC is not None and len(self.CYC) == 1:
                return dashboard(wmo=self.WMO[0], cyc=self.CYC[0], **kw)
            elif len(self.WMO) == 1:
                return dashboard(wmo=self.WMO[0], **kw)
            else:
                warnings.warn("Dashboard only available for a single float or profile request")
