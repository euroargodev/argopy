from typing import Union
import xarray as xr
from pathlib import Path
import pandas as pd
import logging

from ....plot import dashboard
from ....utils import check_wmo, isconnected, argo_split_path
from ....options import OPTIONS
from ... import ArgoIndex, httpstore
from ..spec import ArgoFloatProto


log = logging.getLogger("argopy.stores.ArgoFloat")


class ArgoFloatOnline(ArgoFloatProto):
    _online = True

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.host_protocol == "s3":
            self.host = self.host.replace(
                "/idx", ""
            )  # Fix s3 anomaly whereby index files are not at the 'dac' level

        # Load some data (in a perfect world, this should be done asynchronously):
        # self.load_index()
        self.load_metadata()
        self.load_dac()

    def load_metadata(self):
        """Load float meta data from Euro-Argo fleet-monitoring API"""
        api_server = "https://fleetmonitoring.euro-argo.eu"
        # api_point = f"{api_server}/floats/basic/{self.WMO}"
        api_point = f"{api_server}/floats/{self.WMO}"
        self._metadata = httpstore(
            cache=self.cache, cachedir=self.cachedir
        ).open_json(api_point)

        # Fix data type for some useful keys:
        self._metadata["deployment"]["launchDate"] = pd.to_datetime(
            self._metadata["deployment"]["launchDate"]
        )

    def load_dac(self):
        """Load the DAC short name for this float"""
        try:
            # Get DAC from EA-Metadata API:
            self._dac = self._metadata["dataCenter"]["name"].lower()
        except:
            raise ValueError(
                f"DAC name for Float {self.WMO} cannot be found from {self.host}"
            )

        # For the record, another method to get the DAC name, based on the profile index
        # self._dac = self.idx.search_wmo(self.WMO).read_dac_wmo()[0][0] # Get DAC from Argo index

    def __repr__(self):
        summary = ["<argofloat.%i.%s>" % (self.WMO, self.host_protocol)]
        status = "online ✅" if isconnected(self.path, maxtry=1) else "offline 🚫"
        summary.append("GDAC host: %s [%s]" % (self.host, status))
        summary.append("DAC name: %s" % self.dac)
        summary.append("Network(s): %s" % self.metadata['networks'])

        launchDate = self.metadata["deployment"]["launchDate"]
        today = pd.to_datetime('now', utc=True)
        summary.append(
            "Deployment date: %s [%s ago]"
            % (launchDate, (today-launchDate).days)
        )
        summary.append(
            "Float type and manufacturer: %s [%s]"
            % (
                self.metadata["platform"]["type"],
                self.metadata["maker"],
            )
        )
        summary.append("Number of cycles: %s" % self.N_CYCLES)
        summary.append("Dashboard: %s" % dashboard(wmo=self.WMO, url_only=True))
        summary.append("Netcdf dataset available: %s" % list(self.avail_dataset().keys()))

        return "\n".join(summary)
