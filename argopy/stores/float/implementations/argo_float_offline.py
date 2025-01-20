import pandas as pd
import numpy as np
from pathlib import Path
import logging

from ....errors import InvalidOption
from ..spec import ArgoFloatProto


log = logging.getLogger("argopy.stores.ArgoFloat")


class ArgoFloatOffline(ArgoFloatProto):
    """Offline :class:`ArgoFloat` implementation"""
    _online = False

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.host_protocol != "file":
            raise InvalidOption(
                "Trying to work with the offline store using a remote host !"
            )

        # Load some data (in a perfect world, this should be done asynchronously):
        self.load_dac()
        self.load_metadata()  # must come after dac because metadata are read from netcdf files requiring dac folder name

    def load_metadata(self):
        """Method to load float meta-data"""
        data = {}

        ds = self.open_dataset("meta")
        data.update(
            {
                "deployment": {
                    "launchDate": pd.to_datetime(ds["LAUNCH_DATE"].values, utc=True)
                }
            }
        )
        data.update(
            {"platform": {"type": ds["PLATFORM_TYPE"].values[np.newaxis][0].strip()}}
        )
        data.update({"maker": ds["PLATFORM_MAKER"].values[np.newaxis][0].strip()})

        def infer_network(this_ds):
            if this_ds["PLATFORM_FAMILY"].values[np.newaxis][0].strip() == "FLOAT_DEEP":
                network = ["DEEP"]
                if len(this_ds["SENSOR"].values) > 4:
                    network.append("BGC")

            elif this_ds["PLATFORM_FAMILY"].values[np.newaxis][0].strip() == "FLOAT":
                if len(this_ds["SENSOR"].values) > 4:
                    network = ["BGC"]
                else:
                    network = ["CORE"]

            else:
                network = ["?"]

            return network

        data.update({"networks": infer_network(ds)})

        data.update({"cycles": np.unique(self.open_dataset("prof")["CYCLE_NUMBER"])})

        self._metadata = data

    def load_dac(self):
        """Load the DAC short name for this float"""
        try:
            dac = [
                p.parts[-2]
                for p in Path(self.host).glob(
                    self.host_sep.join(["dac", "*", "%i" % self.WMO])
                )
            ]
            if len(dac) > 0:
                self._dac = dac[0]

        except:
            raise ValueError(
                f"DAC name for Float {self.WMO} cannot be found from {self.host}"
            )

        # For the record, another method to get the DAC name, based on the profile index:
        # self._dac = self.idx.search_wmo(self.WMO).read_dac_wmo()[0][0] # Get DAC from Argo index
