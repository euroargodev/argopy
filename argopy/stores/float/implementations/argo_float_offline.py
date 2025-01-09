from errors import InvalidOption
from pathlib import Path
import logging

from ..spec import ArgoFloatProto


log = logging.getLogger("argopy.stores.ArgoFloat")


class ArgoFloatOffline(ArgoFloatProto):
    _online = False

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self.host_protocol != "file":
            raise InvalidOption("Trying to work with the offline store using a remote host !")

        # Load some data (in a perfect world, this should be done asynchronously):
        # self.load_index()
        self.load_metadata()
        self.load_dac()

    def load_metadata(self):
        """Method to load float meta-data"""
        data = {}
        data.update({"deployment": {"launchDate": "?"}})
        data.update({"cycles": []})
        data.update({"networks": ['?']})
        data.update({"platform": {'type': '?'}})
        data.update({"maker": '?'})

        self._metadata = data
        return self._metadata

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
