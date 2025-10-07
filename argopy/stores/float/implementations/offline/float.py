from pathlib import Path
import logging

from .....errors import InvalidOption
from ...spec import FloatStoreProto


log = logging.getLogger("argopy.stores.offline.FloatStore")


class FloatStore(FloatStoreProto):
    """:class:`FloatStore` implementation for local GDAC files, no internet connection."""

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

    def load_metadata(self):
        """Method to load float meta-data"""
        return self.load_metadata_from_meta_file()

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

        except Exception:
            raise ValueError(
                f"DAC name for Float {self.WMO} cannot be found from {self.host}"
            )

        # For the record, another method to get the DAC name, based on the profile index:
        # self._dac = self.idx.search_wmo(self.WMO).read_dac_wmo()[0][0] # Get DAC from Argo index
        return self
