from pathlib import Path
import logging
import warnings

from argopy.errors import OptionValueError
from argopy.stores.float.spec import FloatStoreProto


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
            log.debug(
                "Working with the offline store using a remote host !"
            )

    def load_metadata(self):
        """Method to load float meta-data"""
        return self.load_metadata_from_meta_file()

    def load_dac(self):
        """Load the DAC short name for this float"""
        try:
            self._dac = self.idx.query.wmo(self.WMO).read_dac_wmo()[0][
                0
            ]  # Get DAC from Argo index

        except Exception:
            raise ValueError(
                f"DAC name for Float {self.WMO} cannot be found from {self.host}"
            )

        return self
