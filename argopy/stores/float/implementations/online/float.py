import pandas as pd
import logging

from .....errors import DataNotFound
from .... import httpstore
from ...spec import FloatStoreProto


log = logging.getLogger("argopy.stores.online.FloatStore")


class FloatStore(FloatStoreProto):
    """:class:`ArgoFloat` implementation using web access and full advantage of Argo web-API."""

    _online = True
    _eafleetmonitoring_server = "https://fleetmonitoring.euro-argo.eu"
    _technicaldata = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.host_protocol == "s3":
            self.host = self.host.replace(
                "/idx", ""
            )  # Fix s3 anomaly whereby index files are not at the 'dac' level

    @property
    def api_point(self):
        """Euro-Argo fleet-monitoring API points"""
        points = {}

        # points['meta'] = f"{self._eafleetmonitoring_server}/floats/basic/{self.WMO}"
        points["meta"] = f"{self._eafleetmonitoring_server}/floats/{self.WMO}"

        # points['technical'] = f"{self._eafleetmonitoring_server}/technical-data/basic/{self.WMO}"
        points["technical"] = (
            f"{self._eafleetmonitoring_server}/technical-data/{self.WMO}"
        )

        return points

    def load_metadata(self):
        """Load float metadata from Euro-Argo fleet-monitoring API

        Notes
        -----
        API point is stored in the :class:`ArgoFloat.api_point` attribute.

        See Also
        --------
        :class:`ArgoFloat.load_technicaldata`
        """
        try:
            self._metadata = httpstore(
                cache=self.cache, cachedir=self.cachedir
            ).open_json(self.api_point["meta"], errors="raise")
        except DataNotFound:
            # Try to load metadata from the meta file
            # to so, we first need the DAC name
            self._dac = self.idx.search_wmo(self.WMO).read_dac_wmo()[0][0]
            self.load_metadata_from_meta_file()

        # Fix data type for some useful keys:
        self._metadata["deployment"]["launchDate"] = pd.to_datetime(
            self._metadata["deployment"]["launchDate"]
        )

        return self

    def load_technicaldata(self):
        """Load float technical data from Euro-Argo fleet-monitoring API

        Notes
        -----
        API point is stored in the :class:`ArgoFloat.api_point` attribute.

        See Also
        --------
        :class:`ArgoFloat.load_metadata`
        """
        self._technicaldata = httpstore(
            cache=self.cache, cachedir=self.cachedir
        ).open_json(self.api_point["technical"])

        return self

    @property
    def technicaldata(self) -> dict:
        """A dictionary holding float technical data"""
        if self._technicaldata is None:
            self.load_technicaldata()
        return self._technicaldata

    def load_dac(self):
        """Load the DAC short name for this float"""
        try:
            # Get DAC from EA-Metadata API:
            self._dac = self.metadata["dataCenter"]["name"].lower()
        except Exception:
            try:
                self._dac = self.idx.query.wmo(self.WMO).read_dac_wmo()[0][
                    0
                ]  # Get DAC from Argo index
            except Exception:
                raise ValueError(
                    f"DAC name for Float {self.WMO} cannot be found from {self.host}"
                )
        return self
