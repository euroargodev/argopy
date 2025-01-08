from typing import Union
import xarray as xr
from pathlib import Path
import pandas as pd

from ...plot import dashboard
from ...utils import check_wmo, isconnected, argo_split_path
from ...options import OPTIONS
from .. import ArgoIndex, httpstore

class ArgoFloat:
    _dac = None
    _fleetmonitoring_data = None
    _df_profiles = None

    def __init__(
        self,
        wmo: Union[int, str],
        host: str = None,
        cache: bool = False,
        cachedir: str = "",
        timeout: int = 0,
    ):

        self.WMO = check_wmo(wmo)[0]
        self.host = OPTIONS["gdac"] if host is None else host
        self.cache = bool(cache)
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout

        self.idx = ArgoIndex(
            index_file="core",
            host=self.host,
            cache=self.cache,
            cachedir=self.cachedir,
            timeout=self.timeout,
        )

        self.host = self.idx.host  # Fix host shortcuts with correct values
        if self.host_protocol == "s3":
            self.host = self.host.replace(
                "/idx", ""
            )  # Fix s3 anomaly whereby index files are not at the 'dac' level

        self.fs = self.idx.fs["src"]

        # Load some data (in a perfect world, this should be done asynchrounesly):
        # self.load_index()
        if self.host_protocol != "file":
            self.load_fleetmonitoring_metadata()
        self.load_dac()

    def load_fleetmonitoring_metadata(self):
        """Load float meta data from Euro-Argo fleet-monitoring API"""
        api_server = "https://fleetmonitoring.euro-argo.eu"
        # api_point = f"{api_server}/floats/basic/{self.WMO}"
        api_point = f"{api_server}/floats/{self.WMO}"
        self._fleetmonitoring_data = httpstore(
            cache=self.cache, cachedir=self.cachedir
        ).open_json(api_point)

        # Fix data type for some useful keys:
        self._fleetmonitoring_data["deployment"]["launchDate"] = pd.to_datetime(
            self._fleetmonitoring_data["deployment"]["launchDate"]
        )

    @property
    def fleetmonitoring_metadata(self):
        if self._fleetmonitoring_data is None:
            self.load_fleetmonitoring_metadata()
        return self._fleetmonitoring_data

    def load_index(self):
        """Load the Argo full index in memory and search for this float"""
        self.idx.load().search_wmo(self.WMO)

    def load_dac(self):
        """Load the DAC short name for this float"""
        try:
            if self.host_protocol == "file":
                # With local files, we assume to work preferentially offline
                # Get DAC from local files:
                dac = [
                    p.parts[-2]
                    for p in Path(self.host).glob(
                        self.host_sep.join(["dac", "*", "%i" % self.WMO])
                    )
                ]
                if len(dac) > 0:
                    self._dac = dac[0]

            else:
                # Get DAC from EA-Metadata API:
                self._dac = self._fleetmonitoring_data["dataCenter"]["name"].lower()
        except:
            raise ValueError(
                f"DAC name for Float {self.WMO} cannot be found from {self.host}"
            )

        # For the record, another method to get the DAC name, based on the profile index
        # self._dac = self.idx.search_wmo(self.WMO).read_dac_wmo()[0][0] # Get DAC from Argo index

    @property
    def host_protocol(self) -> str:
        """Protocol of the GDAC host"""
        return self.idx.fs["src"].protocol

    @property
    def host_sep(self) -> str:
        """Host path separator"""
        return self.fs.fs.sep

    @property
    def dac(self) -> str:
        """Name of the DAC responsible for this float"""
        return self._dac

    @property
    def path(self) -> str:
        """Return float path

        Since path type depends on the host protocol, this property is always a string
        """
        return self.host_sep.join([self.host, "dac", self.dac, "%i" % self.WMO])

    def ls(self) -> list:
        """Return the list of files in float path"""
        paths = self.fs.glob(self.host_sep.join([self.path, "*"]))
        paths = [p for p in paths if Path(p).suffix != ""]
        paths.sort()
        return paths

    def lsprofiles(self) -> list:
        """Return the list of files in float profiles path"""
        paths = self.fs.glob(self.host_sep.join([self.path, "profiles", "*"]))
        paths = [p for p in paths if Path(p).suffix != ""]
        paths.sort()
        return paths

    def avail_dataset(self) -> dict:
        """Dictionary of available dataset for this float"""
        avail = {}
        for file in self.ls():
            filename = file.split(self.host_sep)[-1]
            if Path(filename).suffix == ".nc":
                name = Path(filename).stem.split("_")[-1]
                avail.update({name: file})
        return dict(sorted(avail.items()))

    def open_dataset(self, name: str = "prof", casted: bool = True) -> xr.Dataset:
        if name not in self.avail_dataset():
            raise ValueError(
                "Dataset '%s' not found. Available dataset for this float are: %s"
                % (name, self.avail_dataset().keys())
            )
        else:
            file = self.avail_dataset()[name]
            xr_opts = {"engine": "argo"} if casted else {}
            return self.fs.open_dataset(file, xr_opts=xr_opts)

    @property
    def N_CYCLES(self) -> int:
        return len(self.fleetmonitoring_metadata['cycles'])

    def describe_profiles(self) -> pd.DataFrame:
        """Return a :class:`pd.DataFrame` describing profile files"""
        if self._df_profiles is None:
            prof = []
            for file in self.lsprofiles():
                desc = {}
                desc['stem'] = Path(file).stem
                desc = {**desc, **argo_split_path(file)}
                for v in ['dac', 'wmo', 'extension', 'name', 'origin', 'path']:
                    desc.pop(v)
                desc['path'] = file
                prof.append(desc)
            df = pd.DataFrame(data=prof)
            stem2cyc = lambda s: int(s.split("_")[-1][0:-1]) if s.split("_")[-1][-1] == 'D' else int(s.split("_")[-1][:])
            row2cyc = lambda row: stem2cyc(row['stem'])
            df['cyc'] = df.apply(row2cyc, axis=1)
            df['long_type'] = df.apply(lambda row: row['type'].split(',')[-1].lstrip(), axis=1)
            df['type'] = df.apply(lambda row: row['type'][0], axis=1)
            df['data_mode'] = df.apply(lambda row: row['data_mode'][0], axis=1)
            self._df_profiles = df
        return self._df_profiles


    def __repr__(self):
        summary = ["<argofloat.%i.%s>" % (self.WMO, self.host_protocol)]
        status = "online ✅" if isconnected(self.path, maxtry=1) else "offline 🚫"
        summary.append("GDAC host: %s [%s]" % (self.host, status))
        summary.append("DAC name: %s" % self.dac)
        summary.append("Network(s): %s" % self.fleetmonitoring_metadata['networks'])

        launchDate = self.fleetmonitoring_metadata["deployment"]["launchDate"]
        today = pd.to_datetime('now', utc=True)
        summary.append(
            "Deployment date: %s [%s ago]"
            % (launchDate, (today-launchDate).days)
        )
        summary.append(
            "Float type and manufacturer: %s [%s]"
            % (
                self.fleetmonitoring_metadata["platform"]["type"],
                self.fleetmonitoring_metadata["maker"],
            )
        )
        summary.append("Number of cycles: %s" % self.N_CYCLES)
        summary.append("Dashboard: %s" % dashboard(wmo=self.WMO, url_only=True))
        summary.append("Netcdf dataset available: %s" % list(self.avail_dataset().keys()))

        return "\n".join(summary)
