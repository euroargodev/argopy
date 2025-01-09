from typing import Union
import xarray as xr
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import logging

from ...errors import InvalidOption
from ...plot import dashboard
from ...utils import check_wmo, isconnected, argo_split_path
from ...options import OPTIONS
from .. import ArgoIndex, httpstore


log = logging.getLogger("argopy.stores.ArgoFloat")


class ArgoFloatProto(ABC):
    _metadata = None  # Private holder for float meta-data dictionary
    _dac = None
    _df_profiles = None
    _online = None  # Web access status

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
        self.fs = self.idx.fs["src"]

        if not self._online and self.host_protocol != "file":
            raise InvalidOption("Trying to work with remote host '%s' without a web connection. Check your connection parameters or try to work with a local GDAC path." % self.host)


    @property
    def metadata(self) -> dict:
        """A dictionary holding float meta-data, based on the EA fleet-monitoring API json schema

        Must return a dict with:
        self._metadata["deployment"]["launchDate"]  # pd.Datetime
        self._metadata['cycles']  # list
        self.metadata['networks']  # list of str
        self.metadata["platform"]["type"]  #
        self.metadata["maker"]  #
        """
        if self._metadata is None:
            self.load_metadata()
        return self._metadata

    @abstractmethod
    def load_metadata(self):
        """Method to load float meta-data"""
        raise NotImplementedError("Not implemented")

    def load_index(self):
        """Load the Argo full index in memory and trigger search for this float"""
        self.idx.load().search_wmo(self.WMO)

    @abstractmethod
    def load_dac(self):
        """Load the DAC short name for this float"""
        raise NotImplementedError("Not implemented")

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
        """Number of cycles

        If the float is still active, this is the current value.
        """
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
