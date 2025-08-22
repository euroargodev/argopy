from typing import Union

import fsspec.core

import xarray as xr
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import logging
import numpy as np

from ...errors import InvalidOption
from ...plot import dashboard
from ...utils import check_wmo, argo_split_path, shortcut2gdac
from ...options import OPTIONS
from .. import ArgoIndex


log = logging.getLogger("argopy.stores.ArgoFloat")


class FloatStoreProto(ABC):
    def __init__(
        self,
        wmo: Union[int, str],
        host: str = None,
        aux: bool = False,
        cache: bool = False,
        cachedir: str = "",
        timeout: int = 0,
        **kwargs,
    ):
        """Create an Argo float store

        Parameters
        ----------
        wmo: int or str
            The float WMO number. It will be validated against the Argo convention and raise an :class:`ValueError` if not compliant.
        host: str, optional, default: OPTIONS['gdac']
            Local or remote (http, ftp or s3) path where a ``dac`` folder is to be found (compliant with GDAC structure).

            This parameter takes values like:

            - a local absolute path
            - ``https://data-argo.ifremer.fr``, shortcut with ``http`` or ``https``
            - ``https://usgodae.org/pub/outgoing/argo``, shortcut with ``us-http`` or ``us-https``
            - ``ftp://ftp.ifremer.fr/ifremer/argo``, shortcut with ``ftp``
            - ``s3://argo-gdac-sandbox/pub``, shortcut with ``s3`` or ``aws``
        aux: bool, default = False
            Should we include dataset from the auxiliary data folder. The 'aux' folder is expected to be at the same path level as the 'dac' folder on the GDAC host.
        cache : bool, optional, default: False
            Use cache or not.
        cachedir: str, optional, default: OPTIONS['cachedir']
            Folder where to store cached files.
        timeout: int, optional, default: OPTIONS['api_timeout']
            Time out in seconds to connect to a remote host (ftp or http).
        """
        self.WMO = check_wmo(wmo)[0]
        self.host = OPTIONS["gdac"] if host is None else shortcut2gdac(host)
        self.cache = bool(cache)
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout
        self._aux = bool(aux)

        if not self._online and (self.host.startswith('http') or self.host.startswith('ftp') or self.host.startswith('s3')):
            raise InvalidOption(
                "Trying to work with remote host '%s' without a web connection. Check your connection parameters or try to work with a local GDAC path."
                % self.host
            )

        if "idx" not in kwargs:
            self.idx = ArgoIndex(
                index_file="core",
                host=self.host,
                cache=self.cache,
                cachedir=self.cachedir,
                timeout=self.timeout,
            )
        else:
            self.idx = kwargs["idx"]

        self.host = self.idx.host  # Fix host shortcuts with correct values
        self.fs = self.idx.fs["src"]

        # Load some data (in a perfect world, this should be done asynchronously):
        # self.load_index()

        # Init Internal placeholder for this instance:
        self._dataset = {}  # xarray datasets
        self._metadata = None  # Float meta-data dictionary
        self._dac = None  # DAC name (string)
        self._df_profiles = None  # Dataframe with profiles index
        self._online = None  # Web access status

    def load_index(self):
        """Load the Argo full index in memory and trigger search for this float"""
        self.idx.load().query.wmo(self.WMO)
        return self

    @property
    def metadata(self) -> dict:
        """A dictionary of float meta-data

        The meta-data dictionary will have at least the following keys:

        .. code-block:: python

            metadata["deployment"]["launchDate"]  # pd.Datetime
            metadata['cycles']  # list
            metadata['networks']  # list of str, eg ['BGC', 'DEEP']
            metadata["platform"]["type"]  # str from Reference table 23
            metadata["maker"]  # str from Reference table 24

        The exact dictionary content depends on the :class:`ArgoFloat` backend (online vs offline) providing meta-data.

        """
        if self._metadata is None:
            self.load_metadata()
        return self._metadata

    @abstractmethod
    def load_metadata(self):
        """Method to load float meta-data"""
        raise NotImplementedError("Not implemented")

    def load_metadata_from_meta_file(self):
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

        return self

    @abstractmethod
    def load_dac(self):
        """Load the DAC short name for this float"""
        raise NotImplementedError("Not implemented")

    @property
    def dac(self) -> str:
        """Name of the DAC responsible for this float"""
        if self._dac is None:
            self.load_dac()
        return self._dac

    @property
    def host_protocol(self) -> str:
        """Protocol of the GDAC host"""
        return self.idx.fs["src"].protocol

    @property
    def host_sep(self) -> str:
        """Host path separator"""
        return self.fs.fs.sep

    @property
    def path(self) -> str:
        """Return root path for all float datasets

        Since path type depends on the host protocol, this property is always a string
        """
        return self.host_sep.join([self.host, "dac", self.dac, "%i" % self.WMO])

    def ls(self) -> list:
        """Return the list of files in float path

        Protocol is included

        Examples
        --------
        >>> ArgoFloat(4902640).ls()
        ['https://data-argo.ifremer.fr/dac/meds/4902640/4902640_Sprof.nc',
         'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_meta.nc',
         'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_prof.nc',
         'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_tech.nc']

        >>> ArgoFloat(3901682, aux=True).ls()
        ['https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_Rtraj_aux.nc',
         'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_meta_aux.nc',
         'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_tech_aux.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_Rtraj.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_meta.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_prof.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_tech.nc']

        See Also
        --------
        :class:`ArgoFloat.ls_dataset`
        """
        paths = self.fs.glob(self.host_sep.join([self.path, "*"]))

        if self._aux:
            paths += self.fs.glob(self.host_sep.join([self.path.replace('dac', 'aux'), "*"]))

        paths = [p for p in paths if Path(p).suffix != ""]

        # Ensure the protocol is included for non-local files on FTP server:
        for ip, p in enumerate(paths):
            if self.host_protocol == 'ftp':
                paths[ip] = "ftp://" + self.fs.fs.host + fsspec.core.split_protocol(p)[-1]
            if self.host_protocol == 's3':
                paths[ip] = "s3://" + fsspec.core.split_protocol(p)[-1]

        paths.sort()
        return paths

    def lsprofiles(self) -> list:
        """Return the list of files in float profiles path

        See Also
        --------
        :class:`ArgoFloat.ls`
        """
        paths = self.fs.glob(self.host_sep.join([self.path, "profiles", "*"]))
        paths = [p for p in paths if Path(p).suffix != ""]
        paths.sort()
        return paths

    def ls_dataset(self) -> dict:
        """List all available dataset for this float in a dictionary

        Note that:

        - Dictionary keys are dataset short name to be used with :class:`ArgoFloat.open_dataset`.
        - Dictionary values hold absolute path toward the dataset file.

        Examples
        --------
        >>> ArgoFloat(4902640).ls_dataset()
        {'Sprof': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_Sprof.nc',
         'meta': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_meta.nc',
         'prof': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_prof.nc',
         'tech': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_tech.nc'}

        >>> ArgoFloat(4902640, aux=True).ls_dataset()
        {'Rtraj': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_Rtraj.nc',
         'Rtraj_aux': 'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_Rtraj_aux.nc',
         'meta': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_meta.nc',
         'meta_aux': 'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_meta_aux.nc',
         'prof': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_prof.nc',
         'tech': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_tech.nc',
         'tech_aux': 'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_tech_aux.nc'}

        """
        avail = {}
        for file in self.ls():
            filename = file.split(self.host_sep)[-1]
            if Path(filename).suffix == ".nc":
                split = Path(filename).stem.split("_")
                if split[0] == str(self.WMO):
                    name = "_".join(split[1:])
                    avail.update({name: file})
        return dict(sorted(avail.items()))

    def open_dataset(self, name: str = "prof", cast: bool = True, **kwargs) -> xr.Dataset:
        """Open and decode a dataset

        Parameters
        ----------
        name: str, optional, default = "prof"
            Name of the dataset to open. It can be any key from the dictionary returned by :class:`ArgoFloat.ls_dataset`.
        cast: bool, optional, default = True
            Determine if the dataset variables should be cast or not. This is similar to opening the dataset directly with :class:`xr.open_dataset` using the ``engine=`argo``` option.
            This will be ignored if the ``netCDF4` kwarg is set to True.
        **kwargs
            All the other parameters are passed to the GDAC store `open_dataset` method.

        Returns
        -------
        :class:`xarray.Dataset`

        Notes
        -----
        Use the ``netCDF4=True`` option to return a :class:`netCDF4.Dataset` object instead of a :class:`xarray.Dataset`.

        """
        if name not in self.ls_dataset():
            raise ValueError(
                "Dataset '%s' not found. Available dataset for this float are: %s"
                % (name, self.ls_dataset().keys())
            )
        else:
            file = self.ls_dataset()[name]

            if 'xr_opts' not in kwargs and cast is True:
                kwargs.update({'xr_opts': {"engine": "argo"}})

            ds = self.fs.open_dataset(file, **kwargs)
            self._dataset[name] = ds
            return self.dataset(name)

    def dataset(self, name: str = "prof"):
        if name not in self._dataset:
            self._dataset[name] = self.open_dataset(name)
        return self._dataset[name]

    @property
    def N_CYCLES(self) -> int:
        """Number of cycles

        If the float is still active, this is the current value.
        """
        return len(self.metadata["cycles"])

    def describe_profiles(self) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` describing profile files"""
        if self._df_profiles is None:
            prof = []
            for file in self.lsprofiles():
                desc = {}
                desc["stem"] = Path(file).stem
                desc = {**desc, **argo_split_path(file)}
                for v in ["dac", "wmo", "extension", "name", "origin", "path"]:
                    desc.pop(v)
                desc["path"] = file
                prof.append(desc)
            df = pd.DataFrame(data=prof)
            stem2cyc = lambda s: (  # noqa: E731
                int(s.split("_")[-1][0:-1])
                if s.split("_")[-1][-1] == "D"
                else int(s.split("_")[-1][:])
            )
            row2cyc = lambda row: stem2cyc(row["stem"])  # noqa: E731
            df["cyc"] = df.apply(row2cyc, axis=1)
            df["long_type"] = df.apply(
                lambda row: row["type"].split(",")[-1].lstrip(), axis=1
            )
            df["type"] = df.apply(lambda row: row["type"][0], axis=1)
            df["data_mode"] = df.apply(lambda row: row["data_mode"][0], axis=1)
            self._df_profiles = df
        return self._df_profiles

    def __repr__(self):
        backend = "online" if self._online else "offline"
        summary = ["<argofloat.%i.%s.%s>" % (self.WMO, self.host_protocol, backend)]
        # status = "online ✅" if isconnected(self.path, maxtry=1) else "offline 🚫"
        # summary.append("GDAC host: %s [%s]" % (self.host, status))
        summary.append("GDAC host: %s" % self.host)
        summary.append("DAC name: %s" % self.dac)
        summary.append("Network(s): %s" % self.metadata["networks"])

        launchDate = self.metadata["deployment"]["launchDate"]
        today = pd.to_datetime("now", utc=True)
        summary.append(
            "Deployment date: %s [%s days ago]"
            % (launchDate.strftime("%Y-%m-%d %H:%M"), (today - launchDate).days)
        )
        summary.append(
            "Float type and manufacturer: %s [%s]"
            % (
                self.metadata["platform"]["type"],
                self.metadata["maker"],
            )
        )
        summary.append("Number of cycles: %s" % self.N_CYCLES)
        if self._online:
            summary.append("Dashboard: %s" % dashboard(wmo=self.WMO, url_only=True))
        summary.append(
            "Netcdf dataset available: %s" % list(self.ls_dataset().keys())
        )

        return "\n".join(summary)
