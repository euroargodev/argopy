from typing import Union, Any, Literal, Optional, List
import fsspec.core
import xarray as xr
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from argopy.options import OPTIONS
from argopy.errors import InvalidOption, DataNotFound
from argopy.plot import dashboard
from argopy.stores import ArgoIndex
from argopy.utils.format import argo_split_path
from argopy.utils.lists import shortcut2gdac
from argopy.utils.checkers import check_wmo, check_cyc, to_list
from argopy.utils.decorators import deprecated


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
        self.WMO: int = check_wmo(wmo)[0]
        self.host: str = OPTIONS["gdac"] if host is None else shortcut2gdac(host)
        self.cache: bool = bool(cache)
        self.cachedir: str = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.timeout: int = OPTIONS["api_timeout"] if timeout == 0 else timeout
        self._aux: bool = bool(aux)

        if not self._online and (
            self.host.startswith("http")
            or self.host.startswith("ftp")
            or self.host.startswith("s3")
        ):
            if self.host.startswith("http://127.0.0.1"):
                pass
                # Allow to use the online implementation with a local server.
            else:
                raise InvalidOption(
                    "Trying to work with remote host '%s' without a web connection. Check your connection parameters or try to work with a local GDAC path."
                    % self.host
                )

        if "idx" not in kwargs:
            self.idx: ArgoIndex = ArgoIndex(
                index_file="core",
                host=self.host,
                cache=self.cache,
                cachedir=self.cachedir,
                timeout=self.timeout,
            )
        else:
            self.idx: ArgoIndex = kwargs["idx"]

        self.host: str = self.idx.host  # Fix host shortcuts with correct values
        self.fs = self.idx.fs["src"]

        ########################
        # Internal placeholder #
        ########################
        # These are used to improve performance by limiting data fetching
        # Data remained in memory, attached to the instance
        # This is NOT similar to cache=True, for which data are writen on file

        # Filled by self.load_metadata(), returned by self.metadata:
        self._metadata: dict | None = None

        # Filled by self.load_dac(), returned by self.dac:
        self._dac: str | None = None

        # Filled & returned by self._ls():
        self.__ls: list[str] | None = None

        # Filled & returned by self._lsp():
        self.__lsp: list[str] | None = None

        # Filled & returned by self.ls_profiles():
        self._ls_prof: dict[str | int, str] | None = None

        # Filled & returned by self.open_dataset(), returned by self.dataset():
        self._ds_datasets: dict[str, xr.Dataset | Any] = {}

        # Filled & returned by self.open_profile(), returned by self.profile():
        self._ds_profiles: dict[str, xr.Dataset | Any] = {}

        # Filled & returned by self.profiles_to_dataframe():
        self._df_profiles: pd.DataFrame | None = None

        ########
        # Misc #
        ########

        # Load some data (in a perfect world, this should be done asynchronously):
        # self.load_index()
        # self.ls_profiles()

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
        summary.append("Netcdf dataset available: %s" % list(self.ls_datasets().keys()))

        return "\n".join(summary)

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
        """Load float meta-data"""
        raise NotImplementedError("Not implemented")

    def load_metadata_from_meta_file(self):
        """Load float meta-data from the netcdf file as a dictionary"""
        data = {}

        ds = self.dataset("meta")
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

        cycles_ldict = []
        for cyc in np.unique(self.open_dataset("prof")["CYCLE_NUMBER"]):
            cycles_ldict.append({"id": int(cyc)})
        data.update({"cycles": cycles_ldict})

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
        return self.fs.protocol

    @property
    def host_sep(self) -> str:
        """Host path separator"""
        return self.fs.fs.sep

    @property
    def path(self) -> str:
        """Root path of float datasets

        Since path type depends on the host protocol, this property is always a string
        """
        return self.host_sep.join([self.host, "dac", self.dac, "%i" % self.WMO])

    def _ls(self) -> list[str]:
        """Return the list of files in float root path

        Protocol is included, all files are listed (not only netcdf, if any).

        This is similar to running ``ls dac/<DAC>/<WMO>/*`` from the command line,
        (and possibly appending results from the auxiliary folder as well).

        Examples
        --------
        >>> ArgoFloat(4902640)._ls()
        ['https://data-argo.ifremer.fr/dac/meds/4902640/4902640_Sprof.nc',
         'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_meta.nc',
         'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_prof.nc',
         'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_tech.nc']

        >>> ArgoFloat(3901682, aux=True)._ls()
        ['https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_Rtraj_aux.nc',
         'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_meta_aux.nc',
         'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_tech_aux.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_Rtraj.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_meta.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_prof.nc',
         'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_tech.nc']

        See Also
        --------
        :class:`ArgoFloat.ls_datasets`
        """
        if self.__ls is None:
            paths = self.fs.glob(self.host_sep.join([self.path, "*"]))

            if self._aux:
                paths += self.fs.glob(
                    self.host_sep.join(
                        [
                            self.path.replace(
                                f"{self.host_sep}dac{self.host_sep}",
                                f"{self.host_sep}aux{self.host_sep}",
                            ),
                            "*",
                        ]
                    )
                )

            paths = [p for p in paths if Path(p).suffix != ""]

            # Ensure the protocol is included for non-local files on FTP and S3 servers:
            for ip, p in enumerate(paths):
                if self.host_protocol == "ftp":
                    paths[ip] = (
                        "ftp://" + self.fs.fs.host + fsspec.core.split_protocol(p)[-1]
                    )
                if self.host_protocol == "s3":
                    paths[ip] = "s3://" + fsspec.core.split_protocol(p)[-1]

            paths.sort()
            self.__ls = paths

        return self.__ls

    def _lsp(self) -> list[str]:
        """Return the list of files in float 'profiles' path

        The GDAC host protocol is included, all files are listed (not only netcdf, if any other exist).

        This is similar to running ``ls dac/<DAC>/<WMO>/profiles/*`` from the command line,
        (and possibly appending results from the auxiliary folder as well).

        Examples
        --------
        >>> ArgoFloat(4902640)._lsp()
        >>> ArgoFloat(4902640, aux=True)._lsp()
        """
        if self.__lsp is None:

            paths = self.fs.glob(self.host_sep.join([self.path, "profiles", "*"]))

            if self._aux:
                paths += self.fs.glob(
                    self.host_sep.join(
                        [
                            self.path.replace(
                                f"{self.host_sep}dac{self.host_sep}",
                                f"{self.host_sep}aux{self.host_sep}",
                            ),
                            "profiles",
                            "*",
                        ]
                    )
                )

            # Ensure the protocol is included for non-local files on FTP and S3 servers:
            for ip, p in enumerate(paths):
                if self.host_protocol == "ftp":
                    paths[ip] = (
                        "ftp://" + self.fs.fs.host + fsspec.core.split_protocol(p)[-1]
                    )
                if self.host_protocol == "s3":
                    paths[ip] = "s3://" + fsspec.core.split_protocol(p)[-1]

            paths = [p for p in paths if Path(p).suffix != ""]
            paths.sort()
            self.__lsp = paths

        return self.__lsp

    @deprecated("Replaced by the 'ls_datasets' method", "v1.5.0")
    def ls_dataset(self) -> dict:
        """Deprecated, see ``ls_datasets()``"""
        return self.ls_datasets()

    def ls_datasets(self) -> dict:
        """List all available datasets as a dictionary

        Note that:

        - Dictionary keys are dataset short names to be used with :class:`ArgoFloat.open_dataset`.
        - Dictionary values hold absolute paths to the dataset files.

        Examples
        --------
        >>> ArgoFloat(4902640).ls_datasets()
        {'Sprof': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_Sprof.nc',
         'meta': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_meta.nc',
         'prof': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_prof.nc',
         'tech': 'https://data-argo.ifremer.fr/dac/meds/4902640/4902640_tech.nc'}

        >>> ArgoFloat(4902640, aux=True).ls_datasets()
        {'Rtraj': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_Rtraj.nc',
         'Rtraj_aux': 'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_Rtraj_aux.nc',
         'meta': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_meta.nc',
         'meta_aux': 'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_meta_aux.nc',
         'prof': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_prof.nc',
         'tech': 'https://data-argo.ifremer.fr/dac/coriolis/3901682/3901682_tech.nc',
         'tech_aux': 'https://data-argo.ifremer.fr/aux/coriolis/3901682/3901682_tech_aux.nc'}

        """
        avail = {}
        for file in self._ls():
            filename = file.split(self.host_sep)[-1]
            if Path(filename).suffix == ".nc":
                split = Path(filename).stem.split("_")
                if split[0] == str(self.WMO):
                    name = "_".join(split[1:])
                    avail.update({name: file})
        return dict(sorted(avail.items()))

    def open_dataset(
        self, name: str = "prof", cast: bool = True, **kwargs
    ) -> xr.Dataset:
        """Open and decode a dataset file

        The dataset is fetched and loaded in memory on every call. For performances, please consider using the :meth:`ArgoFloat.dataset` method instead.

        Parameters
        ----------
        name: str, optional, default = "prof"
            Name of the dataset to open. It can be any key from the dictionary returned by :class:`ArgoFloat.ls_datasets`.
        cast: bool, optional, default = True
            Determine if the dataset variables should be cast or not. This is similar to opening the dataset directly with :class:`xarray.open_dataset` using the ``engine=`argo``` option.
            This will be ignored if the ``netCDF4` kwarg is set to True.
        \**kwargs
            All the other arguments are passed to the GDAC store `open_dataset` method.

        Returns
        -------
        :class:`xarray.Dataset`

        Notes
        -----
        Use the ``netCDF4=True`` option to return a :class:`netCDF4.Dataset` object instead of a :class:`xarray.Dataset`.
        """
        if name not in self.ls_datasets():
            raise ValueError(
                "Dataset '%s' not found. Available dataset for this float are: %s"
                % (name, self.ls_datasets().keys())
            )

        file = self.ls_datasets()[name]

        if "xr_opts" not in kwargs and cast is True:
            kwargs.update({"xr_opts": {"engine": "argo"}})

        ds = self.fs.open_dataset(file, **kwargs)
        self._ds_datasets[name] = ds
        return ds

    def dataset(self, name: str = "prof", **kwargs) -> xr.Dataset:
        """Open and decode a dataset file, once

        This method is similar to :meth:`ArgoFloat.open_dataset` except that data are fetched only once to improve performances.

        Parameters
        ----------
        name: str, optional, default = "prof"
            Name of the dataset to open. It can be any key from the dictionary returned by :class:`ArgoFloat.ls_datasets`.
        \**kwargs
            All the other arguments are passed to the :meth:`ArgoFloat.open_dataset` method.

        Returns
        -------
        :class:`xarray.Dataset`

        """
        if name not in self._ds_datasets:
            self.open_dataset(
                name, **kwargs
            )  # will commit this dataset to self._ds_datasets dict
        return self._ds_datasets[name]

    @deprecated("Superseded by the '_lsp' method", "v1.5.0")
    def lsprofiles(self) -> list:
        """Deprecated, see ``_lsp()``"""
        return self._lsp()

    @deprecated("Superseded by the 'profiles_to_dataframe' method", "v1.5.0")
    def describe_profiles(self) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` describing all profile files"""
        return self.profiles_to_dataframe()

    def profiles_to_dataframe(self) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` describing all profile files

        Here we use the *profile* word to designate a mono-cycle netcdf file under the GDAC 'profiles' folder of a WMO.

        Returns
        -------
        :class:`pandas.DataFrame`
            A DataFrame with ["cycle_number", "dataset", "direction", "data_mode", "stem", "path"] columns for each mono-cycle netcdf files. The extra columns `auxiliary` is added if the instance was created with the appropriate option.
        """
        def stem2cyc(s) -> int:
            ii = -2 if "aux" in s else -1
            if s.split("_")[ii][-1] == "D":
                return int(s.split("_")[ii][0:-1])
            else:
                return int(s.split("_")[ii][:])

        if self._df_profiles is None:
            # Read the list of netcdf files under '<wmo>/profiles/*':
            paths = self._lsp()
            paths = [p for p in paths if p.split(".")[-1] == "nc"]

            # Scan each file name to extract information:
            prof = []
            for file in paths:
                desc = {}
                desc["stem"] = Path(file).stem
                desc = {**desc, **argo_split_path(file)}
                for v in ["dac", "wmo", "extension", "name", "origin", "path"]:
                    desc.pop(v)
                desc["path"] = file
                prof.append(desc)

            # Package all information as a DataFrame:
            df = pd.DataFrame(data=prof)
            row2cyc = lambda row: stem2cyc(row["stem"])  # noqa: E731
            df["cyc"] = df.apply(row2cyc, axis=1)
            columns = ["cyc", "type", "direction", "data_mode", "stem", "path"]
            if self._aux:
                columns.append("auxiliary")
            df = df[columns]
            df = df.rename({"cyc": "cycle_number", "type": "dataset"}, axis=1)
            df = df.sort_values(by=["cycle_number", "dataset", "direction"], axis=0)
            df = df.reset_index(drop=True)
            self._df_profiles = df

        return self._df_profiles

    def _ls_profiles_for(
        self,
        cycle_number: Optional[int | list[int]] = None,
        dataset: Literal["C", "B", "S"] = "C",
        direction: Literal["A", "D"] = "A",
        auxiliary: bool = False,
    ) -> dict[int | str, str]:
        """List all available profile files *for* a specific dataset and direction

        Notes
        -----
        Since mono-cycle profile files are either 'R' for real-time or 'D' for adjusted or delayed-mode data, there is no need to select one or the other, they can't exist at the same time.

        Parameters
        ----------
        cycle_number: int | list[int], optional, default = None
            The cycle number, or list, to return files for.

            If set to None (default), all cycle numbers are returned.
        dataset: Literal['C', 'B', 'S'], default = 'C'
            The profile dataset to return files for.

            - 'C': 'core' profile files (default),
            - 'B': BGC mono-cycle profile files,
            - 'S': Synthetic BGC mono-cycle profile files.

        direction: Literal['A', 'D'], default = 'A'
            The profile direction to return files for.

            - 'A' (default): Ascending profile files,
            - 'D': Descending profile files.

        auxiliary: Bool, default = False
            Return files from the auxiliary folder. This requires the object to have been instanciated with the `aux=True` option.

        Returns
        -------
        dict[int|str, str]
            A dictionary where:

            - keys are file short name to be used with :class:`ArgoFloat.open_profile`,
            - values are absolute path toward profile files.
        """
        CYCs = self.CYCLE_NUMBERS if cycle_number is None else cycle_number
        CYCs: list[int] = check_cyc(CYCs)

        if dataset.upper() not in ["C", "B", "S"]:
            raise ValueError(
                f"Invalid profile dataset '{dataset}' (type of mono-cycle file), must be None, 'B' or 'S'."
            )
        else:
            ds: str = dataset[0].upper()
            ds: str = {"C": "M", "B": "B", "S": "S"}[
                ds
            ]  # 'C' > 'M' because profiles_to_dataframe['dataset'] starts with a 'M' for 'C'ore profiles.

        direction = direction[0].upper()
        if direction not in ["A", "D"]:
            raise ValueError(
                f"Invalid profile direction '{direction}', must be 'A' or 'D'."
            )

        if auxiliary and not self._aux:
            raise ValueError(
                "To list auxiliary profile files, you need to create the ArgoFloat object with `aux=True` option, which was not the case with this instance."
            )

        df = self.profiles_to_dataframe()

        results = {}
        for cycle_number in CYCs:
            if cycle_number in df["cycle_number"].values:
                this_df = df[df["cycle_number"] == cycle_number]
                this_df = this_df[
                    this_df["dataset"].apply(
                        lambda x: x.replace("Auxiliary", "").strip()[0]
                    )
                    == ds
                ]
                this_df = this_df[
                    this_df["direction"].apply(lambda x: x[0]) == direction
                ]

                if "auxiliary" in this_df:
                    this_df = this_df[this_df["auxiliary"] == auxiliary]

                if this_df.shape[0] == 1:
                    pds = {"M": "", "B": "B", "S": "S"}[ds]
                    pdi = {"A": "", "D": "D"}[direction]
                    if pds == "" and pdi == "":
                        key = cycle_number
                        # key = f"{cycle_number}"
                    else:
                        key = f"{pds}{cycle_number}{pdi}"
                    if auxiliary:
                        key = f"{key}aux"
                    results[key] = this_df["path"].item()

        if len(results) >= 1:
            return results

        if not self._aux:
            msg = f"No mono-cycle file matches this description: cycle_number={CYCs}, dataset='{dataset}' and direction='{direction}' !"
        else:
            msg = f"No mono-cycle file matches this description: cycle_number={CYCs}, dataset='{dataset}', direction='{direction}' and auxiliary={auxiliary} !"
        raise ValueError(msg)

    def ls_profiles(self) -> dict[int | str, str]:
        """List all available profile files as a dictionary

        Notes
        -----
        In the output dictionary:

        - keys are integer for 'core' and ascending profile files (eg: 12 for 'R6903076_012.nc'),
        - keys are string for all other profile files, with the following convention:

            - ends with a 'D' for 'core' descending profile files (eg: '1D' for 'R6903076_001D.nc'),
            - starts with a 'B' for BGC ascending profile files (eg: 'B12' for 'BD6903091_012.nc'),
            - starts with a 'B' and ends with a 'D' for BGC descending profile files (eg: 'B12D' for 'BD6903091_012D.nc'),
            - starts with a 'S' for Synthetic profile files (eg: 'S134' for 'S6903091_134.nc').
            - starts with a 'S' and ends with a 'D' for Synthetic descending profile files (eg: 'S2D' for 'SR3902492_002D.nc').

        - Data from the auxiliary folder have regular keys with ``aux`` appended at the end of the key (eg: '11aux' for 'aux/coriolis/2903797/profiles/R2903797_011_aux.nc').

        Returns
        -------
        dict[int|str, str]
            A dictionary where:

            - keys are file short name to be used with :class:`ArgoFloat.open_profile`,
            - values are absolute path toward profile files.
        """
        if self._ls_prof is None:
            self._ls_prof = {}

            for ds in ["C", "B", "S"]:
                for di in ["A", "D"]:
                    try:
                        fl = self._ls_profiles_for(
                            dataset=ds,
                            direction=di,
                        )
                        for key, uri in fl.items():
                            self._ls_prof[key] = uri
                    except:
                        pass

            if self._aux:
                for ds in ["C", "B", "S"]:
                    for di in ["A", "D"]:
                        try:
                            fl = self._ls_profiles_for(
                                dataset=ds, direction=di, auxiliary=True
                            )
                            for key, uri in fl.items():
                                self._ls_prof[key] = uri
                        except:
                            pass

        return self._ls_prof

    def open_profile(
        self,
        name: str,
        cast: bool = True,
        **kwargs,
    ) -> xr.Dataset | Any | list[xr.Dataset | Any]:
        """Open and decode a single profile file

        Data are fetched on every call to this method.

        Parameters
        ----------
        name: str
            Name of the profile file to open.

            It can be any key from the dictionary returned by :class:`ArgoFloat.ls_profiles`.
        cast: bool, optional, default = True
            Determine if dataset variables should be cast or not.

            This is similar to opening the dataset directly with :class:`xarray.open_dataset` using the ``engine=`argo``` option.
            This will be ignored if the ``netCDF4` kwarg is set to True.
        **kwargs:
            All the other arguments are passed to the GDAC store `open_dataset` method.

        Returns
        -------
        :class:`xarray.Dataset`

        Notes
        -----
        Use the ``netCDF4=True`` option to return a :class:`netCDF4.Dataset` object instead of a :class:`xarray.Dataset`.

        See Also
        --------
        :class:`ArgoFloat.ls_profiles`

        Examples
        --------
        .. code-block:: python
            :caption: Open a 'core' profile file

            from argopy import ArgoFloat

            WMO = 6903076 # A 'core' float
            af = ArgoFloat(WMO)

            # Open the ascending profile file for cycle number 12:
            ds = af.open_profile(12)

            # Open the descending profile file, for cycle number 1:
            ds = af.open_profile('1D')

        .. code-block:: python
            :caption: Open a 'BGC' profile file

            from argopy import ArgoFloat

            WMO = 6903091 # A 'BGC' float
            af = ArgoFloat(WMO)

            # Open the ascending 'BGC' profile file for cycle number 12:
            ds = af.open_profile('B12')

            # Open the descending 'BGC' profile file for cycle number 1:
            ds = af.open_profile('B1D')

        .. code-block:: python
            :caption: Open a BGC 'Synthetic' profile file

            from argopy import ArgoFloat

            WMO = 6903091 # A 'BGC' float
            af = ArgoFloat(WMO)

            # Open the BGC 'Synthetic' profile file for cycle number 12:
            ds = af.open_profile('S12')
        """
        # d2s = (
        #     lambda x: str(x)
        #     .replace("'", "")
        #     .replace(":", "_")
        #     .replace("{", "")
        #     .replace("}", "")
        #     .replace(" ", "")
        # )

        if name not in self.ls_profiles():
            raise ValueError(
                f"The profile key '{name}' does not match any of the known profile files ({self.ls_profiles().keys()})"
            )

        if "xr_opts" not in kwargs and cast is True:
            kwargs.update({"xr_opts": {"engine": "argo"}})

        # key = f"{name}-{d2s(kwargs)}"
        file = self.ls_profiles()[name]
        ds = self.fs.open_dataset(file, **kwargs)
        self._ds_profiles[name] = ds
        return ds

    def profile(self, name: str, **kwargs) -> xr.Dataset:
        """Open and decode a profile file, once

        This method is similar to :meth:`ArgoFloat.open_profile` except that data are fetched only once to improve performances.

        Parameters
        ----------
        name: str
            Name of the profile file to open. It can be any key from the dictionary returned by :class:`ArgoFloat.ls_profiles`.
        \**kwargs
            All the other arguments are passed to the :meth:`ArgoFloat.open_profile` method.

        Returns
        -------
        :class:`xarray.Dataset`

        """
        if name not in self._ds_profiles:
            self.open_profile(
                name, **kwargs
            )  # will commit this dataset to self._ds_profiles dict
        return self._ds_profiles[name]

    def open_profiles(
        self,
        cycle_number: Optional[int | list[int]] = None,
        dataset: Literal["C", "B", "S"] = "C",
        direction: Literal["A", "D"] = "A",
        auxiliary: bool = False,
        cast: bool = True,
        **kwargs,
    ):
        """Open and decode one or more profile files

        Parameters
        ----------
        cycle_number: int | list[int], optional, default = None
            The cycle number, or list, to return files for.

            If set to None (default), all cycle numbers are returned.
        dataset: Literal['C', 'B', 'S'], default = 'C'
            The profile dataset to return files for.

            - 'C': 'core' profile files (default),
            - 'B': BGC mono-cycle profile files,
            - 'S': Synthetic BGC mono-cycle profile files.
        direction: Literal['A', 'D'], default = 'A'
            The profile direction to return files for.

            - 'A' (default): Ascending profile files,
            - 'D': Descending profile files.
        auxiliary: Bool, default = False
            Return files from the auxiliary folder. This requires the object to have been instanciated with the `aux=True` option.
        cast: bool, optional, default = True
            Determine if dataset variables should be cast or not.

            This is similar to opening the dataset directly with :class:`xarray.open_dataset` using the ``engine=`argo``` option.
            This will be ignored if the ``netCDF4` kwarg is set to True.
        **kwargs
            All the other arguments are passed to the file store `open_mfdataset` method. Interesting arguments are:

            - 'preprocess' and 'preprocess_opts' to apply some pre-processing function to each profile file,
            - 'progress' to display a fetching progress bar,
            - 'method' to impose a parallelization method,
            - 'errors' to control what to do if an error occur with one profile file,
            - 'open_dataset_opts' to provide options when opening each netcdf files, in particular 'netCDF4' to return a legacy netcdf dataset instead of a :class:`xr.Dataset`.

            Depending on the GDAC host, more details can be found from: :class:`argopy.stores.httpstore.open_mfdataset`, :class:`argopy.stores.local.open_mfdataset`, :class:`argopy.stores.ftppstore.open_mfdataset` or :class:`argopy.stores.s3store.open_mfdataset`.

        Returns
        -------
        xr.Dataset | Any | list[xr.Dataset | Any]
            If no pre-processing is done with profile files, return one, or a list, of :class:`xr.Dataset`. Otherwise, return the list of pre-processing output.

        Notes
        -----
        When called on 1 profile file, this method return the same :class:`xarray.Dataset` as the :class:`ArgoFloat.open_profile` method.

        .. code-block:: python

            from argopy import ArgoFloat

            WMO = 6903076 # some float
            af = ArgoFloat(WMO)

            ds1 = af.open_profile(1)
            # or
            ds2 = af.open_profiles(1, dataset='C', direction='A')
            # is similar:
            assert ds1.equals(ds2)

            ds1 = af.open_profile('1D')
            # or
            ds2 = af.open_profiles(1, dataset='C', direction='D')
            # is similar:
            assert ds1.equals(ds2)

        Examples
        --------
        .. code-block:: python
            :caption: Open 'core' profile files

            from argopy import ArgoFloat

            WMO = 6903076 # a 'core' float
            af = ArgoFloat(WMO)

            # Open some core ascending profile files (default):
            ds_list = af.open_profiles([1, 2])

            # Open some descending profile files:
            ds_list = af.open_profiles([1, 2], direction='D')

            # Open *all* profile files (only ascending):
            ds_list = af.open_profiles()

        .. code-block:: python
            :caption: Open 'BGC' profile file(s)

            from argopy import ArgoFloat

            WMO = 6903091 # a 'BGC' float
            af = ArgoFloat(WMO)

            # Open some 'BGC' ascending profile files (default):
            ds_list = af.open_profiles([1, 2], dataset='B')

            # Open some 'BGC' descending profile files:
            ds_list = af.open_profiles([1, 2], dataset='B', direction='D')

            # Open *all* 'BGC' profile files (only ascending):
            ds_list = af.open_profiles(dataset='B')

        .. code-block:: python
            :caption: Open BGC 'Synthetic' profile file(s)

            from argopy import ArgoFloat

            WMO = 6903091 # Some 'BGC' float
            af = ArgoFloat(WMO)

            # Open some BGC 'Synthetic' ascending profile files:
            ds_list = af.open_profiles([1, 2], dataset='S')

            # Open *all* BGC 'Synthetic' profile files:
            ds_list = af.open_profiles(dataset='S')

        """
        self.ls_profiles()  # Just making sure the instance as the internal placeholder filled

        def fname2key(file_name):
            for k, v in self.ls_profiles().items():
                if v == file_name:
                    return k
            raise ValueError(
                f"This file name '{file_name}' is not a valid profile file."
            )

        fnames = list(
            self._ls_profiles_for(
                cycle_number=cycle_number,
                dataset=dataset,
                direction=direction,
                auxiliary=auxiliary,
            ).values()
        )

        if "xr_opts" not in kwargs and cast is True:
            if "open_dataset_opts" in kwargs:
                kwargs["open_dataset_opts"].update({"xr_opts": {"engine": "argo"}})
            else:
                kwargs.update({"open_dataset_opts": {"xr_opts": {"engine": "argo"}}})
        if "concat" not in kwargs:
            kwargs.update({"concat": False})

        _myprocessing = False
        if "preprocess" not in kwargs:
            # Create a pre-processing function that will simply return the dataset in a dictionary
            # where the key will be used to commit the dataset in the internal placeholder later on.
            _myprocessing = True
            kwargs["preprocess"] = lambda ds: {ds.encoding["source"]: ds}
            kwargs["concat"] = False

        results: xr.Dataset | Any | list[xr.Dataset | Any] = self.fs.open_mfdataset(
            fnames, **kwargs
        )

        if _myprocessing:
            # results: list[dict[str, xr.Dataset]]
            # Commit data we just loaded and reformat the output as a list of dataset
            r = []
            for res in results:
                for fname in res.keys():
                    key = fname2key(fname)
                    self._ds_profiles[key] = res[fname]
                    r.append(res[fname])

        if isinstance(results, list) and len(results) == 1:
            return results[0]

        return results

    @property
    def CYCLE_NUMBERS(self) -> List[int]:
        """List of cycle numbers, according to the list of mono-profile files.

        The list is computed from the analysis of the GDAC 'profiles' folder of the float.
        So this relies on :meth:`ArgoFloat.profiles_to_dataframe()` that in turns relies on :meth:`ArgoFloat._lsp()`.

        Notes
        -----
        This `CYCLE_NUMBERS` attribute is not to be confused with the netcdf variable `CYCLE_NUMBER` that is the
        float cycle number of the measurement in the context of a specific netcdf file. That's why we're using
        the extra `s` at the end.
        """
        return sorted(
            [int(c) for c in self.profiles_to_dataframe()["cycle_number"].unique()]
        )

    @property
    def N_CYCLES(self) -> int:
        """Number of cycles, according to the list of mono-profile files.

        This is simply the length of :attr:`ArgoFloat.CYCLE_NUMBERS`.

        Notes
        -----
        This `N_CYCLES` attribute is not to be confused with the netcdf dimension `N_CYCLE` that is the
        index for cycles in the context of a specific netcdf file (eg: trajectory file). That's why we're using
        the extra `s` at the end.
        """
        return len(self.CYCLE_NUMBERS)

    def _ipython_key_completions_(self):
        """Provide method for key-autocompletions in IPython."""
        keys = self.ls_datasets().copy()
        keys.update(self.ls_profiles())
        return [k for k in keys.keys()]

    def __getitem__(self, args) -> int | str | list[int | str]:
        """Retrieve netcdf file(s)

        .. code-block:: python

            from argopy import ArgoFloat

            af = ArgoFloat(3902492)

            # Get any dataset or profile:
            af['prof']
            af[4]
            af['B3']

            # Get a slice of profiles:
            af[1:4]
            af[::2]
            af[:]

            # Get profile from last available cycle:
            af[af.CYCLE_NUMBERS[-1]]
        """
        if isinstance(args, str) or isinstance(args, int) or isinstance(args, slice):
            obj = args

            if isinstance(obj, str) or isinstance(obj, int):
                if obj in self.ls_datasets():
                    return self.dataset(obj)
                elif obj in self.ls_profiles():
                    return self.profile(obj)
                else:
                    raise DataNotFound

            elif isinstance(obj, slice):
                cycs = []
                for cyc in range(
                    self.CYCLE_NUMBERS[0] if not obj.start else obj.start,
                    self.CYCLE_NUMBERS[-1] + 1 if not obj.stop else obj.stop,
                    1 if not obj.step else obj.step,
                ):
                    if cyc in self.CYCLE_NUMBERS:
                        cycs.append(cyc)

                # Small optimisation to handle a large number of cycles:
                results = []
                ConcurrentExecutor = ThreadPoolExecutor()
                with ConcurrentExecutor as executor:
                    future_todo = {
                        executor.submit(self.profile, cyc): cyc for cyc in cycs
                    }
                    futures = as_completed(future_todo)
                    for future in futures:
                        results.append(future.result())

                return results

        elif isinstance(args, tuple):
            obj: int | slice = args[0]
            dataset: str = args[1]

            if isinstance(obj, str) and obj in self.ls_datasets():
                raise ValueError("This syntax is for profiles only")

            elif isinstance(obj, int):
                key = list(self._ls_profiles_for(obj, dataset=dataset).keys())[0]
                return self.profile(key)

            elif isinstance(obj, slice):
                cycs = []
                for cyc in range(
                    self.CYCLE_NUMBERS[0] if not obj.start else obj.start,
                    self.CYCLE_NUMBERS[-1] + 1 if not obj.stop else obj.stop,
                    1 if not obj.step else obj.step,
                ):
                    if cyc in self.CYCLE_NUMBERS:
                        cycs.append(cyc)

                keys = list(self._ls_profiles_for(cycs, dataset=dataset).keys())

                # Small optimisation to handle a large number of cycles:
                results = []
                ConcurrentExecutor = ThreadPoolExecutor()
                with ConcurrentExecutor as executor:
                    future_todo = {
                        executor.submit(self.profile, key): key for key in keys
                    }
                    futures = as_completed(future_todo)
                    for future in futures:
                        results.append(future.result())

                return results

        raise NotImplementedError
