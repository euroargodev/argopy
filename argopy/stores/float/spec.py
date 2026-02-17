from typing import Union, Any, Literal
import fsspec.core
import xarray as xr
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod
import logging
import numpy as np

from argopy.options import OPTIONS
from argopy.errors import InvalidOption
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
        self.WMO = check_wmo(wmo)[0]
        self.host = OPTIONS["gdac"] if host is None else shortcut2gdac(host)
        self.cache = bool(cache)
        self.cachedir = OPTIONS["cachedir"] if cachedir == "" else cachedir
        self.timeout = OPTIONS["api_timeout"] if timeout == 0 else timeout
        self._aux = bool(aux)

        if not self._online and (
            self.host.startswith("http")
            or self.host.startswith("ftp")
            or self.host.startswith("s3")
        ):
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
        self._ls: list[str] | None = None
        self._lsp: dict[str|int, str] | None = None
        self._df_profiles = None  # Dataframe with profiles index

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
        summary.append("Netcdf dataset available: %s" % list(self.ls_dataset().keys()))

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
        """Method to load float meta-data"""
        raise NotImplementedError("Not implemented")

    def load_metadata_from_meta_file(self):
        """Method to load float meta-data from the netcdf file"""
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
        return self.fs.protocol

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
        if self._ls is None:
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

            # Ensure the protocol is included for non-local files on FTP server:
            for ip, p in enumerate(paths):
                if self.host_protocol == "ftp":
                    paths[ip] = (
                        "ftp://" + self.fs.fs.host + fsspec.core.split_protocol(p)[-1]
                    )
                if self.host_protocol == "s3":
                    paths[ip] = "s3://" + fsspec.core.split_protocol(p)[-1]

            paths.sort()
            self._ls = paths
        return self._ls

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

    def open_dataset(
        self, name: str = "prof", cast: bool = True, **kwargs
    ) -> xr.Dataset:
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

            if "xr_opts" not in kwargs and cast is True:
                kwargs.update({"xr_opts": {"engine": "argo"}})

            ds = self.fs.open_dataset(file, **kwargs)
            self._dataset[name] = ds
            return self.dataset(name)

    def dataset(self, name: str = "prof"):
        if name not in self._dataset:
            self.open_dataset(name)  # will commit this dataset to self._dataset dict
        return self._dataset[name]

    @property
    def N_CYCLES(self) -> int:
        """Number of cycles

        If the float is still active, this is the current value.
        """
        return len(np.unique([c["id"] for c in self.metadata["cycles"]]))

    @deprecated("Superseded by the 'ls_profiles' method", 'v1.5.0')
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

    def describe_profiles(self) -> pd.DataFrame:
        """Return a :class:`pandas.DataFrame` describing profile files"""
        if self._df_profiles is None:
            # Read the list of netcdf files under '<wmo>/profiles/*':
            paths = self.fs.glob(self.host_sep.join([self.path, "profiles", "*.nc"]))
            paths = [p for p in paths if Path(p).suffix != ""]
            paths.sort()

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
            stem2cyc = lambda s: (  # noqa: E731
                int(s.split("_")[-1][0:-1])
                if s.split("_")[-1][-1] == "D"
                else int(s.split("_")[-1][:])
            )
            row2cyc = lambda row: stem2cyc(row["stem"])  # noqa: E731
            df["cyc"] = df.apply(row2cyc, axis=1)
            df = df[["cyc", "type", "data_mode", "direction", "stem", "path"]]
            df = df.sort_values(by=["cyc", "type", "data_mode", "direction"], axis=0)
            df = df.reset_index(drop=True)
            df = df.rename({"type": "dataset"}, axis=1)
            self._df_profiles = df

        return self._df_profiles

    def ls_profiles_for(
        self,
        cycle_number: int | list[int] | None = None,
        dataset: Literal['B', 'S'] | None = None,
        direction: Literal['A', 'D'] = "A",
    ) -> dict[int|str, str]:
        """List all available profile files 'for' specific dataset and direction

        Notes
        -----
        Since mono-cycle profile files are either 'R' for real-time or 'D' for adjusted or delayed-mode data, there is no need to select one or the other, they can't exist at the same time.

        Parameters
        ----------
        cycle_number: int | list[int] | None, optional, default = None
            The cycle number, or list, to return files for.

            If set to None (default), all cycle numbers are returned.
        dataset: Literal['B', 'S'] | None, optional, default = None
            The profile dataset to return files for.

            - None: 'core' profile files (default),
            - 'B': BGC mono-cycle profile files,
            - 'S': Synthetic BGC mono-cycle profile files.
        direction: Literal['A', 'D'], optional, default = 'A'
            The profile direction to return files for.

            - 'A' (default): Ascending profile files,
            - 'D': Descending profile files.

        Returns
        -------
        dict[int|str, str]
            A dictionary where:
            - keys are file short name to be used with :class:`ArgoFloat.open_profile`,
            - values are absolute path toward profile files.
        """
        CYCs = [c+1 for c in np.arange(self.N_CYCLES).tolist()] if cycle_number is None else cycle_number
        CYCs: list[int] = check_cyc(CYCs)
        if dataset not in [None, 'b', 's', 'B', 'S']:
            raise ValueError(
                f"Invalid profile dataset '{dataset}' (type of mono-cycle file), must be None, 'B' or 'S'.")
        else:
            ds: str = '' if dataset is None else dataset.upper()
            ds: str = {'': 'M', 'B': 'B', 'S': 'S'}[ds]

        if direction not in ['a', 'd', 'A', 'D']:
            raise ValueError(f"Invalid profile direction '{direction}', must be 'A' or 'D'.")
        else:
            direction = direction.upper()

        df = self.describe_profiles()

        results = {}
        for cycle_number in CYCs:
            if cycle_number in df['cyc']:
                this_df = df[df['cyc'] == cycle_number]
                this_df = this_df[this_df['dataset'].apply(lambda x: x[0]) == ds]
                this_df = this_df[this_df['direction'].apply(lambda x: x[0]) == direction]
                if this_df.shape[0] == 1:
                    pds = {'M': '', 'B': 'B', 'S': 'S'}[ds]
                    pdi = {'A': '', 'D': 'D'}[direction]
                    if pds == '' and pdi == '':
                        key = cycle_number
                    else:
                        key = f"{pds}{cycle_number}{pdi}"
                    results[key] = this_df['path'].item()

        if len(results) >= 1:
            return results

        raise ValueError(f"No mono-cycle file matches this description: cycle_number={CYCs}, dataset='{dataset}' and direction='{direction}' !")

    def ls_profiles(self) -> dict[int|str, str]:
        """List all available profile files, whatever the profile dataset and direction

        Notes
        -----
        In the output dictionary:
        - keys are integer for 'core' and ascending profile files (eg: 12 for '<R/D>6903076_012.nc'),
        - keys are string for all other profile files, with the following convention:
            - '<cycle>D'  for 'core' descending profile files (eg: '1D' for '<R/D>6903076_001D.nc'),
            - 'B<cycle>'  for BGC ascending profile files (eg: 'B12' for 'B<R/D>6903091_012.nc'),
            - 'B<cycle>D' for BGC descending profile files (eg: 'B12D' for 'B<R/D>6903091_012D.nc'),
            - 'S<cycle>'  for Synthetic ascending profile files (eg: 'S134' for 'S<R/D>6903091_134.nc').

        Returns
        -------
        dict[int|str, str]
            A dictionary where:
            - keys are file short name to be used with :class:`ArgoFloat.open_profile`,
            - values are absolute path toward profile files.
        """
        if self._lsp is None:
            self._lsp = {}
            for ds in [None, 'B', 'S']:
                for di in ['A', 'D']:
                    try:
                        fl = self.ls_profiles_for(dataset=ds, direction=di)
                        for key, uri in fl.items():
                            self._lsp.update({key: uri})
                    except:
                        pass
        return self._lsp

    def open_profile(
        self,
        name: str | list[str],
        cast: bool = True,
        **kwargs,
    ) -> xr.Dataset | Any | list[xr.Dataset | Any]:
        """Open and decode one or more profile file dataset

        Parameters
        ----------
        name: str | list[str]
            Name, or list of names, of profile files to open.

            It can be any key from the dictionary returned by :class:`ArgoFloat.ls_profiles`.
        cast: bool, optional, default = True
            Determine if dataset variables should be cast or not.

            This is similar to opening the dataset directly with :class:`xr.open_dataset` using the ``engine=`argo``` option.
            This will be ignored if the ``netCDF4` kwarg is set to True.
        **kwargs
            All the other arguments are passed to the file store `open_mfdataset` method. Interesting arguments are:

            - 'preprocess' and 'preprocess_opts' to apply some pre-processing function to each profile file,
            - 'progress' to display a fetching progress bar,
            - 'method' to impose a parallelization method,
            - 'errors' to control what to do if an error occur with one profile file,
            - 'open_dataset_opts' to provide options when opening each netcdf files, in particular 'netCDF4' to return a legacy netcdf dataset instead of a :class:`xr.Dataset`.

            Depending on the GDAC host, more details can be found at: :class:`argopy.stores.httpstore.open_mfdataset`, :class:`argopy.stores.local.open_mfdataset`, :class:`argopy.stores.ftppstore.open_mfdataset` or :class:`argopy.stores.s3store.open_mfdataset`.

        Returns
        -------
        xr.Dataset | Any | list[xr.Dataset | Any]
            If no pre-processing is done with profile files, return one or a list of :class:`xr.Dataset`. Otherwise, return the pre-processing output.

        Examples
        --------
        ..code-block: python
            :caption: Open 'core' profile file(s)

            from argopy import ArgoFloat

            WMO = 6903076 # Some 'core' float
            af = ArgoFloat(WMO)

            # Open one ascending profile file, for cycle number 12:
            ds = af.open_profile(12)

            # Open one descending profile file, for cycle number 1:
            ds = af.open_profile('1D')

            # Open more than one profile file:
            ds_list = af.open_profile([1,2,3])

            # Open **all** profile files:
            name_list = af.ls_profiles_for(dataset=None, direction='A').keys()
            ds_list = af.open_profile(name_list, progress=True)

        ..code-block: python
            :caption: Open 'BGC' profile file(s)

            from argopy import ArgoFloat

            WMO = 6903091 # Some 'BGC' float
            af = ArgoFloat(WMO)

            # Open one ascending profile file, for cycle number 12:
            ds = af.open_profile('B12')

            # Open one descending profile file, for cycle number 1:
            ds = af.open_profile('B1D')

            # Open more than one profile file:
            ds_list = af.open_profile(['B1', 'B2', 'B3'])

            # Open **all** profile files:
            name_list = af.ls_profiles_for(dataset='B', direction='A').keys()
            ds_list = af.open_profile(name_list, progress=True)

        ..code-block: python
            :caption: Open BGC 'Synthetic' profile file(s)

            from argopy import ArgoFloat

            WMO = 6903091 # Some 'BGC' float
            af = ArgoFloat(WMO)

            # Open one ascending profile file, for cycle number 12:
            ds = af.open_profile('S12')

            # Open more than one profile file:
            ds_list = af.open_profile(['S1', 'S2', 'S3'])

            # Open **all** profile files:
            name_list = af.ls_profiles_for(dataset='S').keys()
            ds_list = af.open_profile(name_list, progress=True)

        """
        flist: list[str] = []
        names = to_list(name)
        [flist.append(n) for n in names if n in self.ls_profiles()]
        if len(flist) == 0:
            raise ValueError(f"This profile key {names} does not match any of the known profile files ({self.ls_profiles().keys()})")

        if "xr_opts" not in kwargs and cast is True:
            if 'open_dataset_opts' in kwargs:
                kwargs['open_dataset_opts'].update({"xr_opts": {"engine": "argo"}})
            else:
                kwargs.update({"open_dataset_opts": {"xr_opts": {"engine": "argo"}}})
        if "concat" not in kwargs:
            kwargs.update({"concat": False})

        urilist = [self.ls_profiles()[f] for f in flist]
        results: xr.Dataset | Any | list[xr.Dataset | Any] = self.fs.open_mfdataset(urilist, **kwargs)

        if isinstance(results, list) and len(results) == 1:
            return results[0]

        return results
