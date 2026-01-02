from abc import abstractmethod
from typing import NoReturn, Any
import pandas as pd

from argopy.utils.casting_param import cast_config_parameter
from argopy.utils.decorators import register_accessor
from argopy.utils.casting import to_list, to_bool
from argopy.utils.format import group_cycles_by_missions, cfgnameparser


def register_ArgoFloat_extension(name, store):
    """A decorator to register an extension as a custom property on :class:`ArgoFloat` objects.

    Parameters
    ----------
    name : str
        Name under which the extension should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.
    store: :class:`ArgoFloat`

    Examples
    --------
    .. code-block:: python

        @register_ArgoFloat_accessor('config')
        class SearchEngine(ArgoFloatExtension):

             def __init__(self, *args, **kwargs):
                 super().__init__(*args, **kwargs)

             def wmo(self, WMOs):
                 return WMOs

    It will be available to an ArgoFloat object, like this::

        ArgoFloat().config.wmo(WMOs)
    """
    return register_accessor(name, store)


class ArgoFloatExtension:
    """Prototype for ArgoFloat extensions

    All extensions should inherit from this class

    This prototype makes available:

    - the :class:`ArgoFloat` instance as ``self._obj``
    """

    __slots__ = "_obj"

    def __init__(self, obj):
        self._obj = obj


class ArgoFloatPlotProto(ArgoFloatExtension):
    """Extension providing plot methods"""

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.plot cannot be called directly. Use "
            "an explicit plot method, e.g. ArgoFloat.plot.trajectory(...)"
        )

    @abstractmethod
    def trajectory(self):
        raise NotImplementedError

    @abstractmethod
    def map(self):
        raise NotImplementedError

    @abstractmethod
    def scatter(self):
        raise NotImplementedError


class ArgoFloatAnyConfigParametersProto(ArgoFloatExtension):
    @property
    def n_missions(self) -> int:
        """Total number of missions"""
        return len(self.missions)

    @property
    def n_params(self) -> int:
        """Total number of parameters"""
        return len(self.parameters)

    @property
    @abstractmethod
    def parameters(self):
        """List of configuration parameter names"""
        raise NotImplementedError

    def __len__(self):
        return self.n_params

    def cast(self, param: str, pvalue: Any) -> float | int | bool:
        """Cast a configuration parameter

        Parameters
        ----------
        param: str
            Name of the configuration parameter. Name is used to infer unit, hence dtype.

        Returns
        -------
        float, int, bool
            Cast parameter value, according to parameter name inferred unit
        """
        return cast_config_parameter(param, pvalue)

    def _ipython_key_completions_(self):
        """Provide method for key-autocompletions in IPython."""
        return [p.replace('CONFIG_','') for p in self.parameters]


class ArgoFloatLaunchConfigParametersProto(ArgoFloatAnyConfigParametersProto):
    """Extension providing access to LAUNCH configuration parameters

    Pre-deployment or launch configuration parameters are the ‘configured’ start settings of the float and the initial mission configuration parameters for the first cycle.

    Warnings
    --------
    Configuration parameters are float settings selected by the PI, not measurements reported by the float.

    Examples
    --------
    .. code-block:: python
        :caption: Launch configuration parameters

        from argopy import ArgoFloat
        af = ArgoFloat(6903091)

        # Total number and list of launch parameters:
        af.launchconfig.n_params
        af.launchconfig.parameters

        # Read one parameter value, with explicit or implicit parameter name:
        # ('CONFIG_' is not mandatory, but string is case-sensitive)
        af.launchconfig['CONFIG_CycleTime_hours']
        af.launchconfig['CycleTime_hours']

        # Export to a DataFrame:
        af.launchconfig.to_dataframe()

    Notes
    -----
    - When called *online*, data are retrieved from the Euro-Argo Fleet-Monitoring web-API.
    - When called *offline*, data are retrieved from a local meta data netcdf file and variables ``N_LAUNCH_CONFIG_PARAM``, ``LAUNCH_CONFIG_PARAMETER_NAME`` and ``LAUNCH_CONFIG_PARAMETER_VALUE``.

    References
    ----------
    .. code-block:: python

        import argopy
        # User manual section on "Configuration parameters"
        argopy.ArgoDocs(29825).open_pdf(55)  # as of Version 3.44.0
    """
    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.launchconfig cannot be called directly. Use "
            "direct indexing or an explicit method."
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Export launch configuration parameters to a class:`pandas.DataFrame`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        data = []
        for param in self.parameters:
            pname = cfgnameparser(param)
            this = {'Name': f"{pname['label']}"}
            this.update({'Unit': f"{pname['unit']}"})
            this.update({f"Value": self[param]})
            this.update({'CONFIG_PARAMETER_NAME': f"{param}"})
            data.append(this)
        return pd.DataFrame(data, dtype=object).sort_values(by='Name').reset_index(drop=True)


class ArgoFloatConfigParametersProto(ArgoFloatAnyConfigParametersProto):
    """Extension providing access to configuration parameters

    Warnings
    --------
    Configuration parameters are float settings selected by the PI, not measurements reported by the float.

    Examples
    --------
    .. code-block:: python
        :caption: Configuration parameters and missions

        from argopy import ArgoFloat
        af = ArgoFloat(6903091)

        # Total number and list of configuration parameters:
        af.config.n_params
        af.config.parameters

        # Total number and list of missions:
        af.config.n_missions
        af.config.missions

        # Read one parameter value, with explicit or implicit parameter name:
        # ('CONFIG_' is not mandatory, but string is case-sensitive)
        af.config['CONFIG_CycleTime_hours']
        af.config['CycleTime_hours']

        # Read parameter value for one or more mission numbers:
        # (! 2nd index is not 0-based, it's an integer key to look for in mission numbers)
        af.config['CycleTime_hours', 1]
        af.config['CycleTime_hours', 1:3]

    .. code-block:: python
        :caption: Configuration parameters and cycle numbers

        from argopy import ArgoFloat
        af = ArgoFloat(6903091)

        # Get a dictionary mapping cycle on mission numbers:
        af.config.cycles

        # Read parameter value for one or more cycle numbers:
        # (! 2nd index is not 0-based, it's an integer key to look for in cycle numbers)
        af.config.for_cycles('CycleTime_hours', 1)
        af.config.for_cycles('CycleTime_hours', [10, 11])

    .. code-block:: python
        :caption: Export configuration parameters

        from argopy import ArgoFloat
        af = ArgoFloat(6903091)

        # Export to a DataFrame:
        af.config.to_dataframe()
        af.config.to_dataframe(missions=1)
        af.config.to_dataframe(missions=[1, 2])

    Notes
    -----
    - When called *online*, data are retrieved from the Euro-Argo Fleet-Monitoring web-API.
    - When called *offline*, data are retrieved from local meta-data and prof-data netcdf files and variables ``N_MISSIONS``, ``CONFIG_PARAMETER_NAME``, ``CONFIG_PARAMETER_VALUE``, ``CONFIG_MISSION_NUMBER`` and ``CYCLE_NUMBER``.

    References
    ----------
    .. code-block:: python

        import argopy
        # User manual section on "Configuration parameters"
        argopy.ArgoDocs(29825).open_pdf(55)  # as of Version 3.44.0
    """

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.config cannot be called directly. Use "
            "direct indexing or an explicit method."
        )

    @property
    @abstractmethod
    def missions(self) -> list[int]:
        """List of mission numbers"""
        raise NotImplementedError

    @property
    @abstractmethod
    def cycles(self) -> dict[int, int]:
        """A dictionary mapping cycle on mission numbers

        Returns
        -------
        dict[int, int]:
            Keys are cycle numbers, values are mission numbers.
        """
        raise NotImplementedError

    def for_cycles(self, param: str, cycle_numbers: int | list[int]) -> int | float | str | bool | list[int | float | str | bool]:
        """Retrieve a configuration parameter for cycle number(s)

        Parameters
        ----------
        param: str
            Name of the configuration parameter to retrieve. Can be any string from :attr:`Argofloat.config.parameters`.
        cycle_numbers: int | list[int]
            Cycle number(s) to retrieve parameter for.

        Returns
        -------
        int | float | str | bool | list[int | float | str | bool]
            One or a list of configuration parameter values.
        """
        cycle_numbers = to_list(cycle_numbers)
        results = []
        for cyc in cycle_numbers:
            mission_number = self.cycles[cyc]
            results.append(self[param, mission_number])
        if len(cycle_numbers) == 1:
            return results[0]
        return results

    def to_dataframe(self, missions: None | int | list[int] = None) -> pd.DataFrame:
        """Export configuration parameters to a class:`pandas.DataFrame`

        Parameters
        ----------
        missions: None | int | list[int], optional, default=None
            Possible export parameters for one or a list of mission numbers. By default, export all missions.

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        if missions is None:
            mlist = self.missions
        else:
            mlist = [m for m in to_list(missions) if m in self.missions]
        if len(mlist) == 0:
            raise ValueError(f"Invalid list of mission numbers. Valid values are: {self.missions}")

        columns = group_cycles_by_missions(self.cycles, output='group')
        data = []
        for param in self.parameters:
            pname = cfgnameparser(param)
            this = {'Name': f"{pname['label']}"}
            this.update({'Unit': f"{pname['unit']}"})
            for m in mlist:
                this.update({f"Mission # {m} / Cycles # {columns[m]}": self[param, m]})
            this.update({'CONFIG_PARAMETER_NAME': f"{param}"})
            data.append(this)
        return pd.DataFrame(data, dtype=object).sort_values(by='Name').reset_index(drop=True)

