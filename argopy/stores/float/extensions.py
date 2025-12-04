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
        return len(self.missions)

    @property
    def n_params(self) -> int:
        return len(self.parameters)

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
        """Provide method for the key-autocompletions in IPython."""
        return [p.replace('CONFIG_','') for p in self.parameters]


class ArgoFloatLaunchConfigParametersProto(ArgoFloatAnyConfigParametersProto):
    """Extension providing easy access to LAUNCH configuration parameters"""

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.launchconfig cannot be called directly. Use "
            "direct indexing or an explicit method."
        )

    def to_dataframe(self, implicit: bool = True) -> pd.DataFrame:
        """Export launch configuration parameters to a class:`pd.DataFrame`

        Parameters
        ----------
        implicit: bool, default = True
            Use implicit parameter's label and unit from the raw 'CONFIG_PARAMETER_NAME' name.
        """
        data = []
        for param in self.parameters:
            if implicit:
                pname = cfgnameparser(param)
                this = {'CONFIG_PARAMETER_NAME': f"{pname['label']} ({pname['unit']})"}
            else:
                this = {'CONFIG_PARAMETER_NAME': f"{param}"}
            this.update({f"Launch": self[param]})
            data.append(this)
        return pd.DataFrame(data, dtype=object).sort_values(by='CONFIG_PARAMETER_NAME').reset_index(drop=True)


class ArgoFloatConfigParametersProto(ArgoFloatAnyConfigParametersProto):
    """Extension providing easy access to configuration parameters"""

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise ValueError(
            "ArgoFloat.config cannot be called directly. Use "
            "direct indexing or an explicit method."
        )

    def for_cycle(self, param: str, cycle_numbers: int | list[int]) -> int | float | str | bool | list[int | float | str | bool]:
        """Retrieve a configuration parameter for a given cycle number"""
        cycle_numbers = to_list(cycle_numbers)
        results = []
        for cyc in cycle_numbers:
            mission_number = self.cycles[cyc]
            results.append(self[param, mission_number])
        if len(cycle_numbers) == 1:
            return results[0]
        return results

    def to_dataframe(self, implicit: bool = True) -> pd.DataFrame:
        """Export configuration parameters to a class:`pd.DataFrame`

        Parameters
        ----------
        implicit: bool, default = True
            Use implicit parameter's label and unit from the raw 'CONFIG_PARAMETER_NAME' name.
        """
        implicit = to_bool(implicit)
        columns = group_cycles_by_missions(self.cycles, output='group')
        data = []
        for param in self.parameters:
            if implicit:
                pname = cfgnameparser(param)
                this = {'CONFIG_PARAMETER_NAME': f"{pname['label']} ({pname['unit']})"}
            else:
                this = {'CONFIG_PARAMETER_NAME': f"{param}"}

            for m in self.missions:
                this.update({f"Mission # {m} / Cycles # {columns[m]}": self[param, m]})
            data.append(this)
        return pd.DataFrame(data, dtype=object).sort_values(by='CONFIG_PARAMETER_NAME').reset_index(drop=True)


