from typing import Any
import pandas as pd

from argopy.utils.format import cfgnameparser
from argopy.stores.float.extensions import ArgoFloatConfigParametersProto, ArgoFloatLaunchConfigParametersProto, register_ArgoFloat_extension
from argopy.stores.float.implementations.online.float import FloatStore


@register_ArgoFloat_extension("launchconfig", FloatStore)
class LaunchParameters(ArgoFloatLaunchConfigParametersProto):
    """Pre-deployment or launch configuration parameters that are the ‘configured’ start settings of the float and the initial mission configuration parameters for the first cycle.

    Warnings
    --------
    Configuration parameters are float settings selected by the PI, not measurements reported by the float.

    Examples
    --------
    ..code-block: python

        from argopy import ArgoFloat
        af = ArgoFloat(6903091)

        # Number of launch parameters:
        af.launchconfig.n_params

        # List of launch parameters:
        af.launchconfig.parameters

        # Read one parameter value, with explicit or implicit parameter name:
        # ('CONFIG_' is not mandatory, but string is case-sensitive)
        af.launchconfig['CONFIG_CycleTime_hours']
        af.launchconfig['CycleTime_hours']

        # Export to a DataFrame:
        af.launchconfig.to_dataframe()
        af.launchconfig.to_dataframe(implicit=False)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata : dict[Any, Any] = self._obj.metadata  # JSON data from the EAfleetmonitoring web-API

    @property
    def parameters(self) -> list[str]:
        """List of launch configuration parameter names"""
        return [param['argoCode'] for param in self._metadata['configDataList']]

    def __getitem__(self, param: str) -> int | float | str | bool:
        if not param.startswith("CONFIG_"):
            param = f"CONFIG_{param}"

        for row in self._metadata['configDataList']:
            if param == row['argoCode']:
                return self.cast(row['argoCode'], row['value'])

        raise ValueError(f"Unknown launch configuration parameter name '{param}'")

@register_ArgoFloat_extension("config", FloatStore)
class ConfigParameters(ArgoFloatConfigParametersProto):
    """

    Warnings
    --------
    Configuration parameters are float settings selected by the PI, not measurements reported by the float.

    Examples
    --------
    ..code-block: python

        from argopy import ArgoFloat
        af = ArgoFloat(6903091)

        af.config.n_missions
        af.config.n_params

        # Read the list of configuration parameters:
        af.config.parameters

        # Read one parameter value, with explicit or implicit parameter name:
        # ('CONFIG_' is not mandatory, but string is case-sensitive)
        af.config['CONFIG_CycleTime_hours']
        af.config['CycleTime_hours']

        # Read parameter value for one or more mission numbers:
        # (! 2nd index is not 0-based, it's a key to look for in mission numbers)
        af.config['CycleTime_hours', 1]
        af.config['CycleTime_hours', 1:3]

        # Read parameter value for one or more cycle numbers:
        af.config.for_cyc('CycleTime_hours', 1)
        af.config.for_cyc('CycleTime_hours', [10, 11])

        # Export to a DataFrame:
        af.config.to_dataframe()
        af.config.to_dataframe(implicit=False)

    References
    ----------
    ..code-block: python

        import argopy
        argopy.ArgoDocs(29825).open_pdf(55)  # User manual section on "Configuration parameters"
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._metadata : dict[Any, Any] = self._obj.metadata  # JSON data from the EAfleetmonitoring web-API
        self._n_missions : list[int] = [int(m) for m in list(self._metadata['configurations']['missionCycles'].keys())]
        self._c2m : dict[int, str] = self._map_cycs2missions()
        self._cycles : dict[int, int] | None = None # Lazy attribute
        self._parameters : list[str] | None = None # Lazy attribute

    @property
    def missions(self) -> list[int]:
        return self._n_missions

    @property
    def cycles(self) -> dict[int, int]:
        """A dictionary mapping cycle on mission numbers

        Returns
        -------
        dict[int, int]:
            Keys are cycle numbers, Values are mission numbers
        """
        if not self._cycles:
            result = {}
            for key, val in self._metadata['configurations']['missionCycles'].items():
                for cyc in [int(v) for v in val]:
                    result.update({cyc: int(key)})
            self._cycles = dict(sorted(result.items()))
        return self._cycles

    @property
    def parameters(self) -> list[str]:
        if not self._parameters:
            plist = []
            for cycs_mission, cfg_list in self._metadata['configurations']['cycles'].items():
                for row in cfg_list:
                    if row['argoCode'] not in plist:
                        plist.append(row['argoCode'])
            sorted(plist)
            self._parameters = plist
        return self._parameters

    def _map_cycs2missions(self) -> dict[int, str]:
        result = {}
        for key, val in self._metadata['configurations']['missionCycles'].items():
            intval = sorted([int(v) for v in val])
            intval = [f"{v}" for v in intval]
            result.update({",".join(intval): int(key)})
        return result

    def __getitem__(self, index) -> int | float | str | bool | list[int | float | str | bool]:
        if isinstance(index, str):
            param = index

            if not param.startswith("CONFIG_"):
                param = f"CONFIG_{param}"

            if param in self.parameters:
                results = {}
                for cycs_mission, cfg_list in self._metadata['configurations']['cycles'].items():
                    for row in cfg_list:
                        if param == row['argoCode'] and row['dimLevel'] == self._c2m[cycs_mission]:
                            results.update({row['dimLevel']: row['value']})
                            break  # Because for unknown reasons they are 3 items per argoCode, with misc 'description', so we keep the 1st one
                results = dict(sorted(results.items()))
                assert set(results.keys()) == set(self.missions), f"Unexpected set of missions {results.keys()} for this parameter. We expect {self.missions}."
                return [self.cast(param, v) for v in results.values()]
            else:
                raise ValueError(f"Unknown configuration parameter name '{param}'")

        elif isinstance(index[1], slice):
            param, s = index[0], index[1]
            results = []
            for ii in range(1 if not s.start else s.start, self.n_missions+1 if not s.stop else s.stop, 1 if not s.step else s.step):
                results.append(self[param, ii])

            if not results:
                raise KeyError(f"Invalid range '{s}'")
            return results

        else:
            param, mission_number = index[0], index[1]
            param_missions = self[param]
            i_mission = self.missions.index(mission_number) if mission_number in self.missions else None
            if i_mission is not None:
                return param_missions[i_mission]
            raise ValueError(f"Invalid mission number {mission_number}. Must be one in {self.missions}")
