from typing import Any
import xarray as xr
import pandas as pd

from argopy.utils.mappers import map_vars_to_dict
from argopy.utils.format import cfgnameparser
from argopy.stores.float.extensions import ArgoFloatConfigParametersProto, ArgoFloatLaunchConfigParametersProto, register_ArgoFloat_extension
from argopy.stores.float.implementations.offline.float import FloatStore


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
        self._metadata : xr.Dataset = self._obj.dataset("meta")
        self._map_launch: dict[str, Any] = map_vars_to_dict(
            self._metadata,
            "LAUNCH_CONFIG_PARAMETER_NAME",
            "LAUNCH_CONFIG_PARAMETER_VALUE",
        )

    @property
    def parameters(self) -> list[str]:
        """List of launch configuration parameter names"""
        return [
            pname.item().strip()
            for pname in self._metadata["LAUNCH_CONFIG_PARAMETER_NAME"]
        ]

    def __getitem__(self, param: str) -> int | float | str | bool:
        if not param.startswith("CONFIG_"):
            param = f"CONFIG_{param}"
        if param in self._map_launch:
            return self.cast(param, self._map_launch[param])
        else:
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

        self._metadata : xr.Dataset = self._obj.dataset("meta")
        self._n_missions : list[int] = list(self._metadata["N_MISSIONS"].values)
        self._cycles : dict[int, int] | None = None # Lazy attribute

    @property
    def missions(self) -> list[int]:
        return self._metadata["CONFIG_MISSION_NUMBER"].to_numpy().tolist()

    @property
    def parameters(self) -> list[str]:
        return [
            pname.item().strip() for pname in self._metadata["CONFIG_PARAMETER_NAME"]
        ]

    @property
    def cycles(self) -> dict[int, int]:
        """A dictionary mapping cycle on mission numbers

        Returns
        -------
        dict[int, int]:
            Keys are cycle numbers, Values are mission numbers
        """
        if not self._cycles:
            prof = self._obj.dataset("prof")
            self._cycles = map_vars_to_dict(prof, 'CYCLE_NUMBER', 'CONFIG_MISSION_NUMBER', duplicate=True)
        return self._cycles

    def __getitem__(self, index) -> int | float | str | bool | list[int | float | str | bool]:
        """Retrieve configuration parameter values for one or a slice of mission numbers"""
        if isinstance(index, str):
            param = index

            if not param.startswith("CONFIG_"):
                param = f"CONFIG_{param}"

            mask = self._metadata["CONFIG_PARAMETER_NAME"] == param.ljust(128)
            if mask.any():
                iparam = self._metadata["N_CONFIG_PARAM"][mask].item()
                da_param = self._metadata["CONFIG_PARAMETER_VALUE"].isel(
                    N_CONFIG_PARAM=iparam
                )
                results = [self.cast(param, val) for val in da_param.values]

                return results

            else:
                raise ValueError(f"Unknown configuration parameter name")

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
