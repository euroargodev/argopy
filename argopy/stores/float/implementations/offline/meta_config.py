from typing import Any
import xarray as xr

from argopy.utils.mappers import map_vars_to_dict
from argopy.stores.float.extensions import (
    ArgoFloatConfigParametersProto,
    ArgoFloatLaunchConfigParametersProto,
    register_ArgoFloat_extension,
)
from argopy.stores.float.implementations.offline.float import FloatStore


@register_ArgoFloat_extension("launchconfig", FloatStore)
class LaunchParameters(ArgoFloatLaunchConfigParametersProto):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata: xr.Dataset = self._obj.dataset("meta")
        self._map_launch: dict[str, Any] = map_vars_to_dict(
            self._metadata,
            "LAUNCH_CONFIG_PARAMETER_NAME",
            "LAUNCH_CONFIG_PARAMETER_VALUE",
        )

    @property
    def parameters(self) -> list[str]:
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._metadata: xr.Dataset = self._obj.dataset("meta")
        self._n_missions: list[int] = list(self._metadata["N_MISSIONS"].values)
        self._cycles: dict[int, int] | None = None  # Lazy attribute

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
        if not self._cycles:
            prof = self._obj.dataset("prof")
            self._cycles = map_vars_to_dict(
                prof, "CYCLE_NUMBER", "CONFIG_MISSION_NUMBER", duplicate=True
            )
        return self._cycles

    def __getitem__(
        self, index
    ) -> int | float | str | bool | list[int | float | str | bool]:
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
                raise ValueError("Unknown configuration parameter name")

        elif isinstance(index[1], slice):
            param, s = index[0], index[1]
            results = []
            for ii in range(
                1 if not s.start else s.start,
                self.n_missions + 1 if not s.stop else s.stop,
                1 if not s.step else s.step,
            ):
                results.append(self[param, ii])

            if not results:
                raise KeyError(f"Invalid range '{s}'")
            return results

        else:
            param, mission_number = index[0], index[1]
            param_missions = self[param]
            i_mission = (
                self.missions.index(mission_number)
                if mission_number in self.missions
                else None
            )
            if i_mission is not None:
                return param_missions[i_mission]
            raise ValueError(
                f"Invalid mission number {mission_number}. Must be one in {self.missions}"
            )
