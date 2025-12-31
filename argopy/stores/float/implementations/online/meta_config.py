from typing import Any

from argopy.stores.float.extensions import (
    ArgoFloatConfigParametersProto,
    ArgoFloatLaunchConfigParametersProto,
    register_ArgoFloat_extension,
)
from argopy.stores.float.implementations.online.float import FloatStore


@register_ArgoFloat_extension("launchconfig", FloatStore)
class LaunchParameters(ArgoFloatLaunchConfigParametersProto):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata: dict[Any, Any] = (
            self._obj.metadata
        )  # JSON data from the EAfleetmonitoring web-API

    @property
    def parameters(self) -> list[str]:
        return [param["argoCode"] for param in self._metadata["configDataList"]]

    def __getitem__(self, param: str) -> int | float | str | bool:
        if not param.startswith("CONFIG_"):
            param = f"CONFIG_{param}"

        for row in self._metadata["configDataList"]:
            if param == row["argoCode"]:
                return self.cast(row["argoCode"], row["value"])

        raise ValueError(f"Unknown launch configuration parameter name '{param}'")


@register_ArgoFloat_extension("config", FloatStore)
class ConfigParameters(ArgoFloatConfigParametersProto):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._metadata: dict[Any, Any] = (
            self._obj.metadata
        )  # JSON data from the EAfleetmonitoring web-API
        self._missions: list[int] = sorted([
            int(m)
            for m in list(self._metadata["configurations"]["missionCycles"].keys())
        ])
        self._c2m: dict[int, str] = self._map_cycs2missions()
        self._cycles: dict[int, int] | None = None  # Lazy attribute
        self._parameters: list[str] | None = None  # Lazy attribute

    @property
    def missions(self) -> list[int]:
        return self._missions

    @property
    def parameters(self) -> list[str]:
        if not self._parameters:
            plist = []
            for cycs_mission, cfg_list in self._metadata["configurations"][
                "cycles"
            ].items():
                for row in cfg_list:
                    if row["argoCode"] not in plist:
                        plist.append(row["argoCode"])
            sorted(plist)
            self._parameters = plist
        return self._parameters

    @property
    def cycles(self) -> dict[int, int]:
        if not self._cycles:
            result = {}
            for key, val in self._metadata["configurations"]["missionCycles"].items():
                for cyc in [int(v) for v in val]:
                    result.update({cyc: int(key)})
            self._cycles = dict(sorted(result.items()))
        return self._cycles

    def _map_cycs2missions(self) -> dict[int, str]:
        result = {}
        for key, val in self._metadata["configurations"]["missionCycles"].items():
            intval = sorted([int(v) for v in val])
            intval = [f"{v}" for v in intval]
            result.update({",".join(intval): int(key)})
        return result

    def __getitem__(
        self, index
    ) -> int | float | str | bool | list[int | float | str | bool]:
        if isinstance(index, str):
            param = index

            if not param.startswith("CONFIG_"):
                param = f"CONFIG_{param}"

            if param in self.parameters:
                results = {}
                for cycs_mission, cfg_list in self._metadata["configurations"][
                    "cycles"
                ].items():
                    for row in cfg_list:
                        if (
                            param == row["argoCode"]
                            and row["dimLevel"] == self._c2m[cycs_mission]
                        ):
                            results.update({row["dimLevel"]: row["value"]})
                            break  # Because for unknown reasons they are 3 items per argoCode, with misc 'description', so we keep the 1st one
                results = dict(sorted(results.items()))
                assert set(results.keys()) == set(
                    self.missions
                ), f"Unexpected set of missions {results.keys()} for this parameter. We expect {self.missions}."
                return [self.cast(param, v) for v in results.values()]
            else:
                raise ValueError(f"Unknown configuration parameter name '{param}'")

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
