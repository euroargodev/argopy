import pandas as pd
import importlib


def _importorskip(modname):
    try:
        importlib.import_module(modname)  # noqa: E402
        has = True
    except ImportError:
        has = False
    return has


has_jsonschema = _importorskip("jsonschema")


class APISensorMetaDataProcessing:

    @classmethod
    def preprocess_df(cls, jsdata, model_name: str = "", **kwargs) -> list[str]:
        """Get the list of sensor metadata for a given model"""
        output = []
        for s in jsdata["sensors"]:
            if model_name in s["model"]:
                this = [jsdata["wmo"]]
                [
                    this.append(s[key])  # type: ignore
                    for key in [
                    "id",
                    "maker",
                    "model",
                    "serial",
                    "units",
                    "accuracy",
                    "resolution",
                ]
                ]
                output.append(this)
        return output

    @classmethod
    def postprocess_df(cls, data, **kwargs):
        d = []
        for this in data:
            for wmo, sid, maker, model, sn, units, accuracy, resolution in this:
                d.append(
                    {
                        "WMO": wmo,
                        "Type": sid,
                        "Model": model,
                        "Maker": maker,
                        "SerialNumber": sn if sn != "n/a" else None,
                        "Units": units,
                        "Accuracy": accuracy,
                        "Resolution": resolution,
                    }
                )
        return pd.DataFrame(d).sort_values(by="WMO").reset_index(drop=True)


