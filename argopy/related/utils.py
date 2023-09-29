import importlib
import os
import json
from . import ArgoNVSReferenceTables


path2assets = importlib.util.find_spec('argopy.static.assets').submodule_search_locations[0]


def load_dict(ptype):
    if ptype == "profilers":
        try:
            nvs = ArgoNVSReferenceTables(cache=True)
            profilers = {}
            for row in nvs.tbl(8).iterrows():
                profilers.update({int(row[1]['altLabel']): row[1]['prefLabel']})
            return profilers
        except Exception:
            with open(os.path.join(path2assets, "profilers.json"), "rb") as f:
                loaded_dict = json.load(f)['data']['profilers']
            return loaded_dict
    elif ptype == "institutions":
        try:
            nvs = ArgoNVSReferenceTables(cache=True)
            institutions = {}
            for row in nvs.tbl(4).iterrows():
                institutions.update({row[1]['altLabel']: row[1]['prefLabel']})
            return institutions
        except Exception:
            with open(os.path.join(path2assets, "institutions.json"), "rb") as f:
                loaded_dict = json.load(f)['data']['institutions']
            return loaded_dict
    else:
        raise ValueError("Invalid dictionary name")


def mapp_dict(Adictionnary, Avalue):
    if Avalue not in Adictionnary:
        return "Unknown"
    else:
        return Adictionnary[Avalue]
