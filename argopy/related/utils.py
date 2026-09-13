import importlib
import logging
from argopy.reference import ArgoReferenceTable

log = logging.getLogger("argopy.related.utils")
path2assets = importlib.util.find_spec(
    "argopy.static.assets"
).submodule_search_locations[0]


def load_dict(ptype):
    if ptype == "profilers":
        art = ArgoReferenceTable('ARGO_WMO_INST_TYPE')
        profilers = {}
        [profilers.update({int(arv.name): arv.long_name}) for arv in art.values()]
        return profilers
    elif ptype == "institutions":
        art = ArgoReferenceTable('DATA_CENTRE_CODES')
        institutions = {}
        [institutions.update({arv.name: arv.long_name}) for arv in art.values()]
        return institutions
    else:
        raise ValueError("Invalid dictionary name")


def mapp_dict(Adictionnary, Avalue):
    if Avalue not in Adictionnary:
        return "Unknown"
    else:
        return Adictionnary[Avalue]
