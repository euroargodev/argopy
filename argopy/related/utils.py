import logging

from argopy.utils.assets import Asset
from . import ArgoNVSReferenceTables


log = logging.getLogger("argopy.related.utils")


def load_dict(ptype):
    if ptype == "profilers":
        try:
            nvs = ArgoNVSReferenceTables(cache=True)
            profilers = {}
            for irow, row in nvs.tbl(8).iterrows():
                profilers.update({int(row['id'].split("/")[-2]): row["prefLabel"]})
            profilers = dict(sorted(profilers.items()))
            return profilers
        except Exception:
            jsdata = Asset.load('profilers')
            log.debug(
                "Failed to load the ArgoNVSReferenceTables R08 for profiler types, fall back on static assets last updated on %s"
                % jsdata["last_update"]
            )
            return jsdata["data"]["profilers"]
    elif ptype == "institutions":
        try:
            nvs = ArgoNVSReferenceTables(cache=True)
            institutions = {}
            for row in nvs.tbl(4).iterrows():
                institutions.update({row[1]["altLabel"]: row[1]["prefLabel"]})
            institutions = dict(sorted(institutions.items()))
            return institutions
        except Exception:
            jsdata = Asset.load('institutions')
            log.debug(
                "Failed to load the ArgoNVSReferenceTables R04 for institutions name, fall back on static assets last updated on %s"
                % jsdata["last_update"]
            )
            return jsdata["data"]["institutions"]
    else:
        raise ValueError("Invalid dictionary name")


def mapp_dict(Adictionnary, Avalue):
    if Avalue not in Adictionnary:
        return "Unknown"
    else:
        return Adictionnary[Avalue]
