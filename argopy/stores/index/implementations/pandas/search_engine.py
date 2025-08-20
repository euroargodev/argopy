import logging
import pandas as pd
import numpy as np
from typing import List

from .....options import OPTIONS
from .....errors import InvalidDatasetStructure
from .....utils import is_indexbox, check_wmo, check_cyc, to_list, conv_lon
from ...extensions import register_ArgoIndex_accessor, ArgoIndexSearchEngine
from ..index_s3 import search_s3
from .index import indexstore

log = logging.getLogger("argopy.stores.index.pd")


@register_ArgoIndex_accessor("query", indexstore)
class SearchEngine(ArgoIndexSearchEngine):

    @search_s3
    def wmo(self, WMOs, nrows=None, composed=False) -> indexstore:
        def checker(WMOs):
            WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
            log.debug(
                "Argo index searching for WMOs=[%s] ..."
                % ";".join([str(wmo) for wmo in WMOs])
            )
            return WMOs

        def namer(WMOs):
            return {"WMO": WMOs}

        def composer(WMOs):
            filt = []
            for wmo in WMOs:
                filt.append(
                    self._obj.index["file"].str.contains(
                        "/%i/" % wmo, regex=True, case=False
                    )
                )
            return self._obj._reduce_a_filter_list(filt, op="or")

        WMOs = checker(WMOs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(WMOs)
        if not composed:
            self._obj.search_type = namer(WMOs)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(WMOs))
            return search_filter

    @search_s3
    def cyc(self, CYCs, nrows=None, composed=False) -> indexstore:
        def checker(CYCs):
            if self._obj.convention in ["ar_index_global_meta"]:
                raise InvalidDatasetStructure(
                    "Cannot search for cycle number in this index)"
                )
            CYCs = check_cyc(CYCs)  # Check and return a valid list of CYCs
            log.debug(
                "Argo index searching for CYCs=[%s] ..."
                % (";".join([str(cyc) for cyc in CYCs]))
            )
            return CYCs

        def namer(CYCs):
            return {"CYC": CYCs}

        def composer(CYCs):
            filt = []
            for cyc in CYCs:
                if cyc < 1000:
                    pattern = "_%0.3d.nc" % (cyc)
                else:
                    pattern = "_%0.4d.nc" % (cyc)
                filt.append(
                    self._obj.index["file"].str.contains(
                        pattern, regex=True, case=False
                    )
                )
            return self._obj._reduce_a_filter_list(filt, op="or")

        CYCs = checker(CYCs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(CYCs)
        if not composed:
            self._obj.search_type = namer(CYCs)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(CYCs))
            return search_filter

    @search_s3
    def wmo_cyc(self, WMOs, CYCs, nrows=None, composed=False):
        def checker(WMOs, CYCs):
            if self._obj.convention in ["ar_index_global_meta"]:
                raise InvalidDatasetStructure(
                    "Cannot search for cycle number in this index)"
                )
            WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
            CYCs = check_cyc(CYCs)  # Check and return a valid list of CYCs
            log.debug(
                "Argo index searching for WMOs=[%s] and CYCs=[%s] ..."
                % (
                    ";".join([str(wmo) for wmo in WMOs]),
                    ";".join([str(cyc) for cyc in CYCs]),
                )
            )
            return WMOs, CYCs

        def namer(WMOs, CYCs):
            return {"WMO": WMOs, "CYC": CYCs}

        def composer(WMOs, CYCs):
            filt = []
            for wmo in WMOs:
                for cyc in CYCs:
                    if cyc < 1000:
                        pattern = "%i_%0.3d.nc" % (wmo, cyc)
                    else:
                        pattern = "%i_%0.4d.nc" % (wmo, cyc)
                    filt.append(
                        self._obj.index["file"].str.contains(
                            pattern, regex=True, case=False
                        )
                    )
            return self._obj._reduce_a_filter_list(filt, op="or")

        WMOs, CYCs = checker(WMOs, CYCs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(WMOs, CYCs)
        if not composed:
            self._obj.search_type = namer(WMOs, CYCs)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(WMOs, CYCs))
            return search_filter

    def date(self, BOX, nrows=None, composed=False):
        def checker(BOX):
            if "date" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for date in this index")
            is_indexbox(BOX)
            log.debug("Argo index searching for date in BOX=%s ..." % BOX)
            return "date"  # Return key to use for time axis

        def namer(BOX):
            return {"DATE": BOX[4:6]}

        def composer(BOX, key):
            tim_min = int(pd.to_datetime(BOX[4]).strftime("%Y%m%d%H%M%S"))
            tim_max = int(pd.to_datetime(BOX[5]).strftime("%Y%m%d%H%M%S"))
            filt = []
            filt.append(self._obj.index[key].ge(tim_min))
            filt.append(self._obj.index[key].le(tim_max))
            return self._obj._reduce_a_filter_list(filt, op="and")

        key = checker(BOX)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(BOX, key)
        if not composed:
            self._obj.search_type = namer(BOX)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(BOX))
            return search_filter

    def lat(self, BOX, nrows=None, composed=False):
        def checker(BOX):
            if "latitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for latitude in this index"
                )
            is_indexbox(BOX)
            log.debug("Argo index searching for latitude in BOX=%s ..." % BOX)

        def namer(BOX):
            return {"LAT": BOX[2:4]}

        def composer(BOX):
            filt = []
            filt.append(self._obj.index["latitude"].ge(BOX[2]))
            filt.append(self._obj.index["latitude"].le(BOX[3]))
            return self._obj._reduce_a_filter_list(filt, op="and")

        checker(BOX)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(BOX)
        if not composed:
            self._obj.search_type = namer(BOX)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(BOX))
            return search_filter

    def lon(self, BOX, nrows=None, composed=False):
        def checker(BOX):
            if "longitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for longitude in this index"
                )
            is_indexbox(BOX)
            log.debug("Argo index searching for longitude in BOX=%s ..." % BOX)

        def namer(BOX):
            return {"LON": BOX[0:2]}

        def composer(BOX):
            filt = []
            if OPTIONS["longitude_convention"] == "360":
                filt.append(
                    self._obj.index["longitude_360"].ge(conv_lon(BOX[0], "360"))
                )
                filt.append(
                    self._obj.index["longitude_360"].le(conv_lon(BOX[1], "360"))
                )
            elif OPTIONS["longitude_convention"] == "180":
                filt.append(self._obj.index["longitude"].ge(conv_lon(BOX[0], "180")))
                filt.append(self._obj.index["longitude"].le(conv_lon(BOX[1], "180")))
            return self._obj._reduce_a_filter_list(filt, op="and")

        checker(BOX)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(BOX)
        if not composed:
            self._obj.search_type = namer(BOX)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(BOX))
            return search_filter

    def lon_lat(self, BOX, nrows=None, composed=False):
        def checker(BOX):
            if "longitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for lon/lat in this index")
            is_indexbox(BOX)
            log.debug("Argo index searching for lon/lat in BOX=%s ..." % BOX)

        def namer(BOX):
            return {"LON": BOX[0:2], "LAT": BOX[2:4]}

        def composer(BOX):
            filt = []
            if OPTIONS["longitude_convention"] == "360":
                filt.append(
                    self._obj.index["longitude_360"].ge(conv_lon(BOX[0], "360"))
                )
                filt.append(
                    self._obj.index["longitude_360"].le(conv_lon(BOX[1], "360"))
                )
            elif OPTIONS["longitude_convention"] == "180":
                filt.append(self._obj.index["longitude"].ge(conv_lon(BOX[0], "180")))
                filt.append(self._obj.index["longitude"].le(conv_lon(BOX[1], "180")))
            filt.append(self._obj.index["latitude"].ge(BOX[2]))
            filt.append(self._obj.index["latitude"].le(BOX[3]))
            return self._obj._reduce_a_filter_list(filt, op="and")

        checker(BOX)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(BOX)
        if not composed:
            self._obj.search_type = namer(BOX)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(BOX))
            return search_filter

    def box(self, BOX, nrows=None, composed=False):
        def checker(BOX):
            if "longitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for coordinates in this index"
                )
            is_indexbox(BOX)
            log.debug("Argo index searching for lat/lon/date in BOX=%s ..." % BOX)
            return "date"  # Return key to use for time axis

        def namer(BOX):
            return {"BOX": BOX}

        def composer(BOX, key):
            tim_min = int(pd.to_datetime(BOX[4]).strftime("%Y%m%d%H%M%S"))
            tim_max = int(pd.to_datetime(BOX[5]).strftime("%Y%m%d%H%M%S"))

            filt = []
            if OPTIONS["longitude_convention"] == "360":
                filt.append(
                    self._obj.index["longitude_360"].ge(conv_lon(BOX[0], "360"))
                )
                filt.append(
                    self._obj.index["longitude_360"].le(conv_lon(BOX[1], "360"))
                )
            elif OPTIONS["longitude_convention"] == "180":
                filt.append(self._obj.index["longitude"].ge(conv_lon(BOX[0], "180")))
                filt.append(self._obj.index["longitude"].le(conv_lon(BOX[1], "180")))
            filt.append(self._obj.index["latitude"].ge(BOX[2]))
            filt.append(self._obj.index["latitude"].le(BOX[3]))
            filt.append(self._obj.index[key].ge(tim_min))
            filt.append(self._obj.index[key].le(tim_max))
            return self._obj._reduce_a_filter_list(filt, op="and")

        key = checker(BOX)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(BOX, key)
        if not composed:
            self._obj.search_type = namer(BOX)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(BOX))
            return search_filter

    def params(self, PARAMs, logical="and", nrows=None, composed=False):
        def checker(PARAMs):
            if "parameters" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for parameters in this index (%s: %s)."
                    % (self._obj.convention, self._obj.convention_title)
                )
            log.debug("Argo index searching for parameters in PARAM=%s." % PARAMs)
            return to_list(PARAMs)  # Make sure we deal with a list

        def namer(PARAMs, logical):
            return {"PARAMS": (PARAMs, logical)}

        def composer(PARAMs, logical):
            filt = []
            self._obj.index["variables"] = self._obj.index["parameters"].apply(
                lambda x: x.split()
            )
            for param in PARAMs:
                filt.append(self._obj.index["variables"].apply(lambda x: param in x))
            self._obj.index = self._obj.index.drop("variables", axis=1)
            return self._obj._reduce_a_filter_list(filt, op=logical)

        PARAMs = checker(PARAMs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(PARAMs, logical)
        if not composed:
            self._obj.search_type = namer(PARAMs, logical)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(PARAMs, logical))
            return search_filter

    def parameter_data_mode(
        self, PARAMs: dict, logical="and", nrows=None, composed=False
    ):
        def checker(PARAMs):
            if self._obj.convention not in [
                "ar_index_global_prof",
                "argo_synthetic-profile_index",
                "argo_bio-profile_index",
            ]:
                raise InvalidDatasetStructure(
                    "Cannot search for parameter data mode in this index)"
                )
            log.debug(
                "Argo index searching for parameter data modes such as PARAM=%s ..."
                % PARAMs
            )

            # Validate PARAMs argument type
            [PARAMs.update({p: to_list(PARAMs[p])}) for p in PARAMs]
            if not np.all(
                [
                    v in ["R", "A", "D", "", " "]
                    for vals in PARAMs.values()
                    for v in vals
                ]
            ):
                raise ValueError("Data mode must be a value in 'R', 'A', 'D', ' ', ''")
            if self._obj.convention in ["argo_aux-profile_index"]:
                raise InvalidDatasetStructure(
                    "Method not available for this index ('%s')" % self._obj.convention
                )
            return PARAMs

        def namer(PARAMs, logical):
            return {"DMODE": (PARAMs, logical)}

        def composer(PARAMs, logical):
            filt = []

            if self._obj.convention in ["ar_index_global_prof"]:
                for param in PARAMs:
                    data_mode = to_list(PARAMs[param])
                    filt.append(
                        self._obj.index["file"].apply(
                            lambda x: str(x.split("/")[-1])[0] in data_mode
                        )
                    )

            elif self._obj.convention in [
                "argo_bio-profile_index",
                "argo_synthetic-profile_index",
            ]:
                self._obj.index["variables"] = self._obj.index["parameters"].apply(
                    lambda x: x.split()
                )
                for param in PARAMs:
                    data_mode = to_list(PARAMs[param])
                    filt.append(
                        self._obj.index.apply(
                            lambda x: (
                                x["parameter_data_mode"][x["variables"].index(param)]
                                if param in x["variables"]
                                else ""
                            )
                            in data_mode,
                            axis=1,
                        )
                    )
                self._obj.index = self._obj.index.drop("variables", axis=1)

            return self._obj._reduce_a_filter_list(filt, op=logical)

        PARAMs = checker(PARAMs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(PARAMs, logical)
        if not composed:
            self._obj.search_type = namer(PARAMs, logical)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(PARAMs, logical))
            return search_filter

    def profiler_type(self, profiler_type: List[int], nrows=None, composed=False):
        def checker(profiler_type):
            if "profiler_type" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for profiler types in this index)"
                )
            log.debug(
                "Argo index searching for profiler type in %s ..." % profiler_type
            )
            return to_list(profiler_type)

        def namer(profiler_type):
            return {"PTYPE": profiler_type}

        def composer(profiler_type):
            return (
                self._obj.index["profiler_type"]
                .fillna(99999)
                .astype(int)
                .isin(profiler_type)
            )

        profiler_type = checker(profiler_type)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(profiler_type)
        if not composed:
            self._obj.search_type = namer(profiler_type)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(profiler_type))
            return search_filter
