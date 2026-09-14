import logging
import pandas as pd
import numpy as np
from typing import List, Literal, Optional, Tuple
from functools import lru_cache


try:
    import pyarrow.csv as csv  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq  # noqa: F401
    import pyarrow.compute as pc  # noqa: F401
except ModuleNotFoundError:
    pass

from argopy.options import OPTIONS
from argopy.errors import InvalidDatasetStructure, OptionValueError
from argopy.utils.monitored_threadpool import pmap
from argopy.utils.checkers import is_indexbox, parse_indexbox, check_wmo, check_cyc
from argopy.utils.casting import to_list
from argopy.utils.geo import conv_lon
from argopy.stores.index.extensions import (
    register_ArgoIndex_accessor,
    ArgoIndexSearchEngine,
)
from argopy.stores.index.implementations.index_s3 import search_s3
from argopy.stores.index.implementations.pyarrow.index import indexstore

log = logging.getLogger("argopy.stores.index.pa")


@lru_cache(maxsize=25_000)
def compute_wmo(wmo: int, obj):
    return pa.compute.match_substring_regex(obj.index["file"], pattern="/%i/" % wmo)


@lru_cache(maxsize=25_000)
def compute_cyc(cyc: int, obj):
    pattern = "_%0.3d.nc" % cyc
    if cyc >= 1000:
        pattern = "_%0.4d.nc" % cyc
    return pa.compute.match_substring_regex(obj.index["file"], pattern=pattern)


@lru_cache(maxsize=25_000)
def compute_wmo_cyc(wmo: int, obj, cyc=None):
    filt = []
    for c in cyc:
        filt.append(compute_cyc(c, obj))
    return np.logical_and.reduce([compute_wmo(wmo, obj), np.logical_or.reduce(filt)])


@lru_cache(maxsize=1_000)
def compute_params(param: str, obj):
    return pa.compute.match_substring_regex(
        obj.index["parameters"],
        options=pa.compute.MatchSubstringOptions(param, ignore_case=True),
    )


@register_ArgoIndex_accessor("query", indexstore)
class SearchEngine(ArgoIndexSearchEngine):

    @search_s3
    def wmo(self, WMOs, nrows=None, composed=False):
        def checker(WMOs):
            WMOs = check_wmo(WMOs)  # Check and return a valid list of WMOs
            log.debug(
                "Argo index searching for WMOs=[%s] ..."
                % ";".join([str(wmo) for wmo in WMOs])
            )
            return WMOs

        def namer(WMOs):
            return {"WMO": WMOs}

        def composer(obj, WMOs):
            filt = pmap(obj, compute_wmo, WMOs)
            return obj._reduce_a_filter_list(filt, op="or")

        WMOs = checker(WMOs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(self._obj, WMOs)
        if not composed:
            self._obj.search_type = namer(WMOs)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(WMOs))
            return search_filter

    @search_s3
    def cyc(self, CYCs, nrows=None, composed=False):
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

        def composer(obj, CYCs):
            filt = pmap(obj, compute_cyc, CYCs)
            return obj._reduce_a_filter_list(filt)

        CYCs = checker(CYCs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(self._obj, CYCs)
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

        def composer(obj, WMOs, CYCs):
            filt = pmap(obj, compute_wmo_cyc, WMOs, kw={"cyc": tuple(CYCs)})
            return obj._reduce_a_filter_list(filt)

        WMOs, CYCs = checker(WMOs, CYCs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(self._obj, WMOs, CYCs)
        if not composed:
            self._obj.search_type = namer(WMOs, CYCs)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(WMOs, CYCs))
            return search_filter

    def date(self, BOX=None, nrows=None, composed=False, **kwargs):
        def checker(BOX, **kwargs):
            BOX = parse_indexbox("date", BOX, **kwargs)
            if "date" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for date in this index")
            is_indexbox(BOX)
            log.debug("Argo index searching for date in BOX=%s ..." % BOX)
            return ("date", BOX)  # Return key to use for time axis

        def namer(BOX):
            return {"DATE": BOX[4:6]}

        def composer(BOX, key):
            filt = []
            filt.append(
                pa.compute.greater_equal(
                    pa.compute.cast(self._obj.index[key], pa.timestamp("ms")),
                    pa.array([pd.to_datetime(BOX[4])], pa.timestamp("ms"))[0],
                )
            )
            filt.append(
                pa.compute.less_equal(
                    pa.compute.cast(self._obj.index[key], pa.timestamp("ms")),
                    pa.array([pd.to_datetime(BOX[5])], pa.timestamp("ms"))[0],
                )
            )
            return self._obj._reduce_a_filter_list(filt, op="and")

        key, BOX = checker(BOX, **kwargs)
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

    def lat(self, BOX=None, nrows=None, composed=False, **kwargs):
        def checker(BOX, **kwargs):
            BOX = parse_indexbox("lat", BOX, **kwargs)
            if "latitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for latitude in this index"
                )
            is_indexbox(BOX)
            log.debug("Argo index searching for latitude in BOX=%s ..." % BOX)
            return BOX

        def namer(BOX):
            return {"LAT": BOX[2:4]}

        def composer(BOX):
            filt = []
            filt.append(pa.compute.greater_equal(self._obj.index["latitude"], BOX[2]))
            filt.append(pa.compute.less_equal(self._obj.index["latitude"], BOX[3]))
            return self._obj._reduce_a_filter_list(filt, op="and")

        BOX = checker(BOX, **kwargs)
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

    def lon(self, BOX=None, nrows=None, composed=False, **kwargs):
        def checker(BOX, **kwargs):
            BOX = parse_indexbox("lon", BOX, **kwargs)
            if "longitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for longitude in this index"
                )
            is_indexbox(BOX)
            log.debug("Argo index searching for longitude in BOX=%s ..." % BOX)
            return BOX

        def namer(BOX):
            return {"LON": BOX[0:2]}

        def composer(BOX):
            filt = []
            if OPTIONS["longitude_convention"] == "360":
                if BOX[0] is not None:
                    filt.append(
                        pc.greater_equal(
                            self._obj.index["longitude_360"], conv_lon(BOX[0], "360")
                        )
                    )
                if BOX[1] is not None:
                    filt.append(
                        pc.less_equal(
                            self._obj.index["longitude_360"], conv_lon(BOX[1], "360")
                        )
                    )
            elif OPTIONS["longitude_convention"] == "180":
                if BOX[0] is not None:
                    filt.append(
                        pc.greater_equal(
                            self._obj.index["longitude"], conv_lon(BOX[0], "180")
                        )
                    )
                if BOX[1] is not None:
                    filt.append(
                        pc.less_equal(
                            self._obj.index["longitude"], conv_lon(BOX[1], "180")
                        )
                    )
            return self._obj._reduce_a_filter_list(filt, op="and")

        BOX = checker(BOX, **kwargs)
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
                    pa.compute.greater_equal(
                        self._obj.index["longitude_360"], conv_lon(BOX[0], "360")
                    )
                )
                filt.append(
                    pa.compute.less_equal(
                        self._obj.index["longitude_360"], conv_lon(BOX[1], "360")
                    )
                )
            elif OPTIONS["longitude_convention"] == "180":
                filt.append(
                    pa.compute.greater_equal(
                        self._obj.index["longitude"], conv_lon(BOX[0], "180")
                    )
                )
                filt.append(
                    pa.compute.less_equal(
                        self._obj.index["longitude"], conv_lon(BOX[1], "180")
                    )
                )
            filt.append(pa.compute.greater_equal(self._obj.index["latitude"], BOX[2]))
            filt.append(pa.compute.less_equal(self._obj.index["latitude"], BOX[3]))
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
            filt = []
            if OPTIONS["longitude_convention"] == "360":
                filt.append(
                    pa.compute.greater_equal(
                        self._obj.index["longitude_360"], conv_lon(BOX[0], "360")
                    )
                )
                filt.append(
                    pa.compute.less_equal(
                        self._obj.index["longitude_360"], conv_lon(BOX[1], "360")
                    )
                )
            elif OPTIONS["longitude_convention"] == "180":
                filt.append(
                    pa.compute.greater_equal(
                        self._obj.index["longitude"], conv_lon(BOX[0], "180")
                    )
                )
                filt.append(
                    pa.compute.less_equal(
                        self._obj.index["longitude"], conv_lon(BOX[1], "180")
                    )
                )
            filt.append(pa.compute.greater_equal(self._obj.index["latitude"], BOX[2]))
            filt.append(pa.compute.less_equal(self._obj.index["latitude"], BOX[3]))
            filt.append(
                pa.compute.greater_equal(
                    pa.compute.cast(self._obj.index[key], pa.timestamp("ms")),
                    pa.array([pd.to_datetime(BOX[4])], pa.timestamp("ms"))[0],
                )
            )
            filt.append(
                pa.compute.less_equal(
                    pa.compute.cast(self._obj.index[key], pa.timestamp("ms")),
                    pa.array([pd.to_datetime(BOX[5])], pa.timestamp("ms"))[0],
                )
            )
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

        def composer(obj, PARAMs, logical):
            filt = pmap(obj, compute_params, PARAMs)
            return obj._reduce_a_filter_list(filt, op=logical)

        PARAMs = checker(PARAMs)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(self._obj, PARAMs, logical)
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

                def filt_parameter_data_mode(this_idx, this_dm):
                    def fct(this_x):
                        dm = str(this_x.split("/")[-1])[0]
                        return dm in this_dm

                    x = this_idx.index["file"].to_numpy()
                    return np.array(list(map(fct, x)))

                for param in PARAMs:
                    data_mode = to_list(PARAMs[param])
                    filt.append(filt_parameter_data_mode(self._obj, data_mode))

            elif self._obj.convention in [
                "argo_bio-profile_index",
                "argo_synthetic-profile_index",
            ]:

                def filt_parameter_data_mode(this_idx, this_param, this_dm):
                    def fct(this_x, this_y):
                        variables = this_x.split()
                        return (
                            this_y[variables.index(this_param)]
                            if this_param in variables
                            else ""
                        ) in this_dm

                    x = this_idx.index["parameters"].to_numpy()
                    y = this_idx.index["parameter_data_mode"].to_numpy()
                    return np.array(list(map(fct, x, y)))

                for param in PARAMs:
                    data_mode = to_list(PARAMs[param])
                    filt.append(filt_parameter_data_mode(self._obj, param, data_mode))

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
            return pa.compute.is_in(
                self._obj.index["profiler_type"], pa.array(profiler_type)
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

    @search_s3
    def institution_code(self, institution_code: List[str], nrows=None, composed=False):
        def checker(institution_code):
            if "institution" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for institution codes in this index)"
                )
            log.debug(
                "Argo index searching for institution code in %s ..." % institution_code
            )
            institution_code = to_list(institution_code)
            valid_codes = []
            for code in institution_code:
                if self._obj.valid("institution_code", code):
                    valid_codes.append(code.upper())
            if len(valid_codes) == 0:
                raise OptionValueError(
                    f"No valid codes found for institution in {institution_code}. Valid codes are: {self._obj.valid.institution_code}"
                )
            else:
                return valid_codes

        def namer(institution_code):
            return {"INST_CODE": institution_code}

        def composer(institution_code):
            return pa.compute.is_in(
                self._obj.index["institution"], pa.array(institution_code)
            )

        institution_code = checker(institution_code)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(institution_code)
        if not composed:
            self._obj.search_type = namer(institution_code)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(institution_code))
            return search_filter

    @search_s3
    def dac(self, dac: list[str], nrows=None, composed=False):
        def checker(dac):
            if "file" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for DAC in this index)")
            log.debug("Argo index searching for DAC in %s ..." % dac)
            return to_list(dac)

        def namer(dac):
            return {"DAC": dac}

        def composer(DACs):
            filt = []
            for dac in DACs:
                filt.append(
                    pa.compute.match_substring_regex(
                        self._obj.index["file"], pattern="%s/" % dac
                    )
                )
            return self._obj._reduce_a_filter_list(filt)

        dac = checker(dac)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(dac)
        if not composed:
            self._obj.search_type = namer(dac)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(dac))
            return search_filter

    def profile_qc(self, PARAMs: dict, logical="and", nrows=None, composed=False):
        def checker(PARAMs):
            if "profile_temp_qc" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for profile QC in this index)"
                )
            # Validate PARAMs
            [PARAMs.update({p: to_list(PARAMs[p])}) for p in PARAMs]
            if not np.all(
                [
                    v in ["", " ", "1", "A", "B", "C", "D", "E", "F"]
                    for vals in PARAMs.values()
                    for v in vals
                ]
            ):
                raise ValueError(
                    "Profile QC must be a value in '', 'A', 'B', 'C', 'D', 'E', 'F'"
                )
            log.debug("Argo index searching for profile QC: %s ..." % PARAMs)
            return PARAMs

        def namer(PARAMs, logical):
            return {"PROFQC": (PARAMs, logical)}

        def composer(PARAMs, logical):
            filt = []

            for param in PARAMs:
                qcflags = PARAMs[param]
                filt.append(
                    pa.compute.is_in(
                        self._obj.index[f"profile_{param.lower()}_qc"],
                        pa.array(qcflags),
                    )
                )

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

    def psal_adj(
        self,
        where: Literal["mean", "std"] = "mean",
        ge: Optional[float] = 0.0,
        le: Optional[float] = None,
        nrows=None,
        composed=False,
    ):
        def checker(where: str, ge: Optional[float], le: Optional[float])-> [str, Optional[float], Optional[float]]:
            if where.lower() not in ['mean', 'std']:
                raise ValueError(f"'{where}': The 'where' argument must be 'mean' or 'std'.")
            if "ad_psal_adjustment_mean" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for salinity adjustment mean in this index"
                )
            if "ad_psal_adjustment_deviation" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for salinity adjustment standard deviation in this index"
                )

            bounds = [where.lower(), ge, le]

            if bounds[0] == 'std' and bounds[2] is not None and bounds[2] < 0:
                raise ValueError(f"Standard deviation lower limit must be zero or positive")

            return bounds

        def namer(bounds):
            return {f"PSAL_ADJ_{bounds[0].upper()}": bounds[1:]}

        def composer(obj, bounds):
            filt = []
            pname: str = (
                "ad_psal_adjustment_mean"
                if bounds[0] == "mean"
                else "ad_psal_adjustment_deviation"
            )
            if bounds[1] is not None:
                filt.append(pc.greater_equal(obj.index[pname], bounds[1]))
            if bounds[2] is not None:
                filt.append(pc.less_equal(obj.index[pname], bounds[2]))
            return obj._reduce_a_filter_list(filt, op="and")

        bounds = checker(where, ge, le)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(self._obj, bounds)
        if not composed:
            self._obj.search_type = namer(bounds)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(bounds))
            return search_filter

    def n_levels(
        self,
        ge: Optional[int] = None,
        le: Optional[int] = None,
        nrows=None,
        composed=False,
    ):
        def checker(ge: Optional[int], le: Optional[int])-> [Optional[int], Optional[int]]:
            if "n_levels" not in self._obj.convention_columns:
                raise InvalidDatasetStructure(
                    "Cannot search for number of levels in this index)"
                )
            bounds = [ge, le]
            if bounds[0] is not None and bounds[0] <= 0:
                raise ValueError(f"The minimum number of levels 'ge' must be positive, {bounds[0]} provided")
            if bounds[1] is not None and bounds[1] <= 0:
                raise ValueError(f"The maximum number of levels 'le' must be positive, {bounds[1]} provided")
            if bounds[0] is not None and bounds[1] is not None and bounds[0] > bounds[1]:
                raise ValueError(f"Upper bound le={bounds[1]} must be small than the lower bound ge={bounds[0]}")
            return bounds

        def namer(bounds):
            return {f"NLEVELS": bounds}

        def composer(obj, bounds):
            filt = []
            if bounds[0] is not None:
                filt.append(pc.greater_equal(obj.index['n_levels'], bounds[0]))
            if bounds[1] is not None:
                filt.append(pc.less_equal(obj.index['n_levels'], bounds[1]))
            return obj._reduce_a_filter_list(filt, op="and")

        bounds = checker(ge, le)
        self._obj.load(nrows=self._obj._nrows_index)
        search_filter = composer(self._obj, bounds)
        if not composed:
            self._obj.search_type = namer(bounds)
            self._obj.search_filter = search_filter
            self._obj.run(nrows=nrows)
            return self._obj
        else:
            self._obj.search_type.update(namer(bounds))
            return search_filter
