import logging
import pandas as pd
import numpy as np
from typing import List

try:
    import pyarrow.csv as csv  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq  # noqa: F401
    import pyarrow.compute as pc  # noqa: F401
except ModuleNotFoundError:
    pass

from .....options import OPTIONS
from .....errors import InvalidDatasetStructure
from .....utils import is_indexbox, check_wmo, check_cyc, to_list, conv_lon
from ...extensions import register_ArgoIndex_accessor, ArgoIndexSearchEngine
from ..index_s3 import search_s3
from .index import indexstore

log = logging.getLogger("argopy.stores.index.pa")


@register_ArgoIndex_accessor('query', indexstore)
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

        def composer(WMOs):
            filt = []
            for wmo in WMOs:
                filt.append(
                    pa.compute.match_substring_regex(
                        self._obj.index["file"], pattern="/%i/" % wmo
                    )
                )
            return self._obj._reduce_a_filter_list(filt)

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

        def composer(CYCs):
            filt = []
            for cyc in CYCs:
                if cyc < 1000:
                    pattern = "_%0.3d.nc" % (cyc)
                else:
                    pattern = "_%0.4d.nc" % (cyc)
                filt.append(
                    pa.compute.match_substring_regex(self._obj.index["file"], pattern=pattern)
                )
            return self._obj._reduce_a_filter_list(filt)

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
                        pa.compute.match_substring_regex(
                            self._obj.index["file"], pattern=pattern
                        )
                    )
            return self._obj._reduce_a_filter_list(filt)

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
                raise InvalidDatasetStructure("Cannot search for latitude in this index")
            is_indexbox(BOX)
            log.debug("Argo index searching for latitude in BOX=%s ..." % BOX)

        def namer(BOX):
            return {"LAT": BOX[2:4]}

        def composer(BOX):
            filt = []
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

    def lon(self, BOX, nrows=None, composed=False):
        def checker(BOX):
            if "longitude" not in self._obj.convention_columns:
                raise InvalidDatasetStructure("Cannot search for longitude in this index")
            is_indexbox(BOX)
            log.debug("Argo index searching for longitude in BOX=%s ..." % BOX)

        def namer(BOX):
            return {"LON": BOX[0:2]}

        def composer(BOX):
            filt = []
            if OPTIONS['longitude_convention'] == '360':
                filt.append(pa.compute.greater_equal(self._obj.index["longitude_360"], conv_lon(BOX[0], '360')))
                filt.append(pa.compute.less_equal(self._obj.index["longitude_360"], conv_lon(BOX[1], '360')))
            elif OPTIONS['longitude_convention'] == '180':
                filt.append(pa.compute.greater_equal(self._obj.index["longitude"], conv_lon(BOX[0], '180')))
                filt.append(pa.compute.less_equal(self._obj.index["longitude"], conv_lon(BOX[1], '180')))
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
            if OPTIONS['longitude_convention'] == '360':
                filt.append(pa.compute.greater_equal(self._obj.index["longitude_360"], conv_lon(BOX[0], '360')))
                filt.append(pa.compute.less_equal(self._obj.index["longitude_360"], conv_lon(BOX[1], '360')))
            elif OPTIONS['longitude_convention'] == '180':
                filt.append(pa.compute.greater_equal(self._obj.index["longitude"], conv_lon(BOX[0], '180')))
                filt.append(pa.compute.less_equal(self._obj.index["longitude"], conv_lon(BOX[1], '180')))
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
                raise InvalidDatasetStructure("Cannot search for coordinates in this index")
            is_indexbox(BOX)
            log.debug("Argo index searching for lat/lon/date in BOX=%s ..." % BOX)
            return "date"  # Return key to use for time axis

        def namer(BOX):
            return {"BOX": BOX}

        def composer(BOX, key):
            filt = []
            if OPTIONS['longitude_convention'] == '360':
                filt.append(pa.compute.greater_equal(self._obj.index["longitude_360"], conv_lon(BOX[0], '360')))
                filt.append(pa.compute.less_equal(self._obj.index["longitude_360"], conv_lon(BOX[1], '360')))
            elif OPTIONS['longitude_convention'] == '180':
                filt.append(pa.compute.greater_equal(self._obj.index["longitude"], conv_lon(BOX[0], '180')))
                filt.append(pa.compute.less_equal(self._obj.index["longitude"], conv_lon(BOX[1], '180')))
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
                raise InvalidDatasetStructure("Cannot search for parameters in this index (%s: %s)." % (self._obj.convention, self._obj.convention_title))
            log.debug("Argo index searching for parameters in PARAM=%s." % PARAMs)
            return to_list(PARAMs)  # Make sure we deal with a list

        def namer(PARAMs, logical):
            return {"PARAMS": (PARAMs, logical)}

        def composer(PARAMs, logical):
            filt = []
            for param in PARAMs:
                filt.append(
                    pa.compute.match_substring_regex(
                        self._obj.index["parameters"],
                        options=pa.compute.MatchSubstringOptions(param, ignore_case=True),
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

    def parameter_data_mode(self, PARAMs: dict, logical="and", nrows=None, composed=False):
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
            [
                PARAMs.update({p: to_list(PARAMs[p])}) for p in PARAMs
            ]
            if not np.all(
                [v in ["R", "A", "D", "", " "] for vals in PARAMs.values() for v in vals]
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
                raise InvalidDatasetStructure("Cannot search for profiler types in this index)")
            log.debug("Argo index searching for profiler type in %s ..." % profiler_type)
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
