import numpy as np
import datetime as dt
# import gsw

from argopy.utils.arco.xarr_utils import get_use_adj_map, get_param
from argopy.utils.arco.nc_parsed import NCParsed
from argopy.utils.arco.flags import UnknownDataMode, MissingParameter, InappropriateAdjustedValues, NegativePressure, \
    UnauthorisedNan, QCCountsMismatch
from argopy.utils.arco.equality import element_wise_nan_equal
from argopy.utils.arco.errors import UnrecognisedDataSelectionMode


class ArgoMultiProf(NCParsed):
    """A super class for an Argo xarray dataset created with a multi-profile file"""
    def __init__(self, arr, data_selection_mode="adj_non_empty"):

        self.wmo = arr["PLATFORM_NUMBER"].values.astype(int)
        self.institute = arr["DATA_CENTRE"].values.astype(str)

        super().__init__(arr)
        self.data_selection_mode = data_selection_mode
        self.temp_found = np.isin(["TEMP", "TEMP_ADJUSTED", "TEMP_QC", "TEMP_ADJUSTED_QC"], self.data_vars).all()
        self.psal_found = np.isin(["PSAL", "PSAL_ADJUSTED", "PSAL_QC", "PSAL_ADJUSTED_QC"], self.data_vars).all()

        self.data_mode = np.array([0 if x == "R" else 1 if x == "A" else 2 if x == "D" else -1 for x in
                                   arr["DATA_MODE"].values.astype(str)])

        if self.data_selection_mode == "data_mode":
            self.temp_selection_mask = (self.data_mode > 0) if self.temp_found else None
            self.psal_selection_mask = (self.data_mode > 0) if self.psal_found else None
            self.pres_selection_mask = (self.data_mode > 0)

        elif self.data_selection_mode == "adj_non_empty":

            self.pres_selection_mask = get_use_adj_map(adjusted_param_name="PRES_ADJUSTED", xarr=arr)
            if self.temp_found:
                self.temp_selection_mask = get_use_adj_map(adjusted_param_name="TEMP_ADJUSTED", xarr=arr)
            else:
                self.temp_selection_mask = None
            if self.psal_found:
                self.psal_selection_mask = get_use_adj_map(adjusted_param_name="PSAL_ADJUSTED", xarr=arr)
            else:
                self.psal_selection_mask = None
        elif self.data_selection_mode == "real-time":
            self.pres_selection_mask = np.zeros(arr["PRES"].values.shape[0]).astype(np.bool)
            if self.temp_found:
                self.temp_selection_mask = np.zeros(arr["TEMP"].values.shape[0]).astype(np.bool)
            else:
                self.temp_selection_mask = None
            if self.psal_found:
                self.psal_selection_mask = np.zeros(arr["PSAL"].values.shape[0]).astype(np.bool)
            else:
                self.psal_selection_mask = None
        elif self.data_selection_mode == "delayed_mode":
            self.pres_selection_mask = np.ones(arr["PRES_ADJUSTED"].values.shape[0]).astype(np.bool)
            if self.temp_found:
                self.temp_selection_mask = np.ones(arr["TEMP_ADJUSTED"].values.shape[0]).astype(np.bool)
            else:
                self.temp_selection_mask = None
            if self.psal_found:
                self.psal_selection_mask = np.ones(arr["PSAL_ADJUSTED"].values.shape[0]).astype(np.bool)
            else:
                self.psal_selection_mask = None

        else:
            raise UnrecognisedDataSelectionMode(institute=self.institute, wmo=self.wmo)

        self.pres = get_param("PRES", "PRES_ADJUSTED", self.pres_selection_mask, arr).astype(np.float32)
        self.pres_qc = get_param("PRES_QC", "PRES_ADJUSTED_QC", self.pres_selection_mask, arr, assign_nan=7).astype(
            np.float32)
        self.pres_error = get_param("PRES_ADJUSTED_ERROR", "PRES_ADJUSTED_ERROR", self.pres_selection_mask, arr).astype(np.float32)

        if self.temp_selection_mask is not None:
            self.temp = get_param("TEMP", "TEMP_ADJUSTED", self.temp_selection_mask, arr).astype(np.float32)
            self.temp_qc = get_param("TEMP_QC", "TEMP_ADJUSTED_QC", self.temp_selection_mask, arr, assign_nan=7).astype(
                np.float32)
            self.temp_error = get_param("TEMP_ADJUSTED_ERROR", "TEMP_ADJUSTED_ERROR", self.temp_selection_mask,
                                        arr).astype(np.float32)
        else:
            self.temp = None
            self.temp_qc = None

        if self.psal_selection_mask is not None:
            self.psal = get_param("PSAL", "PSAL_ADJUSTED", self.psal_selection_mask, arr).astype(np.float32)
            self.psal_qc = get_param("PSAL_QC", "PSAL_ADJUSTED_QC", self.psal_selection_mask, arr, assign_nan=7).astype(
                np.float32)
            self.psal_error = get_param("PSAL_ADJUSTED_ERROR", "PSAL_ADJUSTED_ERROR", self.psal_selection_mask,
                                        arr).astype(np.float32)
        else:
            self.psal = None
            self.psal_qc = None

        self.lon = arr["LONGITUDE"].values.astype(np.float32)
        self.lat = arr["LATITUDE"].values.astype(np.float32)
        self.position_qc = arr["POSITION_QC"].values.astype(np.float32)
        self.position_qc[~np.isfinite(self.position_qc)] = 7

        self.juld = arr["JULD"].values.astype(np.float32)
        self.juld_qc = arr["JULD_QC"].values.astype(np.float32)
        self.juld_qc[~np.isfinite(self.juld_qc)] = 7

        # self.datetime = np.array([dt.timedelta(days=float(x)) for x in np.nan_to_num(self.juld)]) + \
        #                 dt.datetime(year=1950, month=1, day=1, hour=0, minute=0, second=0)
        self.datetime = np.array([dt.datetime.utcfromtimestamp(x) for x in (arr['JULD'].values - np.datetime64(0, 's')) / np.timedelta64(1, 's')])
        self.month = np.array([x.month for x in self.datetime])
        self.year = np.array([x.year for x in self.datetime])
        self.cycle = arr["CYCLE_NUMBER"].values.astype(int)
        self.direction = np.array([1 if x == "A" else -1 for x in arr["DIRECTION"].values.astype(str)])

        self.psal_n = np.sum(np.isin(self.psal_qc, [0, 1, 2, 3, 4, 5, 6, 8]))
        self.temp_n = np.sum(np.isin(self.temp_qc, [0, 1, 2, 3, 4, 5, 6, 8]))
        self.pres_n = np.sum(np.isin(self.pres_qc, [0, 1, 2, 3, 4, 5, 6, 8]))

        # By convention, because various measurements don't match up about 30% of the time,
        # Take the mean number of qcs to be 'the number of measurements'.
        self.n = np.mean([self.psal_n, self.pres_n, self.temp_n])

        self.verify(arr) #  TODO Make this optional??

    def verify(self, arr):

        # Check if parameters are missing. Some floats don't seem to contain either PSAL or TEMP.
        # This is to filter those out.
        # For estimating the number of missing points, we refer to pressure (always available).
        psal_missing = False
        if "PSAL_QC" not in self.data_vars:
            psal_missing = True
            self.flags.append(MissingParameter("PSAL_QC", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if "PSAL_ADJUSTED_QC" not in self.data_vars:
            psal_missing = True
            self.flags.append(MissingParameter("PSAL_ADJUSTED_QC", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if "PSAL" not in self.data_vars:
            psal_missing = True
            self.flags.append(MissingParameter("PSAL", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if "PSAL_ADJUSTED" not in self.data_vars:
            psal_missing = True
            self.flags.append(MissingParameter("PSAL_ADJUSTED", np.sum(np.isfinite(self.pres)), len(self.pres)))

        temp_missing = False
        if "TEMP_QC" not in self.data_vars:
            temp_missing = True
            self.flags.append(MissingParameter("TEMP_QC", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if "TEMP_ADJUSTED_QC" not in self.data_vars:
            temp_missing = True
            self.flags.append(MissingParameter("TEMP_ADJUSTED_QC", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if "TEMP" not in self.data_vars:
            temp_missing = True
            self.flags.append(MissingParameter("TEMP", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if "TEMP_ADJUSTED" not in self.data_vars:
            temp_missing = True
            self.flags.append(MissingParameter("TEMP_ADJUSTED", np.sum(np.isfinite(self.pres)), len(self.pres)))

        if np.isin(-1, self.data_mode):
            self.flags(UnknownDataMode(np.argwhere(self.data_mode == -1)))

        mask_diffs: np.ndarray = (self.pres_selection_mask != (self.data_mode > 0))
        if mask_diffs.any():
            """
            The sum of "all points in the selected profile whose values are different to those in the alt-profile"
            """

            indices = np.argwhere(mask_diffs == True)
            alt_pres = get_param("PRES", "PRES_ADJUSTED", ~self.pres_selection_mask, arr).astype(np.float32)
            affected_points_n = np.sum(~element_wise_nan_equal(alt_pres[indices], self.pres[indices]))
            if affected_points_n:
                self.flags.append(InappropriateAdjustedValues("PRES", indices, affected_points_n))

        if not psal_missing:
            mask_diffs: np.ndarray = (self.psal_selection_mask != (self.data_mode > 0))
            if mask_diffs.any():
                indices = np.argwhere(mask_diffs == True)
                alt_psal = get_param("PSAL", "PSAL_ADJUSTED", ~self.psal_selection_mask, arr).astype(np.float32)
                affected_points_n = np.sum(~element_wise_nan_equal(alt_psal[indices], self.psal[indices]))
                if affected_points_n:
                    self.flags.append(InappropriateAdjustedValues("PSAL", indices, affected_points_n))

        if not temp_missing:
            mask_diffs: np.ndarray = (self.temp_selection_mask != (self.data_mode > 0))
            if mask_diffs.any():
                indices = np.argwhere(mask_diffs == True)
                alt_temp = get_param("TEMP", "TEMP_ADJUSTED", ~self.temp_selection_mask, arr).astype(np.float32)
                affected_points_n = np.sum(~element_wise_nan_equal(alt_temp[indices], self.temp[indices]))
                if affected_points_n:
                    self.flags.append(InappropriateAdjustedValues("TEMP", indices, affected_points_n))

        neg_pres = np.logical_and.reduce([np.isfinite(self.pres), (np.nan_to_num(self.pres) < 0), (self.pres_qc == 1)])
        if neg_pres.any():
            pts = np.argwhere(neg_pres)
            affected_profiles_n = len(np.unique(pts[:, 0]))
            self.flags.append(NegativePressure(pts, affected_profiles_n))

        if not psal_missing:
            _is_finite = np.isfinite(self.psal)
            _qc_not_7 = (self.psal_qc != 7)
            if (_is_finite != _qc_not_7).any():
                self.flags.append(UnauthorisedNan("PSAL", np.argwhere(_is_finite != _qc_not_7)))

        _is_finite = np.isfinite(self.pres)
        _qc_not_7 = (self.pres_qc != 7)
        if (_is_finite != _qc_not_7).any():
            self.flags.append(UnauthorisedNan("PRES", np.argwhere(_is_finite != _qc_not_7)))

        if not temp_missing:
            _is_finite = np.isfinite(self.temp)
            _qc_not_7 = (self.temp_qc != 7)
            if (_is_finite != _qc_not_7).any():
                self.flags.append(UnauthorisedNan("TEMP", np.argwhere(_is_finite != _qc_not_7)))

        _is_finite = np.isfinite(self.juld)
        _qc_not_7 = (self.juld_qc != 7)
        if (_is_finite != _qc_not_7).any():
            self.flags.append(UnauthorisedNan("JULD", np.argwhere(_is_finite != _qc_not_7)))

        _is_finite = np.isfinite(self.lon)
        _qc_not_7 = (self.position_qc != 7)
        if (_is_finite != _qc_not_7).any():
            self.flags.append(UnauthorisedNan("LON", np.argwhere(_is_finite != _qc_not_7)))

        _is_finite = np.isfinite(self.lat)
        _qc_not_7 = (self.position_qc != 7)
        if (_is_finite != _qc_not_7).any():
            self.flags.append(UnauthorisedNan("LON", np.argwhere(_is_finite != _qc_not_7)))

        if self.temp_found and self.psal_found:
            temp_psal_diffs:np.ndarray = self.psal_selection_mask != self.temp_selection_mask
            if temp_psal_diffs.any():
                self.flags.append(QCCountsMismatch(len(temp_psal_diffs), np.sum([x.any() for x in temp_psal_diffs])))
            psal_pres_diffs:np.ndarray = self.psal_selection_mask != self.pres_selection_mask
            if psal_pres_diffs.any():
                self.flags.append(QCCountsMismatch(len(psal_pres_diffs), np.sum([x.any() for x in psal_pres_diffs])))

    # def get_gsw_st0(self):
    #     lons = np.repeat(self.lon, self.psal.shape[1]).reshape(self.psal.shape)
    #     lats = np.repeat(self.lat, self.psal.shape[1]).reshape(self.psal.shape)
    #     sa = gsw.SA_from_SP(self.psal, self.pres, lons, lats)
    #     ct = gsw.CT_from_t(sa, self.temp, self.pres)
    #     return gsw.sigma0(sa, ct)
