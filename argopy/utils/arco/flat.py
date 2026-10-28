import numpy as np
import pandas as pd
import xarray as xr
from functools import cached_property
from argopy.errors import NoDataLeft
from argopy.utils.checkers import check_wmo, check_cyc, is_indexbox

# LEVEL_VARS = ["PRES", "TEMP", "PSAL"]
# PROF_VARS = ["LATITUDE", "LONGITUDE", "JULD", "CYCLE_NUMBER"]

LEVEL_VARS = ['PRES',
              'PRES_QC',
              'PRES_ADJUSTED',
              'PRES_ADJUSTED_QC',
              'PRES_ADJUSTED_ERROR',
              'TEMP',
              'TEMP_QC',
              'TEMP_ADJUSTED',
              'TEMP_ADJUSTED_QC',
              'TEMP_ADJUSTED_ERROR',
              'PSAL',
              'PSAL_QC',
              'PSAL_ADJUSTED',
              'PSAL_ADJUSTED_QC',
              'PSAL_ADJUSTED_ERROR']

PROF_VARS = ['PLATFORM_NUMBER',
             'PROJECT_NAME',
             'PI_NAME',
             'CYCLE_NUMBER',
             'DIRECTION',
             'DATA_CENTRE',
             'DC_REFERENCE',
             'DATA_STATE_INDICATOR',
             'DATA_MODE',
             'PLATFORM_TYPE',
             'FLOAT_SERIAL_NO',
             'FIRMWARE_VERSION',
             'WMO_INST_TYPE',
             'JULD',
             'JULD_QC',
             'JULD_LOCATION',
             'LATITUDE',
             'LONGITUDE',
             'POSITION_QC',
             'POSITIONING_SYSTEM',
             'PROFILE_PRES_QC',
             'PROFILE_TEMP_QC',
             'PROFILE_PSAL_QC',
             'VERTICAL_SAMPLING_SCHEME',
             'CONFIG_MISSION_NUMBER']

class FlatData_v0:
    """Argo dataset as a flat collection of points

    Examples
    --------
    ..code-block:: python

        from argopy.utils.arco import FlatData

        fd = FlatData("argoflat-expert.zarr")

        ds = fd.float(1901514)
        ds = fd.float([1901514, 1902284])
        ds = fd.profile(1901514, [2, 3])
        ds = fd.profile([1901514, 1902284], 3)
        ds = fd.profile([1901514, 1902284], [2, 3])
        ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])

        fg[100]
    """

    def __init__(self, store, broadcast=False):
        self.store = store
        self.broadcast = broadcast
        self.prf = xr.open_zarr(store, group="profiles").load()  # fit in memory, even for the full expert dataset
        self.pts = xr.open_zarr(store, group="points")  # lazy

    def _get_profiles(self, sel, broadcast=None):
        if len(sel) == 0:
            raise ValueError('No data matching your criteria')
        broadcast = self.broadcast if broadcast is None else broadcast

        sel = np.atleast_1d(sel)
        start = self.prf.row_start.values[sel]
        size = self.prf.row_size.values[sel]

        # vectorised construction of the point indices (no python loop)
        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(len(sel)), size)  # position in `sel` for each point
        level = np.arange(size.sum()) - offsets[local]  # level number within its profile
        idx = start[local] + level

        out = self.pts.isel(N_POINTS=idx)  # still lazy
        p = self.prf.isel(N_PROF=sel).drop_vars(["row_start", "row_size"])

        if broadcast:
            # every profile variable repeated along N_POINTS
            pp = p.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
            out = out.assign({k: pp[k] for k in pp.data_vars})
        else:
            # profile variables stay on N_PROF; points carry a pointer to them
            p = p.assign_coords(N_PROF=sel)
            out = out.assign(PROFILE=("N_POINTS", sel[local]))
            out = xr.merge([out, p])

        out = out.assign_coords(N_POINTS=("N_POINTS", level))

        # for v in out:
        #     if v in attrs:
        #         out[v].attrs = attrs[v]

        out = out[PROF_VARS + LEVEL_VARS]
        return out

    def float(self, WMO, broadcast=None):
        WMO = check_wmo(WMO)
        filters = []
        for wmo in WMO:
            filters.append(self.prf['PLATFORM_NUMBER'] == wmo)
        cond = np.logical_or.reduce(filters)
        sel = np.flatnonzero(cond)
        ds = self._get_profiles(sel, broadcast=broadcast)
        return ds

    def profile(self, WMO, CYC, broadcast=None):
        WMO = check_wmo(WMO)
        CYC = check_cyc(CYC)
        sel = []
        for wmo in WMO:
            for cyc in CYC:
                f = np.logical_and.reduce([self.prf['PLATFORM_NUMBER'] == wmo, self.prf['CYCLE_NUMBER'] == cyc])
                [sel.append(i) for i in np.flatnonzero(f)]
        ds = self._get_profiles(sel, broadcast=broadcast)
        return ds

    def region(self, box, broadcast=None):
        is_indexbox(box)
        filters = []
        filters.append(self.prf['LONGITUDE'] >= box[0])
        filters.append(self.prf['LONGITUDE'] < box[1])
        filters.append(self.prf['LATITUDE'] >= box[2])
        filters.append(self.prf['LATITUDE'] < box[3])
        filters.append(self.prf['JULD'] >= pd.to_datetime(box[4]))
        filters.append(self.prf['JULD'] < pd.to_datetime(box[5]))
        cond = np.logical_and.reduce(filters)
        sel = np.flatnonzero(cond)
        ds = self._get_profiles(sel, broadcast=broadcast)
        return ds

    @property
    def N_PROF(self):
        return len(self.prf['N_PROF'])

    @property
    def N_POINTS(self):
        return self.prf['row_size'].sum().item()

    @property
    def domain(self):
        """Space/time domain of the dataset

        This is different from a usual argopy ``box`` because dates are in :class:`numpy.datetime64` format.
        """
        return [
            np.nanmin(self.prf["LONGITUDE"].values),
            np.nanmax(self.prf["LONGITUDE"].values),
            np.nanmin(self.prf["LATITUDE"].values),
            np.nanmax(self.prf["LATITUDE"].values),
            None,
            None,
            np.nanmin(self.prf['JULD'].values),
            np.nanmax(self.prf['JULD'].values),
        ]

    def __repr__(self):
        summary = ["<argopy.flatdata>"]
        summary.append(f"Store: {self.store}")
        summary.append(f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})")
        summary.append(f"Broadcast: {self.broadcast}")
        domain = self.domain
        summary.append("Domain:")
        summary.append(f" - Longitude: [{domain[0]:.2f}, {domain[1]:.2f}]")
        summary.append(f" - Latitude: [{domain[2]:.2f}, {domain[3]:.2f}]")
        summary.append(f" - Time: [{str(domain[6])[0:16]}, {str(domain[7])[0:16]}]")
        return "\n".join(summary)


class FlatData:
    """Argo dataset as a flat (ragged) collection of points, backed by a Zarr store

    The store has two groups:

    - ``points``: dimension ``N_POINTS``, all valid levels of all profiles, concatenated
    - ``profiles``: dimension ``N_PROF``, profile metadata plus ``row_start`` and ``row_size``
      giving the location of each profile in ``points``.

    Returned datasets always list profiles in *store order* (sorted, no duplicates),
    which gives contiguous, efficient reads.

    Examples
    --------
    >>> fd = FlatData("argoflat-expert.zarr")
    >>> fd = FlatData("s3://bucket/argoflat.zarr", anon=True)   # extra kwargs -> storage_options
    >>> ds = fd.float(1901514)
    >>> ds = fd.float([1901514, 1902284])
    >>> ds = fd.profile(1901514, [2, 3])
    >>> ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])
    >>> ds = fd.region([-75., 0., 0., 45.])            # no time constraint
    >>> ds = fd[100]; ds = fd[10:20]; ds = fd[[1, 5, 8]]
    """

    # Number of contiguous runs above which we fall back to a single fancy ``isel``
    # instead of concatenating one slice per run.
    MAX_RUNS = 50

    def __init__(self, store, broadcast=False, **storage_options):
        self.store = store
        self.broadcast = broadcast
        self._kw = {"storage_options": storage_options} if storage_options else {}
        self.refresh()

    # ------------------------------------------------------------------ opening
    def refresh(self):
        """(Re)open the store, e.g. after it has been appended to"""
        self.prf = xr.open_zarr(self.store, group="profiles", **self._kw).load()  # index, fits in memory
        self.pts = xr.open_zarr(self.store, group="points", **self._kw)  # lazy
        for name in ("N_POINTS", "_valid_points"):
            self.__dict__.pop(name, None)  # reset cached properties
        self._check()

    def _check(self):
        n_valid = self._valid_points
        n_pts = self.pts.sizes["N_POINTS"]
        if n_pts < n_valid:
            raise RuntimeError(
                f"Corrupted store: 'profiles' references {n_valid} points but 'points' holds only {n_pts}"
            )
        if n_pts > n_valid:
            raise RuntimeError(
                f"Corrupted store: 'points' holds {n_pts - n_valid} orphan points not referenced by 'profiles' "
                "(half-finished append?)"
            )

    # -------------------------------------------------------------- variables
    # @property
    # def LEVEL_VARS(self):
    #     return list(self.pts.data_vars)
    #
    # @property
    # def PROF_VARS(self):
    #     return [v for v in self.prf.data_vars if v not in ("row_start", "row_size")]

    # ------------------------------------------------------------ core method
    @staticmethod
    def _runs(starts, ends):
        """Merge consecutive [start, end) intervals (given in ascending order) into contiguous runs"""
        breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
        run_starts = starts[np.r_[0, breaks]]
        run_ends = ends[np.r_[breaks - 1, len(ends) - 1]]
        return run_starts, run_ends

    def _get_profiles(self, sel, broadcast=None):
        broadcast = self.broadcast if broadcast is None else broadcast

        sel = np.unique(np.atleast_1d(sel).astype("int64"))  # sorted, unique
        if sel.size == 0:
            raise NoDataLeft("No data matching your criteria")
        if sel[0] < 0 or sel[-1] >= self.N_PROF:
            raise IndexError(f"Profile index out of range [0, {self.N_PROF - 1}]")

        start = self.prf["row_start"].values[sel].astype("int64")
        size = self.prf["row_size"].values[sel].astype("int64")
        end = start + size

        # Position in `sel` and level number of each selected point (vectorised)
        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(sel.size), size)
        level = np.arange(size.sum()) - offsets[local]

        # Lazy selection of the points, using slices whenever possible
        run_starts, run_ends = self._runs(start, end)
        print(len(run_starts), self.MAX_RUNS)
        if len(run_starts) <= self.MAX_RUNS:
            parts = [self.pts.isel(N_POINTS=slice(int(s), int(e))) for s, e in zip(run_starts, run_ends)]
            out = parts[0] if len(parts) == 1 else xr.concat(parts, "N_POINTS")
        else:
            out = self.pts.isel(N_POINTS=start[local] + level)

        out = out.assign_coords(LEVEL=("N_POINTS", level))

        # Profile variables
        p = self.prf[PROF_VARS].isel(N_PROF=sel)
        if broadcast:
            pp = p.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
            out = out.assign({k: pp[k].variable for k in pp.data_vars})
        else:
            p = p.assign_coords(N_PROF=("N_PROF", sel))
            out = out.assign(PROFILE=("N_POINTS", sel[local]))
            out = xr.merge([out, p], combine_attrs="override")

        out.attrs.update(self.pts.attrs)

        out = out[sorted(PROF_VARS) + LEVEL_VARS]

        return out

    # ------------------------------------------------------------ public API
    def float(self, WMO, broadcast=None):
        """All profiles of one or more floats"""
        WMO = check_wmo(WMO)
        cond = np.isin(self.prf["PLATFORM_NUMBER"].values, WMO)
        return self._get_profiles(np.flatnonzero(cond), broadcast=broadcast)

    def profile(self, WMO, CYC, broadcast=None):
        """Profiles of one or more floats, for one or more cycle numbers"""
        WMO, CYC = check_wmo(WMO), check_cyc(CYC)
        cond = np.isin(self.prf["PLATFORM_NUMBER"].values, WMO) & np.isin(self.prf["CYCLE_NUMBER"].values, CYC)
        return self._get_profiles(np.flatnonzero(cond), broadcast=broadcast)

    def region(self, box, broadcast=None):
        """Profiles in a space/time box

        Parameters
        ----------
        box : list
            ``[lon_min, lon_max, lat_min, lat_max]`` or
            ``[lon_min, lon_max, lat_min, lat_max, date_min, date_max]``.
            Intervals are half-open: ``min <= value < max``.
            If ``lon_min > lon_max`` the box is assumed to cross the antimeridian.
        """
        is_indexbox(box)
        lon, lat = self.prf["LONGITUDE"].values, self.prf["LATITUDE"].values

        if box[0] <= box[1]:
            cond = (lon >= box[0]) & (lon < box[1])
        else:  # crosses the antimeridian
            cond = (lon >= box[0]) | (lon < box[1])
        cond &= (lat >= box[2]) & (lat < box[3])

        if len(box) == 6:
            juld = self.prf["JULD"].values
            cond &= (juld >= pd.to_datetime(box[4]).to_datetime64()) & (juld < pd.to_datetime(box[5]).to_datetime64())

        return self._get_profiles(np.flatnonzero(cond), broadcast=broadcast)

    # ------------------------------------------------------- machine learning
    def split(self, fractions=(0.8, 0.1, 0.1), by="PLATFORM_NUMBER", seed=0):
        """Reproducible random split of the profiles, e.g. into train/validation/test

        Parameters
        ----------
        fractions : sequence of float
            Relative size of each subset (normalised to sum to 1).
        by : str or None
            Profile variable whose unique values are kept together in the same subset.
            The default, ``"PLATFORM_NUMBER"``, puts all profiles of a float in the same subset,
            which avoids leakage between train and test (successive profiles of a float are
            strongly correlated). Use ``None`` to split profile by profile.
        seed : int

        Returns
        -------
        list of :class:`numpy.ndarray`
            One array of profile indices per fraction (sorted, in store order).

        Examples
        --------
        >>> train, val, test = fd.split([0.8, 0.1, 0.1])
        >>> train_loader = fd.sampler(sel=train, batch_size=128)
        """
        f = np.asarray(fractions, dtype="float64")
        if (f < 0).any() or f.sum() == 0:
            raise ValueError("fractions must be non-negative and not all zero")
        if by is None:
            n_units, inverse = self.N_PROF, np.arange(self.N_PROF)
        else:
            _, inverse = np.unique(self.prf[by].values, return_inverse=True)
            n_units = int(inverse.max()) + 1

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_units)
        edges = np.round(np.cumsum(f) / f.sum() * n_units).astype(int)
        edges[-1] = n_units
        unit_subset = np.empty(n_units, dtype="int64")
        lo = 0
        for i, hi in enumerate(edges):
            unit_subset[perm[lo:hi]] = i
            lo = hi
        subset = unit_subset[inverse]
        return [np.flatnonzero(subset == i) for i in range(len(f))]

    def sampler(self, sel=None, **kwargs):
        """Random batch sampler for training machine learning models

        See :class:`FlatSampler` for the parameters.

        Examples
        --------
        >>> train, val, test = fd.split()
        >>> loader = fd.sampler(sel=train, variables=["PRES", "TEMP", "PSAL"],
        ...                     batch_size=128, max_levels=500, seed=42)
        >>> for epoch in range(10):
        ...     loader.set_epoch(epoch)
        ...     for batch in loader:
        ...         x = batch["TEMP"]      # (batch, max_levels), NaN where no data
        ...         m = batch["mask"]      # (batch, max_levels), True on valid levels
        """
        return FlatSampler(self, sel=sel, **kwargs)

    # ------------------------------------------------------------ properties
    @property
    def profiles(self):
        """Copy of the profile index, to build custom selections (see :meth:`__getitem__`)

        Examples
        --------
        >>> p = fd.profiles
        >>> idx = np.flatnonzero((p.DATA_MODE == b"D").values)
        >>> ds = fd[idx]
        """
        return self.prf.copy()

    @property
    def N_PROF(self):
        return self.prf.sizes["N_PROF"]

    @cached_property
    def _valid_points(self):
        return int(self.prf["row_size"].sum())

    @cached_property
    def N_POINTS(self):
        return self._valid_points

    @property
    def domain(self):
        """Space/time domain of the dataset, as a dictionary of (min, max) tuples"""
        def mm(name):
            v = self.prf[name].values
            if v.size == 0 or (np.issubdtype(v.dtype, np.floating) and np.isnan(v).all()):
                return (None, None)
            if np.issubdtype(v.dtype, np.datetime64):
                v = v[~np.isnat(v)]
                return (v.min(), v.max()) if v.size else (None, None)
            return (np.nanmin(v), np.nanmax(v))

        e = {"lon": mm("LONGITUDE"), "lat": mm("LATITUDE"), "time": mm("JULD")}
        # return e
        return [e['lon'][0], e['lon'][1], e['lat'][0], e['lat'][1], None, None, e['time'][0], e['time'][1]]

    # --------------------------------------------------------------- dunders
    def __len__(self):
        return self.N_PROF

    def __repr__(self):
        # d = self.domain
        # fmt = lambda a, b, f="{:.2f}": "N/A" if a is None else f"[{f.format(a)}, {f.format(b)}]"
        # t = "N/A" if d["time"][0] is None else f"[{str(d['time'][0])[:16]}, {str(d['time'][1])[:16]}]"
        # return "\n".join([
        #     "<argopy.flatdata>",
        #     f"Store: {self.store}",
        #     f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})",
        #     f"Broadcast: {self.broadcast}",
        #     "Domain:",
        #     f" - Longitude: {fmt(*d['lon'])}",
        #     f" - Latitude: {fmt(*d['lat'])}",
        #     f" - Time: {t}",
        # ])

        summary = ["<argopy.flatdata>"]
        summary.append(f"Store: {self.store}")
        summary.append(f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})")
        summary.append(f"Broadcast: {self.broadcast}")
        domain = self.domain
        summary.append("Domain:")
        summary.append(f" - Longitude: [{domain[0]:.2f}, {domain[1]:.2f}]")
        summary.append(f" - Latitude: [{domain[2]:.2f}, {domain[3]:.2f}]")
        summary.append(f" - Time: [{str(domain[6])[0:16]}, {str(domain[7])[0:16]}]")
        return "\n".join(summary)

    def __getitem__(self, ii):
        n = np.arange(self.N_PROF)
        if isinstance(ii, (int, np.integer)):
            sel = n[[ii]]  # handles negative indices, raises IndexError if out of range
        elif isinstance(ii, slice):
            sel = n[ii]
        elif isinstance(ii, (list, tuple, np.ndarray)):
            ii = np.asarray(ii)
            sel = n[ii]  # works for integer arrays and boolean masks
        else:
            raise NotImplementedError(f"This indexing method ({type(ii)}) is not implemented")
        return self._get_profiles(sel)


class FlatSampler:
    """Random, memory-bounded batch sampler over a :class:`FlatData` store

    Fully random access to a chunked store is slow (one chunk read per profile).
    Instead, shuffling is done in two levels:

    1. the selected profiles are cut into *blocks* of ``block_size`` consecutive profiles
       (contiguous in the store, so each block is read with a few slice reads), and the
       block order is shuffled at each epoch;
    2. ``mix_blocks`` blocks are loaded together into a buffer, whose profiles are shuffled
       before being cut into batches.

    A larger ``mix_blocks`` gives a better shuffle (profiles from more distinct floats and
    regions in each batch) at the cost of memory:
    ``mix_blocks * block_size * max_levels * n_variables * 4`` bytes (float32).

    Parameters
    ----------
    fd : FlatData
    variables : list of str, optional
        Level variables to return, default ``["PRES", "TEMP", "PSAL"]``.
    profile_vars : list of str, optional
        Per-profile variables to return, default ``["LATITUDE", "LONGITUDE", "JULD"]``.
        Datetimes are converted to float days since 1970-01-01.
    sel : array_like of int, optional
        Profile indices to sample from (e.g. from :meth:`FlatData.split`). Default: all.
    batch_size : int
    max_levels : int, optional
        Profiles are padded (or truncated) to this number of levels.
        Default is the largest profile in ``sel``, which can be wasteful if there are outliers.
    shuffle : bool
        If False, profiles are returned in store order (useful for validation/inference).
    seed : int
        Together with the epoch number (see :meth:`set_epoch`), it fully determines the order.
    block_size, mix_blocks : int
        See above.
    drop_last : bool
        Drop the last incomplete batch of each shard.
    rank, world_size : int
        For distributed training: this process is ``rank`` out of ``world_size``.
        Torch ``DataLoader`` workers are detected automatically and get their own share.

    Yields
    ------
    dict
        - one ``(batch, max_levels)`` array per level variable (padded with NaN, or 0 for non-float);
        - ``mask``: ``(batch, max_levels)`` boolean, True where a level exists;
        - ``row_size``: number of levels in each profile (before truncation);
        - one ``(batch,)`` array per profile variable;
        - ``PROFILE``: ``(batch,)`` index of each profile in the store, to trace samples back
          (``fd[batch["PROFILE"]]``).
    """

    def __init__(self, fd, variables=None, profile_vars=("LATITUDE", "LONGITUDE", "JULD"), sel=None,
                 batch_size=64, max_levels=None, shuffle=True, seed=0, block_size=256, mix_blocks=8,
                 drop_last=False, rank=0, world_size=1):
        self.fd = fd
        self.variables = list(variables) if variables is not None else [v for v in ("PRES", "TEMP", "PSAL") if v in fd.pts]
        self.profile_vars = list(profile_vars)
        missing = [v for v in self.variables if v not in fd.pts] + [v for v in self.profile_vars if v not in fd.prf]
        if missing:
            raise ValueError(f"Variables not found in the store: {missing}")

        self.sel = np.arange(fd.N_PROF) if sel is None else np.unique(np.asarray(sel, dtype="int64"))
        if self.sel.size == 0:
            raise ValueError("Empty selection")
        if self.sel[0] < 0 or self.sel[-1] >= fd.N_PROF:
            raise IndexError(f"Profile index out of range [0, {fd.N_PROF - 1}]")

        self.max_levels = int(max_levels or fd.prf["row_size"].values[self.sel].max())
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.block_size = int(block_size)
        self.mix_blocks = int(mix_blocks)
        self.drop_last = drop_last
        self.rank, self.world_size = rank, world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        """Change the shuffle at each epoch (same epoch and seed give the same order)"""
        self.epoch = int(epoch)

    def __len__(self):
        """Number of batches over the whole selection (all shards), approximately"""
        n = self.sel.size
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    # -------------------------------------------------------------- internals
    def _shard(self):
        """(shard_id, n_shards) combining distributed rank and torch DataLoader workers"""
        worker_id, n_workers = 0, 1
        try:
            from torch.utils.data import get_worker_info
            info = get_worker_info()
            if info is not None:
                worker_id, n_workers = info.id, info.num_workers
        except ImportError:
            pass
        return self.rank * n_workers + worker_id, self.world_size * n_workers

    def _read_block(self, sel):
        """Read profiles `sel` (sorted) into padded numpy arrays"""
        fd, L = self.fd, self.max_levels
        start = fd.prf["row_start"].values[sel].astype("int64")
        size = fd.prf["row_size"].values[sel].astype("int64")
        run_starts, run_ends = fd._runs(start, start + size)

        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(sel.size), size)
        level = np.arange(size.sum()) - offsets[local]
        keep = level < L

        out = {}
        for v in self.variables:
            flat = np.concatenate([
                fd.pts[v].isel(N_POINTS=slice(int(s), int(e))).values for s, e in zip(run_starts, run_ends)
            ])
            fill = np.nan if np.issubdtype(flat.dtype, np.floating) else 0
            padded = np.full((sel.size, L), fill, dtype=flat.dtype)
            padded[local[keep], level[keep]] = flat[keep]
            out[v] = padded

        out["mask"] = np.arange(L)[None, :] < size[:, None]
        out["row_size"] = size
        for v in self.profile_vars:
            x = fd.prf[v].values[sel]
            if np.issubdtype(x.dtype, np.datetime64):
                x = (x - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            out[v] = x
        out["PROFILE"] = sel
        return out

    @staticmethod
    def _cat(a, b):
        return b if a is None else {k: np.concatenate([a[k], b[k]]) for k in a}

    @staticmethod
    def _take(d, idx):
        return {k: v[idx] for k, v in d.items()}

    def __iter__(self):
        bs = self.batch_size
        shard, n_shards = self._shard()

        # same block order in all shards, then each shard keeps its own share
        rng = np.random.default_rng([self.seed, self.epoch])
        blocks = [self.sel[i:i + self.block_size] for i in range(0, self.sel.size, self.block_size)]
        order = rng.permutation(len(blocks)) if self.shuffle else np.arange(len(blocks))
        order = order[shard::n_shards]
        rng = np.random.default_rng([self.seed, self.epoch, shard])  # independent row shuffle per shard

        carry = None
        for g in range(0, len(order), self.mix_blocks):
            buf = carry
            for b in order[g:g + self.mix_blocks]:
                buf = self._cat(buf, self._read_block(blocks[b]))
            n = len(buf["PROFILE"])
            if self.shuffle:
                buf = self._take(buf, rng.permutation(n))
            n_full = n // bs
            for i in range(n_full):
                yield self._take(buf, slice(i * bs, (i + 1) * bs))
            carry = self._take(buf, slice(n_full * bs, n)) if n_full * bs < n else None

        if carry is not None and not self.drop_last:
            yield carry

    # ------------------------------------------------------------------ torch
    def torch_dataset(self):
        """Wrap the sampler into a :class:`torch.utils.data.IterableDataset`

        Numeric arrays are converted to tensors. Use ``batch_size=None`` in the ``DataLoader``
        since batching is done here.

        Examples
        --------
        >>> from torch.utils.data import DataLoader
        >>> ds = fd.sampler(sel=train, batch_size=128).torch_dataset()
        >>> loader = DataLoader(ds, batch_size=None, num_workers=4)

        Notes
        -----
        Workers get disjoint shares of the blocks. Call ``sampler.set_epoch(epoch)`` before
        iterating the loader at each epoch (this works when workers are re-created at each epoch,
        i.e. ``persistent_workers=False``, the default).
        """
        import torch

        sampler = self

        class _Dataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for batch in sampler:
                    yield {k: torch.as_tensor(v) if v.dtype.kind in "biuf" else v for k, v in batch.items()}

            def __len__(self):
                return len(sampler)

        return _Dataset()