import glob as _glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from argopy.utils.checkers import check_wmo, check_cyc


class FlatData:
    """Argo dataset as a flat (ragged) collection of points, backed by one or several Zarr stores

    Each store has two groups:

    - ``points``: dimension ``N_POINTS``, all valid levels of all profiles, concatenated
    - ``profiles``: dimension ``N_PROF``, profile metadata plus ``row_start`` and ``row_size``
      giving the location of each profile in the ``points`` of *the same store*.

    With several stores, the profile indexes are concatenated in memory into a single one, so
    that all the selection methods work on the whole collection. The ``points`` stay lazy and
    only the stores holding the selected profiles are read. Profiles are numbered globally,
    following the order of the stores: profile ``i`` of store ``k`` has global index
    ``offsets[k] + i``. This numbering changes if the list of stores changes.

    Returned datasets list profiles in *store order* (sorted, no duplicates), which gives
    contiguous, efficient reads.

    Parameters
    ----------
    stores : str or list of str
        One store, a list of stores, or a glob pattern (``"stores/argo_*.zarr"``,
        ``"s3://bucket/argo_*.zarr"``). Stores from a glob are sorted by name.
    broadcast : bool
        Default for the methods returning datasets, see :meth:`float`.
    **storage_options
        Passed to fsspec for remote stores, e.g. ``anon=True``.

    Examples
    --------
    >>> fd = FlatData("argoflat-expert.zarr")
    >>> fd = FlatData("stores/argo_*.zarr")                 # a collection of stores
    >>> fd = FlatData(["a.zarr", "b.zarr"])
    >>> fd = FlatData("s3://bucket/argo_*.zarr", anon=True)
    >>> ds = fd.float(1901514)
    >>> ds = fd.float([1901514, 1902284])
    >>> ds = fd.profile(1901514, [2, 3])
    >>> ds = fd.region([-75., 0., 0., 45., '20250101', '20250201'])
    >>> ds = fd.region([-75., 0., 0., 45.])            # no time constraint
    >>> ds = fd[100]; ds = fd[10:20]; ds = fd[[1, 5, 8]]
    """

    # Number of contiguous runs above which we fall back to a single fancy ``isel`` per store
    # instead of concatenating one slice per run.
    MAX_RUNS = 50

    def __init__(self, stores, broadcast=False, **storage_options):
        self._spec = stores
        self.broadcast = broadcast
        self._kw = {"storage_options": storage_options} if storage_options else {}
        self.refresh()

    # ------------------------------------------------------------------ opening
    def _resolve(self):
        spec = self._spec
        if isinstance(spec, (str, os.PathLike)):
            spec = str(spec)
            if any(c in spec for c in "*?["):
                if "://" in spec:
                    import fsspec
                    fs, p = fsspec.core.url_to_fs(spec, **self._kw.get("storage_options", {}))
                    proto = spec.split("://")[0]
                    stores = sorted(f"{proto}://{x}" for x in fs.glob(p))
                else:
                    stores = sorted(_glob.glob(spec))
            else:
                stores = [spec]
        else:
            stores = [str(s) for s in spec]
        if not stores:
            raise FileNotFoundError(f"No store found for {self._spec!r}")
        return stores

    def refresh(self):
        """(Re)open the stores, e.g. after one has been appended to, or new stores were added"""
        self.stores = self._resolve()
        prfs, self.pts_stores = [], []
        for path in self.stores:
            prf = xr.open_zarr(path, group="profiles", **self._kw).load()  # index, fits in memory
            pts = xr.open_zarr(path, group="points", **self._kw)  # lazy

            n_valid, n_pts = int(prf["row_size"].sum()), pts.sizes["N_POINTS"]
            if n_pts < n_valid:
                raise RuntimeError(f"Corrupted store {path}: 'profiles' references {n_valid} points "
                                   f"but 'points' holds only {n_pts}")
            if n_pts > n_valid:
                warnings.warn(f"{path}: 'points' holds {n_pts - n_valid} orphan points not referenced "
                              "by 'profiles' (half-finished append?). They are ignored.")
            prfs.append(prf)
            self.pts_stores.append(pts)

        ref_p, ref_l = set(prfs[0].data_vars), set(self.pts_stores[0].data_vars)
        for path, prf, pts in zip(self.stores, prfs, self.pts_stores):
            if set(prf.data_vars) != ref_p or set(pts.data_vars) != ref_l:
                raise ValueError(f"Store {path} has not the same variables as {self.stores[0]}")

        counts = np.array([p.sizes["N_PROF"] for p in prfs])
        self.offsets = np.r_[0, np.cumsum(counts)]  # global index of the first profile of each store
        self.prf = prfs[0] if len(prfs) == 1 else xr.concat(prfs, dim="N_PROF", combine_attrs="override")
        self.prf["STORE"] = ("N_PROF", np.repeat(np.arange(len(prfs)), counts).astype("int32"),
                             {"long_name": "index of the store holding the profile"})
        self.N_POINTS = int(self.prf["row_size"].sum())

    # -------------------------------------------------------------- variables
    @property
    def pts(self):
        """The lazy ``points`` dataset (single store only, see ``pts_stores`` otherwise)"""
        if len(self.pts_stores) == 1:
            return self.pts_stores[0]
        raise AttributeError("A collection of stores has no single 'points' dataset, use `pts_stores[i]`")

    @property
    def LEVEL_VARS(self):
        return list(self.pts_stores[0].data_vars)

    @property
    def PROF_VARS(self):
        return [v for v in self.prf.data_vars if v not in ("row_start", "row_size", "STORE")]

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
            raise ValueError("No data matching your criteria")
        if sel[0] < 0 or sel[-1] >= self.N_PROF:
            raise IndexError(f"Profile index out of range [0, {self.N_PROF - 1}]")

        start = self.prf["row_start"].values[sel].astype("int64")
        size = self.prf["row_size"].values[sel].astype("int64")
        store = self.prf["STORE"].values[sel]

        # Position in `sel` and level number of each selected point (vectorised)
        offsets = np.cumsum(size) - size
        local = np.repeat(np.arange(sel.size), size)
        level = np.arange(size.sum()) - offsets[local]

        # Lazy selection of the points: sel is sorted, so stores are visited in order and the
        # points come out in the same order as the profiles.
        runs = []  # (store, run_starts, run_ends)
        for k in np.unique(store):
            m = store == k
            rs, re = self._runs(start[m], start[m] + size[m])
            runs.append((k, rs, re))

        if sum(len(rs) for _, rs, _ in runs) <= self.MAX_RUNS:
            parts = [self.pts_stores[k].isel(N_POINTS=slice(int(a), int(b)))
                     for k, rs, re in runs for a, b in zip(rs, re)]
        else:
            point_store = store[local]
            parts = []
            for k, _, _ in runs:
                pm = point_store == k
                parts.append(self.pts_stores[k].isel(N_POINTS=start[local][pm] + level[pm]))
        out = parts[0] if len(parts) == 1 else xr.concat(parts, "N_POINTS", combine_attrs="override")

        out = out.assign_coords(LEVEL=("N_POINTS", level))

        # Profile variables
        p = self.prf[self.PROF_VARS].isel(N_PROF=sel)
        if broadcast:
            pp = p.isel(N_PROF=local).rename({"N_PROF": "N_POINTS"})
            out = out.assign({k: pp[k].variable for k in pp.data_vars})
        else:
            p = p.assign_coords(N_PROF=("N_PROF", sel))
            out = out.assign(PROFILE=("N_POINTS", sel[local]))
            out = xr.merge([out, p], combine_attrs="override")

        out.attrs.update(self.pts_stores[0].attrs)
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
        if len(box) not in (4, 6):
            raise ValueError("box must be [lon_min, lon_max, lat_min, lat_max(, date_min, date_max)]")
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
        ...                     batch_size=128, max_pres=2000, seed=42)
        >>> for epoch in range(10):
        ...     loader.set_epoch(epoch)
        ...     for batch in loader:
        ...         x = batch["TEMP"]      # (batch, n_levels), NaN where no data
        ...         m = batch["mask"]      # (batch, n_levels), True on valid levels
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

    @property
    def n_stores(self):
        return len(self.stores)

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

        return {"lon": mm("LONGITUDE"), "lat": mm("LATITUDE"), "time": mm("JULD")}

    # --------------------------------------------------------------- dunders
    def __len__(self):
        return self.N_PROF

    def store_of(self, sel):
        """Index of the store holding each of the given profiles (global indices)"""
        return self.prf["STORE"].values[np.asarray(sel)]

    def __repr__(self):
        d = self.domain
        fmt = lambda a, b, f="{:.2f}": "N/A" if a is None else f"[{f.format(a)}, {f.format(b)}]"
        t = "N/A" if d["time"][0] is None else f"[{str(d['time'][0])[:16]}, {str(d['time'][1])[:16]}]"
        return "\n".join([
            "<argopy.flatdata>",
            f"Stores: {self.n_stores}" + (f" ({self.stores[0]})" if self.n_stores == 1 else f" ({self.stores[0]} ... {self.stores[-1]})"),
            f"Dimensions: (N_PROF: {self.N_PROF:_}, N_POINTS: {self.N_POINTS:_})",
            f"Broadcast: {self.broadcast}",
            "Domain:",
            f" - Longitude: {fmt(*d['lon'])}",
            f" - Latitude: {fmt(*d['lat'])}",
            f" - Time: {t}",
        ])

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


def _fill_value(dtype):
    kind = np.dtype(dtype).kind
    return np.nan if kind == "f" else ("" if kind in "SU" else 0)


def _pad_cols(x, L):
    if x.shape[1] == L:
        return x
    out = np.full((x.shape[0], L), _fill_value(x.dtype), dtype=x.dtype)
    out[:, : x.shape[1]] = x
    return out


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
    ``mix_blocks * block_size * n_levels * n_variables * 4`` bytes (float32), with ``n_levels`` the (padded) length of the longest profile.

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
    max_pres, min_pres : float, optional
        Keep only the levels with ``min_pres <= pressure <= max_pres`` (dbar), e.g. ``max_pres=2000``.
        Pressure is read from ``pres_var``. This is the recommended way to limit the vertical extent,
        since it has a physical meaning, unlike a number of levels. Profiles not reaching ``max_pres``
        simply have fewer valid levels (see ``mask``).
    pres_var : str
        Variable used for the pressure criterion, default ``"PRES"`` (could be ``"PRES_ADJUSTED"``).
    max_levels : int, optional
        Safety cap on the number of levels kept per profile (applied after the pressure criterion);
        extra levels are dropped. Default: no cap.
    fixed_length : bool
        By default each batch is padded to the length of its longest profile, so the second dimension
        varies from batch to batch. With ``fixed_length=True`` all batches are padded to exactly
        ``max_levels`` (required), which is needed by models expecting a fixed input size.
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
        - one ``(batch, n_levels)`` array per level variable (padded with NaN, or 0 for non-float);
        - ``mask``: ``(batch, n_levels)`` boolean, True where a level exists;
        - ``row_size``: number of levels kept in each profile (after the pressure criterion);
        - one ``(batch,)`` array per profile variable;
        - ``PROFILE``: ``(batch,)`` index of each profile in the store, to trace samples back
          (``fd[batch["PROFILE"]]``).
    """

    def __init__(self, fd, variables=None, profile_vars=("LATITUDE", "LONGITUDE", "JULD"), sel=None,
                 batch_size=64, max_pres=None, min_pres=None, pres_var="PRES", max_levels=None,
                 fixed_length=False, shuffle=True, seed=0, block_size=256, mix_blocks=8,
                 drop_last=False, rank=0, world_size=1):
        self.fd = fd
        self.variables = list(variables) if variables is not None else [v for v in ("PRES", "TEMP", "PSAL") if v in fd.pts_stores[0]]
        self.profile_vars = list(profile_vars)
        missing = [v for v in self.variables if v not in fd.pts_stores[0]] + [v for v in self.profile_vars if v not in fd.prf]
        if missing:
            raise ValueError(f"Variables not found in the store: {missing}")

        self.sel = np.arange(fd.N_PROF) if sel is None else np.unique(np.asarray(sel, dtype="int64"))
        if self.sel.size == 0:
            raise ValueError("Empty selection")
        if self.sel[0] < 0 or self.sel[-1] >= fd.N_PROF:
            raise IndexError(f"Profile index out of range [0, {fd.N_PROF - 1}]")

        if (max_pres is not None or min_pres is not None) and pres_var not in fd.pts_stores[0]:
            raise ValueError(f"Pressure variable {pres_var!r} not found in the store")
        if fixed_length and not max_levels:
            raise ValueError("fixed_length=True requires max_levels")
        self.max_pres, self.min_pres, self.pres_var = max_pres, min_pres, pres_var
        self.max_levels = int(max_levels) if max_levels else None
        self.fixed_length = fixed_length
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.block_size = int(block_size)
        self.mix_blocks = int(mix_blocks)
        self.drop_last = drop_last
        self.rank, self.world_size = rank, world_size
        self.epoch = 0

        # Blocks of consecutive profiles, never spanning two stores (one block = a few slice reads
        # in a single store)
        stores = fd.prf["STORE"].values[self.sel]
        groups = np.split(self.sel, np.flatnonzero(np.diff(stores)) + 1)
        self._blocks = [g[i:i + self.block_size] for g in groups for i in range(0, len(g), self.block_size)]

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

    def _read_flat(self, k, name, run_starts, run_ends):
        da = self.fd.pts_stores[k][name]
        return np.concatenate([da.isel(N_POINTS=slice(int(s), int(e))).values for s, e in zip(run_starts, run_ends)])

    def _read_block(self, sel):
        """Read profiles `sel` (sorted) into padded numpy arrays"""
        fd = self.fd
        start = fd.prf["row_start"].values[sel].astype("int64")
        size = fd.prf["row_size"].values[sel].astype("int64")
        run_starts, run_ends = fd._runs(start, start + size)
        k = int(fd.prf["STORE"].values[sel[0]])  # blocks never span several stores
        n = sel.size
        local = np.repeat(np.arange(n), size)

        # 1. levels to keep, from the pressure criterion
        keep = np.ones(local.size, dtype=bool)
        if self.max_pres is not None or self.min_pres is not None:
            pres = self._read_flat(k, self.pres_var, run_starts, run_ends)
            if self.max_pres is not None:
                keep &= pres <= self.max_pres
            if self.min_pres is not None:
                keep &= pres >= self.min_pres

        # 2. new level numbering, compacted to the kept levels of each profile
        local_k = local[keep]
        new_size = np.bincount(local_k, minlength=n).astype("int64")
        new_off = np.cumsum(new_size) - new_size
        level_k = np.arange(local_k.size) - new_off[local_k]

        # 3. width of the block, with optional cap
        if self.fixed_length:
            L = self.max_levels
        else:
            L = int(new_size.max()) if local_k.size else 1
            if self.max_levels:
                L = min(L, self.max_levels)
        L = max(L, 1)
        keep_l = level_k < L
        rows, cols = local_k[keep_l], level_k[keep_l]

        out = {}
        for v in self.variables:
            flat = self._read_flat(k, v, run_starts, run_ends)[keep][keep_l]
            padded = np.full((n, L), _fill_value(flat.dtype), dtype=flat.dtype)
            padded[rows, cols] = flat
            out[v] = padded

        out["mask"] = np.arange(L)[None, :] < np.minimum(new_size, L)[:, None]
        out["row_size"] = new_size
        for v in self.profile_vars:
            x = fd.prf[v].values[sel]
            if np.issubdtype(x.dtype, np.datetime64):
                x = (x - np.datetime64("1970-01-01")) / np.timedelta64(1, "D")
            out[v] = x
        out["PROFILE"] = sel
        return out

    @staticmethod
    def _cat(a, b):
        """Concatenate two blocks along profiles, padding the narrower one"""
        if a is None:
            return b
        out = {}
        for k in a:
            x, y = a[k], b[k]
            if x.ndim == 2 and x.shape[1] != y.shape[1]:
                L = max(x.shape[1], y.shape[1])
                x, y = _pad_cols(x, L), _pad_cols(y, L)
            out[k] = np.concatenate([x, y])
        return out

    @staticmethod
    def _take(d, idx):
        return {k: v[idx] for k, v in d.items()}

    def _finalize(self, batch):
        """Trim the padding to the longest profile of the batch (unless fixed_length)"""
        if self.fixed_length:
            return batch
        w = max(int(batch["mask"].sum(axis=1).max()), 1)
        return {k: (v[:, :w] if v.ndim == 2 else v) for k, v in batch.items()}

    def __iter__(self):
        bs = self.batch_size
        shard, n_shards = self._shard()

        # same block order in all shards, then each shard keeps its own share
        rng = np.random.default_rng([self.seed, self.epoch])
        blocks = self._blocks
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
                yield self._finalize(self._take(buf, slice(i * bs, (i + 1) * bs)))
            carry = self._take(buf, slice(n_full * bs, n)) if n_full * bs < n else None

        if carry is not None and not self.drop_last:
            yield self._finalize(carry)

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