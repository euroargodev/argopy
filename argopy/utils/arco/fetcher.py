import xarray as xr

# import gsw
import pandas as pd
import numpy as np
import datetime as dt
from typing import Literal

from argopy.options import OPTIONS, VALIDATE
from argopy.errors import OptionValueError
from argopy.stores.filesystems import distributed

from argopy.stores.implementations.gdac import gdacfs
from argopy.utils.format import argo_split_path
from argopy.utils.arco.argo_multiprof import ArgoMultiProf


class MassFetcher(object):
    """Fetch Argo data massively

    This class implements a direct approach to fetch Argo data massively, i.e. to fetch a very large selection of measurements, as fast as possible. In other word, this class bypasses all Argopy features to hard code all data processing steps for user modes, making data fetching effective for large selections.

    Warnings
    --------
    This class is highly experimental and API can change without further notice

    Notes
    -----
    This class should work as long as data fit in memory, so check out your RAM levels.

    This class use a :class:`distributed.client.Client` to scale Argo data fetching from a large collection of floats.

    Examples
    --------
    ..code-block:: python
        :caption: Fetch to load global Argo dataset for a given month

        from dask.distributed import Client
        from argopy import ArgoIndex
        from argopy.utils.arco import MassFetcher

        # Create the Dask cluster to work with:
        client = Client(processes=True)
        print(client)

        # Select October 2025 global data:
        # This is about 8_184_913 points (12_287 prof,  3578 floats)
        idx = ArgoIndex(index_file='core')
        idx.query.box([-180, 180, -90, 90, '20251001', '20251101'])

        # Fetch and load data in memory:
        # (~6 mins on laptop with 4 cores and 32Gb of RAM)
        dsp = MassFetcher(idx).to_xarray()

    ..code-block:: python
        :caption: Fetch to save global Argo dataset for a given month

        from dask.distributed import Client
        from argopy import ArgoIndex
        from argopy.utils.arco import MassFetcher

        # Create the Dask cluster to work with:
        client = Client(processes=True)
        print(client)

        # Select October 2025 global data:
        # This is about 8_184_913 points (12_287 prof,  3578 floats)
        idx = ArgoIndex(index_file='core')
        idx.query.box([-180, 180, -90, 90, '20251001', '20251101'])

        # Fetch and save data to a zarr archive:
        # (~6 mins on laptop with 4 cores and 32Gb of RAM)
        zarr_archive = f"zarr/{idx.search_type['BOX'][-2]}_ARGO_STANDARD_POINTS.zarr"
        MassFetcher(idx).to_zarr(zarr_archive)

    ..code-block:: python
        :caption: Work with interpolated data

        from dask.distributed import Client
        import numpy as np
        from argopy import ArgoIndex
        from argopy.utils.arco import MassFetcher

        # Create the Dask cluster to work with:
        client = Client(processes=True)
        print(client)

        # Select October 2025 global data:
        # This is about 8_184_913 points (12_287 prof,  3578 floats)
        idx = ArgoIndex(index_file='core')
        idx.query.box([-180, 180, -90, 90, '20251001', '20251101'])

        # Define standard pressure levels:
        sdl = np.arange(0, 2005., 5)

        # Fetch, interpolate and load data in memory:
        dsp = MassFetcher(idx, sdl=sdl).to_xarray()

        # Fetch, interpolate and save data to zarr:
        zarr_archive = f"zarr/{idx.search_type['BOX'][-2]}_ARGO_STANDARD_POINTS.zarr"
        dsp = MassFetcher(idx, sdl=sdl).to_zarr(zarr_archive)

    ..code-block:: python
        :caption: Load time series of global Argo dataset as points

        # Considering the above example where monthly data have been saved in several zarr archives,
        # one can load back everything like this:

        import xarray as xr

        bigds = xr.open_mfdataset(["./zarr/20250901_ARGO_STANDARD_POINTS.zarr",
                                   "./zarr/20251001_ARGO_STANDARD_POINTS.zarr",
                                   "./zarr/20251101_ARGO_STANDARD_POINTS.zarr"],
                                   combine='nested', concat_dim='N_POINTS')
        bigds

    ..code-block:: python
        :caption: Load time series of global Argo dataset as interpolated profiles

        # Considering the above example where monthly data have been saved in several zarr archives,
        # one can load back everything like this:

        import xarray as xr

        bigds = xr.open_mfdataset(["./zarr/20250901_ARGO_STANDARD_SDLPROF.zarr",
                                   "./zarr/20251001_ARGO_STANDARD_SDLPROF.zarr",
                                   "./zarr/20251101_ARGO_STANDARD_SDLPROF.zarr"],
                                   combine='nested', concat_dim='N_PROF')
        bigds


    """

    def __init__(self, idx: "ArgoIndex", sdl=None, mode : Literal['standard', 'research'] | None = None):
        """
        Parameters
        ----------
        idx: :class:`ArgoIndex`
            An instance of :class:`ArgoIndex` for which a 'box' query has been performed.
        sdl: list, optional, default=None
            Standard pressure levels to interpolate T/S profiles on.
        """
        if "BOX" not in idx.search_type.keys():
            raise OptionValueError(
                f"Argo data mass fetching is only available for an ArgoIndex that was search with a 'box' !"
            )

        self.mode = OPTIONS['mode'] if mode is None else mode
        VALIDATE('mode', self.mode)

        self.domain = idx.domain
        self.cname = idx.cname
        self.box = idx.search_type["BOX"]

        # Possibly read Standard Pressure Levels:
        self.sdl_axis = sdl

        # Handle other options related to file access:
        self.protocol = idx.fs["src"].protocol
        self.fs = gdacfs(cache=idx.cache, cachedir=idx.cachedir)
        self.host = self.fs.fs.path

        # List netcdf files available for processing:
        self.argo_files = idx.read_files(multi=True)
        self.N_RECORDS = len(self.argo_files)


    def __repr__(self):
        summary = [f"<argopy.MassFetcher.{self.protocol}>"]
        summary.append(f"Host: {self.host}")
        summary.append(f"Domain: {self.cname}")
        summary.append(f"Number of floats: {self.N_RECORDS}")
        sdl_str = (
            "-"
            if str(self.sdl_axis) == "None"
            else f"{len(self.sdl_axis)} levels from {min(self.sdl_axis)} to {max(self.sdl_axis)}"
        )
        summary.append(f"Standard pressure levels: {sdl_str}")
        return "\n".join(summary)

    def _to_sdl(self, sdl, var, var_prs, mask, lat=45.0, name="var"):
        """Interpolate a 2D variable onto standard pressure levels, abusively called 'depth' levels"""

        def VI(zi, z, c):
            zi, z = np.abs(zi), np.abs(
                z
            )  # abs ensure depths are sorted for the interpolation to work
            if c.shape[0] > 0:
                ci = np.interp(zi, z, c, left=c[0], right=9999.0)
                if np.any(ci >= 9999.0):
                    return np.array(())
                else:
                    return ci
            else:
                return np.array(())

        var_sdl = []
        ip = []
        for i in range(0, var.shape[0]):
            c = var[i, mask[i, :] == True]
            p = var_prs[i, mask[i, :] == True]

            # Interpolate on z:
            # z = gsw.z_from_p(p, lat[i])
            # ci = VI(sdl, z, c)

            # Interpolate of p:
            ci = VI(sdl, p, c)

            if ci.shape[0] > 0:
                var_sdl.append(ci)
                ip.append(i)
        if len(var_sdl) > 0:
            return xr.DataArray(
                var_sdl,
                dims=["N_PROF", "PRES_INTERPOLATED"],
                coords={"PRES_INTERPOLATED": sdl, "N_PROF": ip},
                name=name,
            )
        else:
            return None

    def _add_dsattributes(self, ds, title="Argo float data", mode='standard'):
        ds.attrs["title"] = title
        ds.attrs["Conventions"] = "CF-1.6"
        ds.attrs["CreationDate"] = pd.to_datetime("now").strftime("%Y/%m/%d")
        if mode == 'standard':
            ds.attrs["Comment"] = (
                "Measurements in this dataset are those with Argo QC flags "
                "1, 5 or 8 on position and date; and with Argo QC flag 1 on pressure, temperature and "
                "salinity. Adjusted variables were used wherever necessary "
                "(depending on the data mode). See the Argo user manual for "
                "more information: https://archimer.ifremer.fr/doc/00187/29825"
            )
        elif mode == 'research':
            ds.attrs["Comment"] = (
                "Measurements in this dataset are those with Argo QC flags "
                "1, 5 or 8 on position and date; and with Argo QC flag 1 on pressure, temperature and "
                "salinity. Only delayed-mode variables were used and samples with a pressure error smaller than 20db. "
                "See the Argo user manual for more information: https://archimer.ifremer.fr/doc/00187/29825"
            )

        # This is following: http://cfconventions.org/Data/cf-standard-names/70/build/cf-standard-name-table.html
        if "TEMP" in ds:
            ds["TEMP"].attrs = {
                "long_name": "Sea temperature in-situ ITS-90 scale",
                "standard_name": "sea_water_temperature",
                "units": "degree_Celsius",
                "valid_min": -2.5,
                "valid_max": 40.0,
                "resolution": 0.001,
            }

        if "PSAL" in ds:
            ds["PSAL"].attrs = {
                "long_name": "Practical salinity",
                "standard_name": "sea_water_salinity",
                "units": "psu",
                "valid_min": 2.0,
                "valid_max": 41.0,
                "resolution": 0.001,
            }

        if "LONGITUDE" in ds:
            ds["LONGITUDE"].attrs = {
                "long_name": "Longitude of the station, best estimate",
                "standard_name": "longitude",
                "units": "degree_east",
                "valid_min": -180.0,
                "valid_max": 180.0,
                "resolution": 0.001,
                "axis": "X",
            }

        if "LATITUDE" in ds:
            ds["LATITUDE"].attrs = {
                "long_name": "Latitude of the station, best estimate",
                "standard_name": "latitude",
                "units": "degree_north",
                "valid_min": -90.0,
                "valid_max": 90.0,
                "resolution": 0.001,
                "axis": "Y",
            }

        if "PRES" in ds:
            ds["PRES"].attrs = {
                "long_name": "Sea water pressure, equals 0 at sea-level",
                "standard_name": "sea_water_pressure",
                "units": "dbar",
                "valid_min": 0.0,
                "valid_max": 12000.0,
                "resolution": 1.0,
                "axis": "Z",
            }

        # if 'PRES_INTERPOLATED' in ds:
        #     ds['PRES_INTERPOLATED'].attrs = {
        #         'long_name': "Vertical distance below the surface",
        #         'standard_name': "depth",
        #         'units': "m",
        #         'valid_min': 0.,
        #         'valid_max': 12000.,
        #         'resolution': 1.,
        #         'axis': 'Z',
        #         'positive': 'down'}

        if "TIME" in ds:
            ds["TIME"].encoding["units"] = "days since 1950-01-01"
            ds["TIME"].attrs = {
                "long_name": "Time",
                "standard_name": "time",
                "axis": "T",
            }

        # Specific to this machinery:
        if "ARCOID" in ds:
            ds["ARCOID"].attrs = {
                "long_name": "Profile unique ID",
                "standard_name": "ID",
                "comment": "Computed as: 1000 * FLOAT_WMO + CYCLE_NUMBER",
            }

        return ds

    def _xload_multiprof_standard(
        self, ds: xr.Dataset | None, domain: list
    ) -> xr.Dataset | None:
        """Load a core-Argo multi-profile file as a collection of points or sdl profiles in 'standard' user mode

        Parameters
        ----------
        ds: :class:`xr.Dataset`
            A raw loaded Xarray dataset to work with
        domain: list[float | datetime]
            The `box` domain to select. Contains: [lon_min, lon_max, lat_min, lat_max, date_min, date_max]

        Returns
        -------
        :class:`xr.Dataset`
        """
        if ds is None:
            return None

        try:
            argo = ArgoMultiProf(ds)
        except Exception as e:
            print("Error with", ds.attrs)
            return None

        if argo.psal_qc is None:
            return None
        if argo.temp_qc is None:
            return None
        if (argo.juld < 0).any():
            return None

        # Profile selection
        metas_finite = np.logical_and.reduce(
            [np.isfinite(argo.lon), np.isfinite(argo.lat), np.isfinite(argo.juld)]
        )
        # pos_good = ~(np.isin(argo.position_qc, [3, 4]))
        # juld_good = ~(np.isin(argo.juld_qc, [3, 4]))
        pos_good = np.isin(argo.position_qc, [1, 5, 8])
        juld_good = np.isin(argo.juld_qc, [1, 5, 8])
        in_domain = np.logical_and.reduce(
            [
                argo.lon >= domain[0],
                argo.lon <= domain[1],
                argo.lat >= domain[2],
                argo.lat <= domain[3],
                argo.datetime >= dt.datetime.utcfromtimestamp(domain[4].tolist() / 1e9),
                argo.datetime < dt.datetime.utcfromtimestamp(domain[5].tolist() / 1e9),
            ]
        )
        good_profiles = np.logical_and.reduce(
            [metas_finite, juld_good, pos_good, in_domain]
        )

        if (~good_profiles).all():
            return None

        temps = argo.temp[
            good_profiles
        ]  # R/A/D measurements have been merged according to DATA_MODE
        psals = argo.psal[
            good_profiles
        ]  # R/A/D measurements have been merged according to DATA_MODE
        pres = argo.pres[
            good_profiles
        ]  # R/A/D measurements have been merged according to DATA_MODE
        lons, lats, dats = (
            argo.lon[good_profiles],
            argo.lat[good_profiles],
            argo.datetime[good_profiles],
        )
        cycs = argo.cycle[good_profiles]

        # Assign an id for each good profile
        meta = argo_split_path(ds.encoding["source"])
        dac_wmo = meta["dac"], meta["wmo"]
        # profile_id = np.array(
        #     [str(dac_wmo[1]) + "_" + x for x in np.arange(pres.shape[0]).astype(str)]
        # )

        assert temps.shape == psals.shape
        assert temps.shape == pres.shape

        # per-point selection
        finite_measurements = np.logical_and.reduce(
            [np.isfinite(temps), np.isfinite(pres), np.isfinite(psals)]
        )

        # Exclude points with QC=3 or 4
        # temp_good = ~(np.isin(argo.temp_qc[good_profiles], [3, 4]))
        # pres_good = ~(np.isin(argo.pres_qc[good_profiles], [3, 4]))
        # psal_good = ~(np.isin(argo.psal_qc[good_profiles], [3, 4]))

        # Keep points with QC=1
        temp_good = np.isin(argo.temp_qc[good_profiles], [1])
        pres_good = np.isin(argo.pres_qc[good_profiles], [1])
        psal_good = np.isin(argo.psal_qc[good_profiles], [1])

        # More sanity checks:
        pres_in_range = np.nan_to_num(pres) >= 0
        temp_in_range = np.logical_and(
            np.nan_to_num(temps) < 40.0, np.nan_to_num(temps) > -2.0
        )
        psal_in_range = np.logical_and(
            np.nan_to_num(psals) < 41.0, np.nan_to_num(psals) > 2.0
        )

        assert finite_measurements.shape == temps.shape
        assert temp_good.shape == temps.shape
        assert pres_good.shape == temps.shape
        assert psal_good.shape == temps.shape
        assert pres_in_range.shape == temps.shape
        assert psal_in_range.shape == temps.shape
        assert temp_in_range.shape == temps.shape
        assert lons.shape == temps[:, 0].shape
        assert lats.shape == temps[:, 0].shape
        assert dats.shape == temps[:, 0].shape
        assert cycs.shape == temps[:, 0].shape

        # good points?
        good = np.logical_and.reduce(
            [
                finite_measurements,
                temp_good,
                pres_good,
                psal_good,
                pres_in_range,
                temp_in_range,
                psal_in_range,
            ]
        )

        assert good.shape == temps.shape

        if np.sum(good) > 0:
            lons = lons[good[:, 0]]
            lats = lats[good[:, 0]]
            dats = dats[good[:, 0]]
            cycs = cycs[good[:, 0]]

            sdl = self.sdl_axis

            if sdl is not None:
                # Return data interpolated onto Standard Depth Levels:
                temps = self._to_sdl(sdl, temps, pres, good, lats, name="TEMP")
                psals = self._to_sdl(sdl, psals, pres, good, lats, name="PSAL")
                if temps is not None and psals is not None:
                    assert np.all(temps["N_PROF"] == psals["N_PROF"]) == True
                    ip = temps["N_PROF"].values
                    try:
                        # Create dataset:
                        ds = xr.merge(
                            (
                                xr.DataArray(lons[ip], name="LONGITUDE", dims="N_PROF"),
                                xr.DataArray(lats[ip], name="LATITUDE", dims="N_PROF"),
                                xr.DataArray(dats[ip], name="TIME", dims="N_PROF"),
                                xr.DataArray(
                                    1000 * argo.wmo[0] + cycs[ip],
                                    name="id",
                                    dims="N_PROF",
                                ),
                                temps,
                                psals,
                            )
                        )
                    except:
                        print(
                            "Error while merging SDL:\n",
                            temps["N_PROF"],
                            "\n",
                            min(ip),
                            max(ip),
                            "\n",
                            good[ip, 0].shape,
                        )
                        return None

                    # Preserve Argo attributes:
                    ds = self._add_dsattributes(
                        ds,
                        title="Argo float profiles interpolated onto Standard Pressure Levels",
                        self.mode,
                    )
                    ds.attrs["DAC"] = dac_wmo[0]
                    ds.attrs["PLATFORM_NUMBER"] = int(dac_wmo[1])
                    ds.attrs["Method"] = "Vertical linear interpolation"
                    ds["N_PROF"].attrs = {"long_name": "Profile samples"}
                    return ds
                else:
                    return None

            else:
                # Return a collection of points:
                pres_pts = pres[good]
                temp_pts = temps[good]
                psal_pts = psals[good]

                repeat_n = [np.sum(x) for x in good]
                lons = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(argo.lon[good_profiles])
                    ]
                )
                lats = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(argo.lat[good_profiles])
                    ]
                )
                dts = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(argo.datetime[good_profiles])
                    ]
                )
                # profile_id_pp = np.concatenate([np.repeat(x, repeat_n[i]) for i, x in enumerate(profile_id)])
                profile_numid = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(
                            1000 * argo.wmo[good_profiles] + argo.cycle[good_profiles]
                        )
                    ]
                )

                ds = xr.merge(
                    (
                        xr.DataArray(lons, name="LONGITUDE", dims="N_POINTS"),
                        xr.DataArray(lats, name="LATITUDE", dims="N_POINTS"),
                        xr.DataArray(pres_pts, name="PRES", dims="N_POINTS"),
                        xr.DataArray(dts, name="TIME", dims="N_POINTS"),
                        xr.DataArray(profile_numid, name="ARCOID", dims="N_POINTS"),
                        xr.DataArray(temp_pts, name="TEMP", dims="N_POINTS"),
                        xr.DataArray(psal_pts, name="PSAL", dims="N_POINTS"),
                    )
                )
                # xr.DataArray(gsw.z_from_p(pres_pts, lats, geo_strf_dyn_height=0), name='PRES_INTERPOLATED', dims='N_POINTS'),

                # Preserve Argo attributes:
                ds = self._add_dsattributes(
                    ds, title="Argo float profiles ravelled data", self.mode,
                )
                ds.attrs["DAC"] = dac_wmo[0]
                ds.attrs["PLATFORM_NUMBER"] = int(dac_wmo[1])
                ds["N_POINTS"].attrs = {"long_name": "Measurement samples"}
                return ds

        else:
            return None

    def _xload_multiprof_research(
        self, ds: xr.Dataset | None, domain: list
    ) -> xr.Dataset | None:
        """Load a core-Argo multi-profile file as a collection of points or sdl profiles in 'research' user mode

        Parameters
        ----------
        ds: :class:`xr.Dataset`
            A raw loaded Xarray dataset to work with
        domain: list[float | datetime]
            The `box` domain to select. Contains: [lon_min, lon_max, lat_min, lat_max, date_min, date_max]

        Returns
        -------
        :class:`xr.Dataset`
        """
        if ds is None:
            return None

        try:
            argo = ArgoMultiProf(ds)
        except Exception as e:
            print("Error with", ds.attrs)
            return None

        if argo.psal_qc is None:
            return None
        if argo.temp_qc is None:
            return None
        if (argo.juld < 0).any():
            return None

        # Profile selection
        metas_finite = np.logical_and.reduce(
            [np.isfinite(argo.lon), np.isfinite(argo.lat), np.isfinite(argo.juld)]
        )
        pos_good = np.isin(argo.position_qc, [1, 5, 8])
        juld_good = np.isin(argo.juld_qc, [1, 5, 8])
        dm_good = np.isin(argo.data_mode, [2])  # Only delayed-mode data
        in_domain = np.logical_and.reduce(
            [
                argo.lon >= domain[0],
                argo.lon <= domain[1],
                argo.lat >= domain[2],
                argo.lat <= domain[3],
                argo.datetime >= dt.datetime.utcfromtimestamp(domain[4].tolist() / 1e9),
                argo.datetime < dt.datetime.utcfromtimestamp(domain[5].tolist() / 1e9),
            ]
        )
        good_profiles = np.logical_and.reduce(
            [metas_finite, juld_good, pos_good, dm_good, in_domain]
        )

        if (~good_profiles).all():
            return None

        temps = argo.temp[
            good_profiles
        ]  # R/A/D measurements have been merged according to DATA_MODE
        psals = argo.psal[
            good_profiles
        ]  # R/A/D measurements have been merged according to DATA_MODE
        pres = argo.pres[
            good_profiles
        ]  # R/A/D measurements have been merged according to DATA_MODE

        lons, lats, dats = (
            argo.lon[good_profiles],
            argo.lat[good_profiles],
            argo.datetime[good_profiles],
        )
        cycs = argo.cycle[good_profiles]

        perrs = argo.pres_error[good_profiles]
        serrs = argo.temp_error[good_profiles]
        terrs = argo.psal_error[good_profiles]

        # Assign an id for each good profile
        meta = argo_split_path(ds.encoding["source"])
        dac_wmo = meta["dac"], meta["wmo"]
        # profile_id = np.array(
        #     [str(dac_wmo[1]) + "_" + x for x in np.arange(pres.shape[0]).astype(str)]
        # )

        assert temps.shape == psals.shape
        assert temps.shape == pres.shape
        assert temps.shape == perrs.shape
        assert temps.shape == serrs.shape
        assert temps.shape == terrs.shape

        # per-point selection
        finite_measurements = np.logical_and.reduce(
            [np.isfinite(temps), np.isfinite(pres), np.isfinite(psals)]
        )

        # Keep points with QC=1
        temp_good = np.isin(argo.temp_qc[good_profiles], [1])
        pres_good = np.isin(argo.pres_qc[good_profiles], [1])
        psal_good = np.isin(argo.psal_qc[good_profiles], [1])

        # Keep points with pressure error smaller than 20db:
        perr_in_range = argo.pres_error[good_profiles] < 20

        # More sanity checks:
        pres_in_range = np.nan_to_num(pres) >= 0
        temp_in_range = np.logical_and(
            np.nan_to_num(temps) < 40.0, np.nan_to_num(temps) > -2.0
        )
        psal_in_range = np.logical_and(
            np.nan_to_num(psals) < 41.0, np.nan_to_num(psals) > 2.0
        )

        assert finite_measurements.shape == temps.shape
        assert temp_good.shape == temps.shape
        assert pres_good.shape == temps.shape
        assert psal_good.shape == temps.shape
        assert pres_in_range.shape == temps.shape
        assert psal_in_range.shape == temps.shape
        assert temp_in_range.shape == temps.shape
        assert perr_in_range.shape == temps.shape
        assert lons.shape == temps[:, 0].shape
        assert lats.shape == temps[:, 0].shape
        assert dats.shape == temps[:, 0].shape
        assert cycs.shape == temps[:, 0].shape

        # good points?
        good = np.logical_and.reduce(
            [
                finite_measurements,
                temp_good,
                pres_good,
                psal_good,
                pres_in_range,
                temp_in_range,
                psal_in_range,
                perr_in_range,
            ]
        )

        assert good.shape == temps.shape

        if np.sum(good) > 0:
            lons = lons[good[:, 0]]
            lats = lats[good[:, 0]]
            dats = dats[good[:, 0]]
            cycs = cycs[good[:, 0]]

            sdl = self.sdl_axis

            if sdl is not None:
                # Return data interpolated onto Standard Depth Levels:
                temps = self._to_sdl(sdl, temps, pres, good, lats, name="TEMP")
                psals = self._to_sdl(sdl, psals, pres, good, lats, name="PSAL")
                if temps is not None and psals is not None:
                    assert np.all(temps["N_PROF"] == psals["N_PROF"]) == True
                    ip = temps["N_PROF"].values
                    try:
                        # Create dataset:
                        ds = xr.merge(
                            (
                                xr.DataArray(lons[ip], name="LONGITUDE", dims="N_PROF"),
                                xr.DataArray(lats[ip], name="LATITUDE", dims="N_PROF"),
                                xr.DataArray(dats[ip], name="TIME", dims="N_PROF"),
                                xr.DataArray(
                                    1000 * argo.wmo[0] + cycs[ip],
                                    name="id",
                                    dims="N_PROF",
                                ),
                                temps,
                                psals,
                            )
                        )
                    except:
                        print(
                            "Error while merging SDL:\n",
                            temps["N_PROF"],
                            "\n",
                            min(ip),
                            max(ip),
                            "\n",
                            good[ip, 0].shape,
                        )
                        return None

                    # Preserve Argo attributes:
                    ds = self._add_dsattributes(
                        ds,
                        title="Argo float profiles interpolated onto Standard Pressure Levels",
                    )
                    ds.attrs["DAC"] = dac_wmo[0]
                    ds.attrs["PLATFORM_NUMBER"] = int(dac_wmo[1])
                    ds.attrs["Method"] = "Vertical linear interpolation"
                    ds["N_PROF"].attrs = {"long_name": "Profile samples"}
                    return ds
                else:
                    return None

            else:
                # Return a collection of points:
                pres_pts = pres[good]
                temp_pts = temps[good]
                psal_pts = psals[good]

                perr_pts = perrs[good]
                terr_pts = terrs[good]
                serr_pts = serrs[good]

                repeat_n = [np.sum(x) for x in good]
                lons = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(argo.lon[good_profiles])
                    ]
                )
                lats = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(argo.lat[good_profiles])
                    ]
                )
                dts = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(argo.datetime[good_profiles])
                    ]
                )
                # profile_id_pp = np.concatenate([np.repeat(x, repeat_n[i]) for i, x in enumerate(profile_id)])
                profile_numid = np.concatenate(
                    [
                        np.repeat(x, repeat_n[i])
                        for i, x in enumerate(
                            1000 * argo.wmo[good_profiles] + argo.cycle[good_profiles]
                        )
                    ]
                )

                ds = xr.merge(
                    (
                        xr.DataArray(profile_numid, name="ARCOID", dims="N_POINTS"),
                        xr.DataArray(lons, name="LONGITUDE", dims="N_POINTS"),
                        xr.DataArray(lats, name="LATITUDE", dims="N_POINTS"),
                        xr.DataArray(dts, name="TIME", dims="N_POINTS"),
                        xr.DataArray(pres_pts, name="PRES", dims="N_POINTS"),
                        xr.DataArray(temp_pts, name="TEMP", dims="N_POINTS"),
                        xr.DataArray(psal_pts, name="PSAL", dims="N_POINTS"),
                        xr.DataArray(perr_pts, name="PRES_ERROR", dims="N_POINTS"),
                        xr.DataArray(terr_pts, name="TEMP_ERROR", dims="N_POINTS"),
                        xr.DataArray(serr_pts, name="PSAL_ERROR", dims="N_POINTS"),
                    )
                )
                # xr.DataArray(gsw.z_from_p(pres_pts, lats, geo_strf_dyn_height=0), name='PRES_INTERPOLATED', dims='N_POINTS'),

                # Preserve Argo attributes:
                ds = self._add_dsattributes(
                    ds, title="Argo float profiles ravelled data"
                )
                ds.attrs["DAC"] = dac_wmo[0]
                ds.attrs["PLATFORM_NUMBER"] = int(dac_wmo[1])
                ds["N_POINTS"].attrs = {"long_name": "Measurement samples"}
                return ds

        else:
            return None

    def to_xarray(
        self,
        client: distributed.client.Client | None = None,
        nfloats: int | None = None,
    ) -> xr.Dataset | None:
        """Massively Fetch data using a :class:`distributed.client.Client`

        Parameters
        ----------
        client: :class:`distributed.client.Client`, optional, default=None
            A Dask client to use. If set to None, we fall back on a possibly existing client otherwise raise an `OptionValueError` if no client is found.
        nfloats: int, optional, default=None
            A number of floats to randomly select a subsample of the dataset for test purposes.

        Returns
        -------
        xr.Dataset

        Notes
        -----
        Selected Argo measurements include:
        - QC=[1,5,8] on position and date
        - QC=1 on pres/temp/psal
        - Real-time, adjusted or delayed, according to data_mode

        """
        if client is None:
            try:
                client = distributed.Client.current()
            except ValueError:
                raise OptionValueError(f"No Distributed client running nor provided !")

        ncfiles = self.argo_files
        if nfloats is not None:
            ncfiles = [
                self.argo_files[f]
                for f in np.random.choice(range(0, len(self.argo_files) - 1), nfloats)
            ]
        concat_dim = "N_PROF" if self.sdl_axis is not None else "N_POINTS"

        preprocess = self._xload_multiprof_standard if self.mode == 'standard' else self._xload_multiprof_research

        ds = self.fs.open_mfdataset(
            ncfiles,
            method=client,
            concat=True,
            concat_dim=concat_dim,
            preprocess=preprocess,
            preprocess_opts={"domain": self.domain},
            open_dataset_opts={"errors": "ignore"},
            errors="ignore",
        )
        ds.attrs.pop("DAC")
        ds.attrs.pop("PLATFORM_NUMBER")
        ds.attrs["domain"] = self.cname
        ds.attrs["nb_floats"] = len(np.unique((ds["ARCOID"] / 1000).astype(int)))
        ds.attrs["nb_profiles"] = len(np.unique((ds["ARCOID"])))
        ds = ds.sortby("TIME")
        ds[concat_dim] = np.arange(0, len(ds[concat_dim]))
        return ds

    def to_zarr(
        self,
        store : str | None = None,
        chunks : dict[str, int] | None = None,
        client: distributed.client.Client | None = None,
        nfloats: int | None = None,
    ):

        ds = self.to_xarray(client, nfloats)

        from zarr.codecs import BloscCodec

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")
        encoding = {}
        for v in ds:
            encoding.update({v: {"compressors": [compressor]}})

        if chunks is None:
            if self.sdl_axis is None:
                ds = ds.chunk({"N_POINTS": len(ds["N_POINTS"])})
            else:
                ds = ds.chunk({"N_PROF": len(ds["N_PROF"]), "PRES_INTERPOLATED": len(ds["PRES_INTERPOLATED"])})
        else:
            ds = ds.chunk(chunks)

        return ds.to_zarr(store, encoding=encoding)
