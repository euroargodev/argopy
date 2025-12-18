import os
import xarray as xr

# import gsw
import pandas as pd
import numpy as np

from argopy.stores.filesystems import distributed

from argopy.stores.implementations.gdac import gdacfs
from argopy.utils.format import argo_split_path
from argopy.utils.arco.argo_multiprof import ArgoMultiProf


class MassFetcher(object):
    def __init__(self, idx: 'ArgoIndex', sdl=None):
        self.sdl_axis = sdl

        # self.client = client
        self.protocol = idx.fs['src'].protocol
        self.fs = gdacfs(cache=idx.cache, cachedir=idx.cachedir)
        self.host = self.fs.fs.path
        self.cname = idx.cname

        # List netcdf files available for processing:
        self.argo_files = idx.read_files(multi=True)
        self.N_RECORDS = len(self.argo_files)

        self.argo_wmos = [
            int(os.path.basename(x).split("_")[0]) for x in self.argo_files
        ]
        self.argo_dacs = [x.split("/")[-3] for x in self.argo_files]

    def __repr__(self):
        summary = [f"<argopy.massfetcher.{self.protocol}>"]
        summary.append(f"Host: {self.host}")
        summary.append(f"Domain: {self.cname}")
        summary.append(f"Number of floats: {self.N_RECORDS}")
        sdl_str = '-' if str(self.sdl_axis) == 'None' else f"{len(self.sdl_axis)} levels from {min(self.sdl_axis)} to {max(self.sdl_axis)}"
        summary.append(f"Standard pressure levels: {sdl_str}")
        return "\n".join(summary)

    def _to_sdl(self, sdl, var, var_prs, mask, lat=45., name='var'):
        """Interpolate a 2D variable onto standard pressure levels, abusively called 'depth' levels """

        def VI(zi, z, c):
            zi, z = np.abs(zi), np.abs(z)  # abs ensure depths are sorted for the interpolation to work
            if c.shape[0] > 0:
                ci = np.interp(zi, z, c, left=c[0], right=9999.)
                if np.any(ci >= 9999.):
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
            return xr.DataArray(var_sdl, dims=['N_PROF', 'PRES_INTERPOLATED'], coords={'PRES_INTERPOLATED': sdl, 'N_PROF': ip}, name=name)
        else:
            return None

    def _add_dsattributes(self, ds, title="Argo float data"):
        ds.attrs["title"] = title
        ds.attrs["Conventions"] = "CF-1.6"
        ds.attrs["CreationDate"] = pd.to_datetime("now").strftime("%Y/%m/%d")
        ds.attrs["Comment"] = (
            "Measurements in this dataset are those with Argo QC flags "
            "1, 5 or 8 on position and date; and with Argo QC flag 1 on pressure, temperature and "
            "salinity. Adjusted variables were used wherever necessary "
            "(depending on the data mode). See the Argo user manual for "
            "more information: https://archimer.ifremer.fr/doc/00187/29825"
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
        if "id" in ds:
            ds["id"].attrs = {
                "long_name": "Profile unique ID",
                "standard_name": "ID",
                "comment": "Computed as: 1000 * FLOAT_WMO + CYCLE_NUMBER",
            }

        return ds

    def _xload_multiprof(self, ds):
        """Load an Argo multi-profile file as a collection of points or sdl profiles"""
        try:
            argo = ArgoMultiProf(ds)
        except ValueError:
            print("Value error on", ds.attrs)
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

        good_profiles = np.logical_and.reduce([metas_finite, juld_good, pos_good])

        if (~good_profiles).all():
            return None

        temps = argo.temp[good_profiles]
        psals = argo.psal[good_profiles]
        pres = argo.pres[good_profiles]

        # Assign an id for each good profile
        meta = argo_split_path(ds.encoding["source"])
        dac_wmo = meta["dac"], meta["wmo"]
        profile_id = np.array(
            [str(dac_wmo[1]) + "_" + x for x in np.arange(pres.shape[0]).astype(str)]
        )

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
            sdl = self.sdl_axis

            if sdl is not None:
                # Return data interpolated onto Standard Depth Levels:
                temps = self._to_sdl(sdl, temps, pres, good, argo.lat, name='TEMP')
                psals = self._to_sdl(sdl, psals, pres, good, argo.lat, name='PSAL')
                if temps is not None and psals is not None:
                    assert np.all(temps['N_PROF'] == psals['N_PROF']) == True
                    try:
                        # Create dataset:
                        ds = xr.merge((
                            xr.DataArray(argo.lon[temps['N_PROF']], name='LONGITUDE', dims='N_PROF'),
                            xr.DataArray(argo.lat[temps['N_PROF']], name='LATITUDE', dims='N_PROF'),
                            xr.DataArray(argo.datetime[temps['N_PROF']], name='TIME', dims='N_PROF'),
                            xr.DataArray(1000 * argo.wmo[temps['N_PROF']] + argo.cycle[temps['N_PROF']],
                                         name='id', dims='N_PROF'),
                            temps, psals))
                    except:
                        print("Error while merging SDL:\n", temps['N_PROF'], "\n", argo.lon.shape)
                        return None

                    # Preserve Argo attributes:
                    ds = self._add_dsattributes(ds,
                                                title='Argo float profiles interpolated onto Standard Pressure Levels')
                    ds.attrs["DAC"] = dac_wmo[0]
                    ds.attrs["PLATFORM_NUMBER"] = int(dac_wmo[1])
                    ds.attrs['Method'] = 'Vertical linear interpolation'
                    ds['N_PROF'].attrs = {'long_name': "Profile samples"}
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
                        xr.DataArray(profile_numid, name="id", dims="N_POINTS"),
                        xr.DataArray(temp_pts, name="TEMP", dims="N_POINTS"),
                        xr.DataArray(psal_pts, name="PSAL", dims="N_POINTS"),
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

    def fetch(self, client : distributed.client.Client, n: int | None = None):
        """Massively Fetch data using a dask distributed client"""

        ncfiles = self.argo_files
        if n is not None:  # Sub-sample for test purposes
            ncfiles = [
                self.argo_files[f]
                for f in np.random.choice(range(0, len(self.argo_files) - 1), n)
            ]
        concat_dim = "N_PROF" if self.sdl_axis is not None else "N_POINTS"

        ds = self.fs.open_mfdataset(
            ncfiles,
            method=client,
            concat=True,
            concat_dim=concat_dim,
            preprocess=self._xload_multiprof,
        )
        ds.attrs.pop("DAC")
        ds.attrs.pop("PLATFORM_NUMBER")
        ds.attrs["domain"] = self.cname
        # ds.attrs["nb_floats"] = len(np.unique((ds["id"] / 1000).astype(int)))
        # ds.attrs["nb_profiles"] = len(np.unique((ds["id"])))
        ds = ds.sortby("TIME")
        ds[concat_dim] = np.arange(0, len(ds[concat_dim]))
        return ds
