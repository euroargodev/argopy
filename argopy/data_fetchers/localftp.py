#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for a local copy of GDAC ftp.
#
# This is not intended to be used directly, only by the facade at fetchers.py
#
# Since the GDAC ftp ir organised by DAC/WMO folders, we start by implementing the 'float' and 'profile' entry points.
#
# Created by gmaze on 18/03/2020
# Building on earlier work from S. Tokunaga (as part of the MOCCA and EARISE H2020 projects)

access_points = ['wmo']
exit_formats = ['xarray']
dataset_ids = ['phy', 'bgc']

import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod

class LocalFTPArgoDataFetcher(ABC):
    """ Manage access to Argo data from a local copy of GDAC ftp

    """
    def __init__(self, argo_root, sdl=None):
        self.argo_root_path = argo_root
        self.sdl_axis = sdl

        # List netcdf files available for processing:
        self.argo_files = sorted(glob(self.argo_root_path + "/*/*/*_prof.nc"))
        if self.argo_files is None:
            raise ValueError("Argo root path doesn't contain any netcdf profiles (under */*/*_prof.nc)")
        self.argo_wmos = [int(os.path.basename(x).split("_")[0]) for x in self.argo_files]
        self.argo_dacs = [x.split("/")[-3] for x in self.argo_files]

    def _to_sdl(self, sdl, var, var_prs, mask, lat=45., name='var'):
        """ Interpolate a 2D variable onto standard depth levels, from pressure levels """

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
            z = gsw.z_from_p(p, lat[i])
            ci = VI(sdl, z, c)
            if ci.shape[0] > 0:
                var_sdl.append(ci)
                ip.append(i)
        if len(var_sdl) > 0:
            return xr.DataArray(var_sdl, dims=['samples', 'depth'], coords={'depth': sdl, 'samples': ip}, name=name)
        else:
            return None

    def _add_dsattributes(self, ds, argo_xarr, title='Argo float data'):
        ds.attrs['title'] = title
        ds.attrs['Conventions'] = 'CF-1.6'
        ds.attrs['CreationDate'] = pd.to_datetime('now').strftime("%Y/%m/%d")
        ds.attrs['Comment'] = "Measurements in this dataset are those with Argo QC flags " \
                              "NOT 3 or 4: on position, date, pressure, temperature and " \
                              "salinity. Adjusted variables were used wherever necessary " \
                              "(depending on the data mode). See the Argo user manual for " \
                              "more information: https://archimer.ifremer.fr/doc/00187/29825"

        # This is following: http://cfconventions.org/Data/cf-standard-names/70/build/cf-standard-name-table.html
        if 'to' in ds:
            ds['to'].attrs = {
                'long_name': "Sea temperature in-situ ITS-90 scale",
                'standard_name': "sea_water_temperature",
                'units': "degree_Celsius",
                'valid_min': -2.5,
                'valid_max': 40.,
                'resolution': 0.001}

        if 'so' in ds:
            ds['so'].attrs = {
                'long_name': "Practical salinity",
                'standard_name': "sea_water_salinity",
                'units': "psu",
                'valid_min': 2.,
                'valid_max': 41.,
                'resolution': 0.001}

        if 'longitude' in ds:
            ds['longitude'].attrs = {
                'long_name': "Longitude of the station, best estimate",
                'standard_name': "longitude",
                'units': "degree_east",
                'valid_min': -180.,
                'valid_max': 180.,
                'resolution': 0.001,
                'axis': 'X'}

        if 'latitude' in ds:
            ds['latitude'].attrs = {
                'long_name': "Latitude of the station, best estimate",
                'standard_name': "latitude",
                'units': "degree_north",
                'valid_min': -90.,
                'valid_max': 90.,
                'resolution': 0.001,
                'axis': 'Y'}

        if 'pressure' in ds:
            ds['pressure'].attrs = {
                'long_name': "Sea water pressure, equals 0 at sea-level",
                'standard_name': "sea_water_pressure",
                'units': "dbar",
                'valid_min': 0.,
                'valid_max': 12000.,
                'resolution': 1.,
                'axis': 'Z'}

        if 'depth' in ds:
            ds['depth'].attrs = {
                'long_name': "Vertical distance below the surface",
                'standard_name': "depth",
                'units': "m",
                'valid_min': 0.,
                'valid_max': 12000.,
                'resolution': 1.,
                'axis': 'Z',
                'positive': 'down'}

        if 'time' in ds:
            ds['time'].encoding['units'] = 'days since 1950-01-01'
            ds['time'].attrs = {
                'long_name': "Time",
                'standard_name': "time",
                'axis': 'T'}

        # Specific to this machinary:
        if 'id' in ds:
            ds['id'].attrs = {
                'long_name': "Profile unique ID",
                'standard_name': "ID",
                'comment': "Computed as: 1000 * FLOAT_WMO + CYCLE_NUMBER"}

        return ds

    def _xload_multiprof(self, dac_wmo):
        """Load an Argo multi-profile file as a collection of points or sdl profiles"""
        dac_name, wmo_id = dac_wmo
        wmo_id = int(wmo_id)

        # instantiate the data loader:
        argo_loader = ArgoMultiProfLocalLoader(argo_root_path=self.argo_root_path)

        with argo_loader.load_from_inst(dac_name, wmo_id) as argo_xarr:
            try:
                argo = ArgoMultiProf(argo_xarr)
            except ValueError:
                print("Value error on", dac_wmo)
                return None

            if argo.psal_qc is None:
                return None
            if argo.temp_qc is None:
                return None
            if (argo.juld < 0).any():
                return None

            # Profile selection
            metas_finite = np.logical_and.reduce([np.isfinite(argo.lon), np.isfinite(argo.lat), np.isfinite(argo.juld)])
            pos_good = ~(np.isin(argo.position_qc, [3, 4]))
            juld_good = ~(np.isin(argo.juld_qc, [3, 4]))

            good_profiles = np.logical_and.reduce([metas_finite, juld_good, pos_good])

            if (~good_profiles).all():
                return None

            temps = argo.temp[good_profiles]
            psals = argo.psal[good_profiles]
            pres = argo.pres[good_profiles]

            # Assign an id for each good profile
            profile_id = np.array([str(dac_wmo[1]) + "_" + x for x in np.arange(pres.shape[0]).astype(np.str)])

            assert temps.shape == psals.shape
            assert temps.shape == pres.shape

            # per-point selection
            finite_measurements = np.logical_and.reduce(
                [np.isfinite(temps), np.isfinite(pres), np.isfinite(psals)])

            # Exclude points with QC=3 or 4
            temp_good = ~(np.isin(argo.temp_qc[good_profiles], [3, 4]))
            pres_good = ~(np.isin(argo.pres_qc[good_profiles], [3, 4]))
            psal_good = ~(np.isin(argo.psal_qc[good_profiles], [3, 4]))

            # More sanity checks:
            pres_in_range = (np.nan_to_num(pres) >= 0)
            temp_in_range = np.logical_and(np.nan_to_num(temps) < 40., np.nan_to_num(temps) > -2.5)
            psal_in_range = np.logical_and(np.nan_to_num(psals) < 41., np.nan_to_num(psals) > 2.)

            assert finite_measurements.shape == temps.shape
            assert temp_good.shape == temps.shape
            assert pres_good.shape == temps.shape
            assert psal_good.shape == temps.shape
            assert pres_in_range.shape == temps.shape
            assert psal_in_range.shape == temps.shape
            assert temp_in_range.shape == temps.shape

            # good points?
            good = np.logical_and.reduce(
                [finite_measurements,
                 temp_good,
                 pres_good,
                 psal_good,
                 pres_in_range,
                 temp_in_range,
                 psal_in_range])

            assert good.shape == temps.shape

            if np.sum(good) > 0:
                sdl = self.sdl_axis

                if sdl is not None:
                    # Return data interpolated onto Standard Depth Levels:
                    temps = self._to_sdl(sdl, temps, pres, good, argo.lat, name='to')
                    psals = self._to_sdl(sdl, psals, pres, good, argo.lat, name='so')
                    if temps is not None and psals is not None:
                        assert np.all(temps['samples'] == psals['samples']) == True
                        try:
                            # Create dataset:
                            ds = xr.merge((
                                xr.DataArray(argo.lon[temps['samples']], name='longitude', dims='samples'),
                                xr.DataArray(argo.lat[temps['samples']], name='latitude', dims='samples'),
                                xr.DataArray(argo.datetime[temps['samples']], name='time', dims='samples'),
                                xr.DataArray(1000 * argo.wmo[temps['samples']] + argo.cycle[temps['samples']],
                                             name='id', dims='samples'),
                                temps, psals))
                        except:
                            print("Error while merging SDL:\n", temps['samples'], "\n", argo.lon.shape)
                            return None

                        # Preserve Argo attributes:
                        ds = self._add_dsattributes(ds, argo_xarr,
                                                    title='Argo float profiles interpolated onto Standard Depth Levels')
                        ds.attrs['DAC'] = dac_name
                        ds.attrs['WMO'] = wmo_id
                        ds.attrs['Method'] = 'Vertical linear interpolation'
                        ds['samples'].attrs = {'long_name': "Profile samples"}
                        return ds
                    else:
                        return None

                else:
                    # Return a collection of points:
                    pres_pts = pres[good]
                    temp_pts = temps[good]
                    psal_pts = psals[good]

                    repeat_n = [np.sum(x) for x in good]
                    lons = np.concatenate([np.repeat(x, repeat_n[i]) for i, x in enumerate(argo.lon[good_profiles])])
                    lats = np.concatenate([np.repeat(x, repeat_n[i]) for i, x in enumerate(argo.lat[good_profiles])])
                    dts = np.concatenate(
                        [np.repeat(x, repeat_n[i]) for i, x in enumerate(argo.datetime[good_profiles])])
                    profile_id_pp = np.concatenate([np.repeat(x, repeat_n[i]) for i, x in enumerate(profile_id)])
                    profile_numid = np.concatenate([np.repeat(x, repeat_n[i]) for i, x in enumerate(
                        1000 * argo.wmo[good_profiles] + argo.cycle[good_profiles])])

                    ds = xr.merge((
                        xr.DataArray(lons, name='longitude', dims='samples'),
                        xr.DataArray(lats, name='latitude', dims='samples'),
                        xr.DataArray(pres_pts, name='pressure', dims='samples'),
                        xr.DataArray(gsw.z_from_p(pres_pts, lats, geo_strf_dyn_height=0), name='depth', dims='samples'),
                        xr.DataArray(dts, name='time', dims='samples'),
                        xr.DataArray(profile_numid, name='id', dims='samples'),
                        xr.DataArray(temp_pts, name='to', dims='samples'),
                        xr.DataArray(psal_pts, name='so', dims='samples')))
                    # Preserve Argo attributes:
                    ds = self._add_dsattributes(ds, argo_xarr, title='Argo float profiles ravelled data')
                    ds.attrs['DAC'] = dac_name
                    ds.attrs['WMO'] = wmo_id
                    ds['samples'].attrs = {'long_name': "Measurement samples"}
                    return ds
            else:
                return None

    def to_xarray(self, client=None, n=None):
        """Fetch data using a dask distributed client"""

        dac_wmo_files = list(zip(*[self.argo_dacs, self.argo_wmos]))
        if n is not None:  # Sub-sample for test purposes
            dac_wmo_files = list(np.array(dac_wmo_files)[np.random.choice(range(0, len(dac_wmo_files) - 1), n)])
        print("NB OF FLOATS TO FETCH:", len(dac_wmo_files))

        if client is not None:
            futures = client.map(self._xload_multiprof, dac_wmo_files)
            results = client.gather(futures, errors='raise')
        else:
            results = []
            for wmo in dac_wmo_files:
                results.append(self._xload_multiprof(wmo))

        results = [r for r in results if r is not None]  # Only keep none empty results
        if len(results) > 0:
            ds = xr.concat(results, dim='samples', data_vars='all', compat='equals')
            ds.attrs.pop('DAC')
            ds.attrs.pop('WMO')
            ds = ds.sortby('time')
            ds['samples'].values = np.arange(0, len(ds['samples']))
            return ds
        else:
            print("CAN'T FETCH ANY DATA !")
            return None