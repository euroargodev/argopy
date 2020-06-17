#!/bin/env python
# -*coding: UTF-8 -*-
#
# Argo data fetcher for Argovis.
#
# This is comprised of functions used to query Argovis api
# query functions either return dictionary objects or error messages.
#


import pandas as pd
import xarray as xr
import requests
import pdb

class ArgovisDataFetcher(object):
    """
    Manage access to Argo data through Argovis
    """

    @staticmethod
    def urlopenjson(url):
        resp = requests.get(url)
        # Consider any status other than 2xx an error
        if not resp.status_code // 100 == 2:
            return "Error: Unexpected response {}".format(resp)
        return resp.json()

    @staticmethod
    def to_dataframe(profiles):
        rows = []
        for profile in profiles:
            keys = [ x for x in profile.keys() if not x in ['measurements', 'bgcMeas']]
            meta_row = dict( (key, profile[key]) for key in keys)
            for row in profile['measurements']:
                row.update(meta_row)
                rows.append(row)
        df = pd.DataFrame(rows)
        return df

    def to_xarray(self, profiles):
        """ Load Argo index and return a xarray Dataset """
        return self.to_dataframe(profiles).to_xarray()

    def get_profile(self, platform_number, cycle_number):
        '''
        Returns either a dictionary object for one profile, or an error string.

        Parameters
        ----------
        platform_number: wmo float id.
        cycle_number: numer signifying the profile number for this particular platform

        Example
        -------
        from argovis import ArgovisDataFetcher
        adf = ArgovisDataFetcher()
        profile = adf.get_profile(3900737, 279)
        '''
        profile_number = str(platform_number) + '_' + str(cycle_number)
        url = 'https://argovis.colorado.edu/catalog/profiles/{}'.format(profile_number)
        profile = self.urlopenjson(url)
        return profile

    def get_platform_profiles(self, platform_number):
        '''
        Returns either a list of dictionary objects representing profile data from a platform, or an error string.

        Parameters
        ----------
        platform_number: wmo float id.

        Example
        -------
        from argovis import ArgovisDataFetcher
        adf = ArgovisDataFetcher()
        platformProfiles = adf.get_platform_profiles(3900737)
        '''
        url = 'https://argovis.colorado.edu/catalog/platforms/{}'.format(str(platform_number))
        platformProfiles = self.urlopenjson(url)
        return platformProfiles

    def get_selection_profiles(self, startDate, endDate, shape, presRange=None, printUrl=True):
        '''
        Returns either a list of dictionary objects representing profile data, or an error string.

        Parameters
        ----------
        start date: str formatted as 'YYYY-MM-DD'
        end date: str formatted as 'YYYY-MM-DD'
        pressure range (optional): list of two numbers formatted as [lowerPressure,upperPressure].
        shape: a list of lists containing [lng, lat] coordinates. First and last coordinate should
        be equal.
        printURL: boolean (optional)

        Example
        -------
        shape = [[[168.6,21.7],[168.6,37.7],[-145.9,37.7],[-145.9,21.7],[168.6,21.7]]]
        startDate='2017-9-15'
        endDate='2017-9-30'
        presRange=[0,50]
        selectionProfiles = adf.get_selection_profiles(startDate, endDate, shape, presRange)
        '''
        url = 'https://argovis.colorado.edu/selection/profiles'
        url += '?startDate={}'.format(startDate)
        url += '&endDate={}'.format(endDate)
        url += '&shape={}'.format( str(shape).replace(' ', ''))
        if presRange:
            pressRangeQuery = '&presRange=' + str(presRange).replace(' ', '')
            url += pressRangeQuery
        if printUrl:
            print(url)
        selectionProfiles = self.urlopenjson(url)
        return selectionProfiles

