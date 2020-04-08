#!/bin/env python
# -*coding: UTF-8 -*-

import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

try:
    import seaborn as sns
    sns.set_style("dark")
    with_seaborn = True
except ModuleNotFoundError:
    warnings.warn("argopy requires seaborn installed for full plotting functionality")
    with_seaborn = False

land_feature = cfeature.NaturalEarthFeature(category='physical',
                                           name='land',
                                           scale='50m',
                                           facecolor=[0.4,0.6,0.7])

# THIS TAKES A DATAFRAME AS INPUT. WE SHOULD SET SOME PLOT FUNCTIONS FOR INDEX AND DATA AT THE SAME TIME
# TRAJECTORY PLOT IS A GOOD EXAMPLE.
# SNS.LINEPLOT AND SNS.SCATTERPLOT SHOULD WORKS FINE WITH A DS.DATASET

def plot_trajectory(idx):
    """ Plot trajectories for an index dataframe """
    if not with_seaborn:
        raise BaseException("This function requires seaborn")

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(land_feature, edgecolor='black')    
    nfloat = len(idx.groupby('wmo').first())
    mypal = sns.color_palette("bright", nfloat)

    sns.lineplot(x="longitude",y="latitude",hue="wmo",data=idx,sort=False,palette=mypal,legend=False)
    sns.scatterplot(x="longitude",y="latitude",hue='wmo',data=idx,palette=mypal)   
    width=np.abs(idx['longitude'].max()-idx['longitude'].min())
    height=np.abs(idx['latitude'].max()-idx['latitude'].min())
    #extent = (idx['longitude'].min()-width/4, 
    #          idx['longitude'].max()+width/4, 
    #          idx['latitude'].min()-height/4, 
    #          idx['latitude'].max()+height/4)    

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.7, linestyle=':')
    gl.xlabels_top = False
    gl.ylabels_left = False    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    #ax.set_extent(extent)     
    plt.legend(loc='upper right', bbox_to_anchor=(1.25,1))
    if(nfloat>15):
        ax.get_legend().remove()

def plot_dac(idx):
    """ Histogram of DAC for an index dataframe """
    if not with_seaborn:
        raise BaseException("This function requires seaborn")
    fig=plt.figure(figsize=(10,5))
    mind=idx.groupby('institution').size().sort_values(ascending=False).index
    sns.countplot(y='institution',data=idx,order=mind)
    plt.ylabel('number of profiles')            

def plot_profilerType(idx):
    """ Histogram of profile types for an index dataframe """
    if not with_seaborn:
        raise BaseException("This function requires seaborn")
    fig=plt.figure(figsize=(10,5))
    mind=idx.groupby('profiler_type').size().sort_values(ascending=False).index
    sns.countplot(y='profiler_type',data=idx,order=mind)
    plt.xlabel('number of profiles')      
    plt.ylabel('') 