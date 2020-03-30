#!/bin/env python
# -*coding: UTF-8 -*-
#
# OUR CUSTOM ARGOPY PLOTS
#
# Created by kbalem on 30/03/2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
land_feature=cfeature.NaturalEarthFeature(category='physical',name='land',scale='50m',facecolor=[0.4,0.6,0.7])

def plot_trajectory(idx):
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(land_feature, edgecolor='black')    
    nfloat=len(idx.groupby('wmo').first())
    mypal=sns.color_palette("bright", nfloat)
    sns.lineplot(x="longitude",y="latitude",hue="wmo",data=idx,sort=False,palette=mypal,legend=False)
    sns.scatterplot(x="longitude",y="latitude",hue='wmo',data=idx,palette=mypal)    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2,1))
    if(nfloat>15):
        ax.get_legend().remove()