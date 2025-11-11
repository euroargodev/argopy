"""
This package contains implementations for data and index fetchers for specific data sources. Most of these fetchers are meant to be used and discovered automatically by the facades (in fetchers.py) and by utilities functions list_available_data_src() and list_available_index_src()
"""

from argopy.data_fetchers.erddap_refdata import Fetch_box as CTDRefDataFetcher
import argopy.data_fetchers.erddap_data
import argopy.data_fetchers.erddap_index
import argopy.data_fetchers.argovis_data
import argopy.data_fetchers.gdac_data
import argopy.data_fetchers.gdac_index

__all__ = (
    "erddap_data",
    "erddap_index",
    "argovis_data",
    "gdac_data",
    "gdac_index",
    "CTDRefDataFetcher",
)
