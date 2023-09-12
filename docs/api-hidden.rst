.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

.. autosummary::
    :toctree: generated/

    argopy

    argopy.fetchers
    argopy.fetchers.ArgoDataFetcher
    argopy.fetchers.ArgoDataFetcher.region
    argopy.fetchers.ArgoDataFetcher.float
    argopy.fetchers.ArgoDataFetcher.profile
    argopy.fetchers.ArgoDataFetcher.load
    argopy.fetchers.ArgoDataFetcher.to_xarray
    argopy.fetchers.ArgoDataFetcher.to_dataframe
    argopy.fetchers.ArgoDataFetcher.to_index
    argopy.fetchers.ArgoDataFetcher.plot
    argopy.fetchers.ArgoDataFetcher.uri
    argopy.fetchers.ArgoDataFetcher.data
    argopy.fetchers.ArgoDataFetcher.index
    argopy.fetchers.ArgoDataFetcher.domain
    argopy.fetchers.ArgoDataFetcher.dashboard
    argopy.fetchers.ArgoDataFetcher.clear_cache

    argopy.fetchers.ArgoIndexFetcher
    argopy.fetchers.ArgoIndexFetcher.region
    argopy.fetchers.ArgoIndexFetcher.float
    argopy.fetchers.ArgoIndexFetcher.profile
    argopy.fetchers.ArgoIndexFetcher.load
    argopy.fetchers.ArgoIndexFetcher.to_xarray
    argopy.fetchers.ArgoIndexFetcher.to_dataframe
    argopy.fetchers.ArgoIndexFetcher.to_csv
    argopy.fetchers.ArgoIndexFetcher.plot
    argopy.fetchers.ArgoIndexFetcher.index
    argopy.fetchers.ArgoIndexFetcher.clear_cache

    argopy.data_fetchers.erddap_data.ErddapArgoDataFetcher
    argopy.data_fetchers.erddap_data.Fetch_wmo
    argopy.data_fetchers.erddap_data.Fetch_box

    argopy.data_fetchers.gdacftp_data.FTPArgoDataFetcher
    argopy.data_fetchers.gdacftp_data.Fetch_wmo
    argopy.data_fetchers.gdacftp_data.Fetch_box

    argopy.data_fetchers.argovis_data.ArgovisDataFetcher
    argopy.data_fetchers.argovis_data.Fetch_wmo
    argopy.data_fetchers.argovis_data.Fetch_box

    argopy.data_fetchers.erddap_refdata.ErddapREFDataFetcher
    argopy.data_fetchers.erddap_refdata.Fetch_box
    argopy.data_fetchers.CTDRefDataFetcher

    argopy.options.set_options

    argopy.tutorial.open_dataset

    argopy.utils.monitor_status

    argopy.utils.show_versions
    argopy.utils.show_options

    argopy.utils.clear_cache
    argopy.utils.lscache

    argopy.utils.list_available_data_src
    argopy.utils.list_available_index_src
    argopy.utils.list_standard_variables
    argopy.utils.list_multiprofile_file_variables

    argopy.utils.Chunker

    argopy.utils.isconnected
    argopy.utils.urlhaskeyword
    argopy.utils.isalive
    argopy.utils.isAPIconnected

    argopy.utils.groupby_remap
    argopy.utils.linear_interpolation_remap

    argopy.utils.format_oneline
    argopy.utils.is_box
    argopy.utils.is_indexbox
    argopy.utils.is_wmo
    argopy.utils.is_cyc
    argopy.utils.check_wmo
    argopy.utils.check_cyc

    argopy.utils.wmo2box

    argopy.utils.deprecated

    argopy.utils.Registry
    argopy.utils.float_wmo

    argopy.utils.drop_variables_not_in_all_datasets
    argopy.utils.fill_variables_not_in_all_datasets

    argopy.utils.MonitoredThreadPoolExecutor

    argopy.related.load_dict
    argopy.related.get_coriolis_profile_id
    argopy.related.get_ea_profile_page

    argopy.related.TopoFetcher.cname
    argopy.related.TopoFetcher.define_constraints
    argopy.related.TopoFetcher.get_url
    argopy.related.TopoFetcher.load
    argopy.related.TopoFetcher.to_xarray
    argopy.related.TopoFetcher.cachepath
    argopy.related.TopoFetcher.uri

    argopy.related.ArgoNVSReferenceTables
    argopy.related.ArgoNVSReferenceTables.search
    argopy.related.ArgoNVSReferenceTables.valid_ref
    argopy.related.ArgoNVSReferenceTables.all_tbl
    argopy.related.ArgoNVSReferenceTables.all_tbl_name
    argopy.related.ArgoNVSReferenceTables.tbl
    argopy.related.ArgoNVSReferenceTables.tbl_name

    argopy.related.OceanOPSDeployments
    argopy.related.OceanOPSDeployments.to_dataframe
    argopy.related.OceanOPSDeployments.status_code

    argopy.related.ArgoDocs
    argopy.related.ArgoDocs.list
    argopy.related.ArgoDocs.search
    argopy.related.ArgoDocs.ris
    argopy.related.ArgoDocs.abstract
    argopy.related.ArgoDocs.pdf
    argopy.related.ArgoDocs.open_pdf
    argopy.related.ArgoDocs.show
    argopy.related.ArgoDocs.js

    argopy.plot
    argopy.plot.dashboard
    argopy.plot.bar_plot
    argopy.plot.scatter_map
    argopy.plot.plot_trajectory
    argopy.plot.latlongrid
    argopy.plot.discrete_coloring
    argopy.plot.open_sat_altim_report

    argopy.plot.ArgoColors
    argopy.plot.ArgoColors.COLORS
    argopy.plot.ArgoColors.quantitative
    argopy.plot.ArgoColors.definition
    argopy.plot.ArgoColors.cmap
    argopy.plot.ArgoColors.lookup
    argopy.plot.ArgoColors.ticklabels
    argopy.plot.ArgoColors.list_valid_known_colormaps

    argopy.stores.filesystems.argo_store_proto

    argopy.stores.filesystems.filestore
    argopy.stores.filestore.open_dataset
    argopy.stores.filestore.read_csv
    argopy.stores.filestore.open
    argopy.stores.filestore.glob
    argopy.stores.filestore.exists
    argopy.stores.filestore.store_path
    argopy.stores.filestore.register
    argopy.stores.filestore.cachepath
    argopy.stores.filestore.clear_cache
    argopy.stores.filestore.open_mfdataset

    argopy.stores.filesystems.httpstore
    argopy.stores.httpstore.open_json
    argopy.stores.httpstore.open_dataset
    argopy.stores.httpstore.read_csv
    argopy.stores.httpstore.open
    argopy.stores.httpstore.glob
    argopy.stores.httpstore.exists
    argopy.stores.httpstore.store_path
    argopy.stores.httpstore.register
    argopy.stores.httpstore.cachepath
    argopy.stores.httpstore.clear_cache
    argopy.stores.httpstore.open_mfdataset
    argopy.stores.httpstore.open_mfjson

    argopy.stores.filesystems.memorystore
    argopy.stores.memorystore.open
    argopy.stores.memorystore.glob
    argopy.stores.memorystore.exists
    argopy.stores.memorystore.store_path
    argopy.stores.memorystore.register
    argopy.stores.memorystore.cachepath
    argopy.stores.memorystore.clear_cache
    argopy.stores.memorystore.open_dataset
    argopy.stores.memorystore.open_mfdataset
    argopy.stores.memorystore.read_csv

    argopy.stores.filesystems.ftpstore
    argopy.stores.ftpstore.open_dataset
    argopy.stores.ftpstore.open_mfdataset

    argopy.stores.argo_index_proto.ArgoIndexStoreProto
    argopy.stores.argo_index_pa.indexstore_pyarrow
    argopy.stores.argo_index_pa.indexstore_pyarrow.load
    argopy.stores.argo_index_pa.indexstore_pyarrow.read_wmo
    argopy.stores.argo_index_pa.indexstore_pyarrow.read_params
    argopy.stores.argo_index_pa.indexstore_pyarrow.records_per_wmo
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_wmo
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_cyc
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_wmo_cyc
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_tim
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_lat_lon
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_lat_lon_tim
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_params
    argopy.stores.argo_index_pa.indexstore_pyarrow.search_parameter_data_mode
    argopy.stores.argo_index_pa.indexstore_pyarrow.to_dataframe
    argopy.stores.argo_index_pa.indexstore_pyarrow.to_indexfile

    argopy.stores.argo_index_pd.indexstore_pandas
    argopy.stores.argo_index_pd.indexstore_pandas.load
    argopy.stores.argo_index_pd.indexstore_pandas.read_wmo
    argopy.stores.argo_index_pd.indexstore_pandas.read_params
    argopy.stores.argo_index_pd.indexstore_pandas.records_per_wmo
    argopy.stores.argo_index_pd.indexstore_pandas.search_wmo
    argopy.stores.argo_index_pd.indexstore_pandas.search_cyc
    argopy.stores.argo_index_pd.indexstore_pandas.search_wmo_cyc
    argopy.stores.argo_index_pd.indexstore_pandas.search_tim
    argopy.stores.argo_index_pd.indexstore_pandas.search_lat_lon
    argopy.stores.argo_index_pd.indexstore_pandas.search_lat_lon_tim
    argopy.stores.argo_index_pd.indexstore_pandas.search_params
    argopy.stores.argo_index_pd.indexstore_pandas.search_parameter_data_mode
    argopy.stores.argo_index_pd.indexstore_pandas.to_dataframe
    argopy.stores.argo_index_pd.indexstore_pandas.to_indexfile

    argopy.stores.ArgoIndex
    argopy.ArgoIndex
    argopy.ArgoIndex.N_MATCH
    argopy.ArgoIndex.N_RECORDS
    argopy.ArgoIndex.convention_supported
    argopy.ArgoIndex.load
    argopy.ArgoIndex.read_wmo
    argopy.ArgoIndex.read_params
    argopy.ArgoIndex.search_wmo
    argopy.ArgoIndex.search_cyc
    argopy.ArgoIndex.search_wmo_cyc
    argopy.ArgoIndex.search_tim
    argopy.ArgoIndex.search_lat_lon
    argopy.ArgoIndex.search_lat_lon_tim
    argopy.ArgoIndex.search_params
    argopy.ArgoIndex.search_parameter_data_mode
    argopy.ArgoIndex.to_dataframe
    argopy.ArgoIndex.to_indexfile

    argopy.xarray.ArgoAccessor.point2profile
    argopy.xarray.ArgoAccessor.profile2point
    argopy.xarray.ArgoAccessor.interp_std_levels
    argopy.xarray.ArgoAccessor.groupby_pressure_bins
    argopy.xarray.ArgoAccessor.teos10
    argopy.xarray.ArgoAccessor.create_float_source
    argopy.xarray.ArgoAccessor.filter_qc
    argopy.xarray.ArgoAccessor.filter_data_mode
    argopy.xarray.ArgoAccessor.filter_scalib_pres
    argopy.xarray.ArgoAccessor.filter_researchmode
    argopy.xarray.ArgoAccessor.cast_types
    argopy.xarray.ArgoAccessor.index
    argopy.xarray.ArgoAccessor.domain
    argopy.xarray.ArgoAccessor.list_WMO_CYC

    argopy.xarray.ArgoEngine

