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
    argopy.fetchers.ArgoDataFetcher.to_dataset
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

    argopy.data_fetchers.gdac_data.GDACArgoDataFetcher
    argopy.data_fetchers.gdac_data.Fetch_wmo
    argopy.data_fetchers.gdac_data.Fetch_box

    argopy.data_fetchers.argovis_data.ArgovisDataFetcher
    argopy.data_fetchers.argovis_data.Fetch_wmo
    argopy.data_fetchers.argovis_data.Fetch_box

    argopy.data_fetchers.erddap_refdata.ErddapREFDataFetcher
    argopy.data_fetchers.erddap_refdata.Fetch_box
    argopy.data_fetchers.CTDRefDataFetcher

    argopy.options.set_options

    argopy.tutorial.open_dataset

    argopy.utils.monitor_status

    argopy.utils.GreenCoding.measurements
    argopy.utils.GreenCoding.total_measurements
    argopy.utils.GreenCoding.footprint_for_release
    argopy.utils.GreenCoding.footprint_since_last_release
    argopy.utils.GreenCoding.footprint_all_releases
    argopy.utils.GreenCoding.footprint_baseline
    argopy.utils.GreenCoding.shieldsio_badge
    argopy.utils.GreenCoding.shieldsio_endpoint

    argopy.utils.Github.releases
    argopy.utils.Github.lastrelease_date
    argopy.utils.Github.lastrelease_tag
    argopy.utils.Github.get_PRtitle
    argopy.utils.Github.ls_PRs
    argopy.utils.Github.ls_PRmerged
    argopy.utils.Github.ls_PRmerged_since_last_release
    argopy.utils.Github.ls_PRmerged_in_release
    argopy.utils.Github.ls_PRbaseline

    argopy.utils.show_versions
    argopy.utils.show_options
    argopy.utils.Asset

    argopy.utils.clear_cache
    argopy.utils.lscache

    argopy.utils.list_available_data_src
    argopy.utils.list_available_index_src
    argopy.utils.list_standard_variables
    argopy.utils.list_multiprofile_file_variables
    argopy.utils.list_core_parameters
    argopy.utils.list_bgc_s_variables
    argopy.utils.list_bgc_s_parameters
    argopy.utils.list_radiometry_variables
    argopy.utils.list_radiometry_parameters
    argopy.utils.list_gdac_servers
    argopy.utils.shortcut2gdac

    argopy.utils.Chunker

    argopy.utils.isconnected
    argopy.utils.urlhaskeyword
    argopy.utils.isalive
    argopy.utils.isAPIconnected

    argopy.utils.groupby_remap
    argopy.utils.linear_interpolation_remap

    argopy.utils.argo_split_path
    argopy.utils.format_oneline
    argopy.utils.UriCName

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

    argopy.utils.optical_modeling.Z_euphotic
    argopy.utils.optical_modeling.Z_firstoptic
    argopy.utils.optical_modeling.Z_iPAR_threshold
    argopy.utils.optical_modeling.DCM

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

    argopy.related.ArgoDOI
    argopy.related.ArgoDOI.search
    argopy.related.ArgoDOI.download
    argopy.related.ArgoDOI.dates
    argopy.related.ArgoDOI.file
    argopy.related.ArgoDOI.dx
    argopy.related.ArgoDOI.doi
    argopy.related.doi_snapshot.DOIrecord

    argopy.plot
    argopy.plot.dashboard
    argopy.plot.bar_plot
    argopy.plot.scatter_map
    argopy.plot.scatter_plot
    argopy.plot.plot_trajectory
    argopy.plot.latlongrid
    argopy.plot.open_sat_altim_report

    argopy.plot.ArgoColors
    argopy.plot.ArgoColors.COLORS
    argopy.plot.ArgoColors.quantitative
    argopy.plot.ArgoColors.definition
    argopy.plot.ArgoColors.cmap
    argopy.plot.ArgoColors.lookup
    argopy.plot.ArgoColors.ticklabels
    argopy.plot.ArgoColors.list_valid_known_colormaps

    argopy.stores.spec.ArgoStoreProto

    argopy.stores.implementations.local.filestore
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

    argopy.stores.implementations.http.httpstore
    argopy.stores.httpstore.download_url
    argopy.stores.httpstore.open_json
    argopy.stores.httpstore.open_mfjson
    argopy.stores.httpstore.open_dataset
    argopy.stores.httpstore.open_mfdataset
    argopy.stores.httpstore.read_csv
    argopy.stores.httpstore.open
    argopy.stores.httpstore.glob
    argopy.stores.httpstore.exists
    argopy.stores.httpstore.store_path
    argopy.stores.httpstore.register
    argopy.stores.httpstore.cachepath
    argopy.stores.httpstore.clear_cache

    argopy.stores.implementations.memory.memorystore
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

    argopy.stores.implementations.ftp.ftpstore
    argopy.stores.ftpstore.open_dataset
    argopy.stores.ftpstore.open_mfdataset

    argopy.stores.implementations.http_erddap.httpstore_erddap_auth
    argopy.stores.httpstore_erddap_auth.get_auth_client
    argopy.stores.httpstore_erddap_auth.connect
    argopy.stores.httpstore_erddap_auth.connected
    argopy.stores.httpstore_erddap_auth.open
    argopy.stores.httpstore_erddap_auth.glob
    argopy.stores.httpstore_erddap_auth.exists
    argopy.stores.httpstore_erddap_auth.store_path
    argopy.stores.httpstore_erddap_auth.register
    argopy.stores.httpstore_erddap_auth.cachepath
    argopy.stores.httpstore_erddap_auth.clear_cache
    argopy.stores.httpstore_erddap_auth.open_mfdataset
    argopy.stores.httpstore_erddap_auth.open_mfjson

    argopy.stores.implementations.http_erddap.httpstore_erddap

    argopy.stores.implementations.s3.s3store
    argopy.stores.s3store.open_json
    argopy.stores.s3store.open_dataset
    argopy.stores.s3store.read_csv
    argopy.stores.s3store.open
    argopy.stores.s3store.glob
    argopy.stores.s3store.exists
    argopy.stores.s3store.store_path
    argopy.stores.s3store.register
    argopy.stores.s3store.cachepath
    argopy.stores.s3store.clear_cache
    argopy.stores.s3store.open_mfdataset
    argopy.stores.s3store.open_mfjson

    argopy.stores.implementations.gdac.gdacfs
    argopy.gdacfs

    argopy.stores.ArgoKerchunker
    argopy.stores.ArgoKerchunker.supported
    argopy.stores.ArgoKerchunker.translate
    argopy.stores.ArgoKerchunker.nc2reference
    argopy.stores.ArgoKerchunker.to_reference
    argopy.stores.ArgoKerchunker.pprint

    argopy.stores.index.spec.ArgoIndexStoreProto

    argopy.stores.ArgoIndex
    argopy.ArgoIndex
    argopy.ArgoIndex.N_MATCH
    argopy.ArgoIndex.N_RECORDS
    argopy.ArgoIndex.N_FILES
    argopy.ArgoIndex.convention_supported
    argopy.ArgoIndex.load

    argopy.ArgoIndex.read_wmo
    argopy.ArgoIndex.read_dac_wmo
    argopy.ArgoIndex.read_domain
    argopy.ArgoIndex.read_params
    argopy.ArgoIndex.read_files
    argopy.ArgoIndex.records_per_wmo

    argopy.ArgoIndex.to_dataframe
    argopy.ArgoIndex.to_indexfile
    argopy.ArgoIndex.copy
    argopy.ArgoIndex.iterfloats
    argopy.ArgoIndex.uri
    argopy.ArgoIndex.uri_full_index
    argopy.ArgoIndex.files
    argopy.ArgoIndex.files_full_index

    argopy.ArgoIndex.query
    argopy.ArgoIndex.query.wmo
    argopy.ArgoIndex.query.cyc
    argopy.ArgoIndex.query.lon
    argopy.ArgoIndex.query.lat
    argopy.ArgoIndex.query.date
    argopy.ArgoIndex.query.params
    argopy.ArgoIndex.query.parameter_data_mode
    argopy.ArgoIndex.query.profiler_type
    argopy.ArgoIndex.query.profiler_label
    argopy.ArgoIndex.query.institution_code
    argopy.ArgoIndex.query.institution_name
    argopy.ArgoIndex.query.dac

    argopy.ArgoIndex.query.wmo_cyc
    argopy.ArgoIndex.query.lon_lat
    argopy.ArgoIndex.query.box
    argopy.ArgoIndex.query.compose

    argopy.ArgoIndex.plot
    argopy.ArgoIndex.plot.trajectory
    argopy.ArgoIndex.plot.bar

    argopy.stores.index.implementations.index_s3.s3index
    argopy.stores.index.implementations.index_s3.s3index_core
    argopy.stores.index.implementations.index_s3.s3index_bgc_bio
    argopy.stores.index.implementations.index_s3.s3index_bgc_synthetic
    argopy.stores.index.implementations.index_s3.search_s3

    argopy.xarray.ArgoAccessor.point2profile
    argopy.xarray.ArgoAccessor.profile2point
    argopy.xarray.ArgoAccessor.interp_std_levels
    argopy.xarray.ArgoAccessor.groupby_pressure_bins
    argopy.xarray.ArgoAccessor.teos10
    argopy.xarray.ArgoAccessor.create_float_source
    argopy.xarray.ArgoAccessor.filter_qc
    argopy.xarray.ArgoAccessor.filter_scalib_pres
    argopy.xarray.ArgoAccessor.filter_researchmode
    argopy.xarray.ArgoAccessor.cast_types
    argopy.xarray.ArgoAccessor.index
    argopy.xarray.ArgoAccessor.domain
    argopy.xarray.ArgoAccessor.list_WMO_CYC
    argopy.xarray.ArgoAccessor.N_POINTS
    argopy.xarray.ArgoAccessor.N_PROF
    argopy.xarray.ArgoAccessor.to_zarr
    argopy.xarray.ArgoAccessor.reduce_profile

    argopy.xarray.ArgoEngine

    argopy.extensions.register_argo_accessor
    argopy.extensions.ArgoAccessorExtension

    argopy.extensions.CanyonMED
    argopy.extensions.CanyonMED.predict
    argopy.extensions.CanyonMED.input_list
    argopy.extensions.CanyonMED.output_list

    argopy.extensions.CanyonB
    argopy.extensions.CanyonB.predict
    argopy.extensions.CanyonB.input_list
    argopy.extensions.CanyonB.output_list

    argopy.extensions.CONTENT
    argopy.extensions.CONTENT.predict
    argopy.extensions.CONTENT.input_list
    argopy.extensions.CONTENT.output_list

    argopy.extensions.ParamsDataMode.compute
    argopy.extensions.ParamsDataMode.merge
    argopy.extensions.ParamsDataMode.filter
    argopy.extensions.ParamsDataMode.split

    argopy.extensions.OpticalModeling
    argopy.extensions.OpticalModeling.Zeu
    argopy.extensions.OpticalModeling.Zpd
    argopy.extensions.OpticalModeling.Z_iPAR_threshold
    argopy.extensions.OpticalModeling.DCM

    argopy.errors.InvalidDatasetStructure

    argopy.stores.float.spec.FloatStoreProto
    argopy.stores.ArgoFloat
    argopy.ArgoFloat.open_dataset
    argopy.ArgoFloat.ls_dataset
    argopy.ArgoFloat.path
    argopy.ArgoFloat.ls
    argopy.ArgoFloat.lsprofiles
    argopy.ArgoFloat.describe_profiles
    argopy.ArgoFloat.metadata
    argopy.ArgoFloat.N_CYCLES
    argopy.ArgoFloat.dac

    argopy.stores.ArgoFloat.plot
    argopy.stores.ArgoFloat.plot.trajectory
    argopy.stores.ArgoFloat.plot.map
    argopy.stores.ArgoFloat.plot.scatter
    argopy.ArgoFloat.plot
    argopy.ArgoFloat.plot.trajectory
    argopy.ArgoFloat.plot.scatter
    argopy.ArgoFloat.plot.map

    argopy.stores.ArgoFloat.config
    argopy.stores.ArgoFloat.config.n_params
    argopy.stores.ArgoFloat.config.parameters
    argopy.stores.ArgoFloat.config.n_missions
    argopy.stores.ArgoFloat.config.missions
    argopy.stores.ArgoFloat.config.cycles
    argopy.stores.ArgoFloat.config.for_cycles
    argopy.stores.ArgoFloat.config.to_dataframe
    argopy.ArgoFloat.config
    argopy.ArgoFloat.config.n_params
    argopy.ArgoFloat.config.parameters
    argopy.ArgoFloat.config.n_missions
    argopy.ArgoFloat.config.missions
    argopy.ArgoFloat.config.cycles
    argopy.ArgoFloat.config.for_cycles
    argopy.ArgoFloat.config.to_dataframe

    argopy.stores.ArgoFloat.launchconfig
    argopy.stores.ArgoFloat.launchconfig.n_params
    argopy.stores.ArgoFloat.launchconfig.parameters
    argopy.stores.ArgoFloat.launchconfig.to_dataframe
    argopy.ArgoFloat.launchconfig
    argopy.ArgoFloat.launchconfig.n_params
    argopy.ArgoFloat.launchconfig.parameters
    argopy.ArgoFloat.launchconfig.to_dataframe

    argopy.reference.concept.ArgoReferenceValue
    argopy.ArgoReferenceValue.from_urn
    argopy.ArgoReferenceValue.from_dict
    argopy.ArgoReferenceValue.to_json
    argopy.ArgoReferenceValue.nvs

    argopy.reference.vocabulary.ArgoReferenceTable
    argopy.ArgoReferenceTable.valid_identifier
    argopy.ArgoReferenceTable.search
    argopy.ArgoReferenceTable.from_urn
    argopy.ArgoReferenceTable.to_dataframe
    argopy.ArgoReferenceTable.to_dict
    argopy.ArgoReferenceTable.nvs

    argopy.reference.mapping.ArgoReferenceMapping
    argopy.ArgoReferenceMapping.subjects
    argopy.ArgoReferenceMapping.objects
    argopy.ArgoReferenceMapping.predicates
    argopy.ArgoReferenceMapping.to_dataframe
    argopy.ArgoReferenceMapping.nvs
