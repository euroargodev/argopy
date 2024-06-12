from .checkers import (
    is_box,
    is_indexbox,
    is_list_of_strings,
    is_list_of_dicts,
    is_list_of_datasets,
    is_list_equal,
    is_wmo,
    check_wmo,
    is_cyc,
    check_cyc,
    check_index_cols,
    check_gdac_path,
    isconnected,
    urlhaskeyword,  # noqa: F401
    isalive,
    isAPIconnected,
    erddap_ds_exists,
)
from .casting import DATA_TYPES, cast_Argo_variable_type, to_list
from .decorators import deprecated, doc_inherit
from .lists import (
    list_available_data_src,
    list_available_index_src,
    list_standard_variables,
    list_multiprofile_file_variables,
)
from .caching import clear_cache, lscache
from .monitored_threadpool import MyThreadPoolExecutor as MonitoredThreadPoolExecutor
from .chunking import Chunker
from .accessories import Registry, float_wmo
from .locals import (
    show_versions,
    show_options,
    modified_environ,
    get_sys_info,   # noqa: F401
    netcdf_and_hdf5_versions,  # noqa: F401
)
from .monitors import monitor_status, badge, fetch_status  # noqa: F401
from .geo import wmo2box, wrap_longitude, conv_lon, toYearFraction, YearFraction_to_datetime
from .compute import linear_interpolation_remap, groupby_remap
from .transform import (
    fill_variables_not_in_all_datasets,
    drop_variables_not_in_all_datasets,
)
from .format import argo_split_path, format_oneline, UriCName
from .loggers import warnUnless, log_argopy_callerstack

import importlib
path2assets = importlib.util.find_spec('argopy.static.assets').submodule_search_locations[0]


__all__ = (
    # Checkers:
    "is_box",
    "is_indexbox",
    "is_list_of_strings",
    "is_list_of_dicts",
    "is_list_of_datasets",
    "is_list_equal",
    "is_wmo",
    "check_wmo",
    "is_cyc",
    "check_cyc",
    "check_index_cols",
    "check_gdac_path",
    "isconnected",
    "isalive",
    "isAPIconnected",
    "erddap_ds_exists",
    # Data type casting:
    "DATA_TYPES",
    "cast_Argo_variable_type",
    "to_list",
    # Decorators:
    "deprecated",
    "doc_inherit",
    # Lists:
    "list_available_data_src",
    "list_available_index_src",
    "list_standard_variables",
    "list_multiprofile_file_variables",
    # Cache management:
    "clear_cache",
    "lscache",
    # Computation and performances:
    "MonitoredThreadPoolExecutor",
    "Chunker",
    # Accessories classes (specific objects):
    "Registry",
    "float_wmo",
    # Locals (environments, versions, systems):
    "show_versions",
    "show_options",
    "modified_environ",
    # Monitors
    "monitor_status",
    # Geo (space/time data utilities)
    "wmo2box",
    "wrap_longitude",
    "conv_lon",
    "toYearFraction",
    "YearFraction_to_datetime",
    # Computation with datasets:
    "linear_interpolation_remap",
    "groupby_remap",
    # Manipulate datasets:
    "fill_variables_not_in_all_datasets",
    "drop_variables_not_in_all_datasets",
    # Formatters:
    "format_oneline",
    "argo_split_path",
    "UriCName",
    # Loggers:
    "warnUnless",
    "log_argopy_callerstack",
)
