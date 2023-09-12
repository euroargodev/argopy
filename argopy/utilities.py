import warnings
import importlib
import inspect
from functools import wraps

warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


def refactored(func1):

    rel = importlib.import_module('argopy.related')
    utils = importlib.import_module('argopy.utils')
    in_related = hasattr(rel, func1.__name__)
    func2 = getattr(rel, func1.__name__) if in_related else getattr(utils, func1.__name__)

    func1_type = 'function'
    if inspect.isclass(func1):
        func1_type = 'class'

    func2_loc = 'utils'
    if in_related:
        func2_loc = 'related'

    msg = "The 'argopy.utilities.{name}' {ftype} has moved to 'argopy.{where}.{name}'. \
You're seeing this message because you called '{name}' imported from 'argopy.utilities'. \
Please update your script to import '{name}' from 'argopy.{where}'. \
After 0.1.15, importing 'utilities' will raise an error."

    @wraps(func1)
    def decorator(*args, **kwargs):
        # warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(
            msg.format(name=func1.__name__, ftype=func1_type, where=func2_loc),
            category=DeprecationWarning,
            stacklevel=2
        )
        # warnings.simplefilter('default', DeprecationWarning)
        return func2(*args, **kwargs)

    return decorator

# Argo related dataset and Meta-data fetchers

@refactored
class TopoFetcher:
    pass

@refactored
class ArgoDocs:
    pass

@refactored
class ArgoNVSReferenceTables:
    pass

@refactored
class OceanOPSDeployments:
    pass

@refactored
def get_coriolis_profile_id(*args, **kwargs):
    pass

@refactored
def get_ea_profile_page(*args, **kwargs):
    pass

@refactored
def load_dict(*args, **kwargs):
    pass

@refactored
def mapp_dict(*args, **kwargs):
    pass

# Checkers
@refactored
def is_box(*args, **kwargs):
    pass

@refactored
def is_indexbox(*args, **kwargs):
    pass

@refactored
def is_list_of_strings(*args, **kwargs):
    pass

@refactored
def is_list_of_dicts(*args, **kwargs):
    pass

@refactored
def is_list_of_datasets(*args, **kwargs):
    pass

@refactored
def is_list_equal(*args, **kwargs):
    pass

@refactored
def check_wmo(*args, **kwargs):
    pass

@refactored
def is_wmo(*args, **kwargs):
    pass

@refactored
def check_cyc(*args, **kwargs):
    pass

@refactored
def is_cyc(*args, **kwargs):
    pass

@refactored
def check_index_cols(*args, **kwargs):
    pass

@refactored
def check_gdac_path(*args, **kwargs):
    pass

@refactored
def isconnected(*args, **kwargs):
    pass

@refactored
def isalive(*args, **kwargs):
    pass

@refactored
def isAPIconnected(*args, **kwargs):
    pass

@refactored
def erddap_ds_exists(*args, **kwargs):
    pass

@refactored
def urlhaskeyword(*args, **kwargs):
    pass


# Data type casting:

@refactored
def to_list(*args, **kwargs):
    pass

@refactored
def cast_Argo_variable_type(*args, **kwargs):
    pass


# Decorators

@refactored
def deprecated(*args, **kwargs):
    pass

@refactored
def doc_inherit(*args, **kwargs):
    pass

# Lists:

@refactored
def list_available_data_src(*args, **kwargs):
    pass

@refactored
def list_available_index_src(*args, **kwargs):
    pass

@refactored
def list_standard_variables(*args, **kwargs):
    pass

@refactored
def list_multiprofile_file_variables(*args, **kwargs):
    pass

# Cache management:
@refactored
def clear_cache(*args, **kwargs):
    pass

@refactored
def lscache(*args, **kwargs):
    pass

# Computation and performances:
@refactored
class Chunker:
    pass

# Accessories classes (specific objects):
@refactored
class float_wmo:
    pass

@refactored
class Registry:
    pass

# Locals (environments, versions, systems):
@refactored
def get_sys_info(*args, **kwargs):
    pass

@refactored
def netcdf_and_hdf5_versions(*args, **kwargs):
    pass

@refactored
def show_versions(*args, **kwargs):
    pass

@refactored
def show_options(*args, **kwargs):
    pass

@refactored
def modified_environ(*args, **kwargs):
    pass


# Monitors
@refactored
def badge(*args, **kwargs):
    pass

@refactored
class fetch_status:
    pass

@refactored
class monitor_status:
    pass

# Geo (space/time data utilities)
@refactored
def toYearFraction(*args, **kwargs):
    pass

@refactored
def YearFraction_to_datetime(*args, **kwargs):
    pass

@refactored
def wrap_longitude(*args, **kwargs):
    pass

@refactored
def wmo2box(*args, **kwargs):
    pass

# Computation with datasets:
@refactored
def linear_interpolation_remap(*args, **kwargs):
    pass

@refactored
def groupby_remap(*args, **kwargs):
    pass

# Manipulate datasets:
@refactored
def drop_variables_not_in_all_datasets(*args, **kwargs):
    pass

@refactored
def fill_variables_not_in_all_datasets(*args, **kwargs):
    pass

# Formatters:
@refactored
def format_oneline(*args, **kwargs):
    pass

@refactored
def argo_split_path(*args, **kwargs):
    pass


# Loggers
@refactored
def warnUnless(*args, **kwargs):
    pass

@refactored
def log_argopy_callerstack(*args, **kwargs):
    pass

if __name__ == "argopy.utilities":
    warnings.warn(
        "The 'argopy.utilities' has moved to 'argopy.utils'. After 0.1.15, importing 'utilities' "
        "will raise an error. Please update your script.",
        category=DeprecationWarning,
        stacklevel=2,
    )

