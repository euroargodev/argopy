from .topography import TopoFetcher
from .ocean_ops_deployments import OceanOPSDeployments
from .argo_documentation import ArgoDocs
from .doi_snapshot import ArgoDOI
from .euroargo_api import get_coriolis_profile_id, get_ea_profile_page
from .utils import load_dict, mapp_dict  # Must come last to avoid circular import, I know, not good
from argopy.reference import ArgoNVSReferenceTables  # For legacy import

#
__all__ = (
    # Classes :
    "TopoFetcher",
    "OceanOPSDeployments",
    "ArgoDocs",
    "ArgoDOI",

    # Functions :
    "get_coriolis_profile_id",
    "get_ea_profile_page",

    # Utilities :
    "load_dict",
    "mapp_dict",
)
