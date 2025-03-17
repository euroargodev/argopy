"""
Use this module directly:
    >>> import argopy.dashboard as dashboard

Or use the methods on a DataFetcher or IndexFetcher:
    >>> DataFetcher().float(**).dashboard()
    >>> DataFetcher().profile(**).dashboard()
"""
import os
import warnings
from packaging import version

from .utils import has_ipython
from ..utils.loggers import warnUnless
from ..related.euroargo_api import get_ea_profile_page
from ..utils import check_wmo, check_cyc
from ..errors import InvalidDashboard
from .. import __version__ as argopy_version


if has_ipython:
    from IPython.display import IFrame, Image


""" Define all dashboards

Use this dictionary to implement or modify a new 3rd party dashboard

"""
dashboard_definitions = {
    "data": {
        "shorts": ["ea"],
        "uri": {
            "base": "https://dataselection.euro-argo.eu",
            "wmo": "https://fleetmonitoring.euro-argo.eu/float/{}".format,
            "cyc": lambda wmo, cyc: get_ea_profile_page(wmo, cyc)[0],
        },
    },
    "meta": {
        "shorts": ["eric"],
        "uri": {
            "base": "https://fleetmonitoring.euro-argo.eu",
            "wmo": "https://fleetmonitoring.euro-argo.eu/float/{}".format,
            "cyc": lambda wmo, cyc: get_ea_profile_page(wmo, cyc)[0],
        },
    },
    "coriolis": {
        "shorts": ["cor"],
        "uri": {
            "base": None,
            "wmo": ("https://co-insitucharts.ifremer.fr/platform/{}/charts").format,
            "cyc": None,
        },
    },
    "argovis": {
        "shorts": [],
        "uri": {
            "base": "https://argovis.colorado.edu/argo",
            "wmo": lambda wmo: "https://argovis.colorado.edu/plots/argo?showAll=true&argoPlatform=%i" % wmo,
            "cyc": lambda wmo, cyc: "https://argovis.colorado.edu/plots/argo?argoPlatform=%i&counterTraces=[%%22%i_%0.3d%%22]" % (wmo, wmo, cyc),
        },
    },
    "ocean-ops": {
        "shorts": ["op"],
        "uri": {
            "base": None,
            "wmo": "https://www.ocean-ops.org/board/wa/Platform?ref={}".format,
            "cyc": None,
        },
    },
    "bgc": {
        "shorts": [],
        "uri": {
            "base": "https://maps.biogeochemical-argo.com/bgcargo",
            "wmo": "https://maps.biogeochemical-argo.com/bgcargo/?&txt={}".format,
            "cyc": lambda wmo, cyc: "https://maps.biogeochemical-argo.com/datamap/jpeg/%i_%0.3d.jpeg"
            % (wmo, cyc),
        },
    },
}


def get_valid_type(defs):
    """Return the list of all boards 'type', including shortcuts"""
    the_list = []
    for board in defs.keys():
        the_list.append(board)
        board_shorts = defs[board]["shorts"]
        if board_shorts is not None:
            [the_list.append(s) for s in board_shorts]
    return the_list


def get_type_name(defs, input_type):
    """Return full board type name, given type or shortcuts"""
    return "%s%s" % (
        "".join(
            [board for board in defs.keys() if input_type in defs[board]["shorts"]]
        ),
        "".join([board for board in defs.keys() if input_type == board]),
    )


def open_dashboard(
    wmo=None, cyc=None, type="ea", url_only=False, width="100%", height=1000,
):
    """ Insert an Argo dashboard page in a notebook cell, or return the corresponding url

        Parameters
        ----------
        wmo: int, optional
            The float WMO to display. By default, this is set to None and will insert the general dashboard.
        cyc: int, optional
            The float CYCLE NUMBER to display. If ``wmo`` is not None, this will open a profile dashboard.
        type: str, optional, default: "ea"
            Type of dashboard to use. This can be any one of the following:

            * "ea", "data": the `Euro-Argo data selection dashboard <https://dataselection.euro-argo.eu>`_
            * "meta": the `Euro-Argo fleet monitoring dashboard <https://fleetmonitoring.euro-argo.eu>`_
            * "op", "ocean-ops": the `Ocean-OPS Argo dashboard <https://www.ocean-ops.org/board?t=argo>`_
            * "bgc": the `Argo-BGC specific dashboard <https://maps.biogeochemical-argo.com/bgcargo>`_
            * "argovis": the `Colorado Argovis dashboard <https://argovis.colorado.edu>`_
        url_only: bool, optional, default: False
            If set to True, will only return the URL toward the dashboard
        width: str, optional, default: "100%"
            Width in percentage or pixel of the returned Iframe or Image
        height: int, optional, default: 1000
            Height in pixel of the returned Iframe or Image

        Returns
        -------
        str or :class:`IPython.display.IFrame` or :class:`IPython.display.Image`

        Examples
        --------
        Directly:
            >>> argopy.dashboard()
            >>> argopy.dashboard(6902745)
            >>> argopy.dashboard(6902745, 12)
            >>> argopy.dashboard(6902745, type='ocean-ops')
            >>> argopy.dashboard(6902745, 12, url_only=True)

        Or from a fetcher with the method ``dashboard``:
            >>> DataFetcher().float(6902745).dashboard()

    """
    warnUnless(has_ipython, "IPython not available, this will fail silently and return URLs to dashboards")
    # This function is 'generic', it consumes the dashboard_definitions dictionary defined above

    if type == "eric":
        type = "meta"
        if version.parse(argopy_version) < version.parse("0.1.13"):
            warnings.warn(
                "The 'eric' option has been replaced by 'meta'. After 0.1.13, this will raise an error.",
                category=DeprecationWarning,
                stacklevel=2,
            )

    if type not in get_valid_type(dashboard_definitions):
        raise InvalidDashboard(
            "Invalid dashboard type. %s not in %s"
            % (type, get_valid_type(dashboard_definitions))
        )

    URIs = dashboard_definitions[get_type_name(dashboard_definitions, type)]["uri"]
    url = URIs["base"] if URIs["base"] is not None else "?"
    if wmo is not None:
        wmo = check_wmo(wmo)[0]
        url = URIs["wmo"](wmo) if URIs["wmo"] is not None else "?"
        if cyc is not None:
            cyc = check_cyc(cyc)[0]
            url = URIs["cyc"](wmo, cyc) if URIs["cyc"] is not None else "?"

    if url == "?":
        raise InvalidDashboard(
            "Dashboard not available for this combination of wmo (%s), cyc (%s) and type (%s)"
            % (str(wmo), str(cyc), type)
        )

    insert = lambda url: url  # noqa: E731
    if has_ipython:
        filename, file_extension = os.path.splitext(url)
        if file_extension in [".jpeg"]:
            insert = lambda x: Image(url=x)  # noqa: E731
        else:
            insert = lambda x: IFrame(x, width=width, height=height)  # noqa: E731

    if url_only:
        return url
    else:
        return insert(url)
