"""
BGC-Argo+ dataset store for :class:`argopy.ArgoFloat`.

The BGC-Argo+ dataset (https://www.bgc-argo-plus.info) is a quality-controlled,
outlier-removed version of BGC-Argo float data curated at SOEST / University of
Hawaiʻi at Mānoa (Bushinsky et al, 2025, Bushinsky et al, submitted).  Individual float files are served on the SOEST FTP server::

    ftp://ftp.soest.hawaii.edu/bgc_argo_plus/outliers_removed/

Usage
-----
The typical access path is through :class:`argopy.ArgoFloat`::

    from argopy import ArgoFloat
    ds = ArgoFloat(6903091).open_dataset('BGCArgoPlus')

You can also use the store directly::

    from argopy.stores.float.bgcargo_plus import BGCArgoPlusStore
    store = BGCArgoPlusStore(6903091)
    ds = store.open_dataset()
    url = store.url 
"""

from __future__ import annotations

import logging

import xarray as xr

from ...stores.implementations.ftp import ftpstore
from ...utils import check_wmo

log = logging.getLogger("argopy.stores.BGCArgoPlusStore")

#: Root of the BGC-Argo+ FTP tree (no trailing slash)
BGCARGO_PLUS_FTP_HOST = "ftp.soest.hawaii.edu"

#: Path template on the FTP server.
#: ``{version}`` can be overridden via :attr:`BGCArgoPlusStore.version`.
BGCARGO_PLUS_PATH_TEMPLATE = (
    "/bgc_argo_plus/outliers_removed/{version}/{wmo}_Sprof_BGCArgoPlus.nc" # for versions v0.1_2025_12 and earlier
    # "/bgc_argo_plus/outliers_removed/{version}/Individual_Floats/{wmo}_Sprof_BGCArgoPlus.nc" # for version v0.1_2026_04 and later, this is managed in the url function
)

#: Default version tag that maps to the latest production release on the FTP.
BGCARGO_PLUS_DEFAULT_VERSION = "v0.1_2026_04"
SUPPORTED_VERSIONS = {BGCARGO_PLUS_DEFAULT_VERSION, 'v0.1_2025_12', 'v0.0_2025_09'}


def bgcargo_plus_url(wmo: int, version: str = BGCARGO_PLUS_DEFAULT_VERSION) -> str:
    """Return the FTP URL for a BGC-Argo+ float file.

    Parameters
    ----------
    wmo : int
        Float WMO number.
    version : str, optional
        Dataset version string, e.g. ``"v0.1_2026_04"``.

    Returns
    -------
    str
        Full FTP URL

    Examples
    --------
    >>> from argopy.stores.float.bgcargo_plus import bgcargo_plus_url
    >>> bgcargo_plus_url(6903091)
    'ftp://ftp.soest.hawaii.edu/bgc_argo_plus/outliers_removed/v0.1_2026_04/6903091_Sprof_BGCArgoPlus.nc'
    """
    path = BGCARGO_PLUS_PATH_TEMPLATE.format(wmo=wmo, version=version) if version != 'v0.1_2026_04' else BGCARGO_PLUS_PATH_TEMPLATE.format(wmo=wmo, version=version+"/Individual_Floats")
    return f"ftp://{BGCARGO_PLUS_FTP_HOST}{path}"


class BGCArgoPlusStore:
    """Store that fetches BGC-Argo+ individual-float files from SOEST FTP.

    Parameters
    ----------
    wmo : int or str
        Float WMO number.
    version : str, optional
        BGC-Argo+ dataset version, default :data:`BGCARGO_PLUS_DEFAULT_VERSION`.
    cache : bool, optional
        Cache downloaded files locally (passed to :class:`ftpstore`).
    cachedir : str, optional
        Local cache directory (passed to :class:`ftpstore`).
    timeout : int, optional
        FTP connection timeout in seconds (passed to :class:`ftpstore`).

    Examples
    --------
    >>> from argopy.stores.float.bgcargo_plus import BGCArgoPlusStore
    >>> store = BGCArgoPlusStore(6903091)
    >>> ds = store.open_dataset()
    >>> store.url
    'ftp://ftp.soest.hawaii.edu/bgc_argo_plus/...'
    """

    def __init__(
        self,
        wmo: int | str,
        version: str = BGCARGO_PLUS_DEFAULT_VERSION,
        cache: bool = False,
        cachedir: str = "",
        timeout: int = 0,
    ):
        self.WMO = check_wmo(wmo)[0]
        self.version = version
        self.cache = cache
        self.cachedir = cachedir
        self.timeout = timeout

        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported BGC-Argo+ version '{self.version}'. "
                f"Supported versions: {sorted(SUPPORTED_VERSIONS)}"
            )

        # Build the FTP store pointing at the SOEST server
        ftp_root = f"{BGCARGO_PLUS_FTP_HOST}"
        self._fs = ftpstore(
            host=ftp_root,
            cache=self.cache,
            cachedir=self.cachedir if self.cachedir else None,
            timeout=self.timeout if self.timeout else None,
        )

    @property
    def url(self) -> str:
        """Full FTP URL of this float's BGC-Argo+ file."""
        return bgcargo_plus_url(self.WMO, version=self.version)

    def open_dataset(self, **kwargs) -> xr.Dataset:
        """Download and open the BGC-Argo+ netCDF file for this float.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`argopy.stores.ftpstore.open_dataset` (e.g. ``lazy=True``,
            ``xr_opts={"decode_times": False}``).

        Returns
        -------
        :class:`xarray.Dataset`

        Examples
        --------
        >>> from argopy.stores.float.bgcargo_plus import BGCArgoPlusStore
        >>> ds = BGCArgoPlusStore(6903091).open_dataset()
        """
        log.debug("BGCArgoPlusStore: fetching %s", self.url)
        try:
            ds = self._fs.open_dataset(self.url, **kwargs)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not retrieve BGC-Argo+ file for WMO {self.WMO} "
                f"(version='{self.version}') from {self.url}.\n"
                f"Original error: {exc}"
            ) from exc
        return ds

    def __repr__(self) -> str:
        return (
            f"<BGCArgoPlusStore>\n"
            f"  WMO     : {self.WMO}\n"
            f"  version : {self.version}\n"
            f"  URL     : {self.url}\n"
        )
