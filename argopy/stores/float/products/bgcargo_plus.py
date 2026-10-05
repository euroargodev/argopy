"""
BGC-Argo+ third-party product for :class:`argopy.ArgoFloat`.

.. warning::

    BGC-Argo+ is a **third-party, non-official** Argo-based product. It is not
    managed nor endorsed by the Argo Data Management Team (ADMT).

The BGC-Argo+ dataset (https://www.bgc-argo-plus.info) is a quality-controlled,
outlier-removed version of BGC-Argo float data curated at SOEST / University of
Hawaiʻi at Mānoa (Bushinsky et al, 2026, submitted: https://doi.org/10.5194/essd-2026-311).
Dataset reference: 10.5281/zenodo.19353191.

Individual float files are served on the SOEST FTP server::

    ftp://ftp.soest.hawaii.edu/bgc_argo_plus/outliers_removed/<version>/

Contact: Raphaël Bajon (rbajon@hawaii.edu).

Usage
-----
The typical access path is through :meth:`argopy.ArgoFloat.open_product`::

    from argopy import ArgoFloat
    ds = ArgoFloat(6903091).open_product('BGCArgoPlus')

You can also use the store directly::

    from argopy.stores.float.products.bgcargo_plus import BGCArgoPlusStore
    store = BGCArgoPlusStore(6903091)
    ds = store.open_dataset()
    url = store.url
    print(store)
"""

from __future__ import annotations

import ftplib
import logging
import re
import socket

import numpy as np
import xarray as xr

from argopy.errors import APIServerError
from argopy.stores.implementations.ftp import ftpstore
from argopy.utils.checkers import check_wmo


log = logging.getLogger("argopy.stores.BGCArgoPlusStore")

# Host of the BGC-Argo+ FTP server
BGCARGO_PLUS_FTP_HOST = "ftp.soest.hawaii.edu"

# Port of the BGC-Argo+ FTP server
BGCARGO_PLUS_FTP_PORT = 21

# Errors raised when the FTP server cannot be reached (refused, timed out, unknown host, busy)
_SERVER_ERRORS = (ConnectionError, TimeoutError, socket.gaierror, ftplib.error_temp)


class BGCArgoPlusServerError(APIServerError):
    """Raise when the BGC-Argo+ FTP server cannot be reached."""

    def __str__(self):
        return str(self.value)


def bgcargo_plus_server_available(timeout: float = 5) -> bool:
    """Check if the BGC-Argo+ FTP server accepts connections.

    Parameters
    ----------
    timeout : float, optional
        Connection timeout in seconds.

    Returns
    -------
    bool
    """
    try:
        with socket.create_connection((BGCARGO_PLUS_FTP_HOST, BGCARGO_PLUS_FTP_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def _server_error(exc: Exception) -> BGCArgoPlusServerError:
    return BGCArgoPlusServerError(
        f"Cannot reach the BGC-Argo+ server ftp://{BGCARGO_PLUS_FTP_HOST}:{BGCARGO_PLUS_FTP_PORT} "
        f"({type(exc).__name__}: {exc}). The server may be down or unreachable from your network, "
        f"please try again later."
    )

# Root folder on the FTP server
BGCARGO_PLUS_ROOT = "/bgc_argo_plus/outliers_removed"

# Path template on the FTP server for each supported dataset version
BGCARGO_PLUS_PATH_TEMPLATES = {
    "v0.0_2025_09": "/bgc_argo_plus/outliers_removed/v0.0_2025_09/{wmo}_Sprof_processed.nc",
    "v0.1_2025_12": "/bgc_argo_plus/outliers_removed/v0.1_2025_12/{wmo}_Sprof_BGCArgoPlus.nc",
    "v0.1_2026_04": "/bgc_argo_plus/outliers_removed/v0.1_2026_04/Individual_Floats/{wmo}_Sprof_BGCArgoPlus.nc",
    "v1.0_2026_08": "/bgc_argo_plus/outliers_removed/v1.0_2026_08/Individual_Floats/{wmo}_Sprof_BGCArgoPlus.nc",
}

# Path template 
BGCARGO_PLUS_PATH_TEMPLATE_NEW = "/bgc_argo_plus/outliers_removed/{version}/Individual_Floats/{wmo}_Sprof_BGCArgoPlus.nc"

# Version described in the reference paper: Bushinsky et al. (2026), https://doi.org/10.5194/essd-2026-311,
# with dataset DOI https://doi.org/10.5281/zenodo.19709012.
BGCARGO_PLUS_PAPER_VERSION = "v0.1_2026_04"

# Default version 
BGCARGO_PLUS_DEFAULT_VERSION = BGCARGO_PLUS_PAPER_VERSION

# Version folder names on the server
_VERSION_PATTERN = re.compile(r"^v(\d+)\.(\d+)_(\d{4})_(\d{2})$")


def _version_key(version: str) -> tuple:
    """Sort key for version tags: (major, minor, year, month)."""
    return tuple(int(x) for x in _VERSION_PATTERN.match(version).groups())


def bgcargo_plus_versions(timeout: int = 0) -> list[str]:
    """List the BGC-Argo+ dataset versions available on the FTP server.

    Parameters
    ----------
    timeout : int, optional
        FTP connection timeout in seconds.

    Returns
    -------
    list[str]
        Version tags sorted from the oldest to the most recent, e.g. ``['v0.0_2025_09', ..., 'v1.0_2026_08']``.

    Raises
    ------
    :class:`BGCArgoPlusServerError`
        If the server cannot be reached.

    Examples
    --------
    >>> from argopy.stores.float.products.bgcargo_plus import bgcargo_plus_versions
    >>> bgcargo_plus_versions()
    """
    try:
        fs = ftpstore(host=BGCARGO_PLUS_FTP_HOST, cache=False, timeout=timeout if timeout else None)
        entries = fs.ls(BGCARGO_PLUS_ROOT)
    except _SERVER_ERRORS as exc:
        raise _server_error(exc) from exc
    names = [(e["name"] if isinstance(e, dict) else e).rstrip("/").split("/")[-1] for e in entries]
    return sorted((n for n in names if _VERSION_PATTERN.match(n)), key=_version_key)


def bgcargo_plus_latest_version(timeout: int = 0) -> str:
    versions = bgcargo_plus_versions(timeout=timeout)
    if not versions:
        raise ValueError(f"No BGC-Argo+ version found on ftp://{BGCARGO_PLUS_FTP_HOST}{BGCARGO_PLUS_ROOT}")
    return versions[-1]


def resolve_bgcargo_plus_version(version: str = BGCARGO_PLUS_DEFAULT_VERSION, timeout: int = 0) -> str:
    """Return the version tag to use for a requested version.

    Parameters
    ----------
    version : str, optional
        One of:

        - ``"paper"``: the version from the reference paper, :data:`BGCARGO_PLUS_PAPER_VERSION`,
        - ``"latest"``: the most recent version available on the FTP server (requires a connection),
        - any version tag, e.g. ``"v1.0_2026_08"``.
    timeout : int, optional
        FTP connection timeout in seconds, only used with ``"latest"``.

    Returns
    -------
    str
    """
    if version == "paper":
        return BGCARGO_PLUS_PAPER_VERSION
    if version == "latest":
        return bgcargo_plus_latest_version(timeout=timeout)
    return version


def _decode_bytes_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Decode byte-string variables to str.

    Fixed-length string variables may be returned as numpy bytes dtype
    (``'|S...'``) instead of str.
    """
    updates = {}
    for name, var in ds.data_vars.items():
        if var.dtype.kind == "S":  # fixed-length bytes, e.g. dtype='|S64'
            decoded = np.char.decode(var.values, "utf-8")
            updates[name] = var.copy(data=decoded)
        elif var.dtype.kind == "O":  # object array, may contain variable-length bytes
            first = next(
                (
                    x
                    for x in var.values.flat
                    if x is not None and not (isinstance(x, float) and np.isnan(x))
                ),
                None,
            )
            if isinstance(first, bytes):
                decoded = np.vectorize(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )(var.values)
                updates[name] = var.copy(data=decoded)
    if updates:
        ds = ds.assign(updates)

    # PLATFORM_NUMBER is stored as a numeric string (e.g. '6903091') but uid() arithmetic requires it as an integer.
    if "PLATFORM_NUMBER" in ds.data_vars:
        try:
            pn = ds["PLATFORM_NUMBER"]
            ds = ds.assign(
                {
                    "PLATFORM_NUMBER": pn.copy(
                        data=np.char.strip(pn.values.astype(str)).astype(np.int64)
                    )
                }
            )
        except (ValueError, TypeError):
            pass  # leave as-is if conversion fails (e.g. non-numeric WMO)

    return ds


def bgcargo_plus_url(wmo: int, version: str = BGCARGO_PLUS_DEFAULT_VERSION, check: bool = True) -> str:
    """Return the FTP URL for a BGC-Argo+ float file.

    Parameters
    ----------
    wmo : int
        Float WMO number.
    version : str, optional
        Dataset version tag, e.g. ``"v0.1_2026_04"``, or ``"paper"`` or ``"latest"``,
        see :func:`resolve_bgcargo_plus_version`.
    check : bool, optional
        If True, only accept versions listed in :data:`BGCARGO_PLUS_PATH_TEMPLATES`. If False, a version
        tag unknown to this module (e.g. a new release on the server) is accepted and assumed to follow
        :data:`BGCARGO_PLUS_PATH_TEMPLATE_NEW`.

    Returns
    -------
    str
        Full FTP URL

    Examples
    --------
    >>> from argopy.stores.float.products.bgcargo_plus import bgcargo_plus_url
    >>> bgcargo_plus_url(6903091)
    'ftp://ftp.soest.hawaii.edu/bgc_argo_plus/outliers_removed/v0.1_2026_04/Individual_Floats/6903091_Sprof_BGCArgoPlus.nc'
    """
    if version == "latest":
        check = False  # The latest version on the server may be unknown to this module
    version = resolve_bgcargo_plus_version(version)
    if version in BGCARGO_PLUS_PATH_TEMPLATES:
        path = BGCARGO_PLUS_PATH_TEMPLATES[version].format(wmo=wmo)
    elif not check and _VERSION_PATTERN.match(version):
        log.debug("BGC-Argo+ version '%s' is not known by argopy, assuming the most recent file layout", version)
        path = BGCARGO_PLUS_PATH_TEMPLATE_NEW.format(version=version, wmo=wmo)
    else:
        raise _unsupported_version(version)
    return f"ftp://{BGCARGO_PLUS_FTP_HOST}{path}"


def _unsupported_version(version: str) -> ValueError:
    return ValueError(
        f"Unsupported BGC-Argo+ version '{version}'. "
        f"Supported versions: {sorted(BGCARGO_PLUS_PATH_TEMPLATES, key=_version_key)}, "
        f"or 'paper' ({BGCARGO_PLUS_PAPER_VERSION}) or 'latest' (most recent version on the server)."
    )


class BGCArgoPlusStore:
    """Store that fetches BGC-Argo+ individual-float files from SOEST FTP.

    Parameters
    ----------
    wmo : int or str
        Float WMO number.
    version : str, optional
        BGC-Argo+ dataset version, default :data:`BGCARGO_PLUS_DEFAULT_VERSION`. It can be:

        - ``"paper"``: the version from the reference paper, :data:`BGCARGO_PLUS_PAPER_VERSION`,
        - ``"latest"``: the most recent version available on the FTP server, looked up on first use,
        - any version tag from :data:`BGCARGO_PLUS_PATH_TEMPLATES`, e.g. ``"v1.0_2026_08"``.
    cache : bool, optional
        Cache downloaded files locally (passed to :class:`ftpstore`).
    cachedir : str, optional
        Local cache directory (passed to :class:`ftpstore`).
    timeout : int, optional
        FTP connection timeout in seconds (passed to :class:`ftpstore`).

    Examples
    --------
    >>> from argopy.stores.float.products.bgcargo_plus import BGCArgoPlusStore
    >>> store = BGCArgoPlusStore(6903091)
    >>> ds = store.open_dataset()
    >>> store.url
    'ftp://ftp.soest.hawaii.edu/bgc_argo_plus/...'
    >>> BGCArgoPlusStore(6903091, version='latest').version  # Most recent version on the server
    'v1.0_2026_08'
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
        if version not in ("paper", "latest") and version not in BGCARGO_PLUS_PATH_TEMPLATES:
            raise _unsupported_version(version)
        self.requested_version = version
        # 'latest' is resolved on first access to :attr:`version`, so that no connection is made here
        self._version = None if version == "latest" else resolve_bgcargo_plus_version(version)
        self.cache = cache
        self.cachedir = cachedir
        self.timeout = timeout
        self._server_available = None  # Checked on first access, see :attr:`server_available`

        # The FTP connection is only opened on first access, see :attr:`fs`
        self._fs = None

    @property
    def fs(self) -> ftpstore:
        """FTP file system connected to the BGC-Argo+ server.

        Raises
        ------
        :class:`BGCArgoPlusServerError`
            If the server cannot be reached.
        """
        if self._fs is None:
            try:
                self._fs = ftpstore(
                    host=BGCARGO_PLUS_FTP_HOST,
                    cache=self.cache,
                    cachedir=self.cachedir if self.cachedir else None,
                    timeout=self.timeout if self.timeout else None,
                )
            except _SERVER_ERRORS as exc:
                raise _server_error(exc) from exc
        return self._fs

    @property
    def server_available(self) -> bool:
        """Whether the BGC-Argo+ FTP server accepts connections.

        The server is checked once, on first access, with :func:`bgcargo_plus_server_available`.
        """
        if self._server_available is None:
            self._server_available = bgcargo_plus_server_available()
        return self._server_available

    @property
    def version(self) -> str:
        """BGC-Argo+ dataset version tag used by this store.

        With ``version='latest'``, the most recent version on the server is looked up once, on first access.
        """
        if self._version is None:
            self._version = bgcargo_plus_latest_version(timeout=self.timeout)
        return self._version

    @property
    def url(self) -> str:
        """Full FTP URL of this float's BGC-Argo+ file."""
        return bgcargo_plus_url(self.WMO, version=self.version, check=self.requested_version != "latest")

    def open_dataset(self, **kwargs) -> xr.Dataset:
        """Download and open the BGC-Argo+ netCDF file for this float.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`argopy.stores.ftpstore.open_dataset`.

        Returns
        -------
        :class:`xarray.Dataset`

        Raises
        ------
        :class:`FileNotFoundError`
            If the float is not part of this BGC-Argo+ version.
        :class:`BGCArgoPlusServerError`
            If the server cannot be reached.

        Examples
        --------
        >>> from argopy.stores.float.products.bgcargo_plus import BGCArgoPlusStore
        >>> ds = BGCArgoPlusStore(6903091).open_dataset()
        """
        log.debug("BGCArgoPlusStore: fetching %s", self.url)
        try:
            ds = _decode_bytes_dataset(self.fs.open_dataset(self.url, **kwargs))
        except _SERVER_ERRORS as exc:
            raise _server_error(exc) from exc
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
            f"  server available : {self.server_available}\n"
        )
