import aiohttp
import logging
from urllib.parse import urlparse, parse_qs
import copy

from ...options import OPTIONS
from ...errors import ErddapHTTPNotFound, ErddapHTTPUnauthorized
from .http import httpstore


log = logging.getLogger("argopy.stores.implementation.http_erddap")


class httpstore_erddap_auth(httpstore):
    """Argo http file system with authentication for erddap server

    Examples
    --------
    >>> fs = httpstore_erddap_auth(login='https://erddap.ifremer.fr/erddap/login.html', auto=False)
    >>> fs.connect()

    """
    async def get_auth_client(self, **kwargs):
        session = aiohttp.ClientSession(**kwargs)

        async with session.post(self._login_page, data=self._login_payload) as resp:
            resp_query = dict(parse_qs(urlparse(str(resp.url)).query))

            if resp.status == 404:
                raise ErddapHTTPNotFound(
                    "Error %s: %s. This erddap server does not support log-in"
                    % (resp.status, resp.reason)
                )

            elif resp.status == 200:
                has_expected = (
                    "message" in resp_query
                )  # only available when there is a form page response
                if has_expected:
                    message = resp_query["message"][0]
                    if "failed" in message:
                        caviard = copy(self._login_payload)
                        if 'password' in caviard:
                            caviard['password'] = '******'
                        raise ErddapHTTPUnauthorized(
                            "Error %i: %s (%s)" % (401, message, caviard)
                        )
                else:
                    raise ErddapHTTPUnauthorized(
                        "This erddap server does not support log-in with a user/password"
                    )

            else:
                log.debug("resp.status", resp.status)
                log.debug("resp.reason", resp.reason)
                log.debug("resp.headers", resp.headers)
                log.debug("resp.url", urlparse(str(resp.url)))
                log.debug("resp.url.query", resp_query)
                data = await resp.read()
                log.debug("data", data)

        return session

    def __init__(
        self,
        cache: bool = False,
        cachedir: str = "",
        login: str = None,
        auto: bool = True,
        **kwargs,
    ):
        if login is None:
            raise ValueError("Invalid login url")
        else:
            self._login_page = login

        self._login_auto = (
            auto  # Should we try to log-in automatically at instantiation ?
        )

        payload = kwargs.get(
            "payload", {"user": OPTIONS["user"], "password": OPTIONS["password"]}
        )
        self._login_payload = payload.copy()

        fsspec_kwargs = {**kwargs, **{"get_client": self.get_auth_client}}
        super().__init__(cache=cache, cachedir=cachedir, **fsspec_kwargs)

        if auto:
            assert isinstance(self.connect(), bool)

    # def __repr__(self):
    #     # summary = ["<httpstore_erddap_auth.%i>" % id(self)]
    #     summary = ["<httpstore_erddap_auth>"]
    #     summary.append("login page: %s" % self._login_page)
    #     summary.append("login data: %s" % (self._login_payload))
    #     if hasattr(self, '_connected'):
    #         summary.append("connected: %s" % (self._connected))
    #     else:
    #         summary.append("connected: ?")
    #     return "\n".join(summary)

    def _repr_html_(self):
        td_title = (
            lambda title: '<td colspan="2"><div style="vertical-align: middle;text-align:left"><strong>%s</strong></div></td>'
            % title
        )  # noqa: E731
        tr_title = lambda title: "<thead><tr>%s</tr></thead>" % td_title(  # noqa: E731
            title
        )
        a_link = lambda url, txt: '<a href="%s">%s</a>' % (url, txt)  # noqa: E731
        td_key = (  # noqa: E731
            lambda prop: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>'
            % str(prop)
        )
        td_val = (
            lambda label: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>'
            % str(label)
        )  # noqa: E731
        tr_tick = lambda key, value: "<tr>%s%s</tr>" % (  # noqa: E731
            td_key(key),
            td_val(value),
        )
        td_vallink = (
            lambda url, label: '<td style="border-width:0px;padding-left:10px;text-align:left">%s</td>'
            % a_link(url, label)
        )
        tr_ticklink = lambda key, url, value: "<tr>%s%s</tr>" % (  # noqa: E731
            td_key(key),
            td_vallink(url, value),
        )

        html = []
        html.append("<table style='border-collapse:collapse;border-spacing:0'>")
        html.append("<thead>")
        html.append(tr_title("httpstore_erddap_auth"))
        html.append("</thead>")
        html.append("<tbody>")
        html.append(tr_ticklink("login page", self._login_page, self._login_page))
        payload = self._login_payload.copy()
        payload["password"] = "*" * len(payload["password"])
        html.append(tr_tick("login data", payload))
        if hasattr(self, "_connected"):
            html.append(tr_tick("connected", "✅" if self._connected else "⛔"))
        else:
            html.append(tr_tick("connected", "?"))
        html.append("</tbody>")
        html.append("</table>")

        html = "\n".join(html)
        return html

    def connect(self) -> bool:
        """Try to connect to the login page and authenticate. Return connection result as bool"""
        try:
            payload = self._login_payload.copy()
            payload["password"] = "*" * len(payload["password"])
            self.fs.info(self._login_page)
            self._connected = True
        except ErddapHTTPUnauthorized:
            self._connected = False
        except:  # noqa: E722
            raise
        return self._connected

    @property
    def connected(self):
        if not hasattr(self, "_connected"):
            self.connect()
        return self._connected


def httpstore_erddap(url: str = "", cache: bool = False, cachedir: str = "", **kwargs):
    """Argo http file system that automatically try to login to an Erddap server"""
    erddap = OPTIONS["erddap"] if url == "" else url
    login_page = "%s/login.html" % erddap.rstrip("/")
    login_store = httpstore_erddap_auth(
        cache=cache, cachedir=cachedir, login=login_page, auto=False, **kwargs
    )
    try:
        login_store.connect()
        keep = True
    except ErddapHTTPNotFound:
        keep = False
        pass

    if keep:
        return login_store
    else:
        return httpstore(cache=cache, cachedir=cachedir, **kwargs)
