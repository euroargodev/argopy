import pandas as pd
from ..options import OPTIONS
from ..utils.checkers import check_wmo, check_cyc
from ..stores import httpstore


def get_coriolis_profile_id(WMO, CYC=None, **kwargs) -> pd.DataFrame:
    """Get Coriolis ID of WMO/CYC

        Return a :class:`pandas.DataFrame` with CORIOLIS ID for WMO/CYC profile pairs

        This method get ID using the https://dataselection.euro-argo.eu trajectory API.

        Parameters
        ----------
        WMO: int, list(int)
            Define the list of Argo floats. This is a list of integers with WMO float identifiers.
            WMO is the World Meteorological Organization.
        CYC: int, list(int)
            Define the list of cycle numbers to load ID for each Argo floats listed in ``WMO``.

        Returns
        -------
        :class:`pandas.DataFrame`
    """
    WMO_list = check_wmo(WMO)
    if CYC is not None:
        CYC_list = check_cyc(CYC)
    if 'api_server' in kwargs:
        api_server = kwargs['api_server']
    elif OPTIONS['server'] is not None:
        api_server = OPTIONS['server']
    else:
        api_server = "https://dataselection.euro-argo.eu/api"
    URIs = [api_server + "/trajectory/%i" % wmo for wmo in WMO_list]

    def prec(data, url):
        # Transform trajectory json to dataframe
        # See: https://dataselection.euro-argo.eu/swagger-ui.html#!/cycle-controller/getCyclesByPlatformCodeUsingGET
        WMO = check_wmo(url.split("/")[-1])[0]
        rows = []
        for profile in data:
            keys = [x for x in profile.keys() if x not in ["coordinate"]]
            meta_row = dict((key, profile[key]) for key in keys)
            for row in profile["coordinate"]:
                meta_row[row] = profile["coordinate"][row]
            meta_row["WMO"] = WMO
            rows.append(meta_row)
        return pd.DataFrame(rows)

    fs = httpstore(cache=True, cachedir=OPTIONS['cachedir'])
    data = fs.open_mfjson(URIs, preprocess=prec, errors="raise", url_follow=True)

    # Merge results (list of dataframe):
    key_map = {
        "id": "ID",
        "lat": "LATITUDE",
        "lon": "LONGITUDE",
        "cvNumber": "CYCLE_NUMBER",
        "level": "level",
        "WMO": "PLATFORM_NUMBER",
    }
    for i, df in enumerate(data):
        df = df.reset_index()
        df = df.rename(columns=key_map)
        df = df[[value for value in key_map.values() if value in df.columns]]
        data[i] = df
    df = pd.concat(data, ignore_index=True)
    df.sort_values(by=["PLATFORM_NUMBER", "CYCLE_NUMBER"], inplace=True)
    df = df.reset_index(drop=True)
    # df = df.set_index(["PLATFORM_NUMBER", "CYCLE_NUMBER"])
    df = df.astype({"ID": int})
    if CYC is not None:
        df = pd.concat([df[df["CYCLE_NUMBER"] == cyc] for cyc in CYC_list]).reset_index(
            drop=True
        )
    return df[
        ["PLATFORM_NUMBER", "CYCLE_NUMBER", "ID", "LATITUDE", "LONGITUDE", "level"]
    ]


def get_ea_profile_page(WMO, CYC=None, **kwargs):
    """ Return a list of URL

        Parameters
        ----------
        WMO: int, list(int)
            WMO must be an integer or an iterable with elements that can be casted as integers
        CYC: int, list(int), default (None)
            CYC must be an integer or an iterable with elements that can be casted as positive integers

        Returns
        -------
        list(str)

        See also
        --------
        get_coriolis_profile_id
    """
    df = get_coriolis_profile_id(WMO, CYC, **kwargs)
    url = "https://dataselection.euro-argo.eu/cycle/{}"
    return [url.format(this_id) for this_id in sorted(df["ID"])]
