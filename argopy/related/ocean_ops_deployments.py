import pandas as pd
import numpy as np
from ..stores import httpstore
from ..errors import DataNotFound
from ..plot import scatter_map


class OceanOPSDeployments:
    """Argo floats deployment plans

    Use the OceanOPS API for metadata access to retrieve Argo floats deployment information.

    The API is documented here: https://www.ocean-ops.org/api/swagger/?url=https://www.ocean-ops.org/api/1/oceanops-api.yaml

    Description of deployment status name:

    =========== == ====
    Status      Id Description
    =========== == ====
    PROBABLE    0  Starting status for some platforms, when there is only a few metadata available, like rough deployment location and date. The platform may be deployed
    CONFIRMED   1  Automatically set when a ship is attached to the deployment information. The platform is ready to be deployed, deployment is planned
    REGISTERED  2  Starting status for most of the networks, when deployment planning is not done. The deployment is certain, and a notification has been sent via the OceanOPS system
    OPERATIONAL 6  Automatically set when the platform is emitting a pulse and observations are distributed within a certain time interval
    INACTIVE    4  The platform is not emitting a pulse since a certain time
    CLOSED      5  The platform is not emitting a pulse since a long time, it is considered as dead
    =========== == ====

    Examples
    --------

    Import the class:

    >>> from argopy.related import OceanOPSDeployments
    >>> from argopy import OceanOPSDeployments

    Possibly define the space/time box to work with:

    >>> box = [-20, 0, 42, 51]
    >>> box = [-20, 0, 42, 51, '2020-01', '2021-01']
    >>> box = [-180, 180, -90, 90, '2020-01', None]

    Instantiate the metadata fetcher:

    >>> deployment = OceanOPSDeployments()
    >>> deployment = OceanOPSDeployments(box)
    >>> deployment = OceanOPSDeployments(box, deployed_only=True) # Remove planification

    Load information:

    >>> df = deployment.to_dataframe()
    >>> data = deployment.to_json()

    Useful attributes and methods:

    >>> deployment.uri
    >>> deployment.uri_decoded
    >>> deployment.status_code
    >>> fig, ax = deployment.plot_status()
    >>> plan_virtualfleet = deployment.plan

    """

    api = "https://www.ocean-ops.org"
    """URL to the API"""

    model = "api/1/data/platform"
    """This model represents a Platform entity and is used to retrieve a platform information (schema model
     named 'Ptf')."""

    api_server_check = "https://www.ocean-ops.org/api/1/oceanops-api.yaml"
    """URL to check if the API is alive"""

    def __init__(self, box: list = None, deployed_only: bool = False):
        """

        Parameters
        ----------
        box: list, optional, default=None
            Define the domain to load the Argo deployment plan for. By default, **box** is set to None to work with the
            global deployment plan starting from the current date.
            The list expects one of the following format:

            - [lon_min, lon_max, lat_min, lat_max]
            - [lon_min, lon_max, lat_min, lat_max, date_min]
            - [lon_min, lon_max, lat_min, lat_max, date_min, date_max]

            Longitude and latitude values must be floats. Dates are strings.
            If **box** is provided with a regional domain definition (only 4 values given), then ``date_min`` will be
            set to the current date.

        deployed_only: bool, optional, default=False
            Return only floats already deployed. If set to False (default), will return the full
            deployment plan (floats with all possible status). If set to True, will return only floats with one of the
            following status: ``OPERATIONAL``, ``INACTIVE``, and ``CLOSED``.
        """
        if box is None:
            box = [
                None,
                None,
                None,
                None,
                pd.to_datetime("now", utc=True).strftime("%Y-%m-%d"),
                None,
            ]
        elif len(box) == 4:
            box.append(pd.to_datetime("now", utc=True).strftime("%Y-%m-%d"))
            box.append(None)
        elif len(box) == 5:
            box.append(None)

        if len(box) != 6:
            raise ValueError(
                "The 'box' argument must be: None or of lengths 4 or 5 or 6\n%s"
                % str(box)
            )

        self.box = box
        self.deployed_only = deployed_only
        self.data = None

        self.fs = httpstore(cache=False)

    def __format(self, x, typ: str) -> str:
        """string formatting helper"""
        if typ == "lon":
            return str(x) if x is not None else "-"
        elif typ == "lat":
            return str(x) if x is not None else "-"
        elif typ == "tim":
            return pd.to_datetime(x).strftime("%Y-%m-%d") if x is not None else "-"
        else:
            return str(x)

    def __repr__(self):
        summary = ["<argo.deployment_plan>"]
        summary.append("API: %s/%s" % (self.api, self.model))
        summary.append("Domain: %s" % self.box_name)
        summary.append("Deployed only: %s" % self.deployed_only)
        if self.data is not None:
            summary.append("Nb of floats in the deployment plan: %s" % self.size)
        else:
            summary.append(
                "Nb of floats in the deployment plan: - [Data not retrieved yet]"
            )
        return "\n".join(summary)

    def __encode_inc(self, inc):
        """Return encoded uri expression for 'include' parameter

        Parameters
        ----------
        inc: str

        Returns
        -------
        str
        """
        return inc.replace('"', "%22").replace("[", "%5B").replace("]", "%5D")

    def __encode_exp(self, exp):
        """Return encoded uri expression for 'exp' parameter

        Parameters
        ----------
        exp: str

        Returns
        -------
        str
        """
        return (
            exp.replace('"', "%22")
            .replace("'", "%27")
            .replace(" ", "%20")
            .replace(">", "%3E")
            .replace("<", "%3C")
        )

    def __get_uri(self, encoded=False):
        uri = "exp=%s&include=%s" % (
            self.exp(encoded=encoded),
            self.include(encoded=encoded),
        )
        url = "%s/%s?%s" % (self.api, self.model, uri)
        return url

    def include(self, encoded=False):
        """Return an Ocean-Ops API 'include' expression

        This is used to determine which variables the API call should return

        Parameters
        ----------
        encoded: bool, default=False

        Returns
        -------
        str
        """
        # inc = ["ref", "ptfDepl.lat", "ptfDepl.lon", "ptfDepl.deplDate", "ptfStatus", "wmos"]
        # inc = ["ref", "ptfDepl.lat", "ptfDepl.lon", "ptfDepl.deplDate", "ptfStatus.id", "ptfStatus.name", "wmos"]
        # inc = ["ref", "ptfDepl.lat", "ptfDepl.lon", "ptfDepl.deplDate", "ptfStatus.id", "ptfStatus.name"]
        inc = [
            "ref",
            "ptfDepl.lat",
            "ptfDepl.lon",
            "ptfDepl.deplDate",
            "ptfStatus.id",
            "ptfStatus.name",
            "ptfStatus.description",
            "program.nameShort",
            "program.country.nameShort",
            "ptfModel.nameShort",
            "ptfDepl.noSite",
        ]
        inc = "[%s]" % ",".join(['"%s"' % v for v in inc])
        return inc if not encoded else self.__encode_inc(inc)

    def exp(self, encoded=False):
        """Return an Ocean-Ops API deployment search expression for an argopy region box definition

        Parameters
        ----------
        encoded: bool, default=False

        Returns
        -------
        str
        """
        exp, arg = "networkPtfs.network.name='Argo'", []
        if self.box[0] is not None:
            exp += " and ptfDepl.lon>=$var%i" % (len(arg) + 1)
            arg.append(str(self.box[0]))
        if self.box[1] is not None:
            exp += " and ptfDepl.lon<=$var%i" % (len(arg) + 1)
            arg.append(str(self.box[1]))
        if self.box[2] is not None:
            exp += " and ptfDepl.lat>=$var%i" % (len(arg) + 1)
            arg.append(str(self.box[2]))
        if self.box[3] is not None:
            exp += " and ptfDepl.lat<=$var%i" % (len(arg) + 1)
            arg.append(str(self.box[3]))
        if len(self.box) > 4:
            if self.box[4] is not None:
                exp += " and ptfDepl.deplDate>=$var%i" % (len(arg) + 1)
                arg.append(
                    '"%s"' % pd.to_datetime(self.box[4]).strftime("%Y-%m-%d %H:%M:%S")
                )
            if self.box[5] is not None:
                exp += " and ptfDepl.deplDate<=$var%i" % (len(arg) + 1)
                arg.append(
                    '"%s"' % pd.to_datetime(self.box[5]).strftime("%Y-%m-%d %H:%M:%S")
                )

        if self.deployed_only:
            exp += " and ptfStatus>=$var%i" % (len(arg) + 1)
            arg.append(str(4))  # Allow for: 4, 5 or 6

        exp = '["%s", %s]' % (exp, ", ".join(arg))
        return exp if not encoded else self.__encode_exp(exp)

    @property
    def size(self):
        return len(self.data["data"]) if self.data is not None else None

    @property
    def status_code(self):
        """Return a :class:`pandas.DataFrame` with the definition of status"""
        status = {
            "status_code": [0, 1, 2, 6, 4, 5],
            "status_name": [
                "PROBABLE",
                "CONFIRMED",
                "REGISTERED",
                "OPERATIONAL",
                "INACTIVE",
                "CLOSED",
            ],
            "description": [
                "Starting status for some platforms, when there is only a few metadata available, like rough deployment location and date. The platform may be deployed",
                "Automatically set when a ship is attached to the deployment information. The platform is ready to be deployed, deployment is planned",
                "Starting status for most of the networks, when deployment planning is not done. The deployment is certain, and a notification has been sent via the OceanOPS system",
                "Automatically set when the platform is emitting a pulse and observations are distributed within a certain time interval",
                "The platform is not emitting a pulse since a certain time",
                "The platform is not emitting a pulse since a long time, it is considered as dead",
            ],
        }
        return pd.DataFrame(status).set_index("status_code")

    @property
    def box_name(self):
        """Return a string to print the box property"""
        BOX = self.box
        cname = ("[lon=%s/%s; lat=%s/%s]") % (
            self.__format(BOX[0], "lon"),
            self.__format(BOX[1], "lon"),
            self.__format(BOX[2], "lat"),
            self.__format(BOX[3], "lat"),
        )
        if len(BOX) == 6:
            cname = ("[lon=%s/%s; lat=%s/%s; t=%s/%s]") % (
                self.__format(BOX[0], "lon"),
                self.__format(BOX[1], "lon"),
                self.__format(BOX[2], "lat"),
                self.__format(BOX[3], "lat"),
                self.__format(BOX[4], "tim"),
                self.__format(BOX[5], "tim"),
            )
        return cname

    @property
    def uri(self):
        """Return encoded URL to post an Ocean-Ops API request

        Returns
        -------
        str
        """
        return self.__get_uri(encoded=True)

    @property
    def uri_decoded(self):
        """Return decoded URL to post an Ocean-Ops API request

        Returns
        -------
        str
        """
        return self.__get_uri(encoded=False)

    @property
    def plan(self):
        """Return a dictionary to be used as argument in a :class:`virtualargofleet.VirtualFleet`

        This method is for dev, but will be moved to the VirtualFleet software utilities
        """
        df = self.to_dataframe()
        plan = (
            df[["lon", "lat", "date"]]
            .rename(columns={"date": "time"})
            .to_dict("series")
        )
        for key in plan.keys():
            plan[key] = plan[key].to_list()
        plan["time"] = np.array(plan["time"], dtype="datetime64")
        return plan

    def to_json(self):
        """Return OceanOPS API request response as a json object"""
        if self.data is None:
            self.data = self.fs.open_json(self.uri)
        return self.data

    def to_dataframe(self):
        """Return the deployment plan as :class:`pandas.DataFrame`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        data = self.to_json()
        if data["total"] == 0:
            raise DataNotFound("Your search matches no results")

        # res = {'date': [], 'lat': [], 'lon': [], 'wmo': [], 'status_name': [], 'status_code': []}
        # res = {'date': [], 'lat': [], 'lon': [], 'wmo': [], 'status_name': [], 'status_code': [], 'ship_name': []}
        res = {
            "date": [],
            "lat": [],
            "lon": [],
            "wmo": [],
            "status_name": [],
            "status_code": [],
            "program": [],
            "country": [],
            "model": [],
        }
        # status = {'REGISTERED': None, 'OPERATIONAL': None, 'INACTIVE': None, 'CLOSED': None,
        #           'CONFIRMED': None, 'OPERATIONAL': None, 'PROBABLE': None, 'REGISTERED': None}

        for irow, ptf in enumerate(data["data"]):
            # if irow == 0:
            # print(ptf)
            res["lat"].append(ptf["ptfDepl"]["lat"])
            res["lon"].append(ptf["ptfDepl"]["lon"])
            res["date"].append(ptf["ptfDepl"]["deplDate"])
            res["wmo"].append(ptf["ref"])
            # res['wmo'].append(ptf['wmos'][-1]['wmo'])
            # res['wmo'].append(float_wmo(ptf['ref'])) # will not work for some CONFIRMED, PROBABLE or REGISTERED floats
            # res['wmo'].append(float_wmo(ptf['wmos'][-1]['wmo']))
            res["status_code"].append(ptf["ptfStatus"]["id"])
            res["status_name"].append(ptf["ptfStatus"]["name"])

            # res['ship_name'].append(ptf['ptfDepl']['shipName'])
            program = (
                ptf["program"]["nameShort"].replace("_", " ")
                if ptf["program"]["nameShort"]
                else ptf["program"]["nameShort"]
            )
            res["program"].append(program)
            res["country"].append(ptf["program"]["country"]["nameShort"] if ptf["program"]["country"] is not None else None)
            res["model"].append(ptf["ptfModel"]["nameShort"])

            # if status[ptf['ptfStatus']['name']] is None:
            #     status[ptf['ptfStatus']['name']] = ptf['ptfStatus']['description']

        df = pd.DataFrame(res)
        df = df.astype({"date": "datetime64[s]"})
        df = df.sort_values(by="date").reset_index(drop=True)
        # df = df[ (df['status_name'] == 'CLOSED') | (df['status_name'] == 'OPERATIONAL')] # Select only floats that have been deployed and returned data
        # print(status)
        return df

    def plot_status(self, **kwargs):
        """Quick plot of the deployment plan

        Named arguments are passed to :class:`plot.scatter_map`

        Returns
        -------
        fig: :class:`matplotlib.figure.Figure`
        ax: :class:`matplotlib.axes.Axes`
        hdl: dict
        """
        df = self.to_dataframe()
        fig, ax, hdl = scatter_map(
            df,
            x="lon",
            y="lat",
            hue="status_code",
            traj=False,
            cmap="deployment_status",
            **kwargs
        )
        ax.set_title(
            "Argo network deployment plan\n%s\nSource: OceanOPS API as of %s"
            % (
                self.box_name,
                pd.to_datetime("now", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
            ),
            fontsize=12,
        )
        return fig, ax, hdl
