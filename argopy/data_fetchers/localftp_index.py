"""Argo index fetcher for a local copy of GDAC ftp.

This is not intended to be used directly, only by the facade at fetchers.py

Deprecation cycle:
Warning: 0.1.11
Error:   0.1.12
Delete:  0.1.13

"""
import os
from abc import ABC, abstractmethod
import warnings
from packaging import version

from ..utilities import load_dict, mapp_dict, format_oneline, deprecated
from ..options import OPTIONS, check_gdac_path
from ..stores import indexstore, indexfilter_wmo, indexfilter_box
from ..errors import InvalidOption
from .. import __version__


access_points = ['wmo', 'box']
exit_formats = ['xarray', 'dataframe']
dataset_ids = ['phy', 'bgc']  # First is default


class LocalFTPArgoIndexFetcher(ABC):
    """ Manage access to Argo index from a local copy of GDAC ftp

    .. warning:

        This fetcher is deprecated. It's been replaced by the ``gdac`` fetcher.

        ================= ======
        Deprecation cycle
        ================= ======
        Warning           0.1.11
        Error             0.1.12
        Delete            0.1.13
        ================= ======

    """
    ###
    # Methods to be customised for a specific request
    ###
    @abstractmethod
    def init(self):
        """ Initialisation for a specific fetcher """
        raise NotImplementedError("Not implemented")

    @property
    def cachepath(self):
        return self.fs.cachepath(self.fcls.uri)

    def cname(self):
        """ Return a unique string defining the request

        """
        return self.fcls.uri

    def filter_index(self):
        """ Return an index filter

        Parameters
        ----------
        index_file: _io.TextIOWrapper

        Returns
        -------
        csv rows matching the request, as a in-memory string. Or None.
        """
        return self.fcls

    @deprecated("The 'localftp' data source is deprecated. It's been replaced by 'gdac'. It will raise an error after argopy 0.1.12")
    def __init__(self,
                 local_ftp: str = "",
                 ds: str = "",
                 cache: bool = False,
                 cachedir: str = "",
                 **kwargs):
        """ Init fetcher

            Parameters
            ----------
            local_path : str
                Path to the directory with the 'dac' folder and index file
            ds: str (optional)
                Dataset to load: 'phy' or 'bgc'

            .. deprecated:: 0.1.11
                `index_file` will be removed in argopy 0.1.13.
        """
        if 'index_file' in kwargs:
            if version.parse(__version__) > version.parse("0.1.11"):
                raise InvalidOption("Invalid option `index_file`")
            else:
                #todo Update version with appropriate release for this PR
                warnings.warn("'index_file option' is deprecated since 0.1.11: the name of the index file is now "
                              "internally determined as a "
                              "function of the 'ds' dataset option. ", DeprecationWarning)

        self.cache = cache
        self.definition = 'Local ftp Argo index fetcher'
        self.dataset_id = OPTIONS["dataset"] if ds == "" else ds
        self.local_ftp = OPTIONS['local_ftp'] if local_ftp == '' else local_ftp
        # Validate server, raise FtpPathError if not valid.
        check_gdac_path(self.local_ftp, errors='raise')

        if self.dataset_id == 'phy':
            self.index_file = "ar_index_global_prof.txt"
        elif self.dataset_id == 'bgc':
            self.index_file = "argo_synthetic-profile_index.txt"
        self.fs = indexstore(cache, cachedir, os.path.sep.join([self.local_ftp, self.index_file]))
        self.init(**kwargs)

    def __repr__(self):
        summary = ["<indexfetcher.localftp>"]
        summary.append("Name: %s" % self.definition)
        summary.append("Index: %s" % self.index_file)
        summary.append("FTP: %s" % self.local_ftp)
        summary.append("Domain: %s" % format_oneline(self.cname()))
        return '\n'.join(summary)

    def to_dataframe(self):
        """ filter local index file and return a pandas dataframe """
        df = self.fs.read_csv(self.filter_index())

        # Post-processing of the filtered index:
        df['wmo'] = df['file'].apply(lambda x: int(x.split('/')[1]))

        # institution & profiler mapping for all users
        # todo: may be we need to separate this for standard and expert users
        institution_dictionnary = load_dict('institutions')
        df['tmp1'] = df.institution.apply(lambda x: mapp_dict(institution_dictionnary, x))
        df = df.rename(columns={"institution": "institution_code", "tmp1": "institution"})

        profiler_dictionnary = load_dict('profilers')
        df['profiler'] = df.profiler_type.apply(lambda x: mapp_dict(profiler_dictionnary, int(x)))
        df = df.rename(columns={"profiler_type": "profiler_code"})

        return df

    def to_xarray(self):
        """ Load Argo index and return a xarray Dataset """
        return self.to_dataframe().to_xarray()

    def clear_cache(self):
        """ Remove cache files and entries from resources open with this fetcher """
        return self.fs.clear_cache()


class Fetch_wmo(LocalFTPArgoIndexFetcher):
    """ Manage access to local ftp Argo data for: a list of WMOs

    """
    def init(self, WMO: list = [], CYC=None, **kwargs):
        """ Create Argo data loader for WMOs

            Parameters
            ----------
            WMO : list(int)
                The list of WMOs to load all Argo data for.
            CYC : int, np.array(int), list(int)
                The cycle numbers to load.
        """
        self.WMO = WMO
        self.CYC = CYC
        self.fcls = indexfilter_wmo(self.WMO, self.CYC)


class Fetch_box(LocalFTPArgoIndexFetcher):
    """ Manage access to local ftp Argo data for: an ocean rectangle

    """
    def init(self, box: list = [-180, 180, -90, 90, '1900-01-01', '2100-12-31'], **kwargs):
        """ Create Argo index loader

            Parameters
            ----------
            box: list()
                Define the domain to load Argo index for. The box list is made of:
                    - lon_min: float, lon_max: float,
                    - lat_min: float, lat_max: float,
                    - date_min: str (optional), date_max: str (optional)

                Longitude and latitude bounds are required, while the two bounding dates are optional.
                If bounding dates are not specified, the entire time series is fetched.
                Eg: [-60, -55, 40., 45., '2007-08-01', '2007-09-01']
        """
        self.BOX = box.copy()
        self.fcls = indexfilter_box(self.BOX)
