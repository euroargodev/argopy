import xarray as xr
import os

from argopy.utils.arco.errors import NetCDF4FileNotFoundError


class Loader(object):
    """
    A generic loader class based on xarray.
    If it can't find a file, it raises a specific error for easy catching.
    """
    def __init__(self):
        self._dac = {'KM': 'kma',
                     'IF': 'coriolis',
                     'AO': 'aoml',
                     'CS': 'csiro',
                     'KO': 'kordi',
                     'JA': 'jma',
                     'HZ': 'csio',
                     'IN': 'incois',
                     'NM': 'nmdis',
                     'ME': 'meds',
                     'BO': 'bodc'}

    @staticmethod
    def _load_nc(file_path, verbose):
        """
        Loads a .nc file using xarray, with a check for file 404s.
        :param file_path:
        :return:
        """
        if os.path.isfile(file_path):
            return xr.open_dataset(file_path, decode_times=False)
        else:
            raise NetCDF4FileNotFoundError(path=file_path, verbose=verbose)


class ArgoMultiProfLoader(Loader):
    """
    Set the snapshot root path when you create the instance.
    Then, it knows how to navigate the folder structure of a snapshot.
    """
    def __init__(self, argo_root_path):
        Loader.__init__(self)
        self.argo_root_path = argo_root_path

    def load_from_inst_code(self, institute_code, wmo, verbose=True):
        """
        Wrapper load function for argo.
        :param institute_code: the code used to identify institutes (e.g. "IF")
        :param wmo: the wmo floater ID (int)
        :param verbose: prints error message
        :return: the contents as an xrarray
        """
        doifile = os.path.join(self.argo_root_path, self._dac[institute_code], str(wmo), ("%i_prof.nc" % wmo))
        return self._load_nc(doifile, verbose=verbose)

    def load_from_inst(self, institute, wmo, verbose=True):
        """
        Wrapper load function for argo.
        :param institute: the name of the institute (e.g. "coriolis")
        :param wmo: the wmo floater ID (int)
        :param verbose: prints error message
        :return: the contents as an xrarray
        """
        doifile = os.path.join(self.argo_root_path, institute, str(wmo), ("%i_prof.nc" % wmo))
        return self._load_nc(doifile, verbose)