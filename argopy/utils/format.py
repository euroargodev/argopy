"""
Manipulate Argo formatted string and print/stdout formatters
"""
import os
from urllib.parse import urlparse
import logging


log = logging.getLogger("argopy.utils.format")


def format_oneline(s, max_width=65):
    """ Return a string formatted for a line print """
    if len(s) > max_width:
        padding = " ... "
        n = (max_width - len(padding)) // 2
        q = (max_width - len(padding)) % 2
        if q == 0:
            return "".join([s[0: n], padding, s[-n:]])
        else:
            return "".join([s[0: n + 1], padding, s[-n:]])
    else:
        return s


def argo_split_path(this_path):  # noqa C901
    """ Split path from a GDAC ftp style Argo netcdf file and return information

    >>> argo_split_path('coriolis/6901035/profiles/D6901035_001D.nc')
    >>> argo_split_path('https://data-argo.ifremer.fr/dac/csiro/5903939/profiles/D5903939_103.nc')

    Parameters
    ----------
    str

    Returns
    -------
    dict
    """
    dacs = [
        "aoml",
        "bodc",
        "coriolis",
        "csio",
        "csiro",
        "incois",
        "jma",
        "kma",
        "kordi",
        "meds",
        "nmdis",
    ]
    output = {}

    start_with = lambda f, x: f[0:len(x)] == x if len(x) <= len(f) else False  # noqa: E731

    def split_path(p, sep='/'):
        """Split a pathname.  Returns tuple "(head, tail)" where "tail" is
        everything after the final slash.  Either part may be empty."""
        # Same as posixpath.py but we get to choose the file separator !
        p = os.fspath(p)
        i = p.rfind(sep) + 1
        head, tail = p[:i], p[i:]
        if head and head != sep * len(head):
            head = head.rstrip(sep)
        return head, tail

    def fix_localhost(host):
        if 'ftp://localhost:' in host:
            return "ftp://%s" % (urlparse(host).netloc)
        if 'http://127.0.0.1:' in host:
            return "http://%s" % (urlparse(host).netloc)
        else:
            return ""

    known_origins = ['https://data-argo.ifremer.fr',
                     'ftp://ftp.ifremer.fr/ifremer/argo',
                     'ftp://usgodae.org/pub/outgoing/argo',
                     fix_localhost(this_path),
                     '']

    output['origin'] = [origin for origin in known_origins if start_with(this_path, origin)][0]
    output['origin'] = '.' if output['origin'] == '' else output['origin'] + '/'
    sep = '/' if output['origin'] != '.' else os.path.sep

    (path, file) = split_path(this_path, sep=sep)

    output['path'] = path.replace(output['origin'], '')
    output['name'] = file

    # Deal with the path:
    # dac/<DAC>/<FloatWmoID>/
    # dac/<DAC>/<FloatWmoID>/profiles
    path_parts = path.split(sep)

    try:
        if path_parts[-1] == 'profiles':
            output['type'] = 'Mono-cycle profile file'
            output['wmo'] = path_parts[-2]
            output['dac'] = path_parts[-3]
        else:
            output['type'] = 'Multi-cycle profile file'
            output['wmo'] = path_parts[-1]
            output['dac'] = path_parts[-2]
    except Exception:
        log.warning(this_path)
        log.warning(path)
        log.warning(sep)
        log.warning(path_parts)
        log.warning(output)
        raise

    if output['dac'] not in dacs:
        log.debug("This is not a Argo GDAC compliant file path: %s" % path)
        log.warning(this_path)
        log.warning(path)
        log.warning(sep)
        log.warning(path_parts)
        log.warning(output)
        raise ValueError("This is not a Argo GDAC compliant file path (invalid DAC name: '%s')" % output['dac'])

    # Deal with the file name:
    filename, file_extension = os.path.splitext(output['name'])
    output['extension'] = file_extension
    if file_extension != '.nc':
        raise ValueError(
            "This is not a Argo GDAC compliant file path (invalid file extension: '%s')" % file_extension)
    filename_parts = output['name'].split("_")

    if "Mono" in output['type']:
        prefix = filename_parts[0].split(output['wmo'])[0]
        if 'R' in prefix:
            output['data_mode'] = 'R, Real-time data'
        if 'D' in prefix:
            output['data_mode'] = 'D, Delayed-time data'

        if 'S' in prefix:
            output['type'] = 'S, Synthetic BGC Mono-cycle profile file'
        if 'M' in prefix:
            output['type'] = 'M, Merged BGC Mono-cycle profile file'
        if 'B' in prefix:
            output['type'] = 'B, BGC Mono-cycle profile file'

        suffix = filename_parts[-1].split(output['wmo'])[-1]
        if 'D' in suffix:
            output['direction'] = 'D, descending profiles'
        elif suffix == "" and "Mono" in output['type']:
            output['direction'] = 'A, ascending profiles (implicit)'

    else:
        typ = filename_parts[-1].split(".nc")[0]
        if typ == 'prof':
            output['type'] = 'Multi-cycle file'
        if typ == 'Sprof':
            output['type'] = 'S, Synthetic BGC Multi-cycle profiles file'
        if typ == 'tech':
            output['type'] = 'Technical data file'
        if typ == 'meta':
            output['type'] = 'Metadata file'
        if 'traj' in typ:
            # possible typ = [Rtraj, Dtraj, BRtraj, BDtraj]
            output['type'], i = 'Trajectory file', 0
            if typ[0] == 'B':
                output['type'], i = 'BGC Trajectory file', 1
            if typ.split("traj")[0][i] == 'D':
                output['data_mode'] = 'D, Delayed-time data'
            elif typ.split("traj")[0][i] == 'R':
                output['data_mode'] = 'R, Real-time data'
            else:
                output['data_mode'] = 'R, Real-time data (implicit)'

    # Adjust origin and path for local files:
    # This ensure that output['path'] is agnostic to users and can be reused on any gdac compliant architecture
    parts = path.split(sep)
    i, stop = len(parts) - 1, False
    while not stop:
        if parts[i] == 'profiles' or parts[i] == output['wmo'] or parts[i] == output['dac'] or parts[i] == 'dac':
            i = i - 1
            if i < 0:
                stop = True
        else:
            stop = True
    output['origin'] = sep.join(parts[0:i + 1])
    output['path'] = output['path'].replace(output['origin'], '')

    return dict(sorted(output.items()))


