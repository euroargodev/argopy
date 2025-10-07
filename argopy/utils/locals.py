import os
import sys
import subprocess  # nosec B404 only used without user inputs
import platform
import locale
import struct
import importlib
from importlib.metadata import version
import contextlib
import copy
import shutil
import json
from ..options import OPTIONS


PIP_INSTALLED = {}
try:
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format', 'json'])
    reqs = json.loads(reqs.decode())
    [PIP_INSTALLED.update({mod['name']: mod['version']}) for mod in reqs]
except:  # noqa E722
    pass


def get_sys_info():
    """Returns system information as a dict"""

    blob = []

    # get full commit hash
    commit = None
    if os.path.isdir(".git") and os.path.isdir("argopy"):
        try:
            pipe = subprocess.Popen(  # nosec No user provided input to control here
                'git log --format="%H" -n 1'.split(" "),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, serr = pipe.communicate()
        except Exception:
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                try:
                    commit = so.decode("utf-8")
                except ValueError:
                    pass
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, nodename, release, version_, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "%s" % (sysname)),
                ("OS-release", "%s" % (release)),
                ("machine", "%s" % (machine)),
                ("processor", "%s" % (processor)),
                ("byteorder", "%s" % sys.byteorder),
                ("LC_ALL", "%s" % os.environ.get("LC_ALL", "None")),
                ("LANG", "%s" % os.environ.get("LANG", "None")),
                ("LOCALE", "%s.%s" % locale.getlocale()),
            ]
        )
    except Exception:
        pass

    return blob


def netcdf_and_hdf5_versions():
    libhdf5_version = None
    libnetcdf_version = None
    try:
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    except ImportError:
        try:
            import h5py

            libhdf5_version = h5py.version.hdf5_version
        except ImportError:
            pass
    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]


def cli_version(cli_name):
    try:
        a = subprocess.run([cli_name, '--version'], capture_output=True)
        return a.stdout.decode().strip("\n").replace(cli_name, '').strip()
    except:  # noqa E722
        if shutil.which(cli_name):
            return "- # installed"
        else:
            return "-"


def pip_version(pip_name):
    version = '-'
    for name in [pip_name, pip_name.replace("_", "-"), pip_name.replace("-", "_")]:
        if name in PIP_INSTALLED:
            version = PIP_INSTALLED[name]
    return version


def get_version(module_name):
    ver = '-'
    try:
        ver = module_name.__version__
    except AttributeError:
        try:
            ver = version(module_name)
        except importlib.metadata.PackageNotFoundError:
            try:
                ver = pip_version(module_name)
            except:  # noqa E722
                try:
                    ver = cli_version(module_name)
                except:  # noqa E722
                    pass
    if sum([int(v == '0') for v in ver.split(".")]) == len(ver.split(".")):
        ver = '-'
    return ver


def show_versions(file=sys.stdout, conda=False):  # noqa: C901
    """Print the versions of argopy and its dependencies

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    conda: bool, optional
        format versions to be copy/pasted on a conda environment file (default, False)
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        print(f"Error collecting netcdf / hdf5 version: {e}")

    DEPS = {
        "core": sorted(
            [
                ("argopy", get_version),

                ("xarray", get_version),
                ("scipy", get_version),
                ("netCDF4", get_version),
                ("h5netcdf", get_version),
                ("erddapy", get_version),
                ("fsspec", get_version),
                ("aiohttp", get_version),
                ("packaging", get_version),
                # will come with xarray, Using 'version' to make API compatible with several fsspec releases
                ("requests", get_version),
                ("toolz", get_version),
                ("decorator", get_version),
            ]
        ),
        "ext.util": sorted(
            [
                (
                    "gsw",
                    get_version,
                ),  # Used by xarray accessor to compute new variables
                ("tqdm", get_version),
            ]
        ),
        "ext.files": sorted(
            [
                ("boto3", get_version),
                ("numcodecs", get_version),
                ("s3fs", get_version),
                ("kerchunk", get_version),
                ("zarr", get_version),
            ]
        ),
        "ext.perf": sorted(
            [
                ("dask", get_version),
                ("distributed", get_version),
                ("pyarrow", get_version),
            ]
        ),
        "ext.plot": sorted(
            [
                ("cartopy", get_version),
                ("IPython", get_version),
                ("ipykernel", get_version),
                ("ipywidgets", get_version),
                ("matplotlib", get_version),
                ("pyproj", get_version),
                ("seaborn", get_version),
            ]
        ),
        "dev": sorted(
            [
                ("aiofiles", get_version),
                ("black", get_version),
                ("bottleneck", get_version),
                ("cftime", get_version),
                ("cfgrib", get_version),
                ("codespell", cli_version),
                ("flake8", get_version),
                ("numpy", get_version),  # will come with xarray and pandas
                ("pandas", get_version),  # will come with xarray
                ("pip", get_version),
                ("pytest", get_version),  # will come with pandas
                ("pytest_env", get_version),  # will come with pandas
                ("pytest_cov", get_version),  # will come with pandas
                ("pytest_localftpserver", get_version),  # will come with pandas
                ("setuptools", get_version),  # Provides : pkg_resources
                ("sphinx", get_version),
            ]
        ),
        "pip": sorted(
            [
                ("pytest-reportlog", get_version),
            ]
        ),
    }

    DEPS_blob = {}
    for level in DEPS.keys():
        deps = DEPS[level]
        deps_blob = list()
        for modname, ver_f in deps:
            try:
                ver = ver_f(modname)
                deps_blob.append((modname, ver))
            except Exception:
                deps_blob.append((modname, "installed"))
        DEPS_blob[level] = deps_blob

    print("\nSYSTEM", file=file)
    print("------", file=file)
    for k, stat in sys_info:
        print(f"{k}: {stat}", file=file)

    for level in DEPS_blob:
        if conda:
            print("\n# %s:" % level.upper(), file=file)
        else:
            title = "INSTALLED VERSIONS: %s" % level.upper()
            print("\n%s" % title, file=file)
            print("-" * len(title), file=file)
        deps_blob = DEPS_blob[level]
        for k, stat in deps_blob:
            if conda:
                if k != "argopy":
                    kf = k.replace("_", "-")
                    comment = " " if stat != "-" else "# "
                    print(
                        f"{comment} - {kf} = {stat}", file=file
                    )  # Format like a conda env line, useful to update ci/requirements
            else:
                print("{:<12}: {:<12}".format(k, stat), file=file)


@contextlib.contextmanager
def modified_environ(*remove, **update):
    """
    Temporarily updates the ``os.environ`` dictionary in-place.

    The ``os.environ`` dictionary is updated in-place so that the modification
    is sure to work in all situations.

    :param remove: Environment variables to remove.
    :param update: Dictionary of environment variables and values to add/update.
    """
    # Source: https://github.com/laurent-laporte-pro/stackoverflow-q2059482
    env = os.environ
    update = update or {}
    remove = remove or []

    # List of environment variables being updated or removed.
    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    # Environment variables and values to restore on exit.
    update_after = {k: env[k] for k in stomped}
    # Environment variables and values to remove on exit.
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


def show_options(file=sys.stdout):  # noqa: C901
    """Print options of argopy

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    print("\nARGOPY OPTIONS", file=file)
    print("--------------", file=file)
    opts = copy.deepcopy(OPTIONS)
    opts = dict(sorted(opts.items()))
    for k, v in opts.items():
        print(f"{k}: {v}", file=file)
