import os
import shutil
import logging
import pickle
import json
import fsspec
import pandas as pd
from packaging import version
from ..options import OPTIONS
from ..errors import FileSystemHasNoCache

log = logging.getLogger("argopy.utils.caching")


def clear_cache(fs=None):
    """Delete argopy cache folder content"""
    if os.path.exists(OPTIONS["cachedir"]):
        # shutil.rmtree(OPTIONS["cachedir"])
        for filename in os.listdir(OPTIONS["cachedir"]):
            file_path = os.path.join(OPTIONS["cachedir"], filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print("Failed to delete %s. Reason: %s" % (file_path, e))
        if fs:
            fs.clear_cache()


def lscache(cache_path: str = "", prt=True, errors='raise'):
    """Decode and list cache folder content

    Parameters
    ----------
    cache_path: str
    prt: bool, default=True
        Return a printable string or a :class:`pandas.DataFrame`
    errors: str, default: ``raise``
            Define how to handle errors raised during listing:

                - ``raise`` (default): Raise any error encountered
                - ``ignore``: Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message

    Returns
    -------
    str or :class:`pandas.DataFrame`
    """
    from datetime import datetime
    import math

    summary = []

    cache_path = OPTIONS["cachedir"] if cache_path == "" else cache_path
    apath = os.path.abspath(cache_path)
    log.debug("Listing cache content at: %s" % cache_path)

    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    cached_files = []
    fn = os.path.join(apath, "cache")
    if os.path.exists(fn):
        if version.parse(fsspec.__version__) <= version.parse("2023.6.0"):
            with open(fn, "rb") as f:
                loaded_cached_files = pickle.load(
                    f
                )  # nosec B301 because files controlled internally
        else:
            with open(fn, "r") as f:
                loaded_cached_files = json.load(f)
        for c in loaded_cached_files.values():
            if isinstance(c["blocks"], list):
                c["blocks"] = set(c["blocks"])
        cached_files.append(loaded_cached_files)
    else:
        if errors == 'raise':
            raise FileSystemHasNoCache("No fsspec cache system at: %s" % apath)
        elif errors == 'ignore':
            log.debug("No fsspec cache system at: %s" % apath)
        else:
            return summary

    cached_files = cached_files or [{}]
    cached_files = cached_files[-1]

    N_FILES = len(cached_files)
    TOTAL_SIZE = 0
    for cfile in cached_files:
        path = os.path.join(apath, cached_files[cfile]["fn"])
        TOTAL_SIZE += os.path.getsize(path)

    summary.append(
        "%s %s"
        % (
            "=" * 20,
            "%i files in fsspec cache folder (%s)"
            % (N_FILES, convert_size(TOTAL_SIZE)),
        )
    )
    summary.append("lscache %s" % os.path.sep.join([apath, ""]))
    summary.append("=" * 20)

    listing = {
        "fn": [],
        "size": [],
        "time": [],
        "original": [],
        "uid": [],
        "blocks": [],
    }
    for cfile in cached_files:
        summary.append("- %s" % cached_files[cfile]["fn"])
        listing["fn"].append(cached_files[cfile]["fn"])

        path = os.path.join(cache_path, cached_files[cfile]["fn"])
        summary.append("\t%8s: %s" % ("SIZE", convert_size(os.path.getsize(path))))
        listing["size"].append(os.path.getsize(path))

        key = "time"
        ts = cached_files[cfile][key]
        tsf = pd.to_datetime(datetime.fromtimestamp(ts)).strftime("%c")
        summary.append("\t%8s: %s (%s)" % (key, tsf, ts))
        listing["time"].append(pd.to_datetime(datetime.fromtimestamp(ts)))

        if version.parse(fsspec.__version__) > version.parse("0.8.7"):
            key = "original"
            summary.append("\t%8s: %s" % (key, cached_files[cfile][key]))
            listing[key].append(cached_files[cfile][key])

        key = "uid"
        summary.append("\t%8s: %s" % (key, cached_files[cfile][key]))
        listing[key].append(cached_files[cfile][key])

        key = "blocks"
        summary.append("\t%8s: %s" % (key, cached_files[cfile][key]))
        listing[key].append(cached_files[cfile][key])

    summary.append("=" * 20)
    summary = "\n".join(summary)
    if prt:
        # Return string to be printed:
        return summary
    else:
        # Return dataframe listing:
        # log.debug(summary)
        return pd.DataFrame(listing)
