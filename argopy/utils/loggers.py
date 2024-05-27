import warnings
import inspect
import pathlib
import os
import logging


log = logging.getLogger("argopy.utils.loggers")


def warnUnless(ok, txt):
    """Function to raise a warning unless condition is True

    This function IS NOT to be used as a decorator anymore

    Parameters
    ----------
    ok: bool
        Condition to raise the warning or not
    txt: str
        Text to display in the warning
    """
    if not ok:
        msg = "%s %s" % (inspect.stack()[1].function, txt)
        warnings.warn(msg)


def log_argopy_callerstack(level="debug"):
    """log the caller’s stack"""
    froot = str(pathlib.Path(__file__).parent.resolve())
    for ideep, frame in enumerate(inspect.stack()[1:]):
        if os.path.join("argopy", "argopy") in frame.filename:
            # msg = ["└─"]
            # [msg.append("─") for ii in range(ideep)]
            msg = [""]
            [msg.append("  ") for ii in range(ideep)]
            msg.append(
                "└─ %s:%i -> %s"
                % (frame.filename.replace(froot, ""), frame.lineno, frame.function)
            )
            msg = "".join(msg)
            if level == "info":
                log.info(msg)
            elif level == "debug":
                log.debug(msg)
            elif level == "warning":
                log.warning(msg)
