"""
This submodule provides utilities for miscellaneous computation tasks with multithreading

We construct the MyThreadPoolExecutor class,
we create a series of classes using multiple inheritance to implement monitoring features

"""
from functools import lru_cache
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from threading import Lock
import logging
from typing import Union
from abc import ABC, abstractmethod
import importlib

try:
    from importlib.resources import files  # New in version 3.9
except ImportError:
    from pathlib import Path

    files = lambda x: Path(  # noqa: E731
        importlib.util.find_spec(x).submodule_search_locations[0]
    )

has_ipython = (spec := importlib.util.find_spec("IPython")) is not None
if has_ipython:
    from IPython.display import display, clear_output, HTML


log = logging.getLogger("argopy.utils.compute")


STATIC_FILES = (
    ("argopy.static.css", "w3.css"),
    ("argopy.static.css", "compute.css"),
)


@lru_cache(None)
def _load_static_files():
    """Lazily load the resource files into memory the first time they are needed"""
    return [
        files(package).joinpath(resource).read_text(encoding="utf-8")
        for package, resource in STATIC_FILES
    ]


class proto_MonitoredThreadPoolExecutor(ABC):
    """
    Add:
        - self.*_fct and self.*_fct_kwargs for all the processing steps
        - self.status list of characters to describe each task status
        - self.status_final character to describe the final computation status
    """

    def __init__(
        self,
        max_workers: int = 10,
        task_fct=None,
        task_fct_kwargs={},
        postprocessing_fct=None,
        postprocessing_fct_kwargs={},
        callback_fct=None,
        callback_fct_kwargs={},
        finalize_fct=None,
        finalize_fct_kwargs={},
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_workers = max_workers

        self.task_fct = task_fct
        self.task_fct_kwargs = task_fct_kwargs

        self.postprocessing_fct = postprocessing_fct
        self.postprocessing_fct_kwargs = postprocessing_fct_kwargs

        self.callback_fct = callback_fct
        self.callback_fct_kwargs = callback_fct_kwargs

        if finalize_fct is None:
            finalize_fct = self._default_finalize_fct
        self.finalize_fct = finalize_fct
        self.finalize_fct_kwargs = finalize_fct_kwargs

    def _default_finalize_fct(self, obj_list, **kwargs):
        return [v for v in dict(sorted(obj_list.items())).values()], True

    def init_status(self, bucket):
        self.status = ["?" for _ in range(len(bucket))]
        if self.finalize_fct:
            self.status_final = "?"
            self.progress = [
                0,
                len(bucket) * 4 + 2,
            ]  # Each task goes by 4 status ('w', 'p', 'c', 'f'/'s') and final by 2 states ('w', 'f'/'s')
        else:
            self.status_final = "n"
            self.progress = [
                0,
                len(bucket) * 4,
            ]  # Each task goes by 4 status ('w', 'p', 'c', 'f'/'s')

    def task(self, obj_id, obj):
        self.update_display_status(obj_id, "w")  # Working
        data, state = self.task_fct(obj, **self.task_fct_kwargs)

        self.update_display_status(obj_id, "p")  # Post-processing
        if self.postprocessing_fct is not None:
            data, state = self.postprocessing_fct(
                data, **self.postprocessing_fct_kwargs
            )

        return obj_id, data, state

    def callback(self, future):
        obj_id, data, state = future.result()
        # self.update_display_status(obj_id, "s" if state else "f")
        self.update_display_status(obj_id, "c")  # Callback
        if self.callback_fct is not None:
            data, state = self.callback_fct(data, **self.callback_fct_kwargs)
        return obj_id, data, state

    def finalize(self, results):
        self.update_display_status_final("w")  # Working
        data, state = self.finalize_fct(results, **self.finalize_fct_kwargs)
        self.update_display_status_final("s" if state else "f")
        return data

    def execute(self, bucket: list = None, list_failed: bool = False):
        self.bucket = bucket
        self.init_status(bucket)
        self.display_status()

        # Execute tasks and post-processing:
        self.lock = Lock()
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.task, ii, obj) for ii, obj in enumerate(bucket)
            ]
            [f.add_done_callback(self.callback) for f in futures]
            for future in as_completed(futures):
                try:
                    obj_id, data, state = future.result()
                    self.update_display_status(obj_id, "s" if state else "f")
                except Exception:
                    raise
                finally:
                    results.update({obj_id: data})

        # Final tasks status:
        failed = [
            obj for obj_id, obj in enumerate(self.bucket) if self.status[obj_id] == "f"
        ]

        # Finalize:
        final = self.finalize(results)

        # Return
        if list_failed:
            return final, failed
        else:
            return final

    @abstractmethod
    def display_status(self):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_display_status(self, task_id: int, st: str):
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_display_status_final(self, st: str):
        raise NotImplementedError("Not implemented")


class proto_MonitoredPoolExecutor_monitor(proto_MonitoredThreadPoolExecutor):
    default_task_legend = {"w": "Working", "p": "Post-processing", "c": "Callback", "f": "Failed", "s": "Success"}
    default_final_legend = {"task": "Processing tasks", "final": "Finalizing"}

    def __init__(
        self,
        show: Union[bool, str] = True,
        task_legend: dict = {},
        final_legend: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(task_legend.keys()) == 0:
            self.task_legend = self.default_task_legend
        else:
            self.task_legend = {**self.default_task_legend, **task_legend}
        if len(final_legend.keys()) == 0:
            self.final_legend = self.default_final_legend
        else:
            self.final_legend = {**self.default_final_legend, **final_legend}
        self.show = bool(show)
        # log.debug(self.runner)

    @property
    def runner(self) -> str:
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return "notebook"  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return "terminal"  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return "standard"  # Probably standard Python interpreter

    @property
    def COLORS(self):
        """task status key to css class and color and label dictionary"""
        return {
            "?": ("gray", "Queued", "⏸"),
            "w": ("yellow", self.task_legend["w"], "⏺"),
            "p": ("blue", self.task_legend["p"], "🔄"),
            "c": ("cyan", self.task_legend["c"], "⏯"),
            "f": ("red", self.task_legend["f"], "🔴"),
            "s": ("green", self.task_legend["s"], "🟢"),
        }

    @property
    def STATE(self):
        """final state key to css class dictionary"""
        return {
            "?": "queue",
            "w": "blinking",
            "s": "success",
            "f": "failure",
            "n": "none",
        }

    @property
    def STATE_COLORS(self):
        """final state key to colors dictionary"""
        return {
            "?": "gray",
            "w": "amber",
            "s": "green",
            "f": "red",
            "n": "blue",
        }

    def display_status(self):
        pass

    def update_display_status(self, *args, **kwargs):
        pass

    def update_display_status_final(self, *args, **kwargs):
        pass


class proto_MonitoredPoolExecutor_notebook(proto_MonitoredPoolExecutor_monitor):
    """
    Add HTML jupyter notebook display
    """

    @property
    def css_style(self):
        return "\n".join(_load_static_files())

    @property
    def status_html(self):
        # Create a legend:
        legend = ["\t<div class='legend'>"]
        # legend.append("\t\t<div style='display:inline-block'><span style='margin-bottom: 5px'>Tasks: </span></div>")
        for key in self.COLORS.keys():
            color, desc, icon = self.COLORS[key]
            legend.append(
                "\t\t<div class='item'><div class='box %s'></div><span class='txt'>%s</span></div>"
                % (color, desc)
            )
        legend.append("\t</div>")
        # legend.append("\t\t<div style='display:inline-block; float:right'><span style='margin-bottom: 5px'>Finalized state: </span><div class='item'><div class='box blinking'></div><span class='txt'>Processing</span></div><div class='item'><div class='box failure'></div><span class='txt'>Failure</span></div><div class='item'><div class='box success'></div><span class='txt'>Success</span></div></div>")
        legend = "\n".join(legend)

        # Create a status bar for tasks:
        content = ["\t<div class='status %s'>" % self.STATE[self.status_final]]
        for s in self.status:
            content.append("\t\t<div class='box %s'></div>" % self.COLORS[s][0])
        content.append("\t</div>")
        content = "\n".join(content)

        # Progress bar:
        val = int(100 * self.progress[0] / self.progress[1])
        color = self.STATE_COLORS[self.status_final]
        txt = self.final_legend["task"]
        if self.status_final != "?":
            txt = "%s" % (self.final_legend["final"])
        if self.status_final == "f":
            txt = "Failed %s" % (self.final_legend["final"])
        if self.status_final == "s":
            txt = "Succeed in %s" % (self.final_legend["final"])
        txt = "%s (%i%% processed)" % (txt, val)
        progress = ["\t<div class='w3-light-grey w3-small w3-round'>"]
        progress.append(
            "\t\t<div class='w3-container w3-round w3-%s' style='width:%i%%'>%s</div>"
            % (color, val, txt)
        )
        progress.append("\t</div>")
        progress = "\n".join(progress)

        # Complete HTML:
        html = (
            "<div class='parent'>\n"
            f"<style>{self.css_style}</style>\n"
            f"{legend}\n"
            f"{content}\n"
            "</div>\n"
            f"{progress}\n"
        )
        return HTML(html)

    def display_status(self):
        super().display_status()
        if self.show and self.runner == "notebook":
            clear_output(wait=True)
            display(self.status_html)

    def update_display_status(self, obj_id, status):
        super().update_display_status()
        with self.lock:
            self.status[obj_id] = "%s" % status
            self.progress[0] += 1
            self.display_status()

    def update_display_status_final(self, state):
        super().update_display_status_final()
        self.status_final = state
        self.progress[0] += 1
        self.display_status()


class proto_MonitoredPoolExecutor_terminal(proto_MonitoredPoolExecutor_monitor):
    """
    Add terminal display
    """

    def __init__(
        self,
        icon: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._text_only = not bool(icon)
        self._reprinter = None

    class _Reprinter:
        def __init__(self, text: str = ""):
            self.text = text
            self.counter = 0

        def moveup(self, lines):
            for _ in range(lines):
                sys.stdout.write("\x1b[A")

        def reprint(self, text):
            if self.counter >= 1:
                self.moveup(self.text.count("\n"))
            print(text, end="\r")
            self.text = text
            self.counter += 1

    def _adjust_for_terminal_width(self, text, max_width=None):
        """Split text if larger than terminal"""
        term_width, _ = os.get_terminal_size()
        term_width = term_width if max_width is None else int(term_width / max_width)
        lines = []
        if len(text) > term_width:
            i_start, i_end = 0, term_width - 1
            while i_end <= len(text):
                lines.append(text[i_start:i_end])
                i_start = i_end
                i_end = i_start + term_width - 1
            lines.append(text[i_start:])
            return "\n".join(lines)
        else:
            return text

    @property
    def status_txt(self):
        def f(text, color=None, bold=0, italic=0, underline=0, crossed=0, negative=0):
            """Format text with color,

            Uses no color by default but accepts any color from the C class.

            Parameters
            ----------
            text: str
            color: str
            bold: bool
            """

            PREF = "\033["
            RESET = f"{PREF}0m"

            class C:
                gray = "37"
                yellow = "33"
                amber = "33"
                blue = "34"
                cyan = "36"
                red = "31"
                green = "32"
                magenta = "35"
                black = "30"
                white = "97"

            dec = []
            if bold:
                dec.append("1")
            if italic:
                dec.append("3")
            if underline:
                dec.append("4")
            if crossed:
                dec.append("9")
            if negative:
                dec.append("7")

            if color is None:
                if len(dec) > 0:
                    dec = ";".join(dec)
                    return f"{PREF}{dec}m" + text + RESET
                else:
                    return f"{PREF}" + text + RESET
            else:
                if len(dec) > 0:
                    dec = ";".join(dec)
                    return f"{PREF}{dec};{getattr(C, color)}m" + text + RESET
                else:
                    return f"{PREF}{getattr(C, color)}m" + text + RESET

        # Text only (no icons):
        if self._text_only:
            # Create a legend:
            legend = []
            for key in self.COLORS.keys():
                color, desc, icon = self.COLORS[key]
                legend.append(f("%s: %s" % (key, desc), color))
            legend = " | ".join(legend)

            # Create a status bar for tasks:
            # with colored brackets color for final status:
            raw_content = "[%s]" % "".join(self.status)
            lines = []
            for status_line in self._adjust_for_terminal_width(raw_content).split("\n"):
                line_content = []
                for s in status_line:
                    if s not in ["[", "]"]:
                        line_content.append(
                            f(s, self.COLORS[s][0], negative=s in ["f"])
                        )
                    else:
                        line_content.append(f(s, self.STATE_COLORS[self.status_final]))
                line_content = "".join(line_content)
                lines.append(line_content)
            content = "\n".join(lines)

        # Icons only
        else:
            # Create a legend:
            legend = []
            for key in self.COLORS.keys():
                color, desc, icon = self.COLORS[key]
                legend.append(f"{icon}: %s" % f(desc, color=color))
            legend = " | ".join(legend)

            # Create a status bar for tasks:
            # with colored brackets color for final status:
            # raw_content = f"[%s]" % "".join(self.status)
            raw_content = "%s" % "".join(self.status)
            lines = []
            for status_line in self._adjust_for_terminal_width(
                raw_content, max_width=4
            ).split("\n"):
                line_content = []
                for s in status_line:
                    if s not in ["[", "]"]:
                        line_content.append("%s " % self.COLORS[s][2])
                    else:
                        line_content.append(f(s, self.STATE_COLORS[self.status_final]))
                line_content = "".join(line_content)
                lines.append(line_content)
            content = "\n".join(lines)

        # Progress bar:
        val = int(100 * self.progress[0] / self.progress[1])
        color = self.STATE_COLORS[self.status_final]
        txt = self.final_legend["task"]
        if self.status_final != "?":
            txt = "%s" % (self.final_legend["final"])
        if self.status_final == "f":
            txt = "Failed %s" % (self.final_legend["final"])
        if self.status_final == "s":
            txt = "Succeed in %s" % (self.final_legend["final"])
        txt = "%s (%i%% processed)" % (txt, val)
        progress = f("%s ..." % txt, color, negative=0)

        # Complete STDOUT:
        txt = f"\n" f"{legend}\n" f"{content}\n" f"{progress: <50}\n"

        return txt

    def display_status(self):
        super().display_status()

        if self.show and self.runner in ["terminal", "standard"]:
            if self._reprinter is None:
                self._reprinter = self._Reprinter(self.status_txt)
            # os.system('cls' if os.name == 'nt' else 'clear')
            self._reprinter.reprint(f"{self.status_txt}")
            # sys.stdout.flush()


if has_ipython:

    class c(proto_MonitoredPoolExecutor_notebook, proto_MonitoredPoolExecutor_terminal):
        pass

else:

    class c(proto_MonitoredPoolExecutor_terminal):
        pass


class MyThreadPoolExecutor(c):
    """
    This is a low-level helper class not intended to be used directly by users

    Examples
    --------
    ::

        from argopy.utils import MyThreadPoolExecutor as MyExecutor
        from random import random
        from time import sleep
        import numpy as np

        def my_task(obj, errors='ignore'):
            data = random()
            sleep(data * 3)
            state = np.random.randint(0,100) >= 25
            if not state:
                if errors == 'raise':
                    raise ValueError('Hello world')
                elif errors == 'ignore':
                    pass
            return data, state

        def my_postprocess(obj, opt=12):
            sleep(random() * 5)
            data = obj**opt
            state = np.random.randint(0,100) >= 25
            return data, state

        def my_callback(obj, opt=2):
            sleep(random() * 2)
            data = obj**opt
            state = np.random.randint(0,100) >= 25
            return data, state

        def my_final(obj_list, opt=True):
            data = random()
            sleep(data * 20)
            results = [v for v in dict(sorted(obj_list.items())).values()]
            return data, np.all(results)

        if __name__ == '__main__':
            run = MyExecutor(max_workers=25,
                              task_fct=my_task,
                              postprocessing_fct=my_postprocess,
                              callback_fct=my_callback,
                              finalize_fct=my_final,
                              show=1,
                             )
            results, failed = run.execute(range(100), list_failed=True)
            print(results)
    """

    pass
