"""
This sub-module provides utilities for miscellaneous computation tasks
"""

from functools import lru_cache

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from threading import Lock
from IPython.display import display, clear_output, HTML
import logging
from typing import Union
import concurrent.futures

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


class CustomThreadPoolExecutor:
    """
    kwargs can be passed to task_fct and postprocessing_fct
    task_fct can return a structure and bool
    Dedicated display methods
    HTML display
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
        show: Union[bool, str] = True,
        task_legend: dict = {'w': 'Working', 'p': 'Post-processing', 'c': 'Callback'},
        final_legend: dict = {'task': 'Processing tasks', 'final': 'Finalizing'},
    ):
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

        self.task_legend = task_legend
        self.final_legend = final_legend
        self.show = bool(show)

    def _default_finalize_fct(self, obj_list, **kwargs):
        return [v for v in dict(sorted(obj_list.items())).values()], True

    def init_status(self, bucket):
        self.status = [f"?" for _ in range(len(bucket))]
        if self.finalize_fct:
            self.status_final = f"?"
            self.progress = [
                0,
                len(bucket) * 4 + 2,
            ]  # Each task goes by 4 status ('w', 'p', 'c', 'f'/'s') and final by 2 states ('w', 'f'/'s')
        else:
            self.status_final = f"n"
            self.progress = [
                0,
                len(bucket) * 4,
            ]  # Each task goes by 4 status ('w', 'p', 'c', 'f'/'s')

    @property
    def runner(self) -> str:
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return 'notebook'  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return 'terminal'  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return 'standard'  # Probably standard Python interpreter

    @property
    def css_style(self):
        return "\n".join(_load_static_files())

    @property
    def COLORS(self):
        """task status key to css class and label dictionary"""
        return {
            "?": ("gray", "Queued"),
            "w": ("yellow", self.task_legend['w']),
            "p": ("blue", self.task_legend['p']),
            "c": ("cyan", self.task_legend['c']),
            "f": ("red", "Failed"),
            "s": ("green", "Succeed"),
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
    def status_html(self):
        # Create a legend:
        legend = ["\t<div class='legend'>"]
        # legend.append("\t\t<div style='display:inline-block'><span style='margin-bottom: 5px'>Tasks: </span></div>")
        for key in self.COLORS.keys():
            color, desc = self.COLORS[key]
            legend.append(
                "\t\t<div class='item'><div class='box %s'></div><span class='txt'>%s</span></div>"
                % (color, desc)
            )
        legend.append("\t</div>")
        # legend.append("\t\t<div style='display:inline-block; float:right'><span style='margin-bottom: 5px'>Finalized state: </span><div class='item'><div class='box blinking'></div><span class='txt'>Processing</span></div><div class='item'><div class='box failure'></div><span class='txt'>Failure</span></div><div class='item'><div class='box success'></div><span class='txt'>Success</span></div></div>")
        legend = f"\n".join(legend)

        # Create a status bar for tasks:
        content = ["\t<div class='status %s'>" % self.STATE[self.status_final]]
        for s in self.status:
            content.append("\t\t<div class='box %s'></div>" % self.COLORS[s][0])
        content.append("\t</div>")
        content = f"\n".join(content)

        # Progress bar:
        val = int(100 * self.progress[0] / self.progress[1])
        color = {
            "?": "grey",
            "w": "amber",
            "s": "green",
            "f": "red",
            "n": "blue",
        }[self.status_final]
        txt = self.final_legend['task']
        if self.status_final != "?":
            txt = "%s" % (self.final_legend['final'])
        if self.status_final == "f":
            txt = "Failed %s" % (self.final_legend['final'])
        if self.status_final == "s":
            txt = "Succeed in %s" % (self.final_legend['final'])
        txt = "%s (%i%% processed)" % (txt, val)
        progress = ["\t<div class='w3-light-grey w3-small w3-round'>"]
        progress.append(
            "\t\t<div class='w3-container w3-round w3-%s' style='width:%i%%'>%s</div>"
            % (color, val, txt)
        )
        progress.append("\t</div>")
        progress = f"\n".join(progress)

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
        if self.show and self.runner == 'notebook':
            clear_output(wait=True)
            display(self.status_html)

    def update_display_status(self, obj_id, status):
        with self.lock:
            self.status[obj_id] = f"%s" % status
            self.progress[0] += 1
            self.display_status()

    def update_display_status_final(self, state):
        self.status_final = state
        self.progress[0] += 1
        self.display_status()

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
        self.update_display_status(obj_id, "c")  # Callback
        if self.callback_fct is not None:
            data, state = self.callback_fct(data, **self.callback_fct_kwargs)
        return obj_id, data, state

    def finalize(self, results):
        self.update_display_status_final("w")  # Working
        data, state = self.finalize_fct(results, **self.finalize_fct_kwargs)
        self.update_display_status_final("s" if state else "f")
        return data

    def execute(self, bucket: list = None, list_failed: bool = False, pool: str = 'thread'):
        self.bucket = bucket
        self.init_status(bucket)
        self.display_status()

        # Execute tasks and post-processing:
        self.lock = Lock()
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.task, ii, obj) for ii, obj in enumerate(bucket)
            ]
            [f.add_done_callback(self.callback) for f in futures]
            for future in as_completed(futures):
                try:
                    obj_id, data, state = future.result()
                    self.update_display_status(obj_id, "s" if state else "f")
                except:
                    raise
                finally:
                    results.update({obj_id: data})

        # Final tasks status:
        failed = [obj for obj_id, obj in enumerate(self.bucket) if self.status[obj_id] == 'f']

        # Finalize:
        final = self.finalize(results)

        # Return
        if list_failed:
            return final, failed
        else:
            return final
