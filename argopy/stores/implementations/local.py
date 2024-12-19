import json
import xarray as xr
import pandas as pd
import types
import concurrent.futures
import multiprocessing
import logging


from ..filesystems import argo_store_proto
from ..filesystems import has_distributed, distributed
from ..filesystems import tqdm
from ...errors import InvalidMethod, DataNotFound


log = logging.getLogger("argopy.stores.implementation.local")


class filestore(argo_store_proto):
    """Argo local file system

    Relies on :class:`fsspec.implementations.local.LocalFileSystem`
    """

    protocol = "file"

    def open_json(self, url, **kwargs):
        """Return a json from a path, or verbose errors

        Parameters
        ----------
        path: str
            Path to resources passed to :func:`json.loads`
        *args, **kwargs:
            Other arguments passed to :func:`json.loads`

        Returns
        -------
        json

        """
        with self.open(url) as of:
            js = json.load(of, **kwargs)
        if len(js) == 0:
            js = None
        return js

    def open_dataset(self, path, *args, **kwargs):
        """Return a xarray.dataset from a path.

        Parameters
        ----------
        path: str
            Path to resources passed to xarray.open_dataset
        *args, **kwargs:
            Other arguments are passed to :func:`xarray.open_dataset`

        Returns
        -------
        :class:`xarray.DataSet`
        """
        xr_opts = {}
        if "xr_opts" in kwargs:
            xr_opts.update(kwargs["xr_opts"])

        with self.open(path) as of:
            # log.debug("Opening dataset: '%s'" % path)  # Redundant with fsspec logger
            ds = xr.open_dataset(of, *args, **xr_opts)
            ds.load()
        if "source" not in ds.encoding:
            if isinstance(path, str):
                ds.encoding["source"] = path
        return ds.copy()

    def _mfprocessor(
        self,
        url,
        preprocess=None,
        preprocess_opts={},
        open_dataset_opts={},
        *args,
        **kwargs,
    ):
        # Load data
        ds = self.open_dataset(url, **open_dataset_opts)
        # Pre-process
        if isinstance(preprocess, types.FunctionType) or isinstance(
            preprocess, types.MethodType
        ):
            ds = preprocess(ds, **preprocess_opts)
        return ds

    def open_mfdataset(
        self,  # noqa: C901
        urls,
        concat_dim="row",
        max_workers: int = 6,
        method: str = "thread",
        progress: bool = False,
        concat: bool = True,
        preprocess=None,
        preprocess_opts={},
        open_dataset_opts={},
        errors: str = "ignore",
        *args,
        **kwargs,
    ):
        """Open multiple urls as a single xarray dataset.

        This is a version of the ``open_dataset`` method that is able to handle a list of urls/paths
        sequentially or in parallel.

        Use a Threads Pool by default for parallelization.

        Parameters
        ----------
        urls: list(str)
            List of url/path to open
        concat_dim: str
            Name of the dimension to use to concatenate all datasets (passed to :class:`xarray.concat`)
        max_workers: int
            Maximum number of threads or processes
        method: str
            The parallelization method to execute calls asynchronously:
                - ``thread`` (Default): use a pool of at most ``max_workers`` threads
                - ``process``: use a pool of at most ``max_workers`` processes
                - (XFAIL) a :class:`distributed.client.Client` object (:class:`distributed.client.Client`)

            Use 'seq' to simply open data sequentially
        progress: bool
            Display a progress bar (True by default)
        preprocess: callable (optional)
            If provided, call this function on each dataset prior to concatenation
        errors: str
            Should it 'raise' or 'ignore' errors. Default: 'ignore'

        Returns
        -------
        :class:`xarray.Dataset`

        """
        if not isinstance(urls, list):
            urls = [urls]

        results = []
        if method in ["thread", "process"]:
            if method == "thread":
                ConcurrentExecutor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                )
            else:
                if max_workers == 6:
                    max_workers = multiprocessing.cpu_count()
                ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=max_workers
                )

            with ConcurrentExecutor as executor:
                future_to_url = {
                    executor.submit(
                        self._mfprocessor,
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    ): url
                    for url in urls
                }
                futures = concurrent.futures.as_completed(future_to_url)
                if progress:
                    futures = tqdm(
                        futures, total=len(urls), disable="disable" in [progress]
                    )

                for future in futures:
                    data = None
                    # url = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as e:
                        if errors == "ignore":
                            log.debug(
                                "Ignored error with this file: %s\nException raised: %s"
                                % (future_to_url[future], str(e.args))
                            )
                            pass
                        else:
                            raise
                    finally:
                        results.append(data)

        elif has_distributed and isinstance(method, distributed.client.Client):
            # Use a dask client:

            if progress:
                from dask.diagnostics import ProgressBar

                with ProgressBar():
                    futures = method.map(
                        self._mfprocessor,
                        urls,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    )
                    results = method.gather(futures)
            else:
                futures = method.map(
                    self._mfprocessor,
                    urls,
                    preprocess=preprocess,
                    preprocess_opts=preprocess_opts,
                    open_dataset_opts=open_dataset_opts,
                    *args,
                    **kwargs,
                )
                results = method.gather(futures)

        elif method in ["seq", "sequential"]:
            if progress:
                urls = tqdm(urls, total=len(urls), disable="disable" in [progress])

            for url in urls:
                data = None
                try:
                    data = self._mfprocessor(
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    )
                except Exception as e:
                    if errors == "ignore":
                        log.debug(
                            "Ignored error with this url: %s\nException raised: %s"
                            % (url, str(e.args))
                        )
                        pass
                    else:
                        raise
                finally:
                    results.append(data)

        else:
            raise InvalidMethod(method)

        # Post-process results
        results = [r for r in results if r is not None]  # Only keep non-empty results
        if len(results) > 0:
            if concat:
                # ds = xr.concat(results, dim=concat_dim, data_vars='all', coords='all', compat='override')
                ds = xr.concat(
                    results,
                    dim=concat_dim,
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                )
                return ds
            else:
                return results
        else:
            raise DataNotFound(urls)

    def read_csv(self, path, **kwargs):
        """Return a pandas.dataframe from a path that is a csv resource

        Parameters
        ----------
        Path: str
            Path to csv resources passed to :func:`pandas.read_csv`

        Returns
        -------
        :class:`pandas.DataFrame`
        """
        log.debug("Reading csv: %s" % path)
        with self.open(path) as of:
            df = pd.read_csv(of, **kwargs)
        return df
