import logging
import types
import xarray as xr
import concurrent.futures
import multiprocessing
import io

from ...errors import InvalidMethod, DataNotFound
from ...utils.transform import drop_variables_not_in_all_datasets
from ..filesystems import has_distributed, distributed
from ..filesystems import tqdm
from .http import httpstore


log = logging.getLogger("argopy.stores.implementation.ftp")


class ftpstore(httpstore):
    """Argo ftp file system

    Inherits from :class:`argopy.stores.httpstore` but relies on :class:`fsspec.implementations.ftp.FTPFileSystem`
    """

    protocol = "ftp"

    def open_dataset(self, url, *args, **kwargs):
        """Open and decode a xarray dataset from an ftp url

        Parameters
        ----------
        url: str

        Returns
        -------
        :class:`xarray.Dataset`
        """
        try:
            this_url = self.fs._strip_protocol(url)
            data = self.fs.cat_file(this_url)
        except Exception:
            log.debug("Error with: %s" % url)
            # except aiohttp.ClientResponseError as e:
            raise

        if data[0:3] != b"CDF" and data[0:3] != b"\x89HD":
            raise TypeError(
                "We didn't get a CDF or HDF5 binary data as expected ! We get: %s"
                % data
            )
        if data[0:3] == b"\x89HD":
            data = io.BytesIO(data)

        xr_opts = {}
        if "xr_opts" in kwargs:
            xr_opts.update(kwargs["xr_opts"])
        ds = xr.open_dataset(data, *args, **xr_opts)

        if "source" not in ds.encoding:
            if isinstance(url, str):
                ds.encoding["source"] = url
        self.register(this_url)
        self.register(url)
        return ds

    def _mfprocessor_dataset(
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
        max_workers: int = 6,
        method: str = "sequential",
        progress: bool = False,
        concat: bool = True,
        concat_dim="row",
        preprocess=None,
        preprocess_opts={},
        open_dataset_opts={},
        errors: str = "ignore",
        *args,
        **kwargs,
    ):
        """Open multiple ftp urls as a single xarray dataset.

        This is a version of the :class:`argopy.stores.ftpstore.open_dataset` method that is able
        to handle a list of urls/paths sequentially or in parallel.

        Use a Threads Pool by default for parallelization.

        Parameters
        ----------
        urls: list(str)
            List of url/path to open
        max_workers: int, default: 6
            Maximum number of threads or processes
        method: str, default: ``thread``
            The parallelization method to execute calls asynchronously:

                - ``seq`` (default): open data sequentially, no parallelization applied
                - ``process``: use a pool of at most ``max_workers`` processes
                - :class:`distributed.client.Client`: Experimental, expect this method to fail !
        progress: bool, default: False
            Display a progress bar
        concat: bool, default: True
            Concatenate results in a single :class:`xarray.Dataset` or not (in this case, function will return a
            list of :class:`xarray.Dataset`)
        concat_dim: str, default: ``row``
            Name of the dimension to use to concatenate all datasets (passed to :class:`xarray.concat`)
        preprocess: callable (optional)
            If provided, call this function on each dataset prior to concatenation
        preprocess_opts: dict (optional)
            If ``preprocess`` is provided, pass this as options
        errors: str, default: ``ignore``
            Define how to handle errors raised during data URIs fetching:

                - ``raise``: Raise any error encountered
                - ``ignore`` (default): Do not stop processing, simply issue a debug message in logging console
                - ``silent``:  Do not stop processing and do not issue log message
        Other args and kwargs: other options passed to :class:`argopy.stores.httpstore.open_dataset`.

        Returns
        -------
        output: :class:`xarray.Dataset` or list of :class:`xarray.Dataset`

        """
        strUrl = lambda x: x.replace("ftps://", "").replace("ftp://", "")  # noqa: E731

        if not isinstance(urls, list):
            urls = [urls]

        results = []
        failed = []
        if method in ["process"]:
            if max_workers == 6:
                max_workers = multiprocessing.cpu_count()
            ConcurrentExecutor = concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            )

            with ConcurrentExecutor as executor:
                future_to_url = {
                    executor.submit(
                        self._mfprocessor_dataset,
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
                    try:
                        data = future.result()
                    except Exception:
                        failed.append(future_to_url[future])
                        if errors == "ignore":
                            log.debug(
                                "Ignored error with this url: %s"
                                % strUrl(future_to_url[future])
                            )
                            # See fsspec.http logger for more
                            pass
                        elif errors == "silent":
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
                        self._mfprocessor_dataset,
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
                    self._mfprocessor_dataset,
                    urls,
                    preprocess=preprocess,
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
                    data = self._mfprocessor_dataset(
                        url,
                        preprocess=preprocess,
                        preprocess_opts=preprocess_opts,
                        open_dataset_opts=open_dataset_opts,
                        *args,
                        **kwargs,
                    )
                except Exception:
                    failed.append(url)
                    if errors == "ignore":
                        log.debug(
                            "Ignored error with this url: %s" % strUrl(url)
                        )  # See fsspec.http logger for more
                        pass
                    elif errors == "silent":
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
                results = drop_variables_not_in_all_datasets(results)
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
