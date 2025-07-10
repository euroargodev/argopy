import numpy as np
import xarray as xr
import logging
import time
from typing import Union, List

from ..utils import to_list, list_core_parameters
from ..utils import (
    split_data_mode,
    merge_param_with_param_adjusted,
    filter_param_by_data_mode,
)
from ..stores import ArgoIndex
from ..stores.index.spec import ArgoIndexStoreProto
from ..errors import InvalidDatasetStructure
from . import register_argo_accessor, ArgoAccessorExtension


log = logging.getLogger("argopy.xtensions.datamode")


@register_argo_accessor("datamode")
class ParamsDataMode(ArgoAccessorExtension):
    """
    Utilities for Argo parameters data mode

    See Also
    --------
    :meth:`datamode.compute`
    :meth:`datamode.merge`
    :meth:`datamode.filter`
    :meth:`datamode.split`

    Examples
    --------
    >>> from argopy import DataFetcher
    >>> ArgoSet = DataFetcher(mode='expert').float(1902605)
    >>> ds = ArgoSet.to_xarray()
    >>> ds.argo.datamode.merge()


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_from_ArgoIndex(
        self, indexfs: Union[None, ArgoIndex]
    ) -> xr.Dataset:  # noqa: C901
        """Compute <PARAM>_DATA_MODE variables from ArgoIndex

        This method consumes a collection of points.

        Parameters
        ----------
        indexfs: :class:`argopy.ArgoIndex`, optional
            An :class:`argopy.ArgoIndex` instance with a :class:`pandas.DataFrame` backend to look for data modes.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        idx = indexfs.copy(deep=True) if isinstance(indexfs, ArgoIndexStoreProto) else ArgoIndex()

        def complete_df(this_df, params):
            """Add 'wmo', 'cyc' and '<param>_data_mode' columns to this dataframe"""
            this_df["wmo"] = this_df["file"].apply(lambda x: int(x.split("/")[1]))
            this_df["cyc"] = this_df["file"].apply(
                lambda x: int(x.split("_")[-1].split(".nc")[0].replace("D", ""))
            )
            this_df["variables"] = this_df["parameters"].apply(lambda x: x.split())
            for param in params:
                this_df["%s_data_mode" % param] = this_df.apply(
                    lambda x: (
                        x["parameter_data_mode"][x["variables"].index(param)]
                        if param in x["variables"]
                        else ""
                    ),
                    axis=1,
                )
            return this_df

        def read_DM(this_df, wmo, cyc, param):
            """Return one parameter data mode for a given wmo/cyc and index dataframe"""
            filt = []
            filt.append(this_df["wmo"].isin([wmo]))
            filt.append(this_df["cyc"].isin([cyc]))
            sub_df = this_df[np.logical_and.reduce(filt)]
            if sub_df.shape[0] == 0:
                log.debug(
                    "Found a profile in the dataset, but not in the index ! wmo=%i, cyc=%i"
                    % (wmo, cyc)
                )
                # This can happen if a Synthetic netcdf file was generated from a non-BGC float.
                # The file exists, but it doesn't have BGC variables. Float is usually not listed in the index.
                return ""
            else:
                return sub_df["%s_data_mode" % param].values[-1]

        def print_etime(txt, t0):
            now = time.process_time()
            print("⏰ %s: %0.2f seconds" % (txt, now - t0))
            return now

        # timer = time.process_time()

        profiles = self._argo.list_WMO_CYC
        idx.query.wmo(self._argo.list_WMO)

        params = [
            p
            for p in idx.read_params()
            if p in self._obj or "%s_ADJUSTED" % p in self._obj
        ]
        # timer = print_etime('Read profiles and params from ds', timer)

        df = idx.to_dataframe(completed=False)
        df = complete_df(df, params)
        # timer = print_etime('Index search wmo and export to dataframe', timer)

        CYCLE_NUMBER = self._obj["CYCLE_NUMBER"].values
        PLATFORM_NUMBER = self._obj["PLATFORM_NUMBER"].values
        N_POINTS = self._obj["N_POINTS"].values

        for param in params:
            # print("=" * 50)
            # print("Filling DATA MODE for %s ..." % param)
            # tims = {'init': 0, 'read_DM': 0, 'isin': 0, 'where': 0, 'fill': 0}

            for iprof, prof in enumerate(profiles):
                wmo, cyc = prof
                # t0 = time.process_time()

                if "%s_DATA_MODE" % param not in self._obj:
                    self._obj["%s_DATA_MODE" % param] = xr.full_like(
                        self._obj["CYCLE_NUMBER"], dtype=str, fill_value=""
                    )
                # now = time.process_time()
                # tims['init'] += now - t0
                # t0 = now

                param_data_mode = read_DM(df, wmo, cyc, param)
                # log.debug("data mode='%s' for %s/%i/%i" % (param_data_mode, param, wmo, cyc))
                # now = time.process_time()
                # tims['read_DM'] += now - t0
                # t0 = now

                i_cyc = CYCLE_NUMBER == cyc
                i_wmo = PLATFORM_NUMBER == wmo
                # now = time.process_time()
                # tims['isin'] += now - t0
                # t0 = now

                i_points = N_POINTS[np.logical_and(i_cyc, i_wmo)]
                # now = time.process_time()
                # tims['where'] += now - t0
                # t0 = now

                # self._obj["%s_DATA_MODE" % param][i_points] = param_data_mode
                self._obj["%s_DATA_MODE" % param].loc[dict(N_POINTS=i_points)] = (
                    param_data_mode
                )
                # now = time.process_time()
                # tims['fill'] += now - t0

            self._obj["%s_DATA_MODE" % param] = self._obj[
                "%s_DATA_MODE" % param
            ].astype("<U1")
            # timer = print_etime('Processed %s (%i profiles)' % (param, len(profiles)), timer)

        # Finalise:
        self._obj = self._obj[np.sort(self._obj.data_vars)]
        return self._obj

    def compute(self, indexfs: Union[None, ArgoIndex]) -> xr.Dataset:
        """Compute <PARAM>_DATA_MODE variables"""
        if "STATION_PARAMETERS" in self._obj and "PARAMETER_DATA_MODE" in self._obj:
            return split_data_mode(self._obj)
        else:
            return self._compute_from_ArgoIndex(indexfs=indexfs)

    def split(self) -> xr.Dataset:
        """Convert PARAMETER_DATA_MODE(N_PROF, N_PARAM) into several <PARAM>_DATA_MODE(N_PROF) variables

        Using the list of *PARAM* found in ``STATION_PARAMETERS``, this method will create ``N_PARAM``
        new variables in the dataset ``<PARAM>_DATA_MODE(N_PROF)``.

        The variable ``PARAMETER_DATA_MODE`` is drop from the dataset at the end of the process.

        Returns
        -------
        :class:`xarray.Dataset`
        """
        return split_data_mode(self._obj)

    def merge(
        self, params: Union[str, List[str]] = "all", errors: str = "raise"
    ) -> xr.Dataset:
        """Merge <PARAM> and <PARAM>_ADJUSTED variables according to DATA_MODE or <PARAM>_DATA_MODE

        Merging is done as follows:

        - For measurements with data mode ``R``: keep <PARAM> (eg: 'DOXY')
        - For measurements with data mode ``D`` or ``A``: keep <PARAM>_ADJUSTED (eg: 'DOXY_ADJUSTED')

        Since adjusted variables are not required anymore after the transformation, all <PARAM>_ADJUSTED variables
        are dropped from the dataset in order to avoid confusion with regard to variable content.
        Variable DATA_MODE or <PARAM>_DATA_MODE are preserved for the record.

        Parameters
        ----------
        params: str, List[str], optional, default='all'
            Parameter or list of parameters to merge.
            Use the default keyword ``all`` to merge all possible parameters in the :class:`xarray.Dataset`.
        errors: str, optional, default='raise'
            If ``raise``, raises a :class:`argopy.errors.InvalidDatasetStructure` error if any of the expected variables is
            not found.
            If ``ignore``, fails silently and return unmodified dataset.

        Returns
        -------
        :class:`xarray.Dataset`

        Notes
        -----
        This method is compatible with core, deep and BGC datasets

        See Also
        --------
        :meth:`filter`
        """
        if self._argo._type != "point":
            raise InvalidDatasetStructure(
                "Method only available to a collection of points"
            )
        else:
            this = self._obj

        # Determine the list of variables to transform:
        params = to_list(params)
        parameters = []
        # log.debug(params)
        if params[0] == "all":
            if "DATA_MODE" in this.data_vars:
                for p in list_core_parameters():
                    if p in this.data_vars or "%s_ADJUSTED" % p in this.data_vars:
                        parameters.append(p)
            else:
                parameters = [
                    p.replace("_DATA_MODE", "")
                    for p in this.data_vars
                    if "_DATA_MODE" in p
                ]
        else:
            [parameters.append(v) for v in params]
        # log.debug(parameters)

        # Transform data:
        for param in parameters:
            this = merge_param_with_param_adjusted(this, param, errors=errors)

        # Finalise:
        this = this[np.sort(this.data_vars)]
        this.argo.add_history(
            "[%s] real-time and adjusted/delayed variables merged according to their data mode"
            % (",".join(parameters))
        )

        return this

    def filter(
        self,  # noqa: C901
        dm: Union[str, List[str]] = ["R", "A", "D"],
        params: Union[str, List[str]] = "all",
        logical: str = "and",
        mask: bool = False,
        errors: str = "raise",
    ) -> xr.Dataset:
        """Filter measurements according to parameters data mode

        Filter the dataset to keep points where all or some of the parameters are in any of the data mode specified.

        This method can return the filtered dataset or the filter mask.

        Parameters
        ----------
        dm: str, List[str], optional, default=[``R``, ``A``, ``D``]
            List of data mode values (string) to keep
        params: str, List[str], optional, default='all'
            List of parameters to apply the filter to. By default, we use all parameters for which a data mode
            can be found
        logical: str, optional, default='and'
            Reduce parameter filters with a logical ``and`` or ``or``. With ``and`` the filter shall be True
            if all parameters match the data mode requested, while with ``or`` it will be True for at least one parameter.
        mask: bool, optional, default=False
            Determine if we should return the filter mask or the filtered dataset
        errors: str, optional, default='raise'
            If ``raise``, raises a :class:`argopy.errors.InvalidDatasetStructure` error if any of the expected variables is
            not found.
            If ``ignore``, fails silently and return unmodified dataset.

        Returns
        -------
        :class:`xarray.Dataset`

        Notes
        -----
        - Method compatible with core, deep and BGC datasets
        - Can be applied after :meth:`merge`

        See Also
        --------
        :meth:`merge`

        """
        if self._argo._type != "point":
            raise InvalidDatasetStructure(
                "Method only available to a collection of points"
            )
        else:
            this = self._obj

        # Make sure we deal with a list of strings:
        if not isinstance(dm, list):
            dm = to_list(dm)
        dm = [str(x).upper() for x in dm]

        if logical not in ["and", "or"]:
            raise ValueError("'logical' must be 'and' or 'or'")

        # Determine the list of variables to filter:
        params = to_list(params)

        if len(params) == 0:
            this.argo.add_history("Found no variables to select according to DATA_MODE")
            return this

        if params[0] == "all":
            if "DATA_MODE" in this.data_vars:
                params = ["PRES", "TEMP"]
                if "PSAL" in this.data_vars:
                    params.append("PSAL")
            else:
                params = [
                    p.replace("_DATA_MODE", "")
                    for p in this.data_vars
                    if "_DATA_MODE" in p
                ]
        elif params[0] == "core":
            params = list_core_parameters()
        else:
            for p in params:
                if p not in this.data_vars:
                    if errors == "raise":
                        raise InvalidDatasetStructure(
                            "Parameter '%s' not found in this dataset" % p
                        )
                    else:
                        log.debug("Parameter '%s' not found in this dataset" % p)
                    params.remove(p)

        if len(params) == 0:
            this.argo.add_history("Found no variables to select according to DATA_MODE")
            return this

        logging.debug(
            "filter_data_mode: Filtering dataset to keep points with DATA_MODE in %s for '%s' fields in %s"
            % (dm, logical, ",".join(params))
        )

        # Get a filter mask for each variables:
        filter = []
        for param in params:
            f = filter_param_by_data_mode(this, param, dm=dm, mask=True)
            [filter.append(f) if len(f) > 0 else None]

        # Reduce dataset:
        if len(filter) > 0:
            if logical == "and":
                filter = np.logical_and.reduce(filter)
            else:
                filter = np.logical_or.reduce(filter)

        if mask:
            # Return mask:
            return filter
        elif len(filter) > 0:
            # Apply mask:
            this = this.loc[dict(N_POINTS=filter)]

            # Finalise:
            this = this[np.sort(this.data_vars)]
            this.argo.add_history(
                "[%s] filtered to retain points with data mode in [%s]"
                % (",".join(params), ",".join(dm))
            )

            if this.argo.N_POINTS == 0:
                log.warning("No data left after DATA_MODE filtering !")

            return this

        else:
            this.argo.add_history(
                "No data mode found for [%s], no filtering applied" % (",".join(params))
            )
            return this
