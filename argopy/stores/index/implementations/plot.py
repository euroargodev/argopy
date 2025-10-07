from typing import Any

from ....plot import plot_trajectory, bar_plot
from ..extensions import ArgoIndexPlotProto


class ArgoIndexPlot(ArgoIndexPlotProto):
    """Extension providing plot methods

    Examples
    --------
    .. code-block:: python
        :caption: Example of :class:`ArgoIndex` trajectory plotting method

        from argopy import ArgoIndex

        idx = ArgoIndex().query.wmo(6903091)

        idx.plot.trajectory()

    .. code-block:: python
        :caption: Examples of :class:`ArgoIndex` trajectory plotting methods with custom arguments

        from argopy import ArgoIndex

        idx = ArgoIndex(index_file='bgc-s')
        idx.query.params('CHLA')

        idx.plot.trajectory(set_global=1,
                            add_legend=0,
                            traj=0,
                            cbar=False,
                            markersize=12,
                            markeredgesize=0.1,
                            dpi=120,
                            figsize=(20,20));

    .. code-block:: python
        :caption: Example of :class:`ArgoIndex` bar plotting methods

        from argopy import ArgoIndex

        idx = ArgoIndex().load()

        idx.plot.bar(by='profiler')

        idx.plot.bar(by='dac')

    See Also
    --------
    :class:`ArgoIndex.plot.trajectory`, :class:`ArgoIndex.plot.bar`

    """

    def trajectory(self, index: bool = False, **kwargs) -> Any:
        """Quick map of profile index trajectories

        Parameters
        ----------
        index: bool, default=False
            Determine if the method makes a plot with the full index (True) or only the query search result (False).
        **kwargs
            All other arguments are passed to :meth:`argopy.plot.plot_trajectory`.

        Returns
        -------
        tuple
            Output from :class:`argopy.plot.plot_trajectory`, typically:
                - fig: :class:`matplotlib.figure.Figure`
                - ax: :class:`matplotlib.axes.Axes`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex

            idx = ArgoIndex().query.wmo(WMO)
            idx.plot.trajectory()
        """
        default_opts = {"style": "white"}
        N = len(self._obj.read_wmo(index=index))
        if N == 1:
            default_opts["hue"] = "cyc"
            default_opts["add_legend"] = False
            default_opts["palette"] = "Spectral_r"
            default_opts["cbar"] = True
        fig, ax = plot_trajectory(
            self._obj.to_dataframe(index=index), **{**default_opts, **kwargs}
        )
        ax.set_title(self.get_title(index))
        return fig, ax

    def bar(self, by: str = "dac", index: bool = False, **kwargs) -> Any:
        """Bar plot of one index property

        Parameters
        ----------
        by: str, default='dac'
            The index property to plot, one in 'date', 'latitude', 'longitude', 'ocean', 'profiler_code', 'institution_code', 'date_update', 'wmo', 'cyc', 'institution', 'dac', 'profiler'.

        **kwargs
            All other arguments are passed to :meth:`argopy.plot.bar_plot`.

        Returns
        -------
        tuple
            Output from :class:`argopy.plot.plot_trajectory`, typically:
                - fig: :class:`matplotlib.figure.Figure`
                - ax: :class:`matplotlib.axes.Axes`

        Examples
        --------
        .. code-block:: python

            from argopy import ArgoIndex

            idx = ArgoIndex()
            idx.plot.bar('institution')
        """
        if by not in [
            "date",
            "latitude",
            "longitude",
            "ocean",
            "profiler_code",
            "institution_code",
            "date_update",
            "wmo",
            "cyc",
            "institution",
            "dac",
            "profiler",
        ]:
            raise ValueError(
                'Invalid value for "by", must be in "date", "latitude", "longitude", "ocean", "profiler_code"'
            )
        fig, ax = bar_plot(self._obj.to_dataframe(index=index), by=by, **kwargs)
        ax.set_title(self.get_title(index))
        return fig, ax
