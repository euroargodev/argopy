import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)

def deprecation_of_plotters():
    warnings.warn(
        "The 'argopy.plotters' has been replaced by 'argopy.plot'. After 0.1.13, importing 'plotters' "
        "will raise an error. You're seeing this message because you called this function through "
        "the argopy 'plotters' module.",
        category=DeprecationWarning,
        stacklevel=2,
    )

def open_dashboard(*args, **kwargs):
    deprecation_of_plotters()
    from .plot import dashboard
    return dashboard(*args, **kwargs)

def open_sat_altim_report(*args, **kwargs):
    deprecation_of_plotters()
    from .plot import open_sat_altim_report
    return open_sat_altim_report(*args, **kwargs)

def plot_trajectory(*args, **kwargs):
    deprecation_of_plotters()
    from .plot import plot_trajectory
    return plot_trajectory(*args, **kwargs)

def bar_plot(*args, **kwargs):
    deprecation_of_plotters()
    from .plot import bar_plot
    return bar_plot(*args, **kwargs)

if __name__ == "argopy.plotters":
    deprecation_of_plotters()
