import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)


def deprecation_of_utilities():
    warnings.warn(
        "The 'argopy.utilities' module has been replaced by 'argopy.utils'. After 0.1.15, importing 'utilities' "
        "will raise an error. You're seeing this message because you called this function through "
        "the argopy 'utilities' module.",
        category=DeprecationWarning,
        stacklevel=2,
    )


def show_versions(*args, **kwargs):
    deprecation_of_utilities()
    from .utils.locals import show_versions
    return show_versions(*args, **kwargs)


if __name__ == "argopy.utilities":
    deprecation_of_utilities()
