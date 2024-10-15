import pytest
import warnings

from argopy.utils.decorators import DocInherit, deprecated


def test_DocInherit():

    class Profile(object):
        def load(self):
            """Dummy"""
            pass

    class Float(Profile):
        @DocInherit
        def load(self):
            pass

    assert Float.load.__doc__ == Profile.load.__doc__


def test_deprecated_no_reason():

    @deprecated
    def dummy_fct():
        """Dummy"""
        pass

    with pytest.deprecated_call():
        dummy_fct()


def test_deprecated_with_a_reason():

    @deprecated("Because !")
    def dummy_fct():
        """Dummy"""
        pass

    with pytest.deprecated_call(match="Because"):
        dummy_fct()



def test_deprecated_with_a_reason_and_version():

    @deprecated("Because !", version='12.0')
    def dummy_fct():
        """Dummy"""
        pass

    with pytest.deprecated_call(match="Deprecated since version"):
        dummy_fct()


def test_deprecated_ignore_caller():

    @deprecated("Because !", ignore_caller='caller_to_be_ignored')
    def dummy_fct():
        """Dummy"""
        pass

    def caller_to_be_ignored():
        dummy_fct()
        pass

    with warnings.catch_warnings():
        # warnings.simplefilter("error")
        caller_to_be_ignored()
