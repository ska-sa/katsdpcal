"""Tests for the parameters module."""

import katcal

def dict_test():
    """Check that parameters are being returned as a dict."""
    from katcal import parameters
    assert(isinstance(parameters.set_params(),dict))

