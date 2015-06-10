"""Tests for the parameters module."""

import unittest

class TestParameters(unittest.TestCase):

    def dict_test(self):
        """Check that parameters are being returned as a dict."""
        from katsdpcal import parameters
        self.AssertIsInstance(parameters.set_params(),dict)

