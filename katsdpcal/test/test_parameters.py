"""Tests for the parameters module."""

import unittest

class TestParameters(unittest.TestCase):

    def dict_test(self):
        """Check that parameters are being returned as a dict."""
        from katcal import parameters
        self.AssertIsInstance(parameters.set_params(),dict)

