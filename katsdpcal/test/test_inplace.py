"""Tests for :mod:`katsdpcal.inplace`."""

from nose.tools import assert_raises, assert_false
import numpy as np
import dask.array as da

from katsdpcal import inplace


class TestStoreInplace:
    def setup(self):
        # Variables with n prefix are numpy arrays
        self.na = np.arange(36).reshape(6, 6)
        self.nx = np.arange(24).reshape(6, 4)
        self.a = da.from_array(self.na, chunks=(3, 2), name=False)
        self.b = da.ones(6, chunks=(2,))
        self.x = da.from_array(self.nx, chunks=(3, 2), name=False)

    def test_simple(self):
        orig = self.na.copy()
        c = self.a + self.b
        inplace.store_inplace(c, self.a)
        np.testing.assert_array_equal(self.na, orig + 1)

    def test_rechunk_merge(self):
        """Computing with finer chunks and merging is safe."""
        orig = self.na.copy()
        a = da.from_array(self.na, chunks=(3, 1))
        b = da.ones(6, chunks=(1,))
        c = a + b
        inplace.store_inplace(c, self.a)
        np.testing.assert_array_equal(self.na, orig + 1)

    def test_unsafe(self):
        c = self.a + self.a.T
        with assert_raises(inplace.UnsafeInplaceError):
            inplace.store_inplace(c, self.a)

    def test_unsafe_with_slice(self):
        c = self.a + self.a.T
        with assert_raises(inplace.UnsafeInplaceError):
            inplace.store_inplace(c[0], self.a[1])
        with assert_raises(inplace.UnsafeInplaceError):
            inplace.store_inplace(c[0:1], self.a[1:2])

    def test_compute_slice(self):
        y = self.a + self.b
        inplace.store_inplace(y[3], self.a[0])
        np.testing.assert_equal(np.arange(6) + 19, self.na[0])

    def test_concatenated(self):
        y = da.concatenate([self.a, self.x], axis=1)
        na_orig = self.na.copy()
        nx_orig = self.nx.copy()
        for key, value in y.dask.items():
            print(key)
            print('   ', value)
        inplace.store_inplace(y * 2, y)
        np.testing.assert_array_equal(na_orig * 2, self.na)
        np.testing.assert_array_equal(nx_orig * 2, self.nx)

    def test_different_shapes(self):
        with assert_raises(ValueError):
            inplace.store_inplace(self.a, self.b)

    def test_duplicate_outputs(self):
        a2 = self.a[:]
        with assert_raises(ValueError):
            inplace.store_inplace([self.a, a2], [self.a, a2])

    def test_target_not_numpy(self):
        with assert_raises(ValueError):
            inplace.store_inplace(self.b, self.b)

    def test_not_dask(self):
        with assert_raises(ValueError):
            inplace.store_inplace(self.a, self.na)
        with assert_raises(ValueError):
            inplace.store_inplace(self.na, self.a)


class TestRename:
    def test(self):
        a = np.array([1, 2, 3, 4, 5, 6], np.int32)
        b = np.array([10, 9, 8, 7, 6, 5], np.int32)
        x = da.from_array(a, chunks=(2,))
        y = da.from_array(b, chunks=(2,))
        z = x * y
        expected = np.array([10, 18, 24, 28, 30, 30], np.int32)
        old_keys = set(z.dask.keys())
        inplace.rename(z)
        new_keys = set(z.dask.keys())
        assert_false(old_keys & new_keys)
        np.testing.assert_array_equal(expected, z.compute())
