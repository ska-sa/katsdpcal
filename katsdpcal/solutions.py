import time

from sortedcontainers import SortedListWithKey
import numpy as np


class CalSolution(object):
    """Calibration solution.

    This represents a solution for a single point in time.
    """
    def __init__(self, soltype, solvalues, soltime):
        self.soltype = soltype
        self.values = solvalues
        self.time = soltime

    def __str__(self):
        """String representation of calibration solution to help identify it."""
        # Obtain human-friendly timestamp representing the centre of solutions
        timestamp = time.strftime("%H:%M:%S", time.gmtime(self.time))
        return "{} {} {}".format(self.soltype, self.values.shape, timestamp)


class CalSolutions(object):
    """Calibration solutions.

    This stores multiple solutions (in time) in a single array.
    """
    def __init__(self, soltype, solvalues, soltimes):
        self.soltype = soltype
        self.values = solvalues
        self.times = soltimes

    def __str__(self):
        """String representation of calibration solution to help identify it."""
        # Obtain human-friendly timestamp representing the centre of solutions
        timestamp = time.strftime("%H:%M:%S", time.gmtime(np.mean(self.times)))
        return "{} {} {}".format(self.soltype, self.values.shape, timestamp)


class CalSolutionStore(object):
    """Calibration solutions indexable by time.

    This stores multiple solutions (in time), but unlike
    :class:`CalSolutions`, each solution is stored separately, and one can query
    all solutions from a time range.
    """
    def __init__(self, soltype):
        self.soltype = soltype
        self._values = SortedListWithKey(key=lambda val: val.time)

    def add(self, solution):
        """Add one :class:`CalSolution`."""
        if solution.soltype != self.soltype:
            raise ValueError('solution has type {}, expected {}'
                             .format(solution.soltype, self.soltype))
        self._values.add(solution)

    @property
    def latest(self):
        """Get the latest solution, or ``None`` if none have been added."""
        if self._values:
            return self._values[-1]
        else:
            return None

    def get_range(self, start_time, end_time):
        """Get the solutions in the interval [start_time, end_time].

        The returned values are combined into a :class:`CalSolutions`.
        """
        parts = list(self._values.irange_key(start_time, end_time))
        if len(parts) > 0:
            values = np.stack([part.values for part in parts])
        else:
            values = np.array([])
        times = np.array([part.time for part in parts])
        return CalSolutions(self.soltype, values, times)


class CalSolutionStoreLatest(object):
    """Tracks the latest value of a calibration solution.

    It provides the same interface as :class:`CalSolutionStore`, but does not
    support :meth:`~CalSolutionStore.get_range`, and only keeps the latest
    solution rather than a full history.
    """
    def __init__(self, soltype):
        self.soltype = soltype
        self._latest = None

    def add(self, solution):
        if solution.soltype != self.soltype:
            raise ValueError('solution has type {}, expected {}'
                             .format(solution.soltype, self.soltype))
        if self._latest is None or solution.time >= self._latest.time:
            self._latest = solution

    @property
    def latest(self):
        return self._latest

    def get_range(self, start_time, end_time):
        raise NotImplementedError('CalSolutionStoreLatest does not support get_range')
