import unittest
import numpy as np
from ciles.tools import runDistribution


class TestConstantDiffusion(unittest.TestCase):

    def test_variance(self):

        B = 0.05
        tmax = 2.
        drift = np.zeros(100)  # no drift field
        diff = np.ones(100) * B  # constant diffusion with 1 deg^2/s

        finals = runDistribution(
            drift,
            diff,
            dt=.01,
            tmax=tmax,
            reps=100000,
            initials=1)

        finals = finals.reshape(-1)
        finals[np.where(finals > np.pi)] = \
            finals[np.where(finals > np.pi)] - 2*np.pi

        # test that variance grows linear in time with slope B
        np.testing.assert_almost_equal(finals.var(), B * tmax, decimal=3)
