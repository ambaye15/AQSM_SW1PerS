import numpy as np
import unittest
from AQSM_SW1PerS.SW1PerS import *
from AQSM_SW1PerS.utils.period_estimation import *

class TestPeriodicityMethods(unittest.TestCase):

    def setUp(self):
        self.t_vals = np.linspace(0, 4, 1000)
        self.fs = 1000 / 4
        self.x = np.cos(2 * np.pi * self.t_vals)
        self.y = np.sin(2 * np.pi * self.t_vals)
        self.X = np.column_stack((self.x, self.y))
        self.X += np.random.normal(scale=0.1, size=self.X.shape)
        spline_funcs = [CubicSpline(self.t_vals, self.X[:, 0]), CubicSpline(self.t_vals, self.X[:, 1])]
        d = 23
        scoring_pipeline = SW1PerS(start_time = 0, end_time = 4, num_points = 1000, method = 'PS1', d = d, prime_coeff = next_prime(2 * d))
        scoring_pipeline.compute_score(spline_funcs)

    def test_estimate_period(self):
        self.assertTrue(0.9 < scoring_pipeline.period < 1.1)

    def test_compute_PS_range(self):
        self.assertTrue(0.5 <= scoring_pipeline.periodicity_score <= 1)

if __name__ == "__main__":
    unittest.main()
