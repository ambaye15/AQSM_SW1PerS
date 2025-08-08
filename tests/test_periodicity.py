import numpy as np
import unittest
from ATSM_SW1PerS.SW1PerS import *

class TestPeriodicityMethods(unittest.TestCase):

    def setUp(self):
        self.t_vals = np.linspace(0, 4, 150)
        self.fs = 150 / 4
        self.x = np.cos(2 * np.pi * self.t_vals)
        self.y = np.sin(2 * np.pi * self.t_vals)
        self.X = np.column_stack((self.x, self.y))
        self.X += np.random.normal(scale=0.1, size=self.X.shape)

    def test_estimate_period(self):
        period = estimate_period(self.X[:, 0], self.X[:, 1], self.fs)
        self.assertTrue(0.9 < period < 1.1)

    def test_compute_PS_range(self):
        d = 23
        tau = 1.0 / (d + 1)
        spline_x = CubicSpline(self.t_vals, self.X[:, 0])
        spline_y = CubicSpline(self.t_vals, self.X[:, 1])
        SW = SW_cloud_nD([spline_x, spline_y], self.t_vals, tau, d, 300, 2)
        diagrams = ripser(SW, coeff=next_prime(2 * d), maxdim=1)['dgms']
        dgm1 = np.array(diagrams[1])
        score = compute_PS(dgm1, method='PS1')
        self.assertTrue(0.5 <= score <= 1)

if __name__ == "__main__":
    unittest.main()
