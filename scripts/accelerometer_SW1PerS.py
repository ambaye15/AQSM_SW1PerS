# Copyright 2025 Austin Amadou MBaye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.interpolate import CubicSpline
from AQSM_SW1PerS.SW1PerS import *
from AQSM_SW1PerS.utils.accelerometer import *

def extract_processed_window(X, meta_entry, expected_length=380, sigma=3):
    """Extract smoothed and interpolated window from raw accelerometer data."""
    window = getAccelerometerRange(X, meta_entry, interpolate_to_fixed_length=True, expected_length=expected_length)
    timestamps = window[:, 0]
    accel = smoothDataGaussian(window[:, 1:], sigma=sigma)
    return np.column_stack((timestamps, accel))

def estimate_sampling_rate(timestamps_ms):
    """Estimate Hz from time deltas (in milliseconds)."""
    deltas = np.diff(timestamps_ms)
    return 1000.0 / np.mean(deltas)
    
def SW1PerS_alt(accelerometer_X, meta_data, method = 'PS1'):

    periodicity_scores = []  

    d = 23

    prime_coeff = next_prime(2 * d)

    for meta in meta_data:

        try:
            
            accelerometer_window = extract_processed_window(accelerometer_X, meta)

            timestamps_sec = (accelerometer_window[:, 0] - accelerometer_window[0, 0]) / 1000.0
            ax, ay, az = accelerometer_window[:, 1], accelerometer_window[:, 2], accelerometer_window[:, 3]
            fx, fy, fz = CubicSpline(timestamps_sec, ax), CubicSpline(timestamps_sec, ay), CubicSpline(timestamps_sec, az)
            spline_funcs = [fx, fy, fz]
            sampling_rate = estimate_sampling_rate(accelerometer_window[:, 0])

            num_points = int(4 * sampling_rate)

            scoring_pipeline = SW1PerS(start_time = 0, end_time = 4, num_points = num_points, method = method, d = d, prime_coeff = prime_coeff)

            score = scoring_pipeline.compute_score(spline_funcs)

            periodicity_scores.append(score)
            
        except Exception as e:
            
            if method == 'PS10':
                periodicity_scores.append(np.zeros(10))
            else:
                periodicity_scores.append(np.zeros(1))
                
    return np.vstack(periodicity_scores)

def processAccel(args):
    '''
    Tool for multiprocessing getFeatures because computations across sensors are easily parallelizable 
    '''
    return SW1PerS_alt(*args)

