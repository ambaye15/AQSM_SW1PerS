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
from AQSM_SW1PerS.SW1PerS import *

def get_label_occurrence_index(window_meta, target_label, occurrence=0):
    """Get the index of the nth occurrence of a label."""
    matches = [i for i, meta in enumerate(window_meta) if meta['label'] == target_label]
    if occurrence >= len(matches):
        raise ValueError(f"Only {len(matches)} occurrences of label '{target_label}' found.")
    return matches[occurrence]

def extract_processed_window(X, meta_entry, expected_length=380, sigma=3):
    """Extract smoothed and interpolated window from raw accelerometer data."""
    window = getAccelerometerRange(X, meta_entry, interpolate_to_fixed_length=True, expected_length=expected_length)
    timestamps = window[:, 0]
    accel = smoothDataGaussian(window[:, 1:], sigma=sigma)
    return np.column_stack((timestamps, accel))

def interpolate_xyz(window_data):
    """Return cubic spline functions for x, y, z."""
    timestamps_sec = (window_data[:, 0] - window_data[0, 0]) / 1000.0
    ax, ay, az = window_data[:, 1], window_data[:, 2], window_data[:, 3]
    return (
        CubicSpline(timestamps_sec, ax),
        CubicSpline(timestamps_sec, ay),
        CubicSpline(timestamps_sec, az),
        timestamps_sec
    )

def estimate_sampling_rate(timestamps_ms):
    """Estimate Hz from time deltas (in milliseconds)."""
    deltas = np.diff(timestamps_ms)
    return 1000.0 / np.mean(deltas)

def resample_and_detrend_splines(t_x, t_y, t_z, sampling_rate, duration_sec=4):
    """Resample splines uniformly, detrend, re-spline, and return interpolated matrix and splines."""
    num_points = int(duration_sec * sampling_rate)
    t_vals = np.linspace(0, duration_sec, num_points)

    # Evaluate and detrend
    keypoint_x = signal.detrend(t_x(t_vals))
    keypoint_y = signal.detrend(t_y(t_vals))
    keypoint_z = signal.detrend(t_z(t_vals))

    # Re-spline (optional but recommended for future use)
    t_x_new = CubicSpline(t_vals, keypoint_x)
    t_y_new = CubicSpline(t_vals, keypoint_y)
    t_z_new = CubicSpline(t_vals, keypoint_z)

    # Stack interpolated coordinates
    X_interp = np.column_stack((keypoint_x, keypoint_y, keypoint_z))
    cs_list = [t_x_new, t_y_new, t_z_new]

    return t_vals, X_interp, cs_list, keypoint_x, keypoint_y, keypoint_z
    
    
def SW1PerS_alt(accelerometer_X, meta_data, method = 'PS1'):

    periodicity_scores = []  

    d = 23

    prime_coeff = next_prime(2 * d)

    for meta in meta_data:

        try:

            accelerometer_window = extract_processed_window(accelerometer_X, meta)
    
            f_x, f_y, f_z, timestamps_sec = interpolate_xyz(accelerometer_window)
    
            sampling_rate = estimate_sampling_rate(accelerometer_window[:, 0])
    
            t_vals, acceleromter_interp, splines, accelerometer_x, accelerometer_y, accelerometer_z = resample_and_detrend_splines(f_x, f_y, f_z, sampling_rate, duration_sec=4)
    
            peirod, top_period, estimated_freq = estimate_period_LAPIS(acceleromter_interp, sampling_rate)
    
            tau = peirod / (d+1)
    
            SW = SW_cloud_nD(splines, t_vals, tau, d, 300, 3)
    
            result = ripser(SW, coeff=prime_coeff, maxdim=1)
            diagrams = result['dgms']
            dgm1 = np.array(diagrams[1])
        
            scores = compute_PS(dgm1, method = method)

            periodicity_scores.append(scores)
            
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




