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

import signal
import pandas as pd
import argparse
from scipy.interpolate import CubicSpline
import os
import numpy as np
from pathlib import Path
import multiprocessing as multi

from AQSM_SW1PerS.utils.accelerometer import *
from AQSM_SW1PerS.SW1PerS import *
from AQSM_SW1PerS.utils.paths import get_data_path


class Timeout:
    def __init__(self, seconds=60, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

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
    
def SW1PerS_accelerometer(accelerometer_X, meta_data, method = 'PS10'):

    periodicity_scores = []  
    periods = []
    d = 23

    prime_coeff = next_prime(2 * d)

    for meta in meta_data:

        try:
            with Timeout(seconds=60): #Add a timeout in case period estimation gets stuck
                accelerometer_window = extract_processed_window(accelerometer_X, meta)
    
                timestamps_sec = (accelerometer_window[:, 0] - accelerometer_window[0, 0]) / 1000.0
                ax, ay, az = accelerometer_window[:, 1], accelerometer_window[:, 2], accelerometer_window[:, 3]
                fx, fy, fz = CubicSpline(timestamps_sec, ax), CubicSpline(timestamps_sec, ay), CubicSpline(timestamps_sec, az)
                spline_funcs = [fx, fy, fz]
                sampling_rate = estimate_sampling_rate(accelerometer_window[:, 0])
    
                num_points = int(4 * sampling_rate)
    
                scoring_pipeline = SW1PerS(start_time = 0, end_time = 4, num_points = num_points, method = method, d = d, prime_coeff = prime_coeff, f_min = 0.25, f_max = 6.0)
                scoring_pipeline.compute_score(spline_funcs)
    
                periodicity_scores.append(scoring_pipeline.periodicity_score)
                periods.append(scoring_pipeline.period)
            
        except Exception as e:
            
            if method == 'PS10':
                periodicity_scores.append(np.zeros(10))
            else:
                periodicity_scores.append(np.zeros(1))
            periods.append(4.0)
                
    return np.vstack(periodicity_scores), np.vstack(periods)

def multi_process_accel(args):
    '''
    Tool for multiprocessing getFeatures because computations across sensors are easily parallelizable 
    '''
    return SW1PerS_accelerometer(*args)


def extract_annotations(meta_data):
    annotations = []
    for meta in meta_data:
        if meta['label'] == 'Normal':
          annotations.append(0)
        elif meta['label'] == 'Rock':
          annotations.append(1)
        elif meta['label'] == 'Flap':
          annotations.append(2)
        else:
          annotations.append(3)
    return np.vstack(annotations)

def process_folder(folder_path):

    method = 'PS1'
    include_period = True
    
    if '001-2010-05-28' in str(folder_path):
        annofile = 'Annotator1Stereotypy.annotation.xlsx'
    else:
        annofile = "Annotator1Stereotypy.annotation.xml"
    
    if 'Study1' in str(folder_path):
        torso_file = "MITes_01_RawCorrectedData_Trunk.RAW_DATA.csv"
        left_wrist_file = "MITes_08_RawCorrectedData_Left-wrist.RAW_DATA.csv"
        right_wrist_file = "MITes_11_RawCorrectedData_Right-wrist.RAW_DATA.csv"
    else:
        torso_file = "Wocket_02_RawCorrectedData_Torso.csv"
        left_wrist_file = "Wocket_01_RawCorrectedData_Left-Wrist.csv"
        right_wrist_file = "Wocket_00_RawCorrectedData_Right-Wrist.csv"
    
    meta_data = process_accelerometer_data(folder_path, annofile, torso_file)
    
    torso_X = get_accel_data(folder_path, torso_file)
    left_wrist_X = get_accel_data(folder_path, left_wrist_file)
    right_wrist_X = get_accel_data(folder_path, right_wrist_file)
    
    args_list = [
        (torso_X, meta_data, method),
        (left_wrist_X, meta_data, method),
        (right_wrist_X, meta_data, method) ]
    
    with multi.Pool(multi.cpu_count()) as pool:
        results = pool.map(multi_process_accel, args_list)
    
    # Unpack the results
    torso_scores, torso_period = results[0]
    lwrist_scores, lwrist_period = results[1]
    rwrist_scores, rwrist_period = results[2]
    
    annotations = extract_annotations(meta_data)
    
    path = Path(folder_path)
    filename_cleaned = path.name
    
    if include_period:
        X_features = np.column_stack((annotations,torso_scores, lwrist_scores, rwrist_scores, torso_period, lwrist_period, rwrist_period))
        if method == 'PS10':
            PS_df = pd.DataFrame(X_features, columns = (["Annotation"] + [f"Torso_{i}" for i in range(1, 11)] + [f"LWrist_{i}" for i in range(1, 11)] + [f"RWrist_{i}" for i in range(1, 11)] + ["Torso_Period", "LWrist_Period", "RWrist_Period"]))
        else:
            PS_df = pd.DataFrame(X_features, columns=["Annotation", "Torso_PS", "Lwrist_PS", "Rwrist_PS", "Torso_Period", "LWrist_Period", "RWrist_Period"] )
    else:
        X_features = np.column_stack((annotations,torso_scores, lwrist_scores, rwrist_scores))
        if method == 'PS10':
            PS_df = pd.DataFrame(X_features, columns = (["Annotation"] + [f"Torso_{i}" for i in range(1, 11)] + [f"LWrist_{i}" for i in range(1, 11)] + [f"RWrist_{i}" for i in range(1, 11)]))
        else:
             PS_df = pd.DataFrame(X_features, columns=["Annotation", "Torso_PS", "Lwrist_PS", "Rwrist_PS"])
    
    # Insert first column with the file/session name
    PS_df.insert(0, "Session", filename_cleaned)
    
    PS_df.to_csv(f'{filename_cleaned}_scores_accelerometer.csv', index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a data folder.')
    parser.add_argument('--input', required=True, help='Path to the input accelerometer folder')
    args = parser.parse_args()
    process_folder(args.input)





