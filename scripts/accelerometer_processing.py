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

import pandas as pd
import argparse
import os
import numpy as np
from pathlib import Path
from AQSM_SW1PerS.utils.accelerometer import *
from AQSM_SW1PerS.SW1PerS import *
from accelerometer_SW1PerS import *
from AQSM_SW1PerS.utils.paths import get_data_path

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
        results = pool.map(processAccel, args_list)

  # Unpack the results
  torso_scores, torso_period = results[0]
  lwrist_scores, lwrist_period = results[1]
  rwrist_scores, rwrist_period = results[2]

  annotations = extract_annotations(meta_data)

  path = Path(folder_path)
  filename_cleaned = path.name
  
  if include_period:
    X_features = np.column_stack((annotations,torso_scores, lwrist_scores, rwrist_scores))
    if method == 'PS10':
      PS_df = pd.DataFrame(
          X_features,
          columns = (
              ["Annotation"] +
              [f"Torso_{i}" for i in range(1, 11)] +
              [f"LWrist_{i}" for i in range(1, 11)] +
              [f"RWrist_{i}" for i in range(1, 11)]
              ) )
    else:
      PS_df = pd.DataFrame(
        X_features,
        columns=["Annotation", "Torso_PS", "Lwrist_PS", "Rwrist_PS"] )
    
  else:
    X_features = np.column_stack((annotations,torso_scores, lwrist_scores, rwrist_scores, torso_period, lwrist_scores, rwrist_scores))

    if method == 'PS10':
      PS_df = pd.DataFrame(
          X_features,
          columns = (
              ["Annotation"] +
              [f"Torso_{i}" for i in range(1, 11)] +
              [f"LWrist_{i}" for i in range(1, 11)] +
              [f"RWrist_{i}" for i in range(1, 11)] +
              ["Torso_Period", "LWrist_Period", "RWrist_Period"]
              ) )
    else:
      PS_df = pd.DataFrame(
        X_features,
        columns=["Annotation", "Torso_PS", "Lwrist_PS", "Rwrist_PS", "Torso_Period", "LWrist_Period", "RWrist_Period"] )
    
  # Insert first column with the file/session name
  PS_df.insert(0, "Session", filename_cleaned)
  
  PS_df.to_csv(f'{filename_cleaned}_scores_accelerometer.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a data folder.')
    parser.add_argument('--input', required=True, help='Path to the input accelerometer folder')
    args = parser.parse_args()
    process_folder(args.input)




  

  
    
  

  

  

  

  
