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
from pathlib import Path
import numpy as np
from AQSM_SW1PerS.utils.data_processing import *
from AQSM_SW1PerS.SW1PerS import *
from AQSM_SW1PerS.utils.paths import get_data_path


'''
IMPORTANT - Ensure that 'Videos' directory is contained in the same directory as this script. Otherwise change bash script to read from a .txt file for ease
'''

def subsample_normal_intervals(segments, segment_annotations):
    # Split segments into positive and negative sets
    positive_segments = []
    positive_segments_labels = []
    
    negative_segments = []
    negative_segments_labels = []
    
    for seg, label in zip(segments, segment_annotations):
        if label in [1, 2, 3]:
            positive_segments.append((seg))
            positive_segments_labels.append(label)
        elif label == 0:
            negative_segments.append((seg))
            negative_segments_labels.append(label)
    
    max_negatives = min(int(len(positive_segments) * 3), len(negative_segments))
    
    if len(negative_segments) > max_negatives:
        print(f"Subsampling {len(negative_segments)} negative annotations...")
        np.random.seed(42)  # For reproducibility
        selected_idx = np.random.permutation(len(negative_segments))[:max_negatives]
        negative_segments = [negative_segments[i] for i in selected_idx]
        negative_segments_labels = [negative_segments_labels[i] for i in selected_idx]
    
    # Combine the positives and the subsampled negatives
    sampled_segments = positive_segments + negative_segments
    sampled_segment_labels = positive_segments_labels + negative_segments_labels

    return sampled_segments, np.vstack(sampled_segment_labels)

    
def process(input_video):

    pkl_file = get_data_path("dataset.pkl")

    video_data = open_pickle(pkl_file)

    path = Path(input_video)
    filename_cleaned = path.stem  
    
    video_entry = next((entry for entry in video_data if entry['name'] == filename_cleaned), None)
        
    fps, frame_times, segments, segment_annotations = segment_video(video_entry, 4) 

    sampled_segments, sampled_segment_labels = subsample_normal_intervals(segments, segment_annotations)

    head = extract_keypoints(video_entry, 0, frame_times, fps)
    lshoulder = extract_keypoints(video_entry, 11, frame_times, fps)
    rshoulder = extract_keypoints(video_entry, 12, frame_times, fps)
    rwrist = extract_keypoints(video_entry, 16, frame_times, fps, do_wrists=True, elbow_index=14, shoulder_index=12)
    lwrist = extract_keypoints(video_entry, 15, frame_times, fps, do_wrists=True, elbow_index=13, shoulder_index=11)
    
    chest = getChest(head, lshoulder, rshoulder)
    
    lshoulder_accel = getAccel(lshoulder,fps)
    rshoulder_accel = getAccel(rshoulder,fps)
    
    rwrist_accel = getAccel(rwrist,fps)
    lwrist_accel = getAccel(lwrist,fps)
    
    head_accel = getAccel(head,fps)
    chest_accel = getAccel(chest,fps)

    #Raw Position Movement Splines

    head_splines = [CubicSpline(frame_times,head[:,0]), CubicSpline(frame_times,head[:,1])]

    rw_splines = [CubicSpline(frame_times,rwrist[:,0]), CubicSpline(frame_times,rwrist[:,1])]
    
    lw_splines = [CubicSpline(frame_times,lwrist[:,0]), CubicSpline(frame_times,lwrist[:,1])]
    
    rs_splines = [CubicSpline(frame_times,rshoulder[:,0]), CubicSpline(frame_times,rshoulder[:,1])]
    
    ls_splines = [CubicSpline(frame_times,lshoulder[:,0]), CubicSpline(frame_times,lshoulder[:,1])]
    
    chest_splines = [CubicSpline(frame_times,chest[:,0]), CubicSpline(frame_times,chest[:,1])]
    
    #Acceleration Representation of Keypoint Movement Splines
    
    head_accel_splines = [CubicSpline(frame_times,head_accel[:,0]), CubicSpline(frame_times,head_accel[:,1])]
    
    rw_accel_splines = [CubicSpline(frame_times,rwrist_accel[:,0]), CubicSpline(frame_times,rwrist_accel[:,1])]
    
    lw_accel_splines = [CubicSpline(frame_times,lwrist_accel[:,0]), CubicSpline(frame_times,lwrist_accel[:,1])]
    
    rs_accel_splines = [CubicSpline(frame_times,rshoulder_accel[:,0]), CubicSpline(frame_times,rshoulder_accel[:,1])]
    
    ls_accel_splines = [CubicSpline(frame_times,lshoulder_accel[:,0]), CubicSpline(frame_times,lshoulder_accel[:,1])]
    
    chest_accel_splines = [CubicSpline(frame_times,chest_accel[:,0]), CubicSpline(frame_times,chest_accel[:,1])]

    args_list = [
        (head_splines, sampled_segments),
        (rw_splines, sampled_segments),
        (lw_splines, sampled_segments),
        (rs_splines, sampled_segments),
        (ls_splines, sampled_segments),
        (chest_splines, sampled_segments),
        (head_accel_splines, sampled_segments),
        (rw_accel_splines, sampled_segments),
        (lw_accel_splines, sampled_segments),
        (rs_accel_splines, sampled_segments),
        (ls_accel_splines, sampled_segments),
        (chest_accel_splines, sampled_segments) ]
    
    # Create a pool of workers and map tasks
    with multi.Pool(multi.cpu_count()) as pool:
        results = pool.map(processFeatures, args_list)

    # Unpack the results
    head_scores = results[0]
    rwrist_scores = results[1]
    lwrist_scores = results[2]
    rshoulder_scores = results[3]
    lshoulder_scores = results[4]
    chest_scores = results[5]
    head_scores_accel = results[6]
    rwrist_scores_accel = results[7]
    lwrist_scores_accel = results[8]
    rshoulder_scores_accel = results[9]
    lshoulder_scores_accel = results[10]
    chest_scores_accel = results[11]
    
    X_features_raw = np.column_stack((sampled_segment_labels,head_scores,rwrist_scores,lwrist_scores,rshoulder_scores,lshoulder_scores,chest_scores))
    X_features_accel = np.column_stack((sampled_segment_labels,head_scores_accel,rwrist_scores_accel,lwrist_scores_accel,rshoulder_scores_accel,lshoulder_scores_accel,chest_scores_accel))

    raw_scores_df = pd.DataFrame(X_features_raw)
    accel_scores_df = pd.DataFrame(X_features_accel)
    
    #Save the scores to csv
    raw_scores_df.to_csv(f'{filename_cleaned}_scores_position.csv', index=False)
    accel_scores_df.to_csv(f'{filename_cleaned}_scores_accel.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video file.')
    parser.add_argument('--input', required=True, help='Path to the input video file')
    args = parser.parse_args()
    process_Feats(args.input)
   
