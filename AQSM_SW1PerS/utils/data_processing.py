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

import pickle
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.ndimage
from collections import Counter
import pandas as pd

# Section: Pickle File Tools
# -------------------------------------------------

def open_pickle(pkl_file = 'dataset.pkl'):
    '''
    Loads the .pkl file
    '''
    with open(pkl_file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
    

def segment_video(entry, segment_size):
    '''    
    Tool for segmenting video into a fixed window size and annotating the segments
    
    :Param filepath: entry of .pkl file
    :Param segment_size: Desired segmentation size
    :Return segments: Array of time intervals based on segment size
    :Return fps, frame times, segments, and their annotations
    '''
    fps, frame_count, duration = entry['fps'], entry['frame_count'], entry['duration']
    frame_times = [1 / fps * i for i in range(int(fps * duration))]
  
    step_size = 1/fps 
  
    segments = []
    annotated_segments = []
    
    start_time = frame_times[0]
    i = start_time
    while i <= np.max(frame_times) - segment_size:
        lower_bound = i
        upper_bound = i + segment_size
        x = np.array([lower_bound,upper_bound])
        segments.append(x)
        i += step_size
    segments = np.vstack(segments)
        
    frame_annotations = entry['annotations']
    
    for segment in segments:
        segment_start = np.min(segment)
        segment_end = np.max(segment)
        frame_start, frame_end = int(fps* segment_start), int(fps * segment_end)
        interval_annos = np.array(frame_annotations[frame_start:frame_end])
        interval_annos_flat = np.ravel(interval_annos)  # Flatten to 1D array
        # Count occurrences
        counts = Counter(interval_annos_flat.tolist())
        # Access counts
        num_normals = int(counts[0])
        num_rocks = int(counts[1])
        num_flaps = int(counts[2])
        num_flap_rock = int(counts[3])
        total_counts = num_normals + num_rocks + num_flaps + num_flap_rock
        # Default annotation
        annotation = 0
        
        # Ensure 100% Overlap
        if num_rocks / total_counts == 1:
            annotation = 1
        elif num_flaps / total_counts == 1:
            annotation = 2
        elif num_flap_rock / total_counts == 1:
            annotation = 3
        elif (0 < num_rocks / total_counts < 1) or \
             (0 < num_flaps / total_counts < 1) or \
             (0 < num_flap_rock / total_counts < 1):
            annotation = -1  

        annotated_segments.append(annotation)
        
    annotated_segments = np.vstack(annotated_segments)
    
    return fps, frame_times, segments, annotated_segments


# Section: MediaPipe Keypoint Handling
# -------------------------------------------------

def getChest(p1, p2, p3):
    '''
    Estimates the chest position as the average of three landmarks. This is a good stabalizer point
    '''
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    midpoint = [(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3]
    return np.column_stack(midpoint)

def compute_joint_angle(x1, y1, x2, y2, x3, y3):
    '''
    Computes the angle between two vectors: (x1,y1)->(x2,y2) and (x2,y2)->(x3,y3).
    '''
    v1 = np.array([x1 - x2, y1 - y2])
    v2 = np.array([x3 - x2, y3 - y2])
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return np.nan
    angle = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0))
    return np.degrees(angle)

def filter_visibility(keypoints, frame_times, landmark_index, do_wrists=False,
                      wrist_index=None, elbow_index=None, visibility_threshold=0.3):
    '''
    Filters out frames with low visibility landmarks.
    '''
    if do_wrists:
        wv = []
        ev = []
        if wrist_index is None or elbow_index is None:
            raise ValueError("wrist_index and elbow_index must be provided when do_wrists=True.")
        wrist_bad_frames = []
        elbow_bad_frames = []
        for idx, frame in enumerate(landmarks):
            wv.append(frame[wrist_index][2])
            ev.append(frame[elbow_index][2])
            if frame[wrist_index][2] < visibility_threshold:
                wrist_bad_frames.append(frame_times[idx])
            if frame[elbow_index][2] < visibility_threshold:
                elbow_bad_frames.append(frame_times[idx])
        return wrist_bad_frames, elbow_bad_frames
    else:
        bad_frames = []
        for idx, frame in enumerate(landmarks):
            if frame[landmark_index][2] < visibility_threshold:
                bad_frames.append(frame_times[idx])
        return bad_frames

def getAccel(values, fps, n_dim = 2):
    '''
    Computes the second-order central derivative of the time series,
    with forward and backward differences applied at the boundaries.
    '''
    h = 1 / fps
    if n_dim == 2:
        f_x = values[:, 0]
        f_y = values[:, 1]
        
        f_x_second_deriv = (f_x[2:] - 2 * f_x[1:-1] + f_x[:-2]) / h**2
        f_x_second_deriv_lower_bound = (f_x[1] - 2 * f_x[0] + f_x[2]) / h**2  
        f_x_second_deriv_upper_bound = (f_x[-3] - 2 * f_x[-2] + f_x[-1]) / h**2  
    
        f_y_second_deriv = (f_y[2:] - 2 * f_y[1:-1] + f_y[:-2]) / h**2
        f_y_second_deriv_lower_bound = (f_y[1] - 2 * f_y[0] + f_y[2]) / h**2  
        f_y_second_deriv_upper_bound = (f_y[-3] - 2 * f_y[-2] + f_y[-1]) / h**2
      
        f_x_second_derivative = np.concatenate(([f_x_second_deriv_lower_bound], f_x_second_deriv, [f_x_second_deriv_upper_bound]))
        f_y_second_derivative = np.concatenate(([f_y_second_deriv_lower_bound], f_y_second_deriv, [f_y_second_deriv_upper_bound]))
        
        return np.column_stack((f_x_second_derivative, f_y_second_derivative))
    else:
        f_x = values
        
        f_x_second_deriv = (f_x[2:] - 2 * f_x[1:-1] + f_x[:-2]) / h**2
        f_x_second_deriv_lower_bound = (f_x[1] - 2 * f_x[0] + f_x[2]) / h**2  
        f_x_second_deriv_upper_bound = (f_x[-3] - 2 * f_x[-2] + f_x[-1]) / h**2  

        return np.concatenate(([f_x_second_deriv_lower_bound], f_x_second_deriv, [f_x_second_deriv_upper_bound]))


def interprolate_missing_frames(df, col1, col2, method="cubic"):
    '''
    Cubic interpolates the two columns in the DataFrame then fills any leading NaNs with the first valid value (backfill) and  trailing NaNs with the last valid value (forward fill).
    '''
    df[col1] = df[col1].interpolate(method=method)
    df[col2] = df[col2].interpolate(method=method)
    
    # --- Handle leading NaNs: backfill from the first valid value ---
    if pd.isna(df[col1].iloc[0]):
        first_valid_idx = df[col1].first_valid_index()
        if first_valid_idx is not None:
            df.loc[:first_valid_idx, col1] = df.loc[first_valid_idx, col1]
            
    if pd.isna(df[col2].iloc[0]):
        first_valid_idx = df[col2].first_valid_index()
        if first_valid_idx is not None:
            df.loc[:first_valid_idx, col2] = df.loc[first_valid_idx, col2]
    
    # --- Handle trailing NaNs: forward fill from the last valid value ---
    if pd.isna(df[col1].iloc[-1]):
        last_valid_idx = df[col1].last_valid_index()
        if last_valid_idx is not None:
            df.loc[last_valid_idx:, col1] = df.loc[last_valid_idx, col1]
            
    if pd.isna(df[col2].iloc[-1]):
        last_valid_idx = df[col2].last_valid_index()
        if last_valid_idx is not None:
            df.loc[last_valid_idx:, col2] = df.loc[last_valid_idx, col2]
            
    return df


def extract_landmarks(entry, landmark_index, frame_times, fps, do_wrists=False,
                      elbow_index=None, shoulder_index=None):
    '''
    Extracts and cleans landmark (x, y) data.
    '''

    landmarks = entry['keypoints']
    
    if do_wrists:
        
        # Get bad visibility frames for wrist and elbow
        wrist_bad_frames, elbow_bad_frames = filter_visibility(landmarks, frame_times, landmark_index, do_wrists=True, wrist_index=landmark_index, elbow_index=elbow_index, visibility_threshold=0.2)
        # Extract (x,y) data for wrist, elbow, and shoulder from each frame.
        landmark_array = [[[x, y] for x, y, _ in frame] for frame in landmarks]
        wrist_points, elbow_points, shoulder_points = [], [], []
        for frame in landmark_array:
            if (landmark_index < len(frame) and elbow_index < len(frame) and shoulder_index < len(frame)):
                wrist_points.append(frame[landmark_index])
                elbow_points.append(frame[elbow_index])
                shoulder_points.append(frame[shoulder_index])
            else:
                continue
        
        wrist_points = np.vstack(wrist_points)
        elbow_points = np.vstack(elbow_points)
        shoulder_points = np.vstack(shoulder_points)

        #Dataframe seems to be easiest and cleanest way to do computations
        df = pd.DataFrame({
            "Time": frame_times[:len(wrist_points)],
            "X_Wrist": wrist_points[:, 0],
            "Y_Wrist": wrist_points[:, 1],
            "X_Elbow": elbow_points[:, 0],
            "Y_Elbow": elbow_points[:, 1],
            "X_Shoulder": shoulder_points[:, 0],
            "Y_Shoulder": shoulder_points[:, 1]
        })
        
        # Mark low visibility elbow frames as NaN.
        df.loc[df["Time"].isin(elbow_bad_frames), ["X_Elbow", "Y_Elbow"]] = np.nan
        
        df = interprolate_missing_frames(df, "X_Elbow", "Y_Elbow")

        # Compute elbow flexion angle.
        df["Elbow_Flexion_Angle"] = df.apply(lambda row: compute_joint_angle(
            row["X_Shoulder"], row["Y_Shoulder"],
            row["X_Elbow"], row["Y_Elbow"],
            row["X_Wrist"], row["Y_Wrist"]
        ), axis=1)
        
        angles = df["Elbow_Flexion_Angle"].values
        angular_acceleration = getAccel(angles, fps, n_dim = 1)
        df["Angular_Acceleration"] = angular_acceleration
        
        # Assign extreme angular acceleration to low visible frames
        df.loc[df["Time"].isin(wrist_bad_frames), "Angular_Acceleration"] = 10000

        #Flag frames that have abnormally high acceleration
        bad_frames = df[np.abs(df["Angular_Acceleration"]) > 500].index 
        df["Valid"] = True
        df.loc[bad_frames, "Valid"] = False
        df.loc[bad_frames, ["X_Wrist", "Y_Wrist"]] = np.nan

        # Interpolate and smooth wrist coordinates.
        df = interprolate_missing_frames(df, "X_Wrist", "Y_Wrist")
        
        sigma = 0.5
        df["X_Wrist"] = scipy.ndimage.gaussian_filter1d(df["X_Wrist"], sigma=sigma)
        df["Y_Wrist"] = scipy.ndimage.gaussian_filter1d(df["Y_Wrist"], sigma=sigma)
        
        return df[["X_Wrist", "Y_Wrist"]].to_numpy()
    
    else:
        # For non-wrist processing, filter visibility and extract the desired landmark.
        bad_visibility_frames = filter_visibility(landmarks, frame_times, landmark_index)
        landmark_array = [[[x, y] for x, y, _ in frame] for frame in landmarks]
        extracted = []
        for frame in landmark_array:
            if landmark_index < len(frame):
                extracted.append(frame[landmark_index])
            else:
                continue
        extracted = np.vstack(extracted)
        
        acceleration = getAccel(extracted, fps)
        
        df = pd.DataFrame({
            "Time": frame_times[:len(extracted)],
            "X": extracted[:, 0],
            "Y": extracted[:, 1]
        })
        df["Acceleration X"] = acceleration[:, 0]
        df["Acceleration Y"] = acceleration[:, 1]
        
        df.loc[df["Time"].isin(bad_visibility_frames), ["Acceleration X", "Acceleration Y"]] = 10000
        
        bad_frames = df[(np.abs(df["Acceleration X"]) > 4) | (np.abs(df["Acceleration Y"]) > 10)].index
        df["Valid"] = True
        df.loc[bad_frames, "Valid"] = False
        df.loc[bad_frames, ["X", "Y"]] = np.nan
        
        df = interprolate_missing_frames(df, "X", "Y")
        
        sigma = 0.5
        df["X"] = scipy.ndimage.gaussian_filter1d(df["X"], sigma=sigma)
        df["Y"] = scipy.ndimage.gaussian_filter1d(df["Y"], sigma=sigma)
        
        return df[["X", "Y"]].to_numpy()

