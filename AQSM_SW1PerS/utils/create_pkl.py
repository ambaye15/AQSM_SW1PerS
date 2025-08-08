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
import pickle
import os

def create_pickle_dataset(keypoint_folder = 'MediaPipeData', annotations_csv = 'study_annotations.csv', output_file = 'dataset.pkl')
    '''
    Makes a cohesive .pkl file that includes information for all videos. 
    :Param keypoint_folder = 'MediaPipeData': Directory where all MediaPipe keypoint information was collected
    :Param annotations_csv = 'study_annotations.csv': Dataframe that contains annotations enocded into each frame of every video. (I provide this in the Dataset folder)
    :Param output_file = 'dataset.pkl': Name the .pkl file that all the information gets written to
    '''
    annotations_df = pd.read_csv(annotations_csv)
    
    entries = []
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):  # Process only CSV files
            file_path = os.path.join(folder_path, file_name)
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            total_joints = len([col for col in df.columns if col.startswith('joint_') and '_x' in col])
            
            keypoints = []
            for _, row in df.iterrows():
                frame_keypoints = []
                for i in range(total_joints):
                    x = row.get(f'joint_{i}_x', None)
                    y = row.get(f'joint_{i}_y', None)
                    visibility = row.get(f'joint_{i}_visibility', None)
                    frame_keypoints.append((x, y, visibility))  # Store keypoint as a tuple
                keypoints.append(frame_keypoints)
            
            video_name = file_name.replace('.csv', '')  # Use file name as video name
            video_path = f'Vids/{video_name}.avi' #Change to where videos are located
            fps, frame_count, duration = get_video_info(video_path)
            
            if 'study1' in video_name:
                vid_name = video_name.replace('_study1', '')
            else:
                vid_name = video_name.replace('_study2', '')
                
            video_annotations = np.vstack(annotations_df[vid_name].dropna().to_numpy())[:frame_count]

            #Put all metadata together
            entry = {
                'keypoints': keypoints,
                'annotations': video_annotations,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'name': video_name
            }
            entries.append(entry)
    
    with open(output_file, 'wb') as f:
        pickle.dump(entries, f)
    
    print(f"All video keypoints have been saved to '{output_file}'.")
