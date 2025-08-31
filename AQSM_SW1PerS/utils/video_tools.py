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

import os
import cv2
import math
import numpy as np
import datetime, calendar
import xml.etree.ElementTree as ET
import glob
from collections import Counter
import pandas as pd
from AQSM_SW1PerS.utils.accelerometer import *

#See AQSM_SW1PerS.utils.accelerometer for details on how to access annotations

# Section: Write videos from images in dataset using annotations
# ---------------------------------------------------------------

def get_current_label(annotations, current_time):
    '''
    Get the annotation label for the current frame based on the time within the interval.
    '''
    for annotation in annotations:
        if annotation["start"] <= current_time <= annotation["stop"]:
            if  annotation["label"] == 'Rock' or annotation["label"] == 'Flap' or annotation["label"] == 'Flap-Rock':
                return annotation["label"]
    return "Normal"


def findFrames(folder_path, start_unix, end_unix):
    '''
    This is necessary to get the total number of frames needed to compute the fps of the video
    '''
    total_frames=0
     # Use glob to get all image paths in nested folders
    image_paths = sorted(glob.glob(f"{folder_path}/**/*.jpg", recursive=True))

    for img_path in image_paths:
        img_name = os.path.basename(img_path).split('.')[0]
        timestamp = datetime.datetime.strptime(img_name, '%Y-%m-%d-%H-%M-%S-%f')
        frame_time = to_unix_s(timestamp)

        # Check if frame is within the "Good Data" interval
        if start_unix <= frame_time <= end_unix:
            total_frames += 1
    return total_frames


def write_video(folder_path, output_name, good_data, annotations):
    '''
    This takes in the files from autism data and converts them into videos, one for each 'Good Data' interval.
    Each frame is labeled with its annotation, and the annotations for each video part are saved in a list.
    '''
    video_annotations = []  # List to store annotations for each video part

    for i, (start_unix, end_unix) in enumerate(good_data):
        part_annotations = []  # List to store annotations for the current part
        
        total_frames = findFrames(folder_path, start_unix, end_unix)
        fps = total_frames/(end_unix - start_unix)
        # Set up video writer for each part
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{output_name}_part_{i+1}.mp4', fourcc, fps, (352, 288))  # Adjust dimensions as needed

        # Use glob to get all image paths in nested folders
        image_paths = sorted(glob.glob(f"{folder_path}/**/*.jpg", recursive=True))

        for img_path in image_paths:
            img_name = os.path.basename(img_path).split('.')[0]
            timestamp = datetime.datetime.strptime(img_name, '%Y-%m-%d-%H-%M-%S-%f')
            frame_time = to_unix_s(timestamp)

            # Check if frame is within the "Good Data" interval
            if start_unix <= frame_time <= end_unix:
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip if image could not be read

                # Get the annotation label for the current frame
                annotation_label = get_current_label(annotations[i], frame_time - start_unix)
                num_label = 0
                if annotation_label:
                    if annotation_label == 'Rock':
                        num_label = 1
                    elif annotation_label == 'Flap':
                        num_label = 2
                    elif annotation_label == 'Flap-Rock':
                        num_label = 3  
                        
                    part_annotations.append(num_label)  # Store annotation for this frame
                    # Overlay the annotation label onto the frame
                    cv2.putText(img, annotation_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    part_annotations.append(num_label)
                # Write the frame to the video file
                out.write(img)
                cv2.imshow('Frame',img)
        # Finalize the video for this part
        out.release()
        cv2.destroyAllWindows()

        # Add the collected annotations for this part to the overall list
        video_annotations.append(part_annotations)

    return video_annotations

# Section: Get relevant video information
# ----------------------------------------

def get_video_info(video_path):
    '''
    Tool to get the number of frames in the video, the fps, and the total duration of the video in seconds. 
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
    fps = cap.get(cv2.CAP_PROP_FPS)
    _, image = cap.read()
    count = 0
    success = True
    while success: 
        success,image = cap.read()
        count += 1
    total_frames = count
    duration_seconds = total_frames / fps
    cap.release()
    return fps, total_frames, duration_seconds

