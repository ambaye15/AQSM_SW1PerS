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
from datetime import datetime 
import xml.etree.ElementTree as ET
import glob
from collections import Counter
import pandas as pd


# Section: Convert Times
# -------------------------------------------------

def getTime(s):
    '''
    Convert time from YYYY-MM-DD HH:MM:SS.mmm into seconds 
    '''
    if isinstance(s, datetime):  
        return s.timestamp()
    else:
        t = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
        return t.timestamp()
      

def getUnix(dt_str):
    '''
    Convert to Unix timestamp
    '''
    dt_obj = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt_obj.timestamp()
  
   
def parse_datetime(datetime_str):
    '''
    Parse the datetime string
    '''
    format_str = '%Y-%m-%d-%H-%M-%S-%f'
    return datetime.strptime(datetime_str, format_str)

# Section: Annotation Tools
# -----------------------------------------------

def loadAnnotations(filename):
    '''
    Load annotations into a dictionary format and extract UNIX times for each Good Data interval.
    Each interval's annotations are stored in a distinct list.
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    annotations = []
    good_data = []

    # Identify Good Data intervals
    for m in root: #This line iterates over the top-level elements (or "child" elements) directly under the root element of the XML file.
        for c in m: #This line iterates over the sub-elements (or "children") of each element m
            if c.tag == "LABEL" and c.text == "Good Data": #This line checks if the tag name of the sub-element c is "LABEL" and if its text content (i.e., the text between <LABEL> and </LABEL> in the XML file) is "Good Data"
                good_data_start_unix = getUnix(m.find("START_DT").text)
                good_data_end_unix = getUnix(m.find("STOP_DT").text)
                good_data.append((good_data_start_unix, good_data_end_unix))

    # Process annotations for each Good Data interval
    
    for start_unix, stop_unix in good_data:
        anno = []
        for m in root:
            start = -1
            stop = -1
            label = ""
            for c in m:
                if c.tag == "START_DT":
                    start = getTime(c.text) - start_unix
                elif c.tag == "STOP_DT":
                    stop = getTime(c.text) - start_unix
                elif c.tag == "LABEL":
                    label = c.text

            # Only append valid annotations within the Good Data interval
            if 0 <= start < (stop_unix - start_unix) and stop > 0:
                start = max(start, 0)
                stop = min(stop, stop_unix - start_unix)
                anno.append({"start": start, "stop": stop, "label": label})
        annotations.append(anno)
    return annotations, good_data

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
        timestamp = parse_datetime(img_name)
        frame_time = getTime(timestamp)

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
            timestamp = parse_datetime(img_name)
            frame_time = getTime(timestamp)

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

