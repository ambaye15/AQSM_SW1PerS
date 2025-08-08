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

from AQSM_SW1PerS.utils.video_tools import *
from AQSM_SW1PerS.mediapipe_pose import *
from AQSM_SW1PerS.utils.paths import get_data_path

import pandas as pd
import argparse

'''
IMPORTANT - Ensure that 'Videos' directory is located in same directory as this script
'''
def getKeypoints(video_path):

    windows_os = False #If you are using a windows OS, you may need to turn this to true since we trained YOLO model on UNIX
    if windows_os:
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
        
    model_path = get_data_path("YOLOv5l/yolov5/runs/train/exp/weights", "YOLOv5l_transfer.pt") 
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    fps, total_frames, duration_seconds, frame_times = get_video_info(video_path)
    
    create_mediapipe_file(video_path, frame_times, fps, model, create_video = False) 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video file.')
    parser.add_argument('--input', required=True, help='Path to the input video file')
    args = parser.parse_args()
    getKeypoints(args.input)
