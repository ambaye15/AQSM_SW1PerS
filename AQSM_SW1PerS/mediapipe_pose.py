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
from pathlib import Path
import cv2
import math
import numpy as np
import pandas as pd
#Video Tools
from filterpy.kalman import KalmanFilter
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose


# Section: YOLOv5 Box Tracking (This can be adjusted for any application)
# -----------------------------------------------------------------------

def getClass(video_name):
    '''
    Returns YOLOv5 class that will be used on video for each subject
    '''
    if '001' in str(video_name):
        return 0
    elif '002' in str(video_name):
        return 1
    elif '003' in str(video_name):
        return 2
    elif '004' in str(video_name):
        return 3
    elif '005' in str(video_name):
        return 4
    elif '006' in str(video_name):
        return 5
    else: 
        return 6
        

# Section: Pose Inference
# ------------------------

class MP_pose:
    
    def __init__(self, time_values, fps, model, create_video = True, min_detection_confidence = 0.3, min_tracking_confidence = 0.3, model_complexity = 2):
        self.time_values = time_values
        self.fps = fps
        self.model = model
        self.create_video = create_video
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity

        self.dims = None


    def _apply_clahe(self, frame, clip_limit=2.0, tile_grid_size=(8, 8)):
        '''
        Apply CLAHE (Adaptive Histogram Equalization) to enhance image contrast.
        For color frames, CLAHE is applied to the luminance channel.
        '''
        if len(frame.shape) == 3:  # Color frame
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:  # Grayscale frame
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(frame)
        return enhanced

    def _gamma_correction(self, frame, gamma=1.5):
        '''
        Apply gamma correction to adjust frame brightness.
        Gamma values > 1.0 brighten the frame, while gamma < 1.0 darken it.
        '''
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(frame, table)

    def _getNormalizedFrameLandmark(self, lmrk, dims, xmin, xmax, ymin, ymax):
        '''
        Normalized landmark with respect to frame, instead of YOLOv5 bounding box
        '''
        frame_width, frame_height = self.dims
        x, y = lmrk[0], lmrk[1]
        w, h = np.abs(xmax - xmin), np.abs(ymax - ymin)
        x_unnormalized = x * w + xmin
        y_unnormalized = y * h + ymin
        x_frame = np.abs((x_unnormalized - frame_width) / frame_width)
        y_frame = np.abs((y_unnormalized - frame_height) / frame_height)
        return x_frame, y_frame

    def _create_kalman_filter(self):
        dt = 1 / self.fps  # Time step
        
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State Transition Matrix (F)
        kf.F = np.array([
            [1, 0, 0, 0, dt, 0,  0,  0],
            [0, 1, 0, 0, 0,  dt, 0,  0],
            [0, 0, 1, 0, 0,  0,  dt, 0],
            [0, 0, 0, 1, 0,  0,  0,  dt],
            [0, 0, 0, 0, 1,  0,  0,  0],
            [0, 0, 0, 0, 0,  1,  0,  0],
            [0, 0, 0, 0, 0,  0,  1,  0],
            [0, 0, 0, 0, 0,  0,  0,  1]
        ])
        
        # Measurement Matrix (H)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process Noise Covariance (Q)
        kf.Q *= 0.01
        
        # Measurement Noise Covariance (R)
        kf.R *= 1.0
        
        # State Covariance Matrix (P)
        kf.P *= 10.0
        
        return kf
    
    def _update_kalman_filter(self, kf, measurement):
        # Predict the next state
        kf.predict()
        
        # Update with the current measurement (YOLOv5 bounding box)
        kf.update(measurement)
        
        corrected_state = kf.x
        return corrected_state[:4]  # Return only position and size

    def _run_pose_inference(self, video_path):
        '''
        Using MediaPipe and YOLOv5, we get all (x,y) keypoint coordinates that will automatically be saved to a .csv file
        '''
        path = Path(video_path)
        cleaned_name = path.stem
        
        output_video = f'{cleaned_name}_mp.avi'
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.dims = (frame_width,frame_height)
        
        if self.create_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_video, fourcc, self.fps, self.dims)
    
        #Decide what child we need to track for the YOLOv5 model
        cls = getClass(video_path)
        
        #For tracking centroids of current frame and previous frame
        kf = self._create_kalman_filter()
    
        data = []
        frame_count = 0
    
        # Total number of joints in Mediapipe Pose
        total_joints = len(mp_pose.PoseLandmark)

        with mp_pose.Pose(min_detection_confidence = self.min_detection_confidence, min_tracking_confidence = self.min_tracking_confidence, model_complexity = self.model_complexity) as pose:
            for timestamp in self.time_values:
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                frame_data = {'frame': frame_count}
                
                if not ret:
                    for idx in range(total_joints):
                        frame_data[f'joint_{idx}_x'] = np.nan
                        frame_data[f'joint_{idx}_y'] = np.nan
                        frame_data[f'joint_{idx}_visibility'] = np.nan
                    data.append(frame_data)
                    continue
    
                frame =  self._gamma_correction(frame)
                frame = self._apply_clahe(frame)
    
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                #Run YOLOv5 on the frame
                results = self.model(image)
                annotated_frame = results.render()[0]
                
                current_bbox = None
    
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                #Obtain the index of the most confident bounding box for the child
                detection_index = 0
                max_confidence = 0
                for idx, det in enumerate(results.xyxy[0].tolist()):
                    xmin, ymin, xmax, ymax, confidence, clas = det
                    if clas == cls and confidence >= max_confidence:
                        max_confidence = confidence
                        detection_index = idx
                        
                MARGIN = 15 #Add some padding to bonding box
                detected = False
                for idx2, det in enumerate(results.xyxy[0].tolist()):
                    if idx2 == detection_index:
                        xmin, ymin, xmax, ymax, confidence, clas = det
                        if clas == cls:
                            current_bbox = [xmin, ymin, xmax, ymax]
                
                            # Convert to [x_center, y_center, width, height] for Kalman Filter
                            x_center = (xmin + xmax) / 2
                            y_center = (ymin + ymax) / 2
                            width = xmax - xmin
                            height = ymax - ymin
                            yolo_bbox = [x_center, y_center, width, height]
                            
                            # Smooth bounding box with Kalman Filter
                            smoothed_bbox = self._update_kalman_filter(kf, np.array(yolo_bbox))
                            xmin = smoothed_bbox[0].item() - smoothed_bbox[2].item() / 2
                            ymin = smoothed_bbox[1].item() - smoothed_bbox[3].item() / 2
                            xmax = smoothed_bbox[0].item() + smoothed_bbox[2].item() / 2
                            ymax = smoothed_bbox[1].item() + smoothed_bbox[3].item() / 2
                            
                            x_min = int(max(0, xmin - MARGIN))
                            y_min = int(max(0, ymin - MARGIN))
                            x_max = int(min(image.shape[1], xmax + MARGIN))
                            y_max = int(min(image.shape[0], ymax + MARGIN))
                           
                            cropped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                            cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                            #Run MediaPipe within the bounding box
                            results = pose.process(cropped_rgb)
    
                            if results.pose_landmarks:
                                # Pose detected, extract landmark data
                                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                                    x, y = landmark.x, landmark.y
                                    #Re-normalize with respect to frame
                                    x_new, y_new = self._getNormalizedFrameLandmark([x,y],x_min,x_max,y_min,y_max)
                                    frame_data[f'joint_{idx}_x'] = x_new
                                    frame_data[f'joint_{idx}_y'] = y_new
                                    frame_data[f'joint_{idx}_visibility'] = landmark.visibility
                                    
                                mp_drawing.draw_landmarks(cropped_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                            else:
                                # Pose not detected, fill with NaN
                                for idx in range(total_joints):
                                    frame_data[f'joint_{idx}_x'] = np.nan
                                    frame_data[f'joint_{idx}_y'] = np.nan
                                    frame_data[f'joint_{idx}_visibility'] = np.nan
                        
                            data.append(frame_data)
                
                            detected = True
                            break
    
                if not detected:
                    for idx in range(total_joints):
                        frame_data[f'joint_{idx}_x'] = np.nan
                        frame_data[f'joint_{idx}_y'] = np.nan
                        frame_data[f'joint_{idx}_visibility'] = np.nan
                    data.append(frame_data)
                    
                if self.create_video:
                    out.write(frame)
                
                if cv2.waitKey(29) & 0xFF == ord('q'):
                    break
                    
                frame_count += 1
                
        if self.create_video:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        df = pd.DataFrame(data)
        df.to_csv(f'{cleaned_name}.csv', index=False) #Pick where you want to send file
    
