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
import pandas as pd
import xml.etree.ElementTree as ET
import datetime, calendar
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

# Section: Data Folder Tools
# -------------------------------------------------

def getTime(s):
    """Convert time string to Unix milliseconds."""
    spre, smilli = s.split(".")
    t = calendar.timegm(datetime.datetime.strptime(spre, "%Y-%m-%d %H:%M:%S").timetuple())
    return t * 1000 + float(smilli)

def to_unix_ms(dt):
    return int(calendar.timegm(dt.timetuple()) * 1000 + dt.microsecond // 1000)
    
def loadAnnotations(filename):
    """Load XML annotations with START_DT, STOP_DT, LABEL tags."""
    tree = ET.parse(filename)
    root = tree.getroot()
    anno = []
    for m in list(root):
        start, stop, label = -1, -1, ""
        for c in list(m):
            if c.tag == "START_DT":
                start = getTime(c.text)
            elif c.tag == "STOP_DT":
                stop = getTime(c.text)
            elif c.tag == "LABEL":
                label = c.text
        anno.append({"start": start, "stop": stop, "label": label})
    return anno

def loadAnnotationsFromXLSX(xlsx_path):
    """
    Load all annotations from an Excel file with start/stop/label info.
    Mirrors the format of loadAnnotations() for XML.

    Returns a list of dictionaries with:
    - 'start': start time in Unix ms
    - 'stop' : stop time in Unix ms
    - 'label': string label
    """
    df = pd.read_excel(xlsx_path, engine='openpyxl', header=None)
    anno = []

    for _, row in df.iterrows():
        try:
            label = str(row[3]).strip()
            start_dt = pd.to_datetime(row[10], errors='coerce')
            stop_dt = pd.to_datetime(row[11], errors='coerce')

            if pd.notnull(start_dt) and pd.notnull(stop_dt) and start_dt < stop_dt and label != "":
                anno.append({
                    "start": to_unix_ms(start_dt),
                    "stop": to_unix_ms(stop_dt),
                    "label": label
                })
        except Exception as e:
            print(f"[!!] Error parsing row: {e}")
            continue

    if not anno:
        raise ValueError("No valid annotations found in Excel file.")

    return anno

def getNormalAnnotations(anno, minTime=4000):
    """Create 'normal' annotations for gaps between labeled intervals."""
    nanno = []
    anno = sorted(anno, key=lambda x: x['start'])
    for i in range(len(anno) - 1):
        start = anno[i]['stop']
        stop = anno[i+1]['start']
        if stop - start >= minTime:
            nanno.append({"start": start, "stop": stop, "label": "Normal"})
    return nanno

def expandAnnotations(anno, time=4000, hop=200):
    """Convert each annotation into overlapping sub-windows."""
    newanno = []
    for a in anno:
        t1 = a['start']
        while t1 + time <= a['stop']:
            newanno.append({"start": t1, "stop": t1 + time, "label": a["label"]})
            t1 += hop
    return newanno
  
# Section: Accelerometer Processing
# -------------------------------------------------

def loadAccelerometerData(filepath):
    """Load accelerometer CSV file as numpy array."""
    return np.loadtxt(filepath, delimiter=",")

def smoothDataGaussian(x, sigma=3):
    return np.array([gaussian_filter1d(x[:, k], sigma=sigma) for k in range(x.shape[1])]).T

def getAccelerometerRange(X, a, interpolate_to_fixed_length=False, expected_length=380):
    t1, t2 = a['start'], a['stop']
    mask = (X[:, 0] >= t1) & (X[:, 0] < t2)
    window = X[mask, :]
    print(len(window))
    if interpolate_to_fixed_length:
        if window.shape[0] > 1:
            from scipy.interpolate import interp1d
            timestamps = window[:, 0]
            acc_data = window[:, 1:]
            t_uniform = np.linspace(timestamps[0], timestamps[-1], expected_length)
            interp_func = interp1d(timestamps, acc_data, axis=0, kind='linear', fill_value='extrapolate')
            acc_interp = interp_func(t_uniform)
            return np.column_stack((t_uniform, acc_interp))
        else:
            t_uniform = np.linspace(t1, t2, expected_length)
            if window.shape[0] == 1:
                fill_vals = np.tile(window[0, 1:], (expected_length, 1))
            else:
                fill_vals = np.zeros((expected_length, X.shape[1] - 1))
            return np.column_stack((t_uniform, fill_vals))

    return window  # include timestamp, ax, ay, az

def process_accelerometer_data(folder_path, annofile, accel_file):

    # Load annotations
    if '001-2010-05-28' in str(folder_path): #This file is special exception that only has correct information in the XLSX file rather than the XML file
        anno = loadAnnotationsFromXLSX(f"{folder_path}/{annofile}")
    else:
        anno = loadAnnotations(f"{folder_path}/{annofile}")

    anno = anno[1::]
    anno = [a for a in anno if "Good Data" not in str(a)]

    nanno = getNormalAnnotations(anno)
    
    anno = expandAnnotations(anno, time=4000, hop = 200)
    nanno = expandAnnotations(nanno, time=4000, hop = 200)
   
    positive_segments_labels = []
    negative_segments_labels = []

    max_negatives = min(int(len(anno) * 3), len(nanno))

    if len(nanno) > max_negatives:
        print("Subsampling %i negative annotations"%len(nanno))
        np.random.seed(42) 
        nanno = [nanno[k] for k in np.random.permutation(len(nanno))[0:max_negatives]]
        
    print("There are %i annotations and %i negative annotations"%(len(anno), len(nanno)))
    
    annotation = anno + nanno

    return annotation

def get_accel_data(folder_path, accel_file):
  XsAccel  = loadAccelerometerData(f"{folder_path}/{accel_file}")
  return XsAccel
