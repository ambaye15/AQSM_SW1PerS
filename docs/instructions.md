
# AQSM_SW1PerS: Instructions for Installation and Use

**Author: Austin MBaye** 

---

## Overview

**AQSM_SW1PerS** is a Python package for analyzing repetitive motion patterns using pose estimation and topological data analysis. It was originally developed to support the classification and analysis of *Stereotypical Motor Movements (SMMs)* observed in individuals with Autism Spectrum Disorder (ASD), but can be applied to any dataset where one aims to quantify and interpret reccurrent patterns in time series or motion in videos. The package provides modules for:

- Motion tracking via Pose Estimation
- Period estimation of $\geq$ 1-variable time series
- Sliding Windows Embeddings
- Persistence diagram computation
- Classification tasks

---

## System Requirements

- Python 3.10 or higher  
- PyTorch 2.1.0  
- CUDA 11.8  
- Recommended: virtual environment (`venv` or `conda`)  
- OS: Windows, Linux, or macOS  

---

## Installation

### Option 1: Install via GitHub (recommended)

```bash
pip install git+https://github.com/ambaye15/AQSM_SW1PerS.git
```

### Option 2: Clone and Install Locally

```bash
git clone https://github.com/ambaye15/AQSM_SW1PerS.git
cd AQSM_SW1PerS
pip install -e .
```

### HPC Example Setup

For the bulk of data analysis, we used a high-performance computing (HPC) cluster. Dependencies were installed from the provided \texttt{requirements.txt} file within a custom Conda environment using the following setup:

```bash
# Load required modules
module load anaconda3/2022.05
module load cuda/11.8

# Create and activate environment
conda create --name pytorch_env -c conda-forge python=3.10 -y
source activate pytorch_env

# Install core packages
conda install jupyterlab -y
pip install torch torchvision torchaudio
```
The HPC allowed easier access to GPU partitions for YOLOv5 inference. However, this was also easy to run on laptop or PC with a GPU and a virtual environment.
---

To verify correct installation and behavior, run the included unit tests:

```bash
python -m unittest tests/test_periodicity.py
```

## Directory Structure

- `AQSM_SW1PerS/`: Core package with analysis and processing modules. 
- `classification_tools/`: Code for training and optimizing models.  
- `notebooks/`: Usage tutorials and visualizations.
- `Dataset/`: (**Must Download**) Preprocessed data, YOLO model results, periodicity scores. You must download this seperatley given this link: 

---

## Data Overview

### `data/`

**Goodwin’s paper in UbiComp ’14**  
Matthew S. Goodwin, Marzieh Haghighi, Qu Tang, Murat Akcakaya, Deniz Erdogmus, and Stephen Intille.  
2014. *Moving towards a real-time system for automatically recognizing stereotypical motor movements in individuals on the autism spectrum using wireless accelerometry.*  
In *Proceedings of the 2014 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp '14)*.  
ACM, New York, NY, USA, 861–872.  
DOI: [10.1145/2632048.2632096](http://doi.acm.org/10.1145/2632048.2632096)

**data directory download:**  
Available at: [Bitbucket Repository](https://bitbucket.org/mhealthresearchgroup/stereotypypublicdataset-sourcecodes/download)

Contains the raw accelerometer data and annotation files. Study 1 and Study 2 sessions were collected 2–3 years apart.

| Participant ID | Study 1 Sessions                          | Study 2 Sessions                          |
|----------------|-------------------------------------------|-------------------------------------------|
| 1              | URI-001-01-18-08 <br> URI-001-01-25-08    | 001-2010-05-25 <br> 001-2010-05-28 <br> 001-2010-06-01 |
| 2              | URI-002-01-18-08 <br> URI-002-01-24-08    | 002-2010-06-04 <br> 002-2011-06-02        |
| 3              | URI-003-01-18-08 <br> URI-003-02-08-08    | 003-2010-05-07 <br> 003-2011-05-23        |
| 4              | URI-004-01-17-08 <br> URI-004-02-07-08    | 004-2010-04-27 <br> 004-2010-05-11 <br> 004-2011-03-22 |
| 5              | URI-005-01-16-08 <br> URI-005-02-08-08    | 005-2010-05-17 <br> 005-2011-05-25        |
| 6              | URI-006-01-15-08 <br> URI-006-01-23-08    | 006-2010-03-1                             |

Inside each session folder you can find:

| Study type        | Study 1                                | Study 2                                |
|-------------------|-----------------------------------------|-----------------------------------------|
| **Session name**  | `URI-00X-MM-DD-YY`                     | `00X-YYYY-MM-DD`                        |
| **Raw data files**<br>(in `.csv`) | `MITes_01_RawCorrectedData_Trunk.RAW_DATA.csv` <br> `MITes_08_RawCorrectedData_Left-wrist.RAW_DATA.csv` <br> `MITes_11_RawCorrectedData_Right-wrist.RAW_DATA.csv` <br> `Wocket_00_RawCorrectedData_Right-Wrist.csv` <br> `Wocket_01_RawCorrectedData_Left-Wrist.csv` <br> `Wocket_02_RawCorrectedData_Torso.csv` | *(same structure as Study 1)* |
| **Annotation files**<br>(in `.xlsx`, `.xml`) | `Annotator1Stereotypy.annotation.*` <br> `Phone.annotation.*` <br> `AnnotationPhoneIntervals.xlsx` <br> `AnnotationVideo1Intervals.xlsx` <br> `AnnotationVideo2Intervals.xlsx` | `Annotator1Stereotypy.annotation.*` <br> `Annotator2Stereotypy.annotation.*` <br> `Phone.annotation.*` |

1. **Raw data files**  
   - Stored in `.csv` format.  
   - File naming convention encodes **sensor type**, **sensor ID**, and **sensor location**, separated by `_`.  
   - Example:  
     - `MITes_01_RawCorrectedData_Trunk.RAW_DATA.csv`  
       - **MITes** = sensor type  
       - **01** = sensor ID  
       - **Trunk** = sensor location (i.e., **Torso**)  
   - The same naming convention applies to both **Study 1** and **Study 2** directories.
     The raw data `.csv` file has **four columns** representing:

    1. **Unix timestamp**  
    2. **Raw x value**  
    3. **Raw y value**  
    4. **Raw z value**

    Notes:
    - There is **no header row**.  
    - The raw value is an **integer**:  
      - Range: `0–1023` for **Wockets**  
      - Range: `0–512` for **MITes**  
    - Dynamic range:  
      - ±2g for **MITes** sensors  
      - ±4g for **Wockets** sensors  
    
    **Example rows:**
   | Unix timestamp | Raw X | Raw Y | Raw Z |
    |----------------|-------|-------|-------|
    | 1274781775000  | 475   | 436   | 435   |
    | 1274781775009  | 474   | 437   | 436   |
    | 1274781775017  | 474   | 438   | 433   |
    | ...            | ...   | ...   | ...   |
    | 1274782392242  | 501   | 469   | 415   |
    | 1274782392253  | 501   | 468   | 416   |
    | 1274782392263  | 504   | 466   | 418   |

3. **Annotation files**  
   - Stored in multiple formats: `.xlsx`, `.xml`, and `.csv`.  
   - Contents are the same across formats (provided for user convenience).  

   a) **Annotation types**  
   - `Annotator1Stereotypy.annotation.*` or `Annotator2Stereotypy.annotation.*` → **offline annotations**  
   - `Phone.annotation.*` → **online annotation**  

   b) **Interval annotations**  
   - Files with the pattern:  
     - `Annotation*Intervals.xlsx`  
     are used specifically for 001-2010-05-28 due to legacy reasons

   c) **Annotation data format (`*.annotation.xml`)**
   - Annotated activity labels are encoded in a human and computer readable XML file. This is stored in the Annotation subdirectory inside the session folder. Some datasets may be labeled by more than one annotator using more than one category or set. Each file uses the XML format. Annotations provide time-stamped activity labels aligned to the raw accelerometer data and provide useful timestamps to write videos.
      - **Annotation node**
        - `<ANNOTATION GUID="...">`  
          - **Label (`<LABEL>`)**  
            - `"Good Data"` (always present → defines usable session span)  
            - Other labels: `"Rock"`, `"Flap"`, `"Flap-Rock"` (SMM)  
          - **Start/Stop times**  
            - `<START_DT>` and `<STOP_DT>`  
            - date-time nodes receive values in the form [YEAR]-[MO]-[DY] [HR]:[MI]:[SE].[MSe]. Define the start and stop times of a given **Label**
          - **Ratings (`<RATINGS>`)**  
            - One or more `<RATING>` entries  
            - Attributes: `TIMESTAMP`, `VALUE` (state flag), `METARATING` (e.g., intensity or certainty)  
          - **Properties (`<PROPERTIES>`)**  
            - Metadata: annotation set, creation, last modification time  
         #### Example (simplified)
         
         ```xml
         <DATA DATASET="My Dataset">
           <ANNOTATION GUID="0ee995dc-...">
             <LABEL>Good Data</LABEL>
             <START_DT>2010-05-25 10:04:32.966</START_DT>
             <STOP_DT>2010-05-25 10:21:07.984</STOP_DT>
             <RATINGS>
               <RATING TIMESTAMP="2010-05-25 10:04:32.966" VALUE="1" METARATING="3"/>
               <RATING TIMESTAMP="2010-05-25 10:21:07.984" VALUE="0" METARATING="3"/>
             </RATINGS>
             <PROPERTIES ANNOTATION_SET="Annotator1Stereotypy" ... />
           </ANNOTATION>
         </DATA>
    ``.xlsx​`` ​files are converted from ``​.xml``​ files, so user should be able to find the corresponding column name with the node name described above. ``​.csv​`` ​files have a header row that describes the meaning of each column.

      
   


### `dataset.pkl`

The `.pkl` file contains pose estimation tracking data extracted from MediaPipe’s *BlazePose* model, along with additional metadata for each video in publicly available data from [Goodwin et al. 2014](https://dl.acm.org/doi/10.1145/2632048.2632096) Each entry of the `.pkl` file contains the following:

- `keypoints`: (x, y, visibility) landmark (also known as keypoint) coordinates  
- `annotations`: Frame-wise SMM behavior labels (0–3)
    -  0 - No Stereotypy
    -  1 - Rocking
    -  2 - Flapping
    -  3 - Combination of Flapping and Rocking
- `fps`, `frame_count`, `duration` - proivdes the fps, total number of frames and duration of the video
- `name` - Unique identifier for each entry, structured as (participant-date\_study).

This `.pkl` file contains all necessary de-identifiable data for the analysis, while allowing room for further modifications or new ideas using the raw MediaPipe landmark data.

### `YOLOv5l/`

Trained YOLOv5l model weights and figures.

### `experiments/`

Contains `.csv` files for each particiapnt and session with TDA-derived periodicity scores (and period estimations for accelerometer data) across 4-second windows. Each row includes:

-`Session`: Format - (URI-00*-date) or (00*-date).
-`Annotation `: Behavior annotation lables for each 4-second time window.
-`Landmark/Sensor_n`: TDA-derived periodicity score for the n-th most persistent $H_1$ point for a given pose landmark or sensor.
-`Sensor_Period`: Estimated period of an accelerometer sensor (Accelerometer data only). May add if one has acess to high-temporal rate video data.
-`PersonID`: Unique particiapnt # (e.g. 001 -> 1, 002 -> 2, etc.)

---

## Documentation

Check the GitHub repository for further documentation, examples, and updates:  
 [https://github.com/ambaye15/AQSM_SW1PerS](https://github.com/ambaye15/AQSM_SW1PerS)
