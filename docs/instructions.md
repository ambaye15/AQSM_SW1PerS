
# AQSM-SW1PerS: Instructions for Installation and Use

**Author: Austin MBaye** 

---

## Overview

**AQSM-SW1PerS** is a Python package for analyzing repetitive motion patterns using pose estimation and topological data analysis. It was originally developed to support the classification and analysis of *Stereotypical Motor Movements (SMMs)* observed in individuals with Autism Spectrum Disorder (ASD), but can be applied to any dataset where one aims to quantify and interpret reccurrent patterns in time series or motion in videos. The package provides modules for:

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

### Option 1 - Clone and Install with `venv`
```bash
# Clone the repository
git clone https://github.com/ambaye15/AQSM_SW1PerS.git
cd AQSM_SW1PerS

# Create and activate environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# Install 
pip install -e .
```

### Option 2 - Clone and Install with `conda` environment 
```bash
# Clone the repository
git clone https://github.com/ambaye15/AQSM_SW1PerS.git
cd AQSM_SW1PerS

# Create and activate environment
conda create -n aqsm python=3.11
conda activate aqsm

# Install in editable mode
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
- `Dataset/`: (**Must Download for reproduction of results**) Preprocessed data, YOLO model results, periodicity scores. You must download this seperatley given this link: (Not yet available)

---

## Data Overview (WILL BE AVAILABLE AT TIME OF PUBLCIATION)

### `data/`

Refer to `Motion sensor data access instructions.pdf` (in the same directory as these intructions) for details on obtaining the accelerometer data, which has been previously published [@goodwin2014].

Once downloaded, extract the data folder and place it into the Dataset directory alongside the following files:
      
### `dataset.pkl`

The `.pkl` file contains pose estimation tracking data extracted from MediaPipe’s *BlazePose* model [@lugaresi2019mediapipe; @blaze], along with additional metadata for each video in publicly available data from [@goodwin2014]. Each entry of the `.pkl` file contains the following:

- `keypoints`: (x, y, visibility) landmark (also known as keypoint) coordinates  
- `annotations`: Frame-wise SMM behavior labels (0–3)
    -  0 - No Stereotypy
    -  1 - Rocking
    -  2 - Flapping
    -  3 - Flaprock
- `fps`, `frame_count`, `duration` - proivdes the fps, total number of frames and duration of the video
- `name` - Unique identifier for each entry, structured as (**participant-date\_study** as in the `data/` directory).

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

All Jupyter notebooks and code provide an interpretable and easy-to-follow pipeline to replicate all experiments done in the paper.

---

## Documentation

Check the GitHub repository for further documentation, examples, and updates:  
 [https://github.com/ambaye15/AQSM_SW1PerS](https://github.com/ambaye15/AQSM_SW1PerS)

## Refrences
