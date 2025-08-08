
# ATSM_SW1PerS: Instructions for Installation and Use

**Author:** 
**Date:** *

---

## Overview

**ATSM_SW1PerS** is a Python package for analyzing repetitive motion patterns using pose estimation and topological data analysis. It was developed to support the classification and analysis of *Stereotypical Motor Movements (SMMs)* observed in individuals with Autism Spectrum Disorder (ASD). The package provides modules for:

- Motion tracking
- Period estimation
- Persistence diagram computation
- Topological visualization of time series data

---

## System Requirements

- Python 3.11 or higher  
- PyTorch 2.1.0  
- CUDA 11.8  
- Recommended: virtual environment (`venv` or `conda`)  
- OS: Windows, Linux, or macOS  

---

## Installation

### Option 1: Install via GitHub (recommended)

```bash
pip install git+https://github.com/ambaye15/ATSM_SW1PerS.git
```

### Option 2: Clone and Install Locally

```bash
git clone https://github.com/ambaye15/ATSM_SW1PerS.git
cd ATSM_SW1PerS
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
The HPC allowed easier access to GPU partitions for YOLOv5 inference. However, this was also easy to run on laptop or PC with a GPU.
---

To verify correct installation and behavior, run the included unit tests:

```bash
python -m unittest tests/test_periodicity.py
```

## Directory Structure

- `ATSM_SW1PerS/`: Core package with analysis and processing modules  
- `Classification_experiments/`: Code for training and evaluation  
- `notebooks/`: Usage tutorials and visualization demos  
- `Dataset/`: (Optional) Preprocessed data, YOLO model results, periodicity scores. NOTE- You must download this seperatley given this link: 

---

## Data Overview

### `dataset.pkl`

The `.pkl` file contains pose estimation tracking data extracted from MediaPipe’s *BlazePose* model, along with additional metadata for each video in publicly available data from Goodwin et al. Each entry of the `.pkl` file contains the exact follwoing:

- `keypoints`: (x, y, visibility) coordinates  
- `annotations`: Frame-wise behavior labels (0–3)
    -  0 - No Stereotypy
    -  1 - Rocking
    -  2 - Flapping
    -  3 - Flapping and Rocking
- `fps`, `frame_count`, `duration` - proivdes the fps, total number of frames and duration of video
-  `name` - Unique identifier for each entry, structured as (child-date\_study).

This `.pkl` file contains all necessary pre-processing data for the analysis, while still allowing room for further modifications or new ideas using the raw MediaPipe keypoint data.

### `YOLOv5l/`

Trained YOLOv5l model weights and figures.

### `Periodicity_Scores/`

Contains `.csv` files with TDA-derived periodicity scores across 4-second windows. Each row includes:

- `Person_ID`: Unique identifier of child
- `Annotation_1`: Behavior annotation  
- `TDA_*`: 10 periodicity scores per sensor. We have 6 sensors and analyze both the spatial position changes as well as the acceleration changes. The first 60 scores are for spatial periodicity scores (e.g. Head, Left Wrist, Right Wrist, Left Shoulder, Right Shoulder, Chest) and the following are in the same order but for acceleration representations.
- `Video_ID`: Each session (video) gets assigned a unique label
- `Child_Study_ID`: Format = `child_study_session` (e.g., `1_2_1`)

---

## Basic Usage

See `notebooks/tutorial.ipynb` for a full walkthrough using the `.pkl` file.

### Quick Demo

```python
from ATSM_SW1PerS.SW1PerS import *
from persim import plot_diagrams 
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import CubicSpline
from ripser import ripser

t_vals = np.linspace(0, 4, 150)
sampling_rate = 150 / 4
x = np.cos(2 * np.pi * t_vals)
y = np.sin(2 * np.pi * t_vals)
X = np.column_stack((x, y))
X += np.random.normal(scale=0.1, size=X.shape)

period = estimate_period(X[:, 0], X[:, 1], sampling_rate)
d = 23
tau = period / (d + 1)

spline_x = CubicSpline(t_vals, X[:, 0])
spline_y = CubicSpline(t_vals, X[:, 1])
spline = [spline_x, spline_y]

SW = SW_cloud_nD(spline, t_vals, tau, d, 300, 2)
results = ripser(SW, coeff=next_prime(2 * d), maxdim=1)
diagrams = results['dgms']
score = compute_PS(np.array(diagrams[1]), method='PS1')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
ax1.plot(t_vals, X[:, 0], color='r', label='X')
ax1.plot(t_vals, X[:, 1], color='g', label='Y')
ax1.set_title("Generated 2D Time Series")
ax1.set_yticks([])
ax1.axis("equal")

plot_diagrams(diagrams, plot_only=[1], xy_range=[0, 2, 0, 2], ax=ax2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Persistence Diagram")

ax3.bar(range(1), score, alpha=0.5)
ax3.set_title("Periodicity Score")
ax3.set_xlim(-0.5, 0.5)
ax3.set_ylim(0, 1)
ax3.set_xticks([])

plt.tight_layout()
plt.show()
```

---

## Documentation

Check the GitHub repository for further documentation, examples, and updates:  
 [https://github.com/ambaye15/ATSM_SW1PerS](https://github.com/ambaye15/ATSM_SW1PerS)
