
# AQSM-SW1PerS: Automated Quantification of Stereotypical Motor Movements via Sliding Windows and 1-Persistence Scoring

This repository contains code, models, and analysis tools introduced in the paper *Automated Quantification of Stereotypical Motor Movements in Autism Using Persistent Homology* (currently in submission) associated with the **AQSM-SW1PerS** algorithm, a pipeline for detecting repetitive movements using topological data analysis on time series data derived from pose estimation software.

## Table of Contents
1. [Abstract](#abstract)
2. [Installation](#installation)
3. [Project Structure](#structure)
4. [Basic Usage](#usage)
5. [Citation](#citation)

---

<a name="abstract"></a>
## Abstract

Stereotypical motor movements (SMM) are a core diagnostic feature of autism that remain difficult to quantify efficiently and validly across individuals and developmental stages. The current paper presents a novel pipeline that leverages Topological Data Analysis to quantify and characterize recurrent movement patterns. Specifically, we use persistent homology to construct low-dimensional, interpretable feature vectors that capture geometric properties associated with autistic SMM by extracting periodic structure from time series derived from pose estimation landmarks in video data and accelerometer signals from wearable sensors. We demonstrate that these features, combined with simple classifiers, enable accurate automated quantification of autistic SMM. Visualization of the learned feature space reveals that extracted features generalize across individuals and are not dominated by person-specific SMM. Our results highlight the potential of using mathematically principled features to support more scalable, interpretable, and person-agnostic characterization of autistic SMM in naturalistic settings.

---

<a name="installation"></a>
## Installation

### Requirements
- Windows / Linux / macOS
- Python 3.11
- PyTorch 2.1.0 (with CUDA 11.8 for GPU support)
> Please install PyTorch manually using: [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) and follow the instructions


### Option 1 - Install with `venv`
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

### Option 2 - Install with `conda`
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

### Pose Estimation
- MediaPipe is automatically installed with this package. For more information on MediaPipe and its other solutions: [https://ai.google.dev/edge/mediapipe/solutions/guide](https://ai.google.dev/edge/mediapipe/solutions/guide). 

---

### Data Downloads

You can download the following data (data avilable after publishing of paper):

Create a new directory named `Dataset` in the root and place these in the folder

---

<a name="structure"></a>
## Project Structure

```
AQSM_SW1PerS/
├── SW1PerS.py               # Core script for persistence scoring
├── mediapipe_pose.py        # Pose extraction pipeline (YOLO + MediaPipe)
├── utils/                    ## General helper tools for data processing and analysis
classification_tools/        # Model training, evaluation scripts
notebooks/                   # Demos, tutorials, reproducible figures
scripts/                     # Batch processing and automation scripts
tests/                       # Test script to verify installation
docs/                        # In depth instruction of dataset
Dataset/                     # Must download
Visualizations/              # Directory to save images
```

---

<a name="usage"></a>
## Basic Usage

Run the included test suite to verify installation:

```bash
python -m unittest tests/test_periodicity.py
```

Here is a simple demo of a periodic 2D signal with its persistence diagram and periodicity score:

```python
from AQSM_SW1PerS.SW1PerS import *
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np

num_points = 150
d = 23
prime_coeff = next_prime(2 * d)
method = 'PS1'

t_vals = np.linspace(0, 4, num_points)
x = np.cos(2 * np.pi * t_vals)
y = np.sin(2 * np.pi *t_vals)
X = np.column_stack((x, y))
X += np.random.normal(scale=0.1, size=X.shape) #Add some noise
spline_x = CubicSpline(t_vals, X[:,0])
spline_y = CubicSpline(t_vals, X[:,1])
spline_funcs = [spline_x, spline_y]

scoring_pipeline = SW1PerS(start_time = 0, end_time = 4, num_points = num_points, method = method, d = d, prime_coeff = prime_coeff)
scoring_pipeline.compute_score(spline_funcs)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(t_vals, X[:, 0],color='r',label = 'X')
ax1.plot(t_vals, X[:, 1],color='g',label = 'Y')
ax1.set_title("Generated 2D Time Series")
ax1.set_yticks([])
ax1.axis("equal")

plot_diagrams(scoring_pipeline.diagram, plot_only=[1], xy_range=[0, 2, 0, 2], ax = ax2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title(fr'Persistence Diagram')

ax3.bar(range(1), scoring_pipeline.periodicity_score, alpha=0.5)
ax3.set_title(fr'Periodicity Score')
ax3.set_xlim(-0.5, 0.5)
ax3.set_ylim(0, 1)
ax3.set_xticks([])

plt.tight_layout()
plt.show()    

```

![Time Series Demo](Visualizations/demo_time_series.png)

---

<a name="citation"></a>
## Citation

The preprint for our paper is available:

> MBaye, Austin A., Perea, Jose A., Tralie, Christopher J., & Goodwin, Matthew S.  
> *Automated Quantification of Stereotypical Motor Movements in Autism Using Persistent Homology.*  
> bioRxiv, 2025. https://doi.org/10.1101/2025.09.03.674008  

BibTeX:
```bibtex
@article{MBaye2025AQSM,
  author    = {MBaye, Austin A and Perea, Jose A. and Tralie, Christopher J. and Goodwin, Matthew S.},
  title     = {Automated Quantification of Stereotypical Motor Movements in Autism Using Persistent Homology},
  elocation-id = {2025.09.03.674008},
  year      = {2025},
  doi       = {10.1101/2025.09.03.674008},
  publisher = {Cold Spring Harbor Laboratory},
  url       = {https://www.biorxiv.org/content/early/2025/09/05/2025.09.03.674008},
  eprint    = {https://www.biorxiv.org/content/early/2025/09/05/2025.09.03.674008.full.pdf},
  journal   = {bioRxiv}
}

