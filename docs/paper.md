---
title: "AQSM_SW1PerS: A Topological Data Analysis Pipeline for Quantifying Recurrent Motion in Multimodal and Multivariate Time Series Data"
tags:
  - Python
  - Topological Data Analysis
  - Pose Estimation
  - Persistent Homology
  - Time Series Analysis
authors:
  - name: Austin A. MBaye
    orcid: 0009-0006-8907-8593
    affiliation: 1
  - name: Jose A. Perea
    affiliation: "1, 2"
  - name: Christopher J. Tralie
    affiliation: 3
affiliations:
  - name: Department of Mathematics, Northeastern University, Boston, MA, USA
    index: 1
  - name: Khoury College of Computer Sciences, Northeastern University, Boston, MA, USA
    index: 2
  - name: Department of Mathematics and Computer Science, Ursinus College, Collegeville, PA, USA
    index: 3
date: 27 August 2025
bibliography: paper.bib
---

# Summary

Automated Quantification of Stereotypical Motor Movements via Sliding Windows and 1-Persistence Scoring (**AQSM_SW1PerS**) provides an open-source pipeline for quantifying recurrent motion behaviors from multimodal and multivariate time series data, including 3-axis accelerometer signals and MediaPipe pose estimation–derived trajectories [@mediapipe_pose]. The package integrates topological data analysis with mutimodal and multivariate time series data to extract interpretable topological features that capture periodicity and recurrence in motion. These features can be directly applied in classification, clustering, and longitudinal analyses.

The software is motivated by the challenge of characterizing **stereotypical motor movements (SMM)** in autism, where video data is often low-resolution and sensor data may be noisy or sparse. By leveraging mathematically grounded topological features, AQSM-SW1PerS offers an interpretable and generalizable framework for motion quantification that is accessible to researchers across various fields. This pipeline can transform human movement in raw video data into time series, map those series into geometric shapes through sliding-window embeddings [@article], and finally analyze their structure with persistent homology to produce interpretable feature representations.
 
# Statement of Need

Quantitative analysis of repetitive movement is essential in autism research, biomechanics, and behavioral neuroscience. Deep learning methods have shown strong performance but often require large, high-quality datasets and lack interpretability. Researchers and clinicians increasingly seek transparent, reproducible approaches that can handle noisy or multimodal data.

**AQSM_SW1PerS** addresses this need by offering:

- **Interpretability**: Periodcity Score features derived directly from persistence diagrams.  
- **Modality-agnostic design**: Compatible with both video-based pose trajectories and wearable accelerometer data.  
- **Generalization**: Provides person-agnostic quantification of recurrent movement, capturing behavior independent of individual variability.  

This fills a gap between persistent homology libraries and domain-specific applications, enabling researchers to move directly from raw data to interpretable results.

# State of the Field

Several mature libraries exist for computing persistent homology, including *Ripser* [@Bauer2021Ripser], *GUDHI* [@gudhi], and *Giotto-TDA* [@giotto-tda]. These packages provide efficient implementations of topological algorithms but primarily function as backends. They do not provide an integrated workflow for interpretable, low-dimensional motion quantification or feature extraction tailored to time series analysis.

**AQSM_SW1PerS** builds upon these foundations by providing an end-to-end pipeline that:

- Extracts **pose landmark trajectories** from videos using MediaPipe.  
- Combines **sliding-window embeddings** with **1-persistence scoring** to quantify recurrence and periodicity in time series derived from either pose trajectories or accelerometer signals.  
- Provides utilities for **feature extraction, visualization, and classification**, lowering the barrier for adoption of topological methods.  
- Includes **Clear Tutorials** for any dataset, allowing researchers to quickly prototype analyses.  

# Software Description

The package is structured around a small set of core classes and utilities:

- **Pose Estimation (`MPPose`)**: Uses MediaPipe to automatically store pose landmarks of an individual from video data. 
- **Sliding-Window and 1-Persistence Scoring (`SW1PerS`)**: Transforms time series into high-dimensional geometric objects by constructing delay-coordinate (sliding-window) embeddings. The resulting point cloud encodes the temporal structure of the signal, which is then analyzed using persistent homology via *Ripser*. The resulting persistence diagrams capture topological features that are indicative of periodicity, which are subsequently summarized into low-dimensional, interpretable feature vectors for downstream analysis.
- **Classification Utilities**: Provides integration with *scikit-learn* for supervised learning on topological feature representations. Supports common tasks such as classifying movement types, evaluating model performance, and conducting hyperparameter optimization. Multiple classification schemes are available, enabling researchers to compare approaches and identify the most effective models for their datasets.
- **Visualization**: Provides plotting functions for persistence diagrams, feature distributions, and classifier outputs to aid interpretation and communication of results.
- **Batch Processing**: Includes shell scripts that support processing multiple files at once, enabling efficient large-scale analyses. This functionality streamlines the workflow when working with datasets containing many videos or sensor recordings, reducing manual overhead and improving reproducibility.

Together, these components allow users to progress from **raw multimodal/multivariate motion data** → **topological features** → **classification or applications** with minimal overhead.

To illustrate the workflow, we generate a noisy circular trajectory and analyze it with `SW1PerS`. The time series is embedded into a high-dimensional point cloud via sliding windows, and persistent homology is computed with *Ripser*. The resulting persistence diagram highlights a 1-dimensional loop corresponding to the underlying periodicity of the signal, which is then summarized into a single periodicity score ($PS_1$). The code snippet below reproduces this example, and the outputs are shown in Figure&nbsp;1.


# Example

```python
from AQSM_SW1PerS.SW1PerS import *
from persim import plot_diagrams
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Generate a noisy circular trajectory (periodic signal)
num_points = 150
t_vals = np.linspace(0, 4, num_points)
x = np.cos(2 * np.pi * t_vals)
y = np.sin(2 * np.pi * t_vals)
X = np.column_stack((x, y)) + np.random.normal(scale=0.1, size=(num_points, 2))

# Interpolate trajectory with cubic splines
spline_x = CubicSpline(t_vals, X[:, 0])
spline_y = CubicSpline(t_vals, X[:, 1])
spline_funcs = [spline_x, spline_y]

# Initialize pipeline and compute persistence-based periodicity score
scoring_pipeline = SW1PerS(start_time=0, end_time=4, num_points=num_points,
                           method="PS1", d=23, prime_coeff=next_prime(46))
scoring_pipeline.compute_score(spline_funcs)

# Visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(t_vals, X[:, 0], 'r', label='X')
ax1.plot(t_vals, X[:, 1], 'g', label='Y')
ax1.set_title("Generated 2D Time Series")
ax1.axis("equal"); ax1.set_yticks([])

plot_diagrams(scoring_pipeline.diagram, plot_only=[1], xy_range=[0, 2, 0, 2], ax=ax2)
ax2.set_title("Persistence Diagram"); ax2.set_xticks([]); ax2.set_yticks([])

ax3.bar([0], [scoring_pipeline.periodicity_score], alpha=0.5)
ax3.set_title("Periodicity Score")
ax3.set_xlim(-0.5, 0.5); ax3.set_ylim(0, 1); ax3.set_xticks([])

plt.tight_layout()
plt.show()
```
![Example workflow: (a) a noisy circular trajectory represented as a 2D time series, (b) the corresponding persistence diagram showing a prominent 1-dimensional loop, and (c) the periodicity score presented as a single value derived from the diagram. Together, these demonstrate how AQSM-SW1PerS extracts interpretable topological features that capture recurrent structure in time series data.](../Visualizations/demo_time_series.png)


# Acknowledgements

The National Science Foundation partially supported Austin A. MBaye and Jose A. Perea through CAREER award #DMS-2415445.

# References

