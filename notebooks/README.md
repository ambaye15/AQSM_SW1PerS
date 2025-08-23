# Notebooks

This folder contains interactive Jupyter notebooks that demonstrate the core functionality and applications of the `AQSM_SW1PerS` package.

---

## `Tutorial.ipynb`

A comprehensive, step-by-step tutorial that is divided into two parts:

1) **Data Collection** - Optional walkthrough that shows how videos were created and MediaPipe pose inference is run
2) **Data Analysis** - Walkthrough of data analysis pipline using ```dataset.pkl``` file

This notebook is ideal for first-time users or reviewers who want a reproducible walkthrough of the full workflow.

---

## `accelerometer_tutorial.ipynb`

A comprehensive, step-by-step tutorial of the `Sw1PerS` algorithm applied to 3-axis acceleromter data:

1) **Data Extraction** - Walkthrough that shows how accelerometer data is extracted from `data` directory
2) **Data Analysis** - Walkthrough of data analysis pipline using accelerometer data

---

## `create_concept_video.ipynb`

A notebook for creating a side-by-side video showing the mediapipe-overlayed skeleton on the left, and the SW1PerS pipeline on the spatial and acceleration trajectories of a particular sensor. This is the notebook used for visualizations during presentations.

Use this notebook to generate videos that showcase important aspects of pipeline.

---

## Notes

- All notebooks assume the `AQSM_SW1PerS` package has been installed (e.g., via `pip install -e .`).
- To run the notebooks, ensure dependencies are installed and activate the correct environment:
  
