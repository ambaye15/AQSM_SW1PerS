# Notebooks

This folder contains interactive Jupyter notebooks that demonstrate the core functionality and applications of the `AQSM_SW1PerS` package.

---
## `ClassificationExperiments.ipynb`

Notebook showing how experiments done in the paper were performed, and how to recreate figures.

---

## `Tutorial.ipynb`

A comprehensive, step-by-step tutorial that is divided into two parts:

1) **Data Collection** - Optional walkthrough that shows how videos were created and MediaPipe pose inference is run
2) **Data Analysis** - Walkthrough of data analysis pipline using ```dataset.pkl``` file

This notebook is ideal for first-time users or reviewers who want a reproducible walkthrough of the full workflow.

---

## `accelerometer_tutorial.ipynb`

A tutorial of the `Sw1PerS` algorithm applied to 3-axis acceleromter data:

1) **Data Extraction** - Walkthrough that shows how accelerometer data is extracted from `data` directory
2) **Data Analysis** - Walkthrough of data analysis pipline using accelerometer data

---


## Notes

- All notebooks assume the `AQSM_SW1PerS` package has been installed (e.g., via `pip install -e .`).
- To run the notebooks, ensure dependencies are installed and activate the correct environment
- Ensure that you have a `Dataset` folder in the root that contains the `.pkl` file.
  
