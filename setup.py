from setuptools import setup, find_packages

setup(
    name="AQSM_SW1PerS",
    version="0.1.0",
    description="TDA pipeline for quantification of recurrence in multimodal time series data",
    license = { file = "LICENSE" }
    authors = [
      { name = "Austin MBaye", email = "mbaye.au@northeastern.edu" }
    ]
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "ripser", 
        "scipy",
        "seaborn",
        "persim",
        "opencv-python",
        "mediapipe",
        "filterpy",
        "sympy",
        "imbalanced-learn",
        "scikit-optimize",
        "joblib",
        "tqdm",
        "PyYAML",
        "umap-learn",
    ],
    python_requires=">=3.7",
)
