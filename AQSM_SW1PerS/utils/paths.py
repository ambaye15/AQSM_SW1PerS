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

from pathlib import Path

def get_project_root(marker: str = "setup.py") -> Path:
    """
    Identify the root directory so you can easily access Dataset
    
    Parameters:
        marker (str): A filename to identify the root directory (default: 'setup.py')
    
    Returns:
        Path: The root path of the project
    """
    current = Path().resolve()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root with marker '{marker}'")


def get_data_path(*parts) -> Path:
    """
    Build a path to a file or folder inside the 'Dataset/' directory from the project root.
    
    Example:
        get_data_path("Periodicity_Scores", "final_pose_results.csv")

    Returns:
        Path: Full path to the desired data file
    """
    return get_project_root() / "Dataset" / Path(*parts)
