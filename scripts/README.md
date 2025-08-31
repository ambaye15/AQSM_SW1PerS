# Scripts

This folder contains scripts for batch processing of video and accelerometer data using the *AQSM_SW1PerS* pipeline.

---

## `accelerometer_processing.py`

Alternative Python script for extracting periodicity scores specifically from accelerometer data using the *SW1PerS* algorithm. Used in `exec_accel.sh` to generate a `.csv` file of periodicity scores for a given participant's accelerometer recordings.

---

## `batch_video_sw1pers.sh`

Batch script that:
- Runs the *SW1PerS* algorithm on the pose data from a video.
- Outputs `.csv` files containing periodicity scores.

---

## `exec_accel.sh`

Batch script that:
- Runs the *SW1PerS* algorithm on a directory of accelerometer data.
- Outputs `.csv` files containing periodicity scores.

---

## `mediapipe_batch.sh`

Batch script that:
- Runs MediaPipe pose estimation on a directory of video files.
- Outputs `.csv` files containing pose landmark coordinates.

---
## `process_video_pose.py`

Helper script used in `exec_mp.sh` to run MediaPipe inference on individual video files.

---

## `video_pose_sw1pers.py`

Helper script used in `exec_sw1pers.sh` to extract periodicity scores from individual video files.

---


## Notes

- All scripts assume the `AQSM_SW1PerS` package is installed
