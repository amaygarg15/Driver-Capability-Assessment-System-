# Driver Capability Assessment System

A real-time driver monitoring project that combines computer vision and speed behavior analysis to estimate driving safety throughout a journey.

This system watches key driver signals such as drowsiness, yawning, gaze direction, and head pose, then blends them with a machine-learning speed risk estimate into a single journey score. The goal is simple: turn many noisy signals into one clear, actionable safety view.

## What This Project Does

- Detects drowsiness from eye closure patterns (EAR-based).
- Detects yawning from mouth movement (MAR-based).
- Tracks gaze direction using iris and eye landmarks.
- Estimates head orientation and on-road/off-road attention status.
- Adds speed risk estimation using engineered speed dynamics and an ML model.
- Produces a unified safety score from 0 to 100 (higher is safer).
- Shows live feedback and prints an end-of-journey risk summary.

## Project Status

The current implementation includes:

- Computer vision monitoring pipeline.
- Speed analysis integration.
- Weighted scoring system for overall safety.

Voice input is not part of the active scope.

## Repository Structure

Top-level files you will use most often:

- `driver_monitor.py`: Unified real-time monitor (CV + speed risk + final scoring).
- `drowsiness_detector.py`: Standalone drowsiness module.
- `yawn_detector.py`: Standalone yawn module.
- `gaze.py`: Standalone gaze tracking module.
- `headpose.py`: Standalone head pose module.
- `modules_overview.txt`: Quick summary of monitoring modules.

Machine learning work and datasets:

- `ML/data/`: feature datasets, model comparison outputs, and test predictions.
- `ML/scripts/`: feature building, model comparison, cross-driver evaluation scripts.
- `ML/models/`: saved model artifacts.

## Tech Stack

- Python
- OpenCV
- dlib
- MediaPipe
- NumPy, pandas, SciPy
- scikit-learn
- imutils

## Requirements

1. Python 3.9 or newer.
2. A working webcam.
3. Landmark model file for dlib:
	- `shape_predictor_68_face_landmarks.dat` in the project root.
4. MediaPipe face landmarker task file:
	- `face_landmarker.task` (auto-downloaded by `driver_monitor.py` if missing).

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install opencv-python dlib mediapipe numpy pandas scipy scikit-learn imutils
```

If dlib installation fails on your platform, install it using a platform-compatible wheel or conda package.

## Run

Run the unified system:

```bash
python driver_monitor.py
```

Run individual modules (optional):

```bash
python drowsiness_detector.py
python yawn_detector.py
python gaze.py
python headpose.py
```

## Controls in Unified Monitor

- `q`: Quit
- `c`: Recalibrate head pose baseline
- `g`: Recalibrate gaze baseline
- `u`: Increase demo speed by +2 km/h
- `j`: Decrease demo speed by -2 km/h

## How Scoring Works (High-Level)

Each component outputs a risk value between 0 and 1:

- Yawn risk
- Drowsiness risk
- Gaze risk
- Head pose risk
- Speed ML risk

The system computes a weighted total risk and converts it to a final score:

`final_score = (1 - total_risk) * 100`

An end-of-journey report includes:

- Final score and rating.
- Average risk per component.
- Behavior percentages over the full trip (for example, off-road time).

## Notes

- Current speed in the live demo can be controlled from keyboard input.
- For production use, speed should come from telemetry (GPS/OBD/CAN stream).
- Lighting, camera angle, and face visibility affect CV reliability.

## Future Improvements

- Integrate real telemetry input for speed.
- Add personalization per driver profile.
- Improve temporal modeling for smoother risk estimation.
- Expand validation across larger and more diverse datasets.

## License

This project is licensed under the MIT License.

Copyright (c) 2026 Amay Garg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the Software), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.