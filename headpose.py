import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Minimal 3D model points (approximate) for PnP
model_points = np.array([
    [0.0, 0.0, 0.0],        # Nose tip
    [-30.0, -125.0, -30.0], # Left eye left corner
    [30.0, -125.0, -30.0],  # Right eye right corner
    [-60.0, 50.0, -25.0],   # Left mouth corner
    [60.0, 50.0, -25.0],    # Right mouth corner
    [0.0, 75.0, -50.0]      # Chin
], dtype=np.float64)

LM_POINTS = [1, 33, 263, 61, 291, 199]

cap = cv2.VideoCapture(0)
print("Head Pose Detector Running... Press Q to quit. Press C to recalibrate.")

yaw_history = deque(maxlen=5)
pitch_history = deque(maxlen=5)

# Deviation thresholds after calibration
yaw_threshold = 12.0
pitch_threshold = 12.0

off_road_start = None
status = "Calibrating"

CALIBRATION_FRAMES = 50
calibration_yaw = []
calibration_pitch = []
baseline_yaw = None
baseline_pitch = None
recalibrating = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

#to add face landmark and head position logic 

    cv2.imshow("Head Pose Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        calibration_yaw.clear()
        calibration_pitch.clear()
        baseline_yaw = None
        baseline_pitch = None
        recalibrating = True
        status = "Calibrating"

cap.release()
cv2.destroyAllWindows()
