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

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pts = []
            for idx in LM_POINTS:
                lm = face_landmarks.landmark[idx]
                pts.append([lm.x * w, lm.y * h])
            pts = np.array(pts, dtype=np.float64)

            focal_length = w
            cam_matrix = np.array([[focal_length, 0, w/2],
                                    [0, focal_length, h/2],
                                    [0, 0, 1]])
            dist_coeffs = np.zeros((4,1))

            success, rvec, tvec = cv2.solvePnP(model_points, pts, cam_matrix, dist_coeffs)
            rot_mat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rot_mat[0,0]**2 + rot_mat[1,0]**2)
            yaw = np.degrees(np.arctan2(rot_mat[2,1], rot_mat[2,2]))
            pitch = np.degrees(np.arctan2(-rot_mat[2,0], sy))
            roll = np.degrees(np.arctan2(rot_mat[1,0], rot_mat[0,0]))

            yaw_history.append(yaw)
            pitch_history.append(pitch)
            smoothed_yaw = sum(yaw_history) / len(yaw_history)
            smoothed_pitch = sum(pitch_history) / len(pitch_history)

            if recalibrating:
                calibration_yaw.append(smoothed_yaw)
                calibration_pitch.append(smoothed_pitch)
                remaining = CALIBRATION_FRAMES - len(calibration_yaw)
                if remaining <= 0:
                    baseline_yaw = float(np.mean(calibration_yaw))
                    baseline_pitch = float(np.mean(calibration_pitch))
                    recalibrating = False
                    status = "On Road"
                else:
                    cv2.putText(frame, f"Calibrating: {remaining}", (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

            if baseline_yaw is not None:
                yaw_dev = smoothed_yaw - baseline_yaw
                pitch_dev = smoothed_pitch - baseline_pitch
                looking_forward = abs(yaw_dev) < yaw_threshold and abs(pitch_dev) < pitch_threshold
            else:
                yaw_dev = 0.0
                pitch_dev = 0.0
                looking_forward = False

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
