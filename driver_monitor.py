"""
Combined Driver Monitoring System
Detects: Gaze direction, Head pose, Drowsiness (eye closure), Yawning
Uses MediaPipe Tasks API (0.10.30+) and dlib
"""

import cv2
import dlib
import numpy as np
import time
import urllib.request
import os
from collections import deque
from scipy.spatial import distance as dist
from imutils import face_utils

# MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Download Model if Needed 
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Downloading face landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

# MediaPipe Tasks Setup 
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

#  Dlib Setup (Drowsiness + Yawn) 
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Dlib eye indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Gaze Constants 
# MediaPipe face mesh landmark indices
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_EYE_TOPS = [159, 158, 157]
LEFT_EYE_BOTTOMS = [145, 144, 153]
RIGHT_EYE_TOPS = [386, 385, 384]
RIGHT_EYE_BOTTOMS = [374, 373, 380]

#  Head Pose Constants 
model_points = np.array([
    [0.0, 0.0, 0.0],
    [-30.0, -125.0, -30.0],
    [30.0, -125.0, -30.0],
    [-60.0, 50.0, -25.0],
    [60.0, 50.0, -25.0],
    [0.0, 75.0, -50.0]
], dtype=np.float64)
LM_POINTS = [1, 33, 263, 61, 291, 199]

# Thresholds 
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.8
YAW_THRESHOLD = 12.0
PITCH_THRESHOLD = 12.0
CALIBRATION_FRAMES = 50

# State Variables 
# Drowsiness
drowsy_counter = 0

# Gaze
gaze_alpha = 0.3  # Lower alpha = more smoothing, less noise
smoothed_horizontal = None
smoothed_vertical = None
gaze_vertical_baseline = None
gaze_horizontal_baseline = None
gaze_vertical_samples = []
gaze_horizontal_samples = []

# Fixed thresholds for gaze detection (more sensitive for vertical)
VERTICAL_UP_THRESH = 0.06    # How much iris moves up from baseline
VERTICAL_DOWN_THRESH = 0.06  # How much iris moves down from baseline  
HORIZONTAL_THRESH = 0.08     # How much iris moves left/right from baseline

# Head pose
yaw_history = deque(maxlen=5)
pitch_history = deque(maxlen=5)
head_calibration_yaw = []
head_calibration_pitch = []
head_baseline_yaw = None
head_baseline_pitch = None
head_recalibrating = True
off_road_start = None
head_status = "Calibrating"

# Helper Functions 

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def iris_center(landmarks, iris_indices, w, h):
    pts = np.array([[landmarks[i].x, landmarks[i].y] for i in iris_indices])
    return pts.mean(axis=0), pts

def horizontal_ratio(landmarks, corner_indices, center):
    left = np.array([landmarks[corner_indices[0]].x, landmarks[corner_indices[0]].y])
    right = np.array([landmarks[corner_indices[1]].x, landmarks[corner_indices[1]].y])
    span = right[0] - left[0]
    if span == 0:
        return 0.5
    return (center[0] - left[0]) / span

def vertical_ratio(landmarks, tops, bottoms, center):
    top_y = min(landmarks[i].y for i in tops)
    bottom_y = max(landmarks[i].y for i in bottoms)
    span = bottom_y - top_y
    if span <= 0:
        return 0.5
    return (center[1] - top_y) / span

#Main Loop
cap = cv2.VideoCapture(0)
print("Combined Driver Monitor Running...")
print("Press Q to quit, C to recalibrate head pose, G to recalibrate gaze.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # MediaPipe Processing (Gaze + Head Pose)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    mp_results = face_landmarker.detect(mp_image)
    
    gaze_direction = "N/A"
    head_direction = "N/A"

    if mp_results.face_landmarks:
        for face_landmarks in mp_results.face_landmarks:
            landmarks = face_landmarks

            # Gaze Detection
            # Check if iris landmarks exist (indices 468-477)
            if len(landmarks) > 475:
                left_center, left_iris_pts = iris_center(landmarks, LEFT_IRIS, w, h)
                right_center, right_iris_pts = iris_center(landmarks, RIGHT_IRIS, w, h)

                left_h = horizontal_ratio(landmarks, LEFT_EYE_CORNERS, left_center)
                right_h = horizontal_ratio(landmarks, RIGHT_EYE_CORNERS, right_center)
                avg_h = (left_h + right_h) / 2.0

                left_v = vertical_ratio(landmarks, LEFT_EYE_TOPS, LEFT_EYE_BOTTOMS, left_center)
                right_v = vertical_ratio(landmarks, RIGHT_EYE_TOPS, RIGHT_EYE_BOTTOMS, right_center)
                avg_v = (left_v + right_v) / 2.0

                if smoothed_horizontal is None:
                    smoothed_horizontal = avg_h
                    smoothed_vertical = avg_v
                else:
                    smoothed_horizontal = gaze_alpha * avg_h + (1 - gaze_alpha) * smoothed_horizontal
                    smoothed_vertical = gaze_alpha * avg_v + (1 - gaze_alpha) * smoothed_vertical

                # Calibration phase - collect baseline when looking straight
                if gaze_vertical_baseline is None and len(gaze_vertical_samples) < CALIBRATION_FRAMES:
                    gaze_vertical_samples.append(smoothed_vertical)
                    gaze_horizontal_samples.append(smoothed_horizontal)
                    if len(gaze_vertical_samples) == CALIBRATION_FRAMES:
                        gaze_vertical_baseline = float(np.mean(gaze_vertical_samples))
                        gaze_horizontal_baseline = float(np.mean(gaze_horizontal_samples))
                    gaze_direction = "Calibrating"
                elif gaze_vertical_baseline is not None:
                    # Calculate deviation from baseline
                    v_dev = smoothed_vertical - gaze_vertical_baseline
                    h_dev = smoothed_horizontal - gaze_horizontal_baseline
                    
                    # Prioritize vertical detection (up/down) - critical for phone detection
                    if v_dev < -VERTICAL_UP_THRESH:
                        gaze_direction = "DOWN"
                    elif v_dev > VERTICAL_DOWN_THRESH:
                        gaze_direction = "UP"
                    elif h_dev < -HORIZONTAL_THRESH:
                        gaze_direction = "RIGHT"
                    elif h_dev > HORIZONTAL_THRESH:
                        gaze_direction = "LEFT"
                    else:
                        gaze_direction = "CENTER"
                    
                    # Debug display - show deviation values
                    cv2.putText(frame, f"V:{v_dev:.3f} H:{h_dev:.3f}", (w - 180, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

