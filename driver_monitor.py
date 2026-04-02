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

