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

                # Draw iris circles
                for iris_pts in [left_iris_pts, right_iris_pts]:
                    iris_pixel = np.array([[int(p[0] * w), int(p[1] * h)] for p in iris_pts])
                    (cx, cy), radius = cv2.minEnclosingCircle(iris_pixel)
                    cv2.circle(frame, (int(cx), int(cy)), int(radius), (255, 0, 255), 1)

            # Head Pose Detection
            pts = []
            for idx in LM_POINTS:
                lm = landmarks[idx]
                pts.append([lm.x * w, lm.y * h])
            pts = np.array(pts, dtype=np.float64)

            focal_length = w
            cam_matrix = np.array([[focal_length, 0, w/2],
                                   [0, focal_length, h/2],
                                   [0, 0, 1]])
            dist_coeffs = np.zeros((4, 1))

            success, rvec, tvec = cv2.solvePnP(model_points, pts, cam_matrix, dist_coeffs)
            rot_mat, _ = cv2.Rodrigues(rvec)
            sy = np.sqrt(rot_mat[0, 0]**2 + rot_mat[1, 0]**2)
            yaw = np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2]))
            pitch = np.degrees(np.arctan2(-rot_mat[2, 0], sy))

            yaw_history.append(yaw)
            pitch_history.append(pitch)
            smoothed_yaw = sum(yaw_history) / len(yaw_history)
            smoothed_pitch = sum(pitch_history) / len(pitch_history)

            if head_recalibrating:
                head_calibration_yaw.append(smoothed_yaw)
                head_calibration_pitch.append(smoothed_pitch)
                remaining = CALIBRATION_FRAMES - len(head_calibration_yaw)
                if remaining <= 0:
                    head_baseline_yaw = float(np.mean(head_calibration_yaw))
                    head_baseline_pitch = float(np.mean(head_calibration_pitch))
                    head_recalibrating = False
                    head_status = "On Road"

            if head_baseline_yaw is not None:
                yaw_dev = smoothed_yaw - head_baseline_yaw
                pitch_dev = smoothed_pitch - head_baseline_pitch
                looking_forward = abs(yaw_dev) < YAW_THRESHOLD and abs(pitch_dev) < PITCH_THRESHOLD

                if looking_forward:
                    off_road_start = None
                    head_status = "On Road"
                else:
                    if off_road_start is None:
                        off_road_start = time.time()
                    elif time.time() - off_road_start >= 2.0:
                        head_status = "Off Road"

                if yaw_dev > YAW_THRESHOLD:
                    head_direction = "Up"
                elif yaw_dev < -YAW_THRESHOLD:
                    head_direction = "Down"
                elif pitch_dev > PITCH_THRESHOLD:
                    head_direction = "Left"
                elif pitch_dev < -PITCH_THRESHOLD:
                    head_direction = "Right"
                else:
                    head_direction = "Forward"
            else:
                head_direction = "Calibrating"

            # Draw head direction arrow
            nose_3d = model_points[0].reshape(1, 3)
            nose_3d_forward = (model_points[0] + np.array([0.0, 0.0, 120.0])).reshape(1, 3)
            nose_2d, _ = cv2.projectPoints(nose_3d, rvec, tvec, cam_matrix, dist_coeffs)
            nose_forward_2d, _ = cv2.projectPoints(nose_3d_forward, rvec, tvec, cam_matrix, dist_coeffs)
            p1 = tuple(nose_2d[0].ravel().astype(int))
            p2 = tuple(nose_forward_2d[0].ravel().astype(int))
            line_color = (0, 0, 255) if head_status == "Off Road" else (0, 255, 0)
            cv2.arrowedLine(frame, p1, p2, line_color, 2, tipLength=0.25)

    # Dlib Processing (Drowsiness + Yawn) 
    drowsy_alert = False
    yawn_alert = False
    dlib_rects = dlib_detector(gray, 0)

    for rect in dlib_rects:
        shape = dlib_predictor(gray, rect)
        shape_np = face_utils.shape_to_np(shape)

        # Drowsiness Detection
        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            drowsy_counter += 1
            if drowsy_counter >= EYE_AR_CONSEC_FRAMES:
                drowsy_alert = True
        else:
            drowsy_counter = 0

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)

        # Yawn Detection
        mouth = shape_np[48:68]
        mar = mouth_aspect_ratio(mouth)
        if mar > MOUTH_AR_THRESH:
            yawn_alert = True

        # Draw mouth contour
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

    # Display Status Panel
    panel_y = 30
    cv2.putText(frame, f"Gaze: {gaze_direction}", (20, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    panel_y += 30
    cv2.putText(frame, f"Head: {head_direction}", (20, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    panel_y += 30
    status_color = (0, 0, 255) if head_status == "Off Road" else (0, 255, 0)
    cv2.putText(frame, f"Road Status: {head_status}", (20, panel_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    panel_y += 30

    if drowsy_alert:
        cv2.putText(frame, "DROWSINESS DETECTED!", (20, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panel_y += 30

    if yawn_alert:
        cv2.putText(frame, "YAWNING DETECTED!", (20, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Driver Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        # Recalibrate head pose
        head_calibration_yaw.clear()
        head_calibration_pitch.clear()
        head_baseline_yaw = None
        head_baseline_pitch = None
        head_recalibrating = True
        head_status = "Calibrating"
    if key == ord('g'):
        # Recalibrate gaze
        gaze_vertical_baseline = None
        gaze_horizontal_baseline = None
        gaze_vertical_samples.clear()
        gaze_horizontal_samples.clear()

cap.release()
cv2.destroyAllWindows()
