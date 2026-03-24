import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # required for iris landmarks 468-477
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Iris landmark indices
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Corner landmarks
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# Full eye contour landmarks 
LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_LANDMARKS = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]

# Landmark indices for vertical measurement
LEFT_EYE_TOPS = [159, 158, 157]
LEFT_EYE_BOTTOMS = [145, 144, 153]
RIGHT_EYE_TOPS = [386, 385, 384]
RIGHT_EYE_BOTTOMS = [374, 373, 380]

def iris_center(landmarks, iris_indices):
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
    # Use min of top set and max of bottom set to maximize span and sensitivity
    top_y = min(landmarks[i].y for i in tops)
    bottom_y = max(landmarks[i].y for i in bottoms)
    span = bottom_y - top_y
    if span <= 0:
        return 0.5
    return (center[1] - top_y) / span

alpha = 0.5  # smoothing factor for exponential moving average
smoothed_horizontal = None
smoothed_vertical = None
