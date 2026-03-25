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

# Vertical calibration baseline
vertical_baseline = None
vertical_samples = []
CALIBRATION_FRAMES = 40

# Adaptive vertical range tracking after post calibration
vertical_min = None
vertical_max = None
ADAPT_FRAMES = 150  # frames after which we slowly tighten range


cap = cv2.VideoCapture(0)
print("Gaze Detector Running... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            landmarks = face.landmark

            left_center, left_iris_pts = iris_center(landmarks, LEFT_IRIS)
            right_center, right_iris_pts = iris_center(landmarks, RIGHT_IRIS)

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
                smoothed_horizontal = alpha * avg_h + (1 - alpha) * smoothed_horizontal
                smoothed_vertical = alpha * avg_v + (1 - alpha) * smoothed_vertical

            # Build baseline first frames or on demand if not set
            if vertical_baseline is None and len(vertical_samples) < CALIBRATION_FRAMES:
                vertical_samples.append(smoothed_vertical)
                if len(vertical_samples) == CALIBRATION_FRAMES:
                    vertical_baseline = float(np.mean(vertical_samples))
                    vertical_min = smoothed_vertical
                    vertical_max = smoothed_vertical
                direction = "CENTER"
            else:
                if vertical_baseline is not None:
                    dev = smoothed_vertical - vertical_baseline
                    # Update adaptive min/max
                    if vertical_min is None or smoothed_vertical < vertical_min:
                        vertical_min = smoothed_vertical
                    if vertical_max is None or smoothed_vertical > vertical_max:
                        vertical_max = smoothed_vertical
                    # Dynamic threshold based on observed spread
                    spread = max(vertical_max - vertical_min, 1e-3)
                    # Fraction thresholds; smaller fractions increase sensitivity
                    up_thresh = vertical_baseline - spread * 0.25
                    down_thresh = vertical_baseline + spread * 0.25
                    if smoothed_vertical < up_thresh:
                        direction = "UP"
                    elif smoothed_vertical > down_thresh:
                        direction = "DOWN"
                    else:
                        if smoothed_horizontal < 0.38:
                            direction = "RIGHT"
                        elif smoothed_horizontal > 0.62:
                            direction = "LEFT"
                        else:
                            direction = "CENTER"
                else:
                    # Fallback absolute thresholds (should rarely be used)
                    if smoothed_vertical < 0.35:
                        direction = "UP"
                    elif smoothed_vertical > 0.58:
                        direction = "DOWN"
                    else:
                        if smoothed_horizontal < 0.38:
                            direction = "RIGHT"
                        elif smoothed_horizontal > 0.62:
                            direction = "LEFT"
                        else:
                            direction = "CENTER"

            cv2.putText(frame, f"Gaze: {direction}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if vertical_baseline is not None:
                spread = max(vertical_max - vertical_min, 0.0) if vertical_min is not None else 0.0
                cv2.putText(frame, f"H:{smoothed_horizontal:.2f} V:{smoothed_vertical:.2f} B:{vertical_baseline:.2f} S:{spread:.2f}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            else:
                cv2.putText(frame, f"Calibrating V:{smoothed_vertical:.2f}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            # Draw eye contours
            for eye_landmarks in [LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS]:
                pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in eye_landmarks])
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=1)

            # Draw iris outlines (enclosing circle)
            for iris_pts in [left_iris_pts, right_iris_pts]:
                iris_pixel = np.array([[int(p[0] * w), int(p[1] * h)] for p in iris_pts])
                (cx, cy), radius = cv2.minEnclosingCircle(iris_pixel)
                cv2.circle(frame, (int(cx), int(cy)), int(radius), (0, 0, 255), 1)
                cv2.circle(frame, (int(cx), int(cy)), 1, (0, 0, 255), -1)

    cv2.imshow("Gaze Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        vertical_baseline = None
        vertical_samples.clear()
        vertical_min = None
        vertical_max = None

cap.release()
cv2.destroyAllWindows()
