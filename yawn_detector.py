import cv2
import dlib
import imutils
from scipy.spatial import distance as dist
from imutils import face_utils

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  
    B = dist.euclidean(mouth[4], mouth[8])   
    C = dist.euclidean(mouth[0], mouth[6])   
    mar = (A + B) / (2.0 * C)
    return mar

MOUTH_AR_THRESH = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
