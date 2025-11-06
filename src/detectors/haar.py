import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces_haar(frame):
    """
    Detect faces in a frame using Haar Cascade.
    Returns list of bounding boxes (x, y, w, h).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    return faces
