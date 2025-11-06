import cv2
import os
import urllib.request
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "data/models")
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isfile(PROTO_PATH):
    print("[INFO] Downloading deploy.prototxt...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        PROTO_PATH
    )

if not os.path.isfile(MODEL_PATH):
    print("[INFO] Downloading res10_300x300_ssd_iter_140000.caffemodel...")
    urllib.request.urlretrieve(
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        MODEL_PATH
    )

net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
print("[INFO] Face detector initialized!")

def detect_faces_dnn(frame, conf_threshold=0.5):
    """
    Detect faces in a frame using DNN.
    Returns a list of ((x1, y1, x2, y2), confidence)
    """
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append(((x1, y1, x2, y2), confidence))
    return faces

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera. Make sure macOS camera permissions are granted.")
    exit()

print("[INFO] Camera opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    faces = detect_faces_dnn(frame)
    print(f"[DEBUG] Faces detected: {faces}")  # debug output

    for (box, conf) in faces:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    
    cv2.imshow("Face Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break


cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera released and windows closed.")
