import cv2
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

proto_path = os.path.join(MODEL_DIR, "deploy.prototxt")
model_path = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

def detect_faces_dnn(image, conf_threshold=0.5):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 
        1.0,
        (300, 300), 
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append(((startX, startY, endX, endY), confidence))
            

    print(f"[DEBUG] Number of detections: {detections.shape[2]}")
    print(f"[DEBUG] Faces found: {len(faces)}")

    return faces
