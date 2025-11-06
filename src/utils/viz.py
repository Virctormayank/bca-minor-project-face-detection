# utils/viz.py
import cv2

def draw_boxes(image, faces):
    """
    Draw rectangles on image for list of faces.
    Supports both Haar (x, y, w, h) and DNN ((x1, y1, x2, y2), confidence) formats.
    """
    for f in faces:
        # DNN output: ((x1, y1, x2, y2), conf)
        if isinstance(f, tuple) and len(f) == 2 and isinstance(f[0], (tuple, list)):
            (x1, y1, x2, y2), conf = f
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{conf*100:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Haar output: (x, y, w, h)
        elif isinstance(f, (list, tuple)) and len(f) == 4:
            x, y, w, h = f
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

def draw_hand_landmarks(frame, landmarks, label=None):
    """Draws hand landmarks on the frame with optional label."""
    for point in landmarks:
        cv2.circle(frame, point, 5, (255, 0, 0), -1)
    if label:
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
