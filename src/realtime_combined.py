import cv2
import mediapipe as mp
import time
from detectors.dnn import detect_faces_dnn
from gestures.hand_signs import detect_gesture
from utils.viz import draw_face_box
from recognition.face_recognition_model import load_model
from utils.attendance import mark_attendance

# Load trained LBPH model
recognizer = load_model("face_model.yml")

# Name + Roll mapping (manually add here)
students = {
    "Mayank Singh": "166"
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera. Check permissions.")
    exit()

print("[INFO] Camera opened successfully! Press 'q' to quit.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)  # mirror image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    faces = detect_faces_dnn(frame)  # Returns list of (box, confidence)
    for (box, conf) in faces:
        (startX, startY, endX, endY) = box

        # Crop & convert to grayscale for recognition
        face_crop = frame[startY:endY, startX:endX]
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Predict face
        label, confidence = recognizer.predict(gray)

        # Get name from label
        names = list(students.keys())
        if label < len(names):
            name = names[label]
            roll = students[name]
        else:
            name = "Unknown"
            roll = "--"

        # Draw Box + Name
        draw_face_box(frame, startX, startY, endX - startX, endY - startY,
                    label=f"{name} ({int(confidence)})")

        # ✅ Mark Attendance if recognized
        if name != "Unknown":
            mark_attendance(name, roll)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           
            gesture = detect_gesture(hand_landmarks.landmark)

            if gesture:
                # Get position of index finger tip to show label above hand
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[8].x * w)
                y = int(hand_landmarks.landmark[8].y * h) - 20

                cv2.putText(frame, gesture, (x, max(y, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                print(f"[DEBUG] Gesture detected: {gesture}")

    #  Display FPS
  
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #  Show frame

    cv2.imshow("Face + Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Camera released and windows closed.")
