# src/recognition/recognize_and_mark_attendance.py
import cv2
import numpy as np
import pickle
import csv
import os
from datetime import datetime
import time

# Load trained model
with open(os.path.join("models", "face_model.pkl"), "rb") as f:
    model, label_map = pickle.load(f)

# Prepare attendance folder & file
os.makedirs("attendance", exist_ok=True)
attendance_file = os.path.join("attendance", "attendance.csv")
stop_file = os.path.join("attendance", "stop_attendance")

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Roll", "Date", "Time"])

# Roll Map (normalized)
roll_numbers = {
    "mayank": "166",
    "mayank_pundora": "167",
    "md_aman": "168",
}

roll_numbers_norm = {k.lower().replace(" ", "_"): v for k, v in roll_numbers.items()}

# Load today's attendance
today = datetime.now().strftime("%Y-%m-%d")
marked_today = set()

with open(attendance_file, "r") as f:
    for row in csv.reader(f):
        if len(row) >= 3 and row[2] == today:
            marked_today.add(row[0].lower().replace(" ", "_"))

print("[INFO] Already marked today:", marked_today)

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("[INFO] Multi-face attendance running... Press 'q' to quit.")

try:
    while True:
        if os.path.exists(stop_file):
            print("[INFO] Stop signal received. Exiting.")
            break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ✅ Detect ALL faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y + h, x:x + w]

            # ✅ square crop
            size = min(face_gray.shape[0], face_gray.shape[1])
            face_square = face_gray[0:size, 0:size]

            # resize to model input
            face_resized = cv2.resize(face_square, (100, 100)).flatten()

            # Predict with model
            pred = model.predict([face_resized])[0]
            name_raw = label_map[pred]
            name_norm = name_raw.lower().replace(" ", "_")

            print(f"[DEBUG] {name_raw} detected at X:{x}, Y:{y}")

            # ✅ Mark attendance for each face independently
            if name_norm in roll_numbers_norm and name_norm not in marked_today:
                roll = roll_numbers_norm[name_norm]
                now = datetime.now()

                with open(attendance_file, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [name_raw, roll, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
                    )

                marked_today.add(name_norm)
                print(f"[ATTEND] Marked {name_raw} ({roll})")

            # ✅ Draw box for EACH FACE
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name_raw, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Multi-Face Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if os.path.exists(stop_file):
        os.remove(stop_file)
    print("[INFO] Multi-face attendance stopped.")
