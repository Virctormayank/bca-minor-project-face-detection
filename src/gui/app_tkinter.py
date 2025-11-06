import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import os
import time
import mediapipe as mp
import csv
from datetime import datetime
from ..gestures.hand_signs import detect_gesture
from ..utils.viz import draw_face_box



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

class FaceHandAttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face & Hand Detection + Attendance")
        self.root.geometry("900x700")

        self.cap = None
        self.running = False
        self.current_frame = None
        self.prev_time = 0

        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_names = []
        self.frame_count = 0

        
        # Layout
      
        self.camera_frame = tk.Frame(root, width=900, height=600, bg="black")
        self.camera_frame.grid(row=0, column=0, sticky="nsew")
        self.camera_frame.pack_propagate(0)

        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        self.btn_frame = tk.Frame(root, height=100)
        self.btn_frame.grid(row=1, column=0, sticky="ew", pady=5)

        # Buttons
        self.open_btn = tk.Button(self.btn_frame, text="Open Camera", command=self.start_camera)
        self.open_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(self.btn_frame, text="Stop Camera", command=self.stop_camera)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.screenshot_btn = tk.Button(self.btn_frame, text="Screenshot", command=self.take_screenshot)
        self.screenshot_btn.pack(side=tk.LEFT, padx=5)
        self.quit_btn = tk.Button(self.btn_frame, text="Quit", command=self.quit_app)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        # Load known faces
        self.load_known_faces()

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)


    def load_known_faces(self):
        project_root = os.getcwd()
        faces_dir = os.path.join(project_root, "data/faces")
        if not os.path.exists(faces_dir):
            print(f"[Warning] Faces folder not found: {faces_dir}")
            return

        for person_name in os.listdir(faces_dir):
            person_dir = os.path.join(faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            for file in os.listdir(person_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, file)
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                 
                        if person_name.lower() == "mayank":
                            self.known_face_names.append("Mayank")
                        else:
                            self.known_face_names.append(person_name)
                        print(f"[DEBUG] Loaded {person_name} from {file}")
                    else:
                        print(f"[WARNING] No face found in {file}")

        print(f"[Info] Loaded {len(self.known_face_names)} known faces.")
        
        
    def mark_attendance(self, name):
        if name == "Unknown":
            return

        csv_dir = os.path.join(os.getcwd(), "attendance")
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = os.path.join(csv_dir, "attendance.csv")

        existing_names = set()
        if os.path.exists(csv_file):
            with open(csv_file, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 1:
                        existing_names.add(row[0])

        if name not in existing_names:
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d,%H:%M:%S")
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name, dt_string])
            print(f"[Attendance] {name} marked at {dt_string}")

    # Start camera
   
    def start_camera(self):
        if self.running:
            return

        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return

        self.running = True
        self.update_frame()

    # Stop camera
 
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_label.config(image="")

    # Take screenshot
    
    def take_screenshot(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame to save!")
            return

        screenshot_dir = os.path.join(os.getcwd(), "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")

        cv2.imwrite(file_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
        messagebox.showinfo("Saved", f"Screenshot saved to:\n{file_path}")

    # Update frame
   
    def update_frame(self):
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)  # Mirror the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

       
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1

        if self.frame_count % 3 == 0:
            self.face_locations = face_recognition.face_locations(small_frame)
            self.face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)
            self.face_names = []

            for encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.45)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    if self.known_face_names[first_match_index] == "Mayank1":
                        name = "Mayank"
                    else:
                        name = self.known_face_names[first_match_index]
                self.face_names.append(name)

        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            draw_face_box(frame, left, top, right-left, bottom-top, label=name)
            
            self.mark_attendance(name)

        # Hand detection
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks.landmark)
                if gesture:
                    x = int(hand_landmarks.landmark[8].x * w)
                    y = int(hand_landmarks.landmark[8].y * h) - 20
                    cv2.putText(frame, gesture, (x, max(y, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time != 0 else 0
        self.prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display frame in Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_label.imgtk = imgtk
        self.camera_label.config(image=imgtk)

        self.root.after(10, self.update_frame)


    # Quit app

    def quit_app(self):
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceHandAttendanceApp(root)
    root.mainloop()
