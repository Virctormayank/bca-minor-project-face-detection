import sys
import os
import streamlit as st
import numpy as np
import cv2
import subprocess
import pandas as pd

# ✅ Add src folder to Python path
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from detectors.haar import detect_faces_haar
from detectors.dnn import detect_faces_dnn
from utils.viz import draw_boxes

# ✅ Streamlit Page Config
st.set_page_config(page_title="Smart Vision System", page_icon="🤖")

st.title("🤖 Smart Vision System Dashboard")
st.markdown("### Face Detection | Face Recognition | Hand Gesture | Attendance System")

# ✅ Sidebar Menu
module = st.sidebar.radio(
    "Choose a feature:",
    [
        "📸 Face Detection (Image Upload)",
        "🧠 Face Recognition + Attendance",
        "✋ Hand Gesture Recognition",
        "📊 View Attendance Records"
    ]
)

# ================================================================
# ✅ 1. IMAGE FACE DETECTION
# ================================================================
if module == "📸 Face Detection (Image Upload)":
    st.subheader("Upload an Image for Face Detection")

    method = st.sidebar.selectbox("Choose Detector", ["haar", "dnn"])
    conf = st.sidebar.slider("Confidence (DNN only)", 0.1, 1.0, 0.5)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # ✅ Detect faces
        faces = detect_faces_haar(image) if method == "haar" else detect_faces_dnn(image, conf)

        output = draw_boxes(image.copy(), faces)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Faces")


# ================================================================
# ✅ 2. FACE RECOGNITION + ATTENDANCE
# ================================================================
elif module == "🧠 Face Recognition + Attendance":
    st.subheader("Real-Time Face Recognition + Attendance")

    st.info("Start webcam-based attendance. Marks attendance only once per person.")

    start = st.button("✅ Start Attendance")
    stop = st.button("⛔ Stop Attendance")

    # ✅ Start recognizer script
    if start:
        st.success("Attendance is running... Close camera window to stop.")
        subprocess.Popen([sys.executable, os.path.join(SRC_PATH, "recognition", "recognize_lbph.py")])

    # ✅ Stop recognizer (via flag file)
    if stop:
        with open("attendance/stop_flag", "w") as f:
            f.write("stop")
        st.warning("Stop signal sent!")


# ------------------------------------------------------------------
# ✅ 3. Hand Gesture Recognition
# ------------------------------------------------------------------
elif module == "✋ Hand Gesture Recognition":
    st.subheader("Real-Time Hand Gesture Recognition")

    st.info("Click START to open the gesture camera window. Press 'q' to stop.")

    if st.button("▶ Start Gesture Recognition"):
        st.success("Gesture Recognition Running... Close camera window to stop.")
        
        gesture_path = os.path.join(SRC_PATH, "gestures", "gesture_demo.py")
        subprocess.Popen([sys.executable, gesture_path])


# ================================================================
# ✅ 4. ATTENDANCE VIEW + DELETE + REMOVE DUPLICATES
# ================================================================
elif module == "📊 View Attendance Records":
    st.subheader("Attendance Records")

    att_file = "attendance/attendance.csv"

    if os.path.exists(att_file):
        df = pd.read_csv(att_file)
        st.dataframe(df)

        # ✅ Buttons (side-by-side)
        col1, col2, col3 = st.columns(3)

        # ✅ Download CSV
        with col1:
            st.download_button("⬇ Download CSV", df.to_csv(index=False), "attendance.csv")

        # ✅ Remove duplicates
        with col2:
            if st.button("🧹 Remove Duplicates"):
                df = df.drop_duplicates()
                df.to_csv(att_file, index=False)
                st.success("Duplicates removed successfully!")

        # ✅ Clear attendance sheet
        with col3:
            if st.button("🗑 Clear All"):
                df = pd.DataFrame(columns=["Name", "Roll Number", "Date", "Time"])
                df.to_csv(att_file, index=False)
                st.error("All attendance records cleared!")

    else:
        st.warning("⚠ No attendance file found. Run recognition first.")
