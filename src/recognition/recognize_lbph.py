import cv2
import pickle

MODEL_PATH = "models/lbph_model.xml"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

with open("models/lbph_labels.pkl", "rb") as f:
    label_map = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

print("[INFO] Running LBPH Face Recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            label, confidence = recognizer.predict(roi)
            name = label_map[label]
        except:
            name = "Unknown"
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

    cv2.imshow("LBPH Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
