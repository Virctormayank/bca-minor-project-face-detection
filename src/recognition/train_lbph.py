import cv2
import os
import numpy as np

DATASET = "dataset"
MODEL_PATH = "models/lbph_model.xml"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)
    if not os.path.isdir(person_path):
        continue
    
    label_map[current_label] = person
    
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        faces.append(img)
        labels.append(current_label)
    
    current_label += 1

recognizer.train(faces, np.array(labels))

recognizer.write(MODEL_PATH)

# Save label map
import pickle
with open("models/lbph_labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ LBPH Training Completed")
print("✅ Saved model and labels")
