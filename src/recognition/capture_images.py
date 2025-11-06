import cv2
import os

# User ka naam input
name = input("Enter your name: ").strip()

# Dataset folder
dataset_path = "dataset"
user_folder = os.path.join(dataset_path, name)

os.makedirs(user_folder, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detecting frame.")
        break

    cv2.imshow("Capturing Images - Press 'q' to stop", frame)

    # Save image
    img_path = os.path.join(user_folder, f"{name}_{count}.jpg")
    cv2.imwrite(img_path, frame)
    count += 1

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"✅ Images saved in: {user_folder}")
