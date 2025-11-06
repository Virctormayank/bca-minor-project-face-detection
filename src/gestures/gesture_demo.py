import cv2
from cvzone.HandTrackingModule import HandDetector

detector = HandDetector(detectionCon=0.7, maxHands=1)

print("[INFO] Starting Hand Gesture Recognition...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Gesture mapping
        gesture_text = None
        if fingers == [0,1,0,0,0]:
            gesture_text = "Pointing ☝️"
        elif fingers == [1,1,1,1,1]:
            gesture_text = "Open Hand ✋"
        elif fingers == [0,0,0,0,0]:
            gesture_text = "Fist ✊"
        elif fingers == [0,1,1,0,0]:
            gesture_text = "Peace ✌️"
        elif fingers == [1,0,0,0,0]:
            gesture_text = "Thumbs Up 👍"

        if gesture_text:
            cv2.putText(img, gesture_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,255,0), 3)
            print("[Gesture]:", gesture_text)

    cv2.imshow("Hand Gesture Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting Gesture Recognition...")
        break

cap.release()
cv2.destroyAllWindows()
