import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow/MediaPipe logs

def detect_gesture(landmarks):
    """
    Detects simple hand gestures based on MediaPipe landmarks.
    Returns gesture label as a string.
    """

    # Tip landmarks
    tips = [4, 8, 12, 16, 20]
    thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip = [landmarks[i].y for i in tips]

    # Finger MCP landmarks (base of fingers)
    thumb_mcp = landmarks[2].y
    index_mcp = landmarks[5].y
    middle_mcp = landmarks[9].y
    ring_mcp = landmarks[13].y
    pinky_mcp = landmarks[17].y

    # Open Hand
    if all(tip < mcp for tip, mcp in zip([thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip],
                                         [thumb_mcp, index_mcp, middle_mcp, ring_mcp, pinky_mcp])):
        return "Hand Raised ✋"

    # Thumbs Up
    if thumb_tip < thumb_mcp and all(tip > mcp for tip, mcp in zip([index_tip, middle_tip, ring_tip, pinky_tip],
                                                                    [index_mcp, middle_mcp, ring_mcp, pinky_mcp])):
        return "Thumbs Up 👍"

    # Victory / Peace
    if index_tip < index_mcp and middle_tip < middle_mcp and ring_tip > ring_mcp and pinky_tip > pinky_mcp:
        return "Victory ✌️"

    # Fist
    if all(tip > mcp for tip, mcp in zip([thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip],
                                         [thumb_mcp, index_mcp, middle_mcp, ring_mcp, pinky_mcp])):
        return "Fist ✊"

    # Rock Sign 🤘 (Index + Pinky extended)
    if index_tip < index_mcp and pinky_tip < pinky_mcp and middle_tip > middle_mcp and ring_tip > ring_mcp:
        return "Rock 🤘"

    # OK Sign (Thumb touches Index tip)
    if abs(thumb_tip - index_tip) < 0.03:
        return "OK 👌"

    return None
