import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque

# =====================
# LOAD MODEL & STATS
# =====================
model = tf.keras.models.load_model(
    "static/models/Landmark/asl_landmark_best_model.keras"
)


CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

# =====================
# MEDIAPIPE HANDS
# =====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================
# SMOOTHING BUFFER
# =====================
prediction_buffer = deque(maxlen=15)

def smooth_prediction(buffer):
    if len(buffer) == 0:
        return None
    return max(set(buffer), key=buffer.count)

# =====================
# LANDMARK EXTRACTION
# =====================
def extract_landmarks(handLms):
    landmarks = []
    for lm in handLms.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks, dtype=np.float32)

# =====================
# WEBCAM LOOP
# =====================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural interaction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        handLms = result.multi_hand_landmarks[0]

        # Extract + normalize landmarks
        x = extract_landmarks(handLms)
        x = np.expand_dims(x, axis=0)  # (1, 63)

        # Predict
        pred = model.predict(x, verbose=0)
        cls = CLASS_NAMES[np.argmax(pred)]

        prediction_buffer.append(cls)
        final = smooth_prediction(prediction_buffer)

        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            handLms,
            mp_hands.HAND_CONNECTIONS
        )

        # Draw label
        cv2.putText(
            frame,
            f"Pred: {final}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

    cv2.imshow("ASL Landmark Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
