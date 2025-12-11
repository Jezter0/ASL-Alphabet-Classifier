import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import pyttsx3
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("static/models/EfficientNet/asl_best_model(1).h5")

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N',
    'O','P','Q','R','S','T','U','V','W','X','Y','Z',
]

engine = pyttsx3.init()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Smoothing buffer
prediction_buffer = deque(maxlen=20)

def smooth_prediction(buffer):
    if len(buffer) == 0:
        return None
    return max(set(buffer), key=buffer.count)

def speak(text):
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:

            # Get bounding box
            x_coords = [lm.x for lm in handLms.landmark]
            y_coords = [lm.y for lm in handLms.landmark]

            xmin_raw = int(min(x_coords) * w)
            xmax_raw = int(max(x_coords) * w)
            ymin_raw = int(min(y_coords) * h)
            ymax_raw = int(max(y_coords) * h)

            # Make box bigger
            hand_w = xmax_raw - xmin_raw
            hand_h = ymax_raw - ymin_raw
            scale = 1.5
            xmin = max(0, int(xmin_raw - hand_w * (scale - 1)/2))
            xmax = min(w, int(xmax_raw + hand_w * (scale - 1)/2))
            ymin = max(0, int(ymin_raw - hand_h * (scale - 1)/2))
            ymax = min(h, int(ymax_raw + hand_h * (scale - 1)/2))

            # Crop ROI
            roi = frame[ymin:ymax, xmin:xmax]
            if roi.size != 0:
                # Resize to 224x224
                img = cv2.resize(roi, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_input(img)  # EfficientNet preprocessing
                img = np.expand_dims(img, axis=0)

                # Predict
                pred = model.predict(img, verbose=0)
                cls = CLASS_NAMES[np.argmax(pred)]

                # Smoothing
                prediction_buffer.append(cls)
                final = smooth_prediction(prediction_buffer)

                # Draw bounding box & label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                cv2.putText(frame, f"Pred: {final}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # Speak
                if len(prediction_buffer) == prediction_buffer.maxlen:
                    speak(final)
                    prediction_buffer.clear()

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()