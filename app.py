from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet, resnet

    
# Load model
MODEL_PATHS = {
    "efficientnet": "static/models/EfficientNet/asl_best_model(1).h5",
    "cnn": "static/models/CNN/asl_model_cnn.keras",
    "resnet": "static/models/ResNet34/asl_resnet_best_model.h5",
    # "convnext": "static/models/ConvNeXt-Tiny/best_asl_convnext_model.h5"
}

PREPROCESS = {
    "efficientnet": efficientnet.preprocess_input,
    "resnet": resnet.preprocess_input,
    "convnext": lambda x: x / 255.0,  
    "cnn": lambda x: x / 255.0,  
}

INPUT_SIZE = {
    "efficientnet": 224,
    "resnet": 224,
    "convnext": 224,
    "cnn": 224,
}

loaded_models = {}
for key, path in MODEL_PATHS.items():
    try:
        loaded_models[key] = tf.keras.models.load_model(path)
        print(f"[OK] Loaded {key} model")
    except Exception as e:
        print(f"[ERROR] Could not load {key}: {e}")

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get frame + model name
    file = request.files.get("frame")
    model_name = request.form.get("model", "efficientnet").lower()

    if model_name not in loaded_models:
        return jsonify({"prediction": "None", "confidence": 0})

    model = loaded_models[model_name]
    preprocess = PREPROCESS[model_name]
    size = INPUT_SIZE[model_name]

    if not file:
        return jsonify({"prediction": "None", "confidence": 0})

    # Decode image
    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"prediction": "None", "confidence": 0})
    
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))

    # Preprocess
    img = preprocess(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img, verbose=0)[0]
    idx = np.argmax(pred)

    return jsonify({
        "prediction": CLASS_NAMES[idx],
        "confidence": float(pred[idx]),
        "model_used": model_name
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)