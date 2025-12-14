from flask import Flask, request, jsonify, render_template
import os
import cv2
import gdown
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet, resnet

    
# Load model
MODEL_SOURCES = {
    "efficientnet": {
        "path": "static/models/EfficientNet/model.h5",
        "drive_id": "1u5N396JC-vw-aSpAVHn8EF212SO3aZWv"
    },
    "resnet": {
        "path": "static/models/ResNet/model.h5",
        "drive_id": "1UYrHIHUaR77ku7WphQRD9saPMirqpVhB"
    },
    "cnn": {
        "path": "static/models/CNN/asl_model_cnn.keras",
        "drive_id": None  # already local
    }
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

def is_lfs_pointer(path):
    try:
        with open(path, "rb") as f:
            return b"git-lfs.github.com" in f.read(200)
    except:
        return True

def download_from_drive(file_id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

loaded_models = {}
broken_models = set()
def get_model(name):
    if name in loaded_models:
        return loaded_models[name]

    if name in broken_models:
        return None

    info = MODEL_SOURCES.get(name)
    if not info:
        return None

    path = info["path"]

    if not os.path.exists(path) or is_lfs_pointer(path):
        if not info["drive_id"]:
            print(f"[SKIP] No source for {name}")
            broken_models.add(name)
            return None

        print(f"[INFO] Downloading {name} from Google Drive...")
        try:
            download_from_drive(info["drive_id"], path)
        except Exception as e:
            print(f"[FAIL] Download failed for {name}: {e}")
            broken_models.add(name)
            return None

    try:
        model = tf.keras.models.load_model(path)
        loaded_models[name] = model
        print(f"[OK] Loaded {name}")
        return model
    except Exception as e:
        print(f"[FAIL] Could not load {name}: {e}")
        broken_models.add(name)
        return None
    
    
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

    if model_name not in MODEL_SOURCES:
        return jsonify({"prediction": "Model not available", "confidence": 0})

    model = get_model(model_name)
    if model is None:
        return jsonify({
        "prediction": "Model unavailable",
        "confidence": 0,
    })
    preprocess = PREPROCESS[model_name]
    size = INPUT_SIZE[model_name]

    if not file:
        return jsonify({"prediction": "No Hands", "confidence": 0})

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