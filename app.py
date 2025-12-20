from flask import Flask, request, jsonify, render_template
import os
import cv2
import gdown
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import efficientnet, resnet

    
# Load model
MODEL_SOURCES = {
    "efficientnet": {
        "path": "/temp/models/EfficientNet/model.h5",
        "drive_id": "1u5N396JC-vw-aSpAVHn8EF212SO3aZWv"
    },
    "resnet": {
        "path": "/temp/models/ResNet34/model.h5",
        "drive_id": "1UYrHIHUaR77ku7WphQRD9saPMirqpVhB"
    },
    "cnn": {
        "path": "static/models/CNN/asl_model_cnn.keras",
        "drive_id": None  
    },
    "landmark": {
        "path": "static/models/Landmark/asl_landmark_best_model.keras",
        "drive_id": None  
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
    gdown.download(
        url,
        output_path,
        quiet=False,
        use_cookies=False
    )

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
        model = tf.keras.models.load_model(path, compile=False)
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
    model_name = request.form.get("model", "").lower()

    if model_name not in MODEL_SOURCES:
        return jsonify({"prediction": "Model not available", "confidence": 0})

    model = get_model(model_name)
    if model is None:
        return jsonify({"prediction": "Model unavailable", "confidence": 0})

    # ======================
    # LANDMARK MODEL PATH
    # ======================
    if model_name == "landmark":
        lm_json = request.form.get("landmarks")
        if not lm_json:
            return jsonify({"prediction": "No Hand", "confidence": 0})

        landmarks = np.array(json.loads(lm_json), dtype=np.float32)

        if landmarks.shape != (63,):
            return jsonify({"prediction": "Invalid Input", "confidence": 0})

        x = np.expand_dims(landmarks, axis=0)  # (1,63)
        pred = model.predict(x, verbose=0)[0]
        idx = np.argmax(pred)

        return jsonify({
            "prediction": CLASS_NAMES[idx],
            "confidence": float(pred[idx]),
            "model_used": "landmark"
        })

    # ======================
    # IMAGE MODEL PATH
    # ======================
    file = request.files.get("frame")
    if not file:
        return jsonify({"prediction": "No Image", "confidence": 0})

    arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"prediction": "Invalid Image", "confidence": 0})

    preprocess = PREPROCESS[model_name]
    size = INPUT_SIZE[model_name]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = preprocess(img.astype("float32"))
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    idx = np.argmax(pred)

    return jsonify({
        "prediction": CLASS_NAMES[idx],
        "confidence": float(pred[idx]),
        "model_used": model_name
    })

        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
