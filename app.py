# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
import torch
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "model.pt")
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

def preprocess_image(img):
    # Ensure grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (28, 28))
    img = 255 - img              # MNIST inversion
    img = img.astype(np.float32) / 255.0

    # Shape: [1, 1, 28, 28]
    img = img.reshape(1, 1, 28, 28)

    # Convert to torch tensor
    img_tensor = torch.from_numpy(img)

    return img_tensor

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict-canvas", methods=["POST"])
def predict_canvas():
    data = request.json["image"]
    encoded = data.split(",")[1]
    img_bytes = base64.b64decode(encoded)

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    img_tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        digit = output.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()

    return jsonify({
        "prediction": digit,
        "confidence": confidence
    })

@app.route("/predict-upload", methods=["POST"])
def predict_upload():
    file = request.files["file"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    img_tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        digit = output.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()

    return jsonify({
        "prediction": digit,
        "confidence": confidence
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    img_tensor = preprocess_image(img)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        digit = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()

    return jsonify({
        "prediction": digit,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
