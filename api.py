"""FastAPI backend for static hand-gesture recognition."""

import base64
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.features import landmarks_to_features

# Resolve all project assets relative to this api.py file, not the shell's cwd.
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ONNX_MODEL_PATH = MODELS_DIR / "static_gesture_mlp.onnx"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
INDEX_FILE = BASE_DIR / "frontend" / "index.html"

app = FastAPI(title="Sign Language Recognition API")

# For separate Vercel/Render domains, set FRONTEND_ORIGINS on Render to the
# exact Vercel production origin, e.g. https://hand-tracking.vercel.app
FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Fail at startup with a clear message if the model files are absent.
if not ONNX_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy ONNX model: {ONNX_MODEL_PATH}")
if not METADATA_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy metadata: {METADATA_PATH}")

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
DIGIT_COUNT = metadata.get("modes", {}).get("digit", {}).get("end", 10)

session = ort.InferenceSession(
    str(ONNX_MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

# Use metadata names when present, then validate them against the ONNX graph.
onnx_inputs = session.get_inputs()
onnx_outputs = session.get_outputs()
if not onnx_inputs or not onnx_outputs:
    raise RuntimeError("ONNX model không có input hoặc output.")

INPUT_NAME = metadata.get("input_name") or onnx_inputs[0].name
OUTPUT_NAME = metadata.get("output_name") or onnx_outputs[0].name

if INPUT_NAME not in {item.name for item in onnx_inputs}:
    raise RuntimeError(
        f"Input ONNX '{INPUT_NAME}' không tồn tại. "
        f"Các input: {[item.name for item in onnx_inputs]}"
    )
if OUTPUT_NAME not in {item.name for item in onnx_outputs}:
    raise RuntimeError(
        f"Output ONNX '{OUTPUT_NAME}' không tồn tại. "
        f"Các output: {[item.name for item in onnx_outputs]}"
    )

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


class PredictRequest(BaseModel):
    image: str
    mode: str = "all"  # all, digit, letter


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "classes": len(CLASSES),
    }


@app.post("/predict")
def predict_gesture(req: PredictRequest):
    try:
        encoded_image = req.image.split(",", 1)[-1]
        image_bytes = base64.b64decode(encoded_image, validate=True)
        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Không thể giải mã ảnh.")

        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return {"detected": False}

        hand_landmarks = results.multi_hand_landmarks[0]

        features = landmarks_to_features(
            [hand_landmarks],
            image_width=width,
            image_height=height,
        )
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        raw_output = session.run([OUTPUT_NAME], {INPUT_NAME: features})[0]
        raw_probs = np.asarray(raw_output)[0]

        if req.mode == "digit":
            sub_probs = raw_probs[:DIGIT_COUNT]
            if sub_probs.size == 0:
                raise HTTPException(status_code=400, detail="Model không có nhóm nhãn digit.")
            idx = int(np.argmax(sub_probs))
        elif req.mode == "letter":
            sub_probs = raw_probs[DIGIT_COUNT:]
            if sub_probs.size == 0:
                raise HTTPException(status_code=400, detail="Model không có nhóm nhãn letter.")
            idx = int(DIGIT_COUNT + np.argmax(sub_probs))
        else:
            idx = int(np.argmax(raw_probs))

        if idx >= len(CLASSES):
            raise RuntimeError(
                f"Output model có {len(raw_probs)} giá trị nhưng metadata chỉ có {len(CLASSES)} nhãn."
            )

        label = CLASSES[idx]
        confidence = float(raw_probs[idx])

        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        bbox = {
            "x_min": max(0.0, min(x_coords) - 0.05),
            "y_min": max(0.0, min(y_coords) - 0.05),
            "x_max": min(1.0, max(x_coords) + 0.05),
            "y_max": min(1.0, max(y_coords) + 0.05),
        }

        return {
            "detected": True,
            "prediction": label,
            "confidence": round(confidence, 4),
            "bbox": bbox,
        }

    except HTTPException:
        raise
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Ảnh Base64 không hợp lệ.") from exc
    except Exception as exc:
        print(f"Lỗi predict: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Lỗi xử lý ảnh hoặc dự đoán mô hình.",
        ) from exc


@app.get("/", response_class=HTMLResponse)
def serve_index():
    # Keep this route for local testing; the production frontend is on Vercel.
    if INDEX_FILE.is_file():
        return INDEX_FILE.read_text(encoding="utf-8")
    return "<h1>Hand-Tracking API is running. Open /docs to inspect the API.</h1>"
