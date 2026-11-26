"""
api.py - Backend FastAPI phục vụ nhận diện ngôn ngữ ký hiệu
- Cung cấp endpoint REST API /predict nhận frame ảnh base64
- Trả về nhãn, độ tin cậy, bounding box và landmarks
- Phục vụ giao diện web index.html
"""

import base64
import json
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

# ================= CẤU HÌNH & KHỞI TẠO =================
app = FastAPI(title="Sign Language Recognition API")

# Cho phép gọi API từ mọi domain (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
ONNX_MODEL_PATH = MODELS_DIR / "static_gesture_mlp.onnx"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
INPUT_NAME = metadata.get("input_name", "landmarks")
OUTPUT_NAME = metadata.get("output_name", "probs")
DIGIT_COUNT = metadata.get("modes", {}).get("digit", {}).get("end", 10)

session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ================= SCHEMA DỮ LIỆU =================
class PredictRequest(BaseModel):
    image: str       # Chuỗi ảnh base64 gửi từ trình duyệt
    mode: str = "all"  # "all", "digit", "letter"

# ================= ENDPOINT API =================
@app.post("/predict")
def predict_gesture(req: PredictRequest):
    try:
        # 1. Giải mã chuỗi base64 thành ảnh OpenCV
        img_data = req.image.split(",")[-1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Không thể giải mã ảnh")

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return {"detected": False}

        hand_landmarks = results.multi_hand_landmarks[0]

        # 2. Chuẩn hóa đặc trưng theo hợp đồng 63 chiều
        feat = landmarks_to_features([hand_landmarks], image_width=w, image_height=h)
        feat = np.expand_dims(feat, axis=0)

        # 3. Dự đoán qua ONNX
        raw_probs = session.run([OUTPUT_NAME], {INPUT_NAME: feat})[0][0]

        # 4. Lọc theo mode
        if req.mode == "digit":
            sub_probs = raw_probs[:DIGIT_COUNT]
            idx = int(np.argmax(sub_probs))
            label = CLASSES[idx]
            conf = float(sub_probs[idx])
        elif req.mode == "letter":
            sub_probs = raw_probs[DIGIT_COUNT:]
            idx = int(DIGIT_COUNT + np.argmax(sub_probs))
            label = CLASSES[idx]
            conf = float(raw_probs[idx])
        else:
            idx = int(np.argmax(raw_probs))
            label = CLASSES[idx]
            conf = float(raw_probs[idx])

        # 5. Tọa độ Bounding box và Landmarks để Frontend vẽ lên canvas
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
            "confidence": round(conf, 4),
            "bbox": bbox,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================= PHỤC VỤ TRANG WEB TRỰC TIẾP =================
@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = Path("index.html")
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>Vui lòng tạo file index.html</h1>"