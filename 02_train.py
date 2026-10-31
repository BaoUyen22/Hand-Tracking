from src.model import build_mlp_static
import numpy as np
import os
import tf2onnx

# 1. Đọc dữ liệu
X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")
classes = np.load("data/processed/classes.npy")

# ... (huấn luyện xong) ...

# 2. Xuất ONNX vào thư mục models/
os.makedirs("models", exist_ok=True)
output_path = "models/static_gesture_mlp.onnx"
tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)