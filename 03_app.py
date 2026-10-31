import cv2
import numpy as np
import onnxruntime as ort
from src.utils import normalize_hand_landmarks_84d

classes = np.load("data/processed/classes.npy")
session = ort.InferenceSession("models/static_gesture_mlp.onnx")
# ... tiếp tục vòng lặp webcam ...