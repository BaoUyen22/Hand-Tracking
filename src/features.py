"""Shared 63D feature contract for training and webcam. One detected hand only.
Coordinates are adjusted for image aspect ratio before wrist/scale normalization.
"""

import numpy as np

FEATURE_DIM = 63
FEATURE_VERSION = "hand63_aspect_wrist09_v1"


def normalize_hand(landmarks, image_width, image_height):
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid image size")
    arr = np.asarray([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
    if arr.shape != (21, 3) or not np.isfinite(arr).all():
        raise ValueError("Invalid hand landmarks")
    # Convert x/y into common units (image width); z is MediaPipe's width-relative depth.
    arr[:, 1] *= float(image_height) / float(image_width)
    scale = float(np.linalg.norm(arr[9, :2] - arr[0, :2]))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("Invalid wrist-to-middle-MCP scale")
    arr = (arr - arr[0]) / scale
    result = arr.reshape(FEATURE_DIM).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite features")
    return result


def landmarks_to_features(multi_hand_landmarks, image_width, image_height):
    hands = list(multi_hand_landmarks or [])
    if len(hands) != 1:
        raise ValueError(f"63D yêu cầu đúng 1 bàn tay, thấy {len(hands)}")
    return normalize_hand(hands[0], image_width, image_height)
