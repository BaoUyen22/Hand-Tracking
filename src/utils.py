# src/utils.py
import numpy as np

def normalize_hand_landmarks_84d(hand_landmarks):
    """
    Trích xuất và chuẩn hóa 21 landmark thành vector 84-d
    Công thức: x' = (x - x_wrist) / palm_scale
    """
    raw_coords = []
    for lm in hand_landmarks.landmark:
        # Lấy đủ 4 thông số: x, y, z và visibility
        vis = getattr(lm, 'visibility', 1.0)
        raw_coords.append([lm.x, lm.y, lm.z, vis])
    
    coords = np.array(raw_coords) # shape: (21, 4)

    # 1. Wrist-anchor: Điểm số 0 (cổ tay) làm gốc
    wrist_xyz = coords[0, :3].copy()
    coords[:, :3] = coords[:, :3] - wrist_xyz

    # 2. Palm-scale: Khoảng cách từ cổ tay (điểm 0) đến gốc ngón giữa (điểm 9)
    # Lúc này điểm 0 đã là [0, 0, 0] nên khoảng cách chính là độ lớn vector điểm 9
    palm_scale = np.linalg.norm(coords[9, :3])
    if palm_scale < 1e-6:
        palm_scale = 1.0

    # Chuẩn hóa khoảng cách
    coords[:, :3] = coords[:, :3] / palm_scale

    # Trả về vector 1D phẳng 84 phần tử
    return coords.flatten()