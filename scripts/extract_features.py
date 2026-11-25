"""Create X/y/sample_ids for B, using MediaPipe Hands (legacy solutions API).
Run from repo root after review and group-safe split.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import landmarks_to_features, FEATURE_VERSION
from scripts.split_data import LABELS


def read_image(path):
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def run(manifest, raw_dir, output, min_detection_confidence=0.5):
    dimensions = 63
    try:
        import mediapipe as mp
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài mediapipe. Chạy pip install -r requirements-data.txt"
        ) from e
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
        raise RuntimeError(
            "MediaPipe bản cài không hỗ trợ mp.solutions.hands. Dùng phiên bản tương thích trong requirements-data.txt hoặc chuyển sang Tasks API."
        )
    rows = [
        json.loads(s)
        for s in Path(manifest).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if not rows:
        raise ValueError("Split manifest rỗng")
    if any(r["label"] == "0" for r in rows):
        print(
            "CẢNH BÁO: 63D chỉ nhận một tay; nếu ký hiệu 0 trong Raw dùng hai tay, các mẫu này sẽ bị bỏ qua. Cần kiểm chứng nhãn 0 trước khi bàn giao."
        )
    X = {s: [] for s in ("train", "val", "test")}
    y = {s: [] for s in X}
    ids = {s: [] for s in X}
    failures = []
    label_to_idx = {v: i for i, v in enumerate(LABELS)}
    with mp.solutions.hands.Hands(  # pyright: ignore[reportAttributeAccessIssue]
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
    ) as detector:
        for row in rows:
            split = row["split"]
            if split not in X or row["label"] not in label_to_idx:
                raise ValueError(f"Split/label không hợp lệ: {row['sample_id']}")
            image = read_image(Path(raw_dir) / row["relative_path"])
            if image is None:
                failures.append((row, "image_unreadable"))
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            hands = result.multi_hand_landmarks or []
            if len(hands) == 0:
                failures.append((row, "no_hand_detected"))
                continue
            try:
                feature = landmarks_to_features(hands, image.shape[1], image.shape[0])
            except ValueError as exc:
                failures.append((row, str(exc)))
                continue
            X[split].append(feature)
            y[split].append(label_to_idx[row["label"]])
            ids[split].append(row["sample_id"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "feature_dimensions": dimensions,
        "label_order": LABELS,
        "normalization": "aspect_corrected_xy_then_wrist_origin_and_2d_distance_landmarks_0_9",
        "feature_version": FEATURE_VERSION,
        "split_counts": {},
        "failures": len(failures),
        "min_detection_confidence": min_detection_confidence,
    }
    for split in X:
        features = (
            np.stack(X[split]).astype(np.float32)
            if X[split]
            else np.empty((0, dimensions), dtype=np.float32)
        )
        labels = np.asarray(y[split], dtype=np.int64)
        sample_ids = np.asarray(ids[split], dtype=str)
        np.save(output / f"X_{split}.npy", features)
        np.save(output / f"y_{split}.npy", labels)
        np.save(output / f"sample_ids_{split}.npy", sample_ids)
        missing = [
            label for label in LABELS if label_to_idx[label] not in set(labels.tolist())
        ]
        summary["split_counts"][split] = {
            "samples": len(labels),
            "missing_labels": missing,
            "per_label": dict(Counter(LABELS[i] for i in labels)),
        }
    (output / "classes.json").write_text(
        json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "extraction_failures.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample_id", "relative_path", "label", "participant_id", "split", "reason"]
        )
        for row, reason in failures:
            writer.writerow(
                [
                    row["sample_id"],
                    row["relative_path"],
                    row["label"],
                    row["participant_id"],
                    row["split"],
                    reason,
                ]
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if any(
        v["samples"] == 0 or v["missing_labels"]
        for v in summary["split_counts"].values()
    ):
        raise RuntimeError(
            "Trích xuất xong nhưng có tập rỗng/thiếu nhãn. KHÔNG bàn giao B trước khi kiểm tra metadata.json và extraction_failures.csv."
        )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest", type=Path, default=Path("manifests/split_manifest.jsonl")
    )
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    a = p.parse_args()
    run(a.manifest, a.raw_dir, a.output, a.min_detection_confidence)


if __name__ == "__main__":
    main()
