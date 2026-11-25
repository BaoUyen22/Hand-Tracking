"""Validate B handoff files and no participant/pixel leakage."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def validate(folder, split_manifest):
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["feature_dimensions"] == 63, "Hợp đồng yêu cầu 63D"
    classes = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
    assert (
        classes == meta["label_order"] and len(classes) == 34
    ), "Sai thứ tự hoặc số nhãn"
    rows = [
        json.loads(s)
        for s in Path(split_manifest).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    by_id = {r["sample_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("sample_id trùng trong split manifest")
    seen_ids = set()
    participant_splits, subject_splits, pixel_splits = (
        defaultdict(set),
        defaultdict(set),
        defaultdict(set),
    )
    for split in ("train", "val", "test"):
        X = np.load(folder / f"X_{split}.npy", allow_pickle=False)
        y = np.load(folder / f"y_{split}.npy", allow_pickle=False)
        ids = np.load(folder / f"sample_ids_{split}.npy", allow_pickle=False)
        assert (
            X.ndim == 2 and X.shape[1] == meta["feature_dimensions"]
        ), f"X_{split} sai shape"
        assert (
            len(X) == len(y) == len(ids) and len(X) > 0
        ), f"{split} thiếu/mismatch dữ liệu"
        assert X.dtype == np.float32 and np.issubdtype(
            y.dtype, np.integer
        ), f"{split} sai dtype"
        assert (
            np.isfinite(X).all() and ((y >= 0) & (y < len(classes))).all()
        ), f"{split} dữ liệu không hợp lệ"
        assert set(y.tolist()) == set(range(34)), f"{split} chưa đủ 34 nhãn"
        for sid, target in zip(ids.tolist(), y.tolist()):
            if sid in seen_ids or sid not in by_id:
                raise ValueError(f"sample_id lặp/không tồn tại: {sid}")
            seen_ids.add(sid)
            r = by_id[sid]
            if r["split"] != split or classes[target] != r["label"]:
                raise ValueError(f"Không khớp nhãn/split: {sid}")
            participant_splits[r["participant_id"]].add(split)
            subject_splits[r["subject_group"]].add(split)
            pixel_splits[r["pixel_sha256"]].add(split)
        print(f"{split}: X={X.shape}, y={y.shape}, sample_ids={ids.shape}")
    for name, groups in [
        ("participant", participant_splits),
        ("subject_group", subject_splits),
        ("pixel_sha256", pixel_splits),
    ]:
        leaks = [key for key, values in groups.items() if len(values) > 1]
        if leaks:
            raise ValueError(f"Data leakage theo {name}: {leaks[:5]}")
    print(
        "PASS: 34 nhãn, shape/dtype, alignment và không phát hiện leakage theo metadata."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, default=Path("data/processed"))
    p.add_argument(
        "--manifest", type=Path, default=Path("manifests/split_manifest.jsonl")
    )
    a = p.parse_args()
    validate(a.folder, a.manifest)
