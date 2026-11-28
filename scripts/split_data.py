"""Group-safe subject split. No image/feature extraction here."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = list("0123456789") + [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in "JZ"]
DEFAULT = {
    "train": set("P1 P2 P3 P4 P5 P6".split()),
    "val": {"P7", "P8"},
    "test": {"P9", "P10"},
}


def split_records(records, mapping=DEFAULT):
    all_ids = set().union(*mapping.values())
    if len(all_ids) != sum(len(x) for x in mapping.values()):
        raise ValueError("Participant xuất hiện trong nhiều tập")
    groups = defaultdict(set)
    for r in records:
        if r["participant_id"]:
            groups[r["subject_group"]].add(r["participant_id"])
    for group, members in groups.items():
        matching = [name for name, people in mapping.items() if members & people]
        if len(matching) > 1:
            raise ValueError(
                f"RÒ RỈ: {group} có {sorted(members)} thuộc {matching}. Phải sửa phân chia Participant trước!"
            )
        if not matching:
            raise ValueError(f"{group}: chưa được gán tập")
    result = []
    for r in records:
        if not r["keep"] or r["review_status"] != "approved":
            continue
        if (
            r["label"] not in LABELS
            or r["participant_id"] not in all_ids
            or not r["subject_group"]
        ):
            raise ValueError(f"Ảnh giữ nhưng metadata không hợp lệ: {r['sample_id']}")
        split = next(
            name for name, ids in mapping.items() if r["participant_id"] in ids
        )
        result.append({**r, "split": split})
    hashes = defaultdict(set)
    for r in result:
        hashes[r["pixel_sha256"]].add(r["split"])
    if any(len(v) > 1 for v in hashes.values()):
        raise ValueError("RÒ RỈ: cùng pixel hash ở nhiều tập")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest", type=Path, default=Path("manifests/manifest_final.jsonl")
    )
    p.add_argument(
        "--output", type=Path, default=Path("manifests/split_manifest.jsonl")
    )
    p.add_argument("--train", default="P1,P2,P3,P4,P5,P6")
    p.add_argument("--val", default="P7,P8")
    p.add_argument("--test", default="P9,P10")
    a = p.parse_args()
    mapping = {k: set(getattr(a, k).split(",")) for k in ("train", "val", "test")}
    records = [
        json.loads(s)
        for s in a.manifest.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    result = split_records(records, mapping)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Ảnh giữ sau review và chia tập:", dict(Counter(r["split"] for r in result)))
    for name in ("train", "val", "test"):
        counts = Counter(r["label"] for r in result if r["split"] == name)
        missing = [label for label in LABELS if counts[label] == 0]
        print(name, "số nhãn:", len(counts), "nhãn thiếu:", missing)
        if missing:
            raise ValueError(
                f"{name} thiếu nhãn {missing}. Cần kiểm tra dữ liệu và cách chia."
            )
    print("Đã ghi:", a.output)


if __name__ == "__main__":
    main()
