import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "manifests"

VALID_LABELS = tuple("0123456789") + tuple(
    ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if ch not in "JZ"
)
VALID_PARTICIPANTS = {f"P{i}" for i in range(1, 11)}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r"^(P\d+)_([A-Z0-9])_(\d+)$")

# Ngưỡng khởi đầu, phải hiệu chỉnh bằng ảnh thực tế của bộ dữ liệu.
DEFAULT_MIN_SIDE = 128
DEFAULT_BLUR_THRESHOLD = 20.0
DEFAULT_DARK_THRESHOLD = 45.0
DEFAULT_BRIGHT_THRESHOLD = 230.0
DEFAULT_CONTRAST_THRESHOLD = 15.0


def quality_metrics(img):
    """Các phép đo đơn giản, không khẳng định ảnh có bàn tay rõ hay nhãn đúng."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize trước khi đo blur để giảm ảnh hưởng kích thước ảnh.
    h, w = gray.shape
    target = 256
    scale = target / max(h, w)
    if scale < 1:
        gray_blur = cv2.resize(
            gray,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        gray_blur = gray
    return {
        "blur_score": round(float(cv2.Laplacian(gray_blur, cv2.CV_64F).var()), 3),
        "brightness_mean": round(float(gray.mean()), 3),
        "contrast_std": round(float(gray.std()), 3),
    }


def quality_warnings(
    img,
    min_side=DEFAULT_MIN_SIDE,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    dark_threshold=DEFAULT_DARK_THRESHOLD,
    bright_threshold=DEFAULT_BRIGHT_THRESHOLD,
    contrast_threshold=DEFAULT_CONTRAST_THRESHOLD,
):
    """Trả về (metrics, reason_codes). Cảnh báo không tự động loại ảnh."""
    if img is None:
        return {"blur_score": None, "brightness_mean": None, "contrast_std": None}, []
    metrics = quality_metrics(img)
    h, w = img.shape[:2]
    reasons = []
    if min(h, w) < min_side:
        reasons.append("small_image")
    if metrics["blur_score"] < blur_threshold:
        reasons.append("possibly_blurry")
    if metrics["brightness_mean"] < dark_threshold:
        reasons.append("too_dark")
    if metrics["brightness_mean"] > bright_threshold:
        reasons.append("too_bright")
    if metrics["contrast_std"] < contrast_threshold:
        reasons.append("low_contrast")
    return metrics, reasons


def file_sha256(path):
    """Hash nội dung file (bao gồm dữ liệu mã hóa, metadata ảnh)."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def pixel_sha256(img):
    """Hash pixel đã giải mã, có kèm shape, dtype và thứ tự kênh BGR."""
    sha = hashlib.sha256()
    sha.update(f"BGR|{img.dtype}|{img.shape}".encode("ascii"))
    sha.update(img.tobytes())
    return sha.hexdigest()


def read_image(path):
    """Đọc ảnh bằng OpenCV; np.fromfile hỗ trợ đường dẫn Unicode trên Windows."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def make_record(path, data_dir, index, quality_config=None):
    relative_path = path.relative_to(data_dir)
    parts = relative_path.parts
    label = parts[0] if len(parts) >= 2 else None

    filename_match = FILENAME_PATTERN.fullmatch(path.stem)
    participant = filename_match.group(1) if filename_match else None
    filename_label = filename_match.group(2) if filename_match else None

    reasons = []
    if len(parts) != 2:
        reasons.append("unexpected_folder_layout")
    if filename_match is None:
        reasons.append("invalid_filename")
    if participant not in VALID_PARTICIPANTS:
        reasons.append("invalid_participant")
    if filename_match is not None and filename_label != label:
        reasons.append("filename_label_mismatch")
    if label in {"J", "Z"}:
        reasons.append("out_of_scope")
    if label not in VALID_LABELS and label not in {"J", "Z"}:
        reasons.append("unknown_label")

    digest = None
    try:
        digest = file_sha256(path)
    except OSError:
        reasons.append("file_unreadable")

    img = read_image(path)
    if img is None:
        reasons.append("image_unreadable")
    metrics, quality_reasons = quality_warnings(img, **(quality_config or {}))
    reasons.extend(quality_reasons)

    # Prepare stage is limited to image validity, metadata, quality and duplicates.
    # Hand detection / landmark extraction belongs to extract_features.py.
    fatal = any(
        code in reasons
        for code in ("out_of_scope", "file_unreadable", "image_unreadable")
    )
    status = "rejected" if fatal else "review_needed" if reasons else "approved"

    return {
        "sample_id": f"RAW_{index:06d}",
        "source_version": "ASL-HG_Raw_v1",
        "relative_path": relative_path.as_posix(),
        "label": label,
        "participant_id": participant,
        "session_id": None,
        "width": int(img.shape[1]) if img is not None else None,
        "height": int(img.shape[0]) if img is not None else None,
        "file_sha256": digest,
        "pixel_sha256": pixel_sha256(img) if img is not None else None,
        "duplicate_group": None,
        "subject_group": None,
        "split": None,
        "review_status": status,
        "keep": status == "approved",
        "reason_codes": reasons,
        "reviewer": None,
        "preprocessing_version": "inventory_v1",
        "hand_count": None,
        "feature_status": "not_extracted",
        **metrics,
    }


def group_exact_duplicates(records):
    """Tìm nhóm trùng pixel; chỉ ghi nhóm có ít nhất 2 ảnh."""
    by_pixel = defaultdict(list)
    for record in records:
        if record["pixel_sha256"] is not None:
            by_pixel[record["pixel_sha256"]].append(record)

    groups = []
    for pixel_hash, members in sorted(by_pixel.items()):
        if len(members) < 2:
            continue

        group_id = f"EXACT_{pixel_hash[:16]}"
        labels = sorted({item["label"] for item in members})
        participants = sorted(
            {
                item["participant_id"]
                for item in members
                if item["participant_id"] is not None
            }
        )
        has_conflict = len(labels) > 1

        # Ưu tiên ảnh hợp lệ đầu tiên; thứ tự sample_id có tính xác định.
        approved = [item for item in members if item["review_status"] == "approved"]
        representative = (approved or members)[0] if not has_conflict else None

        for record in members:
            record["duplicate_group"] = group_id
            if has_conflict:
                record["reason_codes"].append("duplicate_exact_diff_label")
                if record["review_status"] != "rejected":
                    record["review_status"] = "review_needed"
                record["keep"] = False
            elif record is representative:
                record["reason_codes"].append("duplicate_representative")
            else:
                record["reason_codes"].append("duplicate_exact")
                record["review_status"] = "rejected"
                record["keep"] = False

        groups.append(
            {
                "duplicate_group": group_id,
                "type": "exact_pixel",
                "sample_ids": [item["sample_id"] for item in members],
                "relative_paths": [item["relative_path"] for item in members],
                "labels": labels,
                "participants": participants,
                "representative_sample_id": (
                    representative["sample_id"] if representative is not None else None
                ),
                "label_conflict": has_conflict,
            }
        )

    return groups


def assign_subject_groups(records, duplicate_groups):
    """Liên kết Participant thông qua các nhóm ảnh trùng, kể cả nối gián tiếp."""
    links = {participant: set() for participant in VALID_PARTICIPANTS}
    for group in duplicate_groups:
        participants = set(group["participants"]) & VALID_PARTICIPANTS
        for participant in participants:
            links[participant].update(participants - {participant})

    subject_groups = []
    participant_to_group = {}
    visited = set()

    for start in sorted(VALID_PARTICIPANTS, key=lambda p: int(p[1:])):
        if start in visited:
            continue
        component = set()
        stack = [start]
        while stack:
            participant = stack.pop()
            if participant in visited:
                continue
            visited.add(participant)
            component.add(participant)
            stack.extend(links[participant] - visited)

        members = sorted(component, key=lambda p: int(p[1:]))
        group_id = "SG_" + "_".join(members)
        subject_groups.append({"subject_group": group_id, "participants": members})
        for participant in members:
            participant_to_group[participant] = group_id

    for record in records:
        record["subject_group"] = participant_to_group.get(record["participant_id"])

    return subject_groups


def prepare_manifest(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    raw_zip=None,
    quality_config=None,
    overwrite_review_queue=False,
):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục Raw: {data_dir}")

    image_paths = sorted(
        (
            path
            for path in data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ),
        key=lambda p: p.relative_to(data_dir).as_posix(),
    )
    if not image_paths:
        raise ValueError(f"Không tìm thấy ảnh JPG/JPEG/PNG trong {data_dir}")

    records = [
        make_record(path, data_dir, i, quality_config)
        for i, path in enumerate(image_paths, start=1)
    ]

    duplicate_groups = group_exact_duplicates(records)
    subject_groups = assign_subject_groups(records, duplicate_groups)

    status_counts = Counter(record["review_status"] for record in records)
    warning_counts = Counter(
        code for record in records for code in record["reason_codes"]
    )
    review_records = [
        record for record in records if record["review_status"] == "review_needed"
    ]
    usable_records = [record for record in records if record["keep"]]

    label_counts = Counter(record["label"] for record in records)
    summary = {
        "raw_image_count": len(records),
        "in_scope_image_count": sum(
            record["label"] in VALID_LABELS for record in records
        ),
        "usable_image_count": sum(record["keep"] for record in records),
        "status_counts": dict(status_counts),
        "warning_counts": dict(sorted(warning_counts.items())),
        "review_queue_count": len(review_records),
        "label_counts": dict(sorted(label_counts.items())),
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_participant_groups": sum(
            len(group["participants"]) > 1 for group in duplicate_groups
        ),
        "conflicting_label_groups": sum(
            group["label_conflict"] for group in duplicate_groups
        ),
        "subject_groups": subject_groups,
        "note": "Thống kê tự động Task 2.1; chưa thay thế rà soát thủ công Task 2.2.",
    }
    if raw_zip is not None:
        summary["raw_zip_sha256"] = file_sha256(Path(raw_zip))

    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "review_queue.csv"
    if queue_path.is_file() and not overwrite_review_queue:
        with queue_path.open(encoding="utf-8-sig", newline="") as existing:
            has_manual_decisions = any(
                (row.get("review_decision") or "").strip()
                for row in csv.DictReader(existing)
            )
        if has_manual_decisions:
            raise FileExistsError(
                f"{queue_path} có quyết định review thủ công. "
                "Hãy lưu quyết định riêng trước khi chạy lại; "
                "chỉ dùng --overwrite-review-queue khi muốn tạo lại từ đầu."
            )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Review thủ công không tự khôi phục; chặn ghi đè nếu queue đã có quyết định.
    review_columns = [
        "sample_id",
        "relative_path",
        "label",
        "participant_id",
        "blur_score",
        "brightness_mean",
        "contrast_std",
        "review_status",
        "reason_codes",
        "review_decision",
        "review_note",
    ]
    with (output_dir / "review_queue.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=review_columns)
        writer.writeheader()
        for record in review_records:
            writer.writerow(
                {
                    key: (
                        {
                            **record,
                            "reason_codes": ";".join(record["reason_codes"]),
                            "review_decision": "",
                            "review_note": "",
                        }
                    ).get(key)
                    for key in review_columns
                }
            )
    # Chỉ các ảnh qua kiểm tra tự động; không phải dữ liệu đã được review hoàn tất.
    with (output_dir / "auto_approved_images.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "relative_path",
                "label",
                "participant_id",
                "subject_group",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {key: record[key] for key in writer.fieldnames} for record in usable_records
        )

    for filename, content in (
        ("duplicate_groups.json", duplicate_groups),
        ("inventory_summary.json", summary),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    print(f"Tổng ảnh Raw: {len(records)}")
    print(f"Thuộc 34 nhãn: {summary['in_scope_image_count']}")
    print(f"Ảnh có thể dùng trước review: {summary['usable_image_count']}")
    print(f"Cần review: {status_counts['review_needed']}")
    print(f"Bị loại theo quy tắc: {status_counts['rejected']}")
    print(f"Nhóm trùng pixel: {len(duplicate_groups)}")
    print(f"Nhóm trùng khác Participant: {summary['cross_participant_groups']}")
    print(f"Nhóm trùng khác nhãn: {summary['conflicting_label_groups']}")
    print(
        "Cảnh báo chất lượng:",
        {
            k: v
            for k, v in warning_counts.items()
            if k
            in {
                "small_image",
                "possibly_blurry",
                "too_dark",
                "too_bright",
                "low_contrast",
            }
        },
    )
    print(f"Đã xuất báo cáo vào: {output_dir}")
    return records, duplicate_groups, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2.1: manifest và exact duplicates"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE)
    parser.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument("--dark-threshold", type=float, default=DEFAULT_DARK_THRESHOLD)
    parser.add_argument(
        "--bright-threshold", type=float, default=DEFAULT_BRIGHT_THRESHOLD
    )
    parser.add_argument(
        "--contrast-threshold", type=float, default=DEFAULT_CONTRAST_THRESHOLD
    )
    parser.add_argument(
        "--raw-zip",
        type=Path,
        default=None,
        help="Đường dẫn ZIP gốc nếu muốn lưu SHA-256 ZIP",
    )
    parser.add_argument(
        "--overwrite-review-queue",
        action="store_true",
        help="Ghi đè review_queue.csv kể cả khi đã có quyết định thủ công. Không thể hoàn tác.",
    )
    args = parser.parse_args()
    config = {
        "min_side": args.min_side,
        "blur_threshold": args.blur_threshold,
        "dark_threshold": args.dark_threshold,
        "bright_threshold": args.bright_threshold,
        "contrast_threshold": args.contrast_threshold,
    }
    prepare_manifest(
        args.data_dir,
        args.output_dir,
        args.raw_zip,
        config,
        overwrite_review_queue=args.overwrite_review_queue,
    )
