"""Apply manual review decisions to manifest.

Task 2.2 - Cleaning & Human Review.
Does not modify Raw, original manifest or review CSV.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HARD_FAILURES = {
    "out_of_scope",
    "unknown_label",
    "invalid_filename",
    "invalid_participant",
    "unexpected_folder_layout",
    "filename_label_mismatch",
    "file_unreadable",
    "image_unreadable",
    "duplicate_exact",
    "duplicate_exact_diff_label",
}


def load_manifest(path):
    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    ids = [record["sample_id"] for record in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Manifest có sample_id bị trùng")

    return records


def load_reviews(path):
    decisions = {}

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "sample_id",
            "review_decision",
            "review_note",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Review CSV thiếu cột bắt buộc")

        for row in reader:
            sample_id = row["sample_id"].strip()
            decision = row["review_decision"].strip().lower()
            note = row["review_note"].strip()

            if not decision:
                continue

            if decision not in {"keep", "reject"}:
                raise ValueError(f"{sample_id}: chỉ dùng keep/reject")

            if sample_id in decisions:
                raise ValueError(f"Trùng quyết định: {sample_id}")

            decisions[sample_id] = {
                "decision": decision,
                "note": note,
            }

    return decisions


def check_duplicates(records):
    seen = {}

    for record in records:
        if not record["keep"]:
            continue

        pixel_hash = record.get("pixel_sha256")

        if not pixel_hash:
            raise ValueError(f'{record["sample_id"]}: thiếu pixel_sha256')

        if pixel_hash in seen:
            raise ValueError(
                "Duplicate còn được giữ: "
                f'{seen[pixel_hash]} và {record["sample_id"]}'
            )

        seen[pixel_hash] = record["sample_id"]


def status_counts(records):
    return dict(Counter(record["review_status"] for record in records))


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/manifest.jsonl"),
    )

    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("manifests/review_queue.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests"),
    )

    args = parser.parse_args()

    records = load_manifest(args.manifest)
    decisions = load_reviews(args.reviews)

    valid_ids = {record["sample_id"] for record in records}

    unknown_ids = set(decisions) - valid_ids

    if unknown_ids:
        raise ValueError(f"Review chứa ID không hợp lệ: {sorted(unknown_ids)[:5]}")

    before_counts = status_counts(records)

    decision_log = []

    for record in records:
        sample_id = record["sample_id"]

        if sample_id not in decisions:
            continue

        if record["review_status"] != "review_needed":
            raise ValueError(f"{sample_id}: ảnh không ở trạng thái " "review_needed")

        review = decisions[sample_id]
        decision = review["decision"]
        reason_codes = set(record.get("reason_codes", []))

        if decision == "keep" and reason_codes & HARD_FAILURES:
            raise ValueError(
                f"{sample_id}: không thể giữ ảnh có " "lỗi metadata hoặc duplicate"
            )

        record["keep"] = decision == "keep"

        record["review_status"] = "approved" if decision == "keep" else "rejected"

        record["reviewer"] = "manual_review"
        record["review_note"] = review["note"]

        decision_log.append(
            {
                "sample_id": sample_id,
                "decision": decision,
                "reviewer": "manual_review",
                "review_note": review["note"],
                "reason_codes": record.get("reason_codes", []),
            }
        )

    # Kiểm tra toàn bộ trước khi ghi output.
    check_duplicates(records)

    after_counts = status_counts(records)

    reason_counts = Counter()

    for record in records:
        if record["review_status"] == "rejected":
            for reason in record.get("reason_codes", []):
                reason_counts[reason] += 1

    keep_count = sum(record["keep"] for record in records)

    pending_count = sum(
        record["review_status"] == "review_needed" for record in records
    )

    manual_keep = sum(item["decision"] == "keep" for item in decision_log)

    manual_reject = sum(item["decision"] == "reject" for item in decision_log)

    summary = {
        "task": "2.2 - Cleaning & Human Review",
        "source_manifest": str(args.manifest),
        "source_reviews": str(args.reviews),
        "total_images": len(records),
        "before_status_counts": before_counts,
        "after_status_counts": after_counts,
        "manual_decisions": {
            "total": len(decision_log),
            "keep": manual_keep,
            "reject": manual_reject,
        },
        "usable_image_count": keep_count,
        "pending_review_count": pending_count,
        "rejected_reason_counts": dict(reason_counts),
        "review_complete": pending_count == 0,
        "policy": {
            "quality_warnings_auto_reject": False,
            "hard_failures_cannot_keep": True,
            "unreviewed_images_auto_approved": False,
            "raw_modified": False,
        },
        "note": (
            "Ảnh còn review_needed chưa được duyệt. "
            "Không sử dụng chúng như ảnh approved "
            "khi tạo tập dữ liệu cuối cùng."
        ),
    }

    output_dir = args.output_dir

    output_paths = [
        output_dir / "manifest_final.jsonl",
        output_dir / "decisions.jsonl",
        output_dir / "cleaning_summary.json",
    ]

    input_paths = [
        args.manifest,
        args.reviews,
    ]

    for output in output_paths:
        if any(output.resolve() == source.resolve() for source in input_paths):
            raise ValueError("Output không được ghi đè input")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_jsonl(output_paths[0], records)
    save_jsonl(output_paths[1], decision_log)

    with output_paths[2].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=== TASK 2.2 - CLEANING SUMMARY ===")
    print(f"Tổng ảnh: {len(records)}")
    print(f"Quyết định thủ công: {len(decision_log)}")
    print(f"Giữ thủ công: {manual_keep}")
    print(f"Loại thủ công: {manual_reject}")
    print(f"Tổng ảnh được giữ: {keep_count}")
    print(f"Còn chờ review: {pending_count}")
    print(f"Review hoàn tất: {pending_count == 0}")
    print(f"Đã xuất kết quả vào: {output_dir}")


if __name__ == "__main__":
    main()
