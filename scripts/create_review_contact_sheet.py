"""Task 2.2: stratified contact sheets and optional MediaPipe audit.

Contact-sheet mode reads review_queue.csv and writes sample sheets / sample CSV.
Audit mode reads manifest_final.jsonl and writes detection-only audit reports.
Neither mode modifies Raw, manifest files, or review_queue.csv.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

GROUPS = [
    "blurry_low",
    "blurry_mid",
    "blurry_near_threshold",
    "low_contrast",
    "too_bright",
    "duplicate",
]
QUALITY_WARNINGS = {
    "possibly_blurry",
    "low_contrast",
    "too_bright",
    "too_dark",
    "small_image",
}
HARD_REASONS = {
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


def get_reasons(row):
    value = row.get("reason_codes", [])
    if isinstance(value, list):
        return set(value)
    return {part.strip() for part in str(value).split(";") if part.strip()}


def belongs_to_group(row, group):
    reasons = get_reasons(row)
    if group.startswith("blurry"):
        if "possibly_blurry" not in reasons:
            return False
        try:
            blur = float(row["blur_score"])
        except (ValueError, TypeError):
            return False
        if group == "blurry_low":
            return blur < 50
        if group == "blurry_mid":
            return 50 <= blur < 65
        return 65 <= blur < 80
    if group == "duplicate":
        return bool(reasons & {"duplicate_exact", "duplicate_representative"})
    return group in reasons


def choose_samples(candidates, count, rng):
    """Sample across participant and label buckets."""
    buckets = defaultdict(list)
    for row in candidates:
        buckets[(row["participant_id"], row["label"])].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    selected = []
    while len(selected) < count:
        found = False
        for key in keys:
            if not buckets[key]:
                continue
            found = True
            selected.append(buckets[key].pop())
            if len(selected) >= count:
                break
        if not found:
            break
    return selected


def read_image(path):
    """Unicode-safe OpenCV reading on Windows."""
    try:
        content = np.fromfile(str(path), dtype=np.uint8)
        if content.size == 0:
            return None
        return cv2.imdecode(content, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def write_jpg(path, img):
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise OSError(f"Không thể mã hóa ảnh: {path}")
    encoded.tofile(str(path))


def create_sheet(batch, raw_dir, group):
    tile_width, tile_height, columns, rows, header_height = 210, 255, 5, 5, 40
    sheet = np.full(
        (header_height + rows * tile_height, columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(sheet, group, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for index, row in enumerate(batch):
        image_path = raw_dir / row["relative_path"]
        image = read_image(image_path)
        x = (index % columns) * tile_width + 10
        y = header_height + (index // columns) * tile_height + 10
        if image is not None:
            image = cv2.resize(image, (190, 190))
            sheet[y : y + 190, x : x + 190] = image
        else:
            cv2.putText(
                sheet,
                "UNREADABLE",
                (x, y + 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        lines = [
            row["sample_id"],
            f'{row["label"]} / {row["participant_id"]}',
            f'Blur: {row["blur_score"]}',
        ]
        for line_index, value in enumerate(lines):
            cv2.putText(
                sheet,
                value,
                (x, y + 207 + line_index * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 0),
                1,
            )
    return sheet


def create_contact_sheets(args):
    rng = random.Random(42)
    with args.reviews.open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_samples = []
    for group in GROUPS:
        candidates = [row for row in records if belongs_to_group(row, group)]
        selected = choose_samples(candidates, args.per_group, rng)
        group_dir = args.output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(selected), 25):
            batch = selected[start : start + 25]
            sheet = create_sheet(batch, args.raw_dir, group)
            output_path = group_dir / f"sheet_{start // 25 + 1:02d}.jpg"
            write_jpg(output_path, sheet)
        for row in selected:
            review_samples.append(
                {
                    "review_group": group,
                    "sample_id": row["sample_id"],
                    "relative_path": row["relative_path"],
                    "label": row["label"],
                    "participant_id": row["participant_id"],
                    "blur_score": row["blur_score"],
                    "brightness_mean": row["brightness_mean"],
                    "contrast_std": row["contrast_std"],
                    "reason_codes": row["reason_codes"],
                    "review_decision": "",
                    "review_note": "",
                }
            )
        print(f"{group}: chọn {len(selected)}/{len(candidates)} ảnh")
    output_csv = args.output_dir / "review_sample.csv"
    fieldnames = [
        "review_group",
        "sample_id",
        "relative_path",
        "label",
        "participant_id",
        "blur_score",
        "brightness_mean",
        "contrast_std",
        "reason_codes",
        "review_decision",
        "review_note",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_samples)
    print(f"Đã tạo {len(review_samples)} dòng review. CSV: {output_csv}")
    print("Raw, manifest và review_queue.csv không thay đổi.")
    print("Một sample_id có thể xuất hiện trong nhiều nhóm cảnh báo.")
    print(
        "Chú ý: với blur threshold=50, blurry_mid và blurry_near_threshold thường rỗng."
    )


def load_pending_records(manifest_path):
    pending = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("review_status") != "review_needed":
                continue
            sid = record.get("sample_id")
            if not sid or sid in seen:
                raise ValueError(f"Thiếu/trùng sample_id tại dòng {line_no}: {sid}")
            seen.add(sid)
            if record.get("keep"):
                raise ValueError(
                    f"{sid}: review_needed nhưng keep=true, cần kiểm tra manifest"
                )
            pending.append(record)
    return pending


def audit_pending_images(args):
    """Detection-only audit; never writes review decisions or labels."""
    try:
        import mediapipe as mp

        hands_api = mp.solutions.hands  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Audit cần MediaPipe phiên bản hỗ trợ mp.solutions.hands trong .venv311. "
            "Kiểm tra cài đặt MediaPipe."
        ) from exc

    pending = load_pending_records(args.manifest)
    if not pending:
        print("Không còn ảnh review_needed trong manifest_final.jsonl.")
        return

    # Do not overwrite existing audit reports accidentally.
    outdir = args.audit_output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    output_csv = outdir / "detection_results.csv"
    summary_json = outdir / "audit_summary.json"
    group_csv = outdir / "quality_group_summary.csv"
    priority_csv = outdir / "review_priority.csv"
    outputs = [output_csv, summary_json, group_csv, priority_csv]
    existing = [str(p) for p in outputs if p.exists()]
    if existing and not args.overwrite_audit:
        raise FileExistsError(
            "Báo cáo audit đã tồn tại, không tự ghi đè: "
            + ", ".join(existing)
            + ". Nếu muốn chạy lại, thêm --overwrite-audit."
        )

    fieldnames = [
        "sample_id",
        "relative_path",
        "label",
        "participant_id",
        "subject_group",
        "reason_codes",
        "blur_score",
        "brightness_mean",
        "contrast_std",
        "detected_hands",
        "detection_status",
        "warning_count",
        "priority",
        "priority_reason",
    ]
    summary_count = Counter()
    by_label = defaultdict(Counter)
    by_participant = defaultdict(Counter)
    by_warning = defaultdict(Counter)
    all_rows = []

    with hands_api.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
    ) as detector:
        for index, record in enumerate(pending, start=1):
            reasons = get_reasons(record)
            image = read_image(args.raw_dir / record["relative_path"])
            if image is None:
                hand_count = 0
                status = "unreadable"
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                hand_count = len(result.multi_hand_landmarks or [])
                status = "detected" if hand_count > 0 else "no_hand"

            quality_count = len(reasons & QUALITY_WARNINGS)
            hard = bool(reasons & HARD_REASONS)
            if hard:
                priority, priority_reason = 1, "hard_reason_in_pending"
            elif status != "detected":
                priority, priority_reason = 1, status
            elif quality_count >= 2:
                priority, priority_reason = 2, "multiple_quality_warnings"
            else:
                priority, priority_reason = 3, "detected_needs_visual_review"
            row = {
                "sample_id": record["sample_id"],
                "relative_path": record["relative_path"],
                "label": record.get("label"),
                "participant_id": record.get("participant_id"),
                "subject_group": record.get("subject_group"),
                "reason_codes": ";".join(sorted(reasons)),
                "blur_score": record.get("blur_score"),
                "brightness_mean": record.get("brightness_mean"),
                "contrast_std": record.get("contrast_std"),
                "detected_hands": hand_count,
                "detection_status": status,
                "warning_count": quality_count,
                "priority": priority,
                "priority_reason": priority_reason,
            }
            all_rows.append(row)
            summary_count[status] += 1
            by_label[str(record.get("label"))][status] += 1
            by_participant[str(record.get("participant_id"))][status] += 1
            for warning in sorted(reasons & QUALITY_WARNINGS):
                by_warning[warning][status] += 1
            if index % 500 == 0 or index == len(pending):
                print(
                    f"Audit MediaPipe: {index}/{len(pending)} | phát hiện: {summary_count['detected']} | không thấy tay: {summary_count['no_hand']} | không đọc được: {summary_count['unreadable']}",
                    flush=True,
                )

    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    priority_fields = fieldnames + ["review_decision", "review_note"]
    # Priority first; stable ordering for repeatable review.
    with priority_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=priority_fields)
        writer.writeheader()
        for row in sorted(
            all_rows,
            key=lambda r: (
                r["priority"],
                r["label"] or "",
                r["participant_id"] or "",
                r["sample_id"],
            ),
        ):
            writer.writerow({**row, "review_decision": "", "review_note": ""})

    def summary_group_rows(by_category, category_name):
        for value, counts in sorted(by_category.items()):
            total = sum(counts.values())
            yield {
                "category": category_name,
                "value": value,
                "total": total,
                "detected": counts["detected"],
                "no_hand": counts["no_hand"],
                "unreadable": counts["unreadable"],
                "detection_rate_pct": (
                    round(100 * counts["detected"] / total, 2) if total else 0
                ),
            }

    group_fields = [
        "category",
        "value",
        "total",
        "detected",
        "no_hand",
        "unreadable",
        "detection_rate_pct",
    ]
    with group_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=group_fields)
        writer.writeheader()
        writer.writerows(summary_group_rows(by_warning, "quality_warning"))
        writer.writerows(summary_group_rows(by_label, "label"))
        writer.writerows(summary_group_rows(by_participant, "participant"))

    total = len(all_rows)
    summary = {
        "task": "Task 2.2 - Pending MediaPipe detection audit",
        "source_manifest": str(args.manifest),
        "pending_audited": total,
        "detected": summary_count["detected"],
        "no_hand": summary_count["no_hand"],
        "unreadable": summary_count["unreadable"],
        "detection_rate_pct": round(summary_count["detected"] * 100 / total, 2),
        "priority_counts": dict(
            sorted(Counter(row["priority"] for row in all_rows).items())
        ),
        "settings": {
            "max_hands": args.max_hands,
            "min_detection_confidence": args.min_detection_confidence,
            "static_image_mode": True,
        },
        "policy": {
            "auto_approve": False,
            "auto_reject": False,
            "raw_modified": False,
            "review_queue_modified": False,
            "manifest_modified": False,
        },
        "important_note": (
            "Detection of 21 landmarks is NOT a guarantee of correct landmark geometry, "
            "correct gesture label, or safe inclusion in the final 63D dataset. "
            "Quality groups can overlap. Review decisions remain manual or require an "
            "independently validated cleaning policy."
        ),
    }
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Đã lưu audit tại: {outdir}")
    print(
        "Chỉ xuất báo cáo. Không thay đổi Raw, manifest hay review_queue.csv; không tự keep/reject."
    )


def create_priority_sheets(args):
    """Visual-review sheets for audit priority 1/2; never applies decisions."""
    priority_path = args.audit_output_dir / "review_priority.csv"
    if not priority_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {priority_path}. Hãy chạy --audit trước."
        )

    with priority_path.open(encoding="utf-8-sig", newline="") as file:
        audit_rows = list(csv.DictReader(file))
    if not audit_rows or "priority" not in audit_rows[0]:
        raise ValueError("review_priority.csv không có dữ liệu hoặc thiếu cột priority")

    # Audit is a snapshot. Never present already resolved samples as pending.
    current_pending = {r["sample_id"]: r for r in load_pending_records(args.manifest)}
    selected = []
    skipped_resolved = 0
    seen = set()
    for row in audit_rows:
        try:
            priority = int(row["priority"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Priority không hợp lệ: {row.get('sample_id')}") from exc
        if priority not in (1, 2):
            continue
        sid = row["sample_id"]
        if sid in seen:
            raise ValueError(f"Trùng sample_id trong review_priority.csv: {sid}")
        seen.add(sid)
        current = current_pending.get(sid)
        if current is None:
            skipped_resolved += 1
            continue
        if row["relative_path"].replace("\\", "/") != current["relative_path"]:
            raise ValueError(f"Dữ liệu audit không khớp manifest: {sid}")
        selected.append((priority, row))

    if not selected:
        print("Không còn ảnh Priority 1/2 ở trạng thái review_needed.")
        return

    try:
        import mediapipe as mp

        hands_api = mp.solutions.hands  # type: ignore
        draw_api = mp.solutions.drawing_utils  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cần MediaPipe có mp.solutions.hands và drawing_utils"
        ) from exc

    outdir = args.priority_output_dir
    output_csv = outdir / "priority_review_sample.csv"
    if output_csv.exists() and not args.overwrite_priority_sheets:
        raise FileExistsError(
            f"{output_csv} đã có. Hãy giữ quyết định cũ hoặc dùng "
            "--overwrite-priority-sheets nếu muốn tạo lại."
        )
    outdir.mkdir(parents=True, exist_ok=True)

    sheet_rows = []
    # 3 pairs per row x 4 rows = 12 images per sheet.
    columns, rows_per_sheet = 3, 4
    tile_width, tile_height, header_height = 470, 300, 45
    image_size = 212

    def render_tile(sheet, index, row, detector):
        x = (index % columns) * tile_width
        y = header_height + (index // columns) * tile_height
        image = read_image(args.raw_dir / row["relative_path"])
        if image is None:
            cv2.putText(
                sheet,
                "UNREADABLE",
                (x + 25, y + 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 200),
                2,
            )
            detected = "unreadable"
        else:
            overlay = image.copy()
            result = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hands = result.multi_hand_landmarks or []
            for hand in hands:
                draw_api.draw_landmarks(overlay, hand, hands_api.HAND_CONNECTIONS)
            detected = f"{len(hands)} hand(s)"
            a = cv2.resize(image, (image_size, image_size))
            b = cv2.resize(overlay, (image_size, image_size))
            sheet[y : y + image_size, x + 5 : x + 5 + image_size] = a
            sheet[y : y + image_size, x + 231 : x + 231 + image_size] = b

        def put(text, y_text, font_scale=0.42):
            cv2.putText(
                sheet,
                str(text),
                (x + 8, y + y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                1,
            )

        put(
            f"{row['sample_id']} | {row['label']}/{row['participant_id']} | {detected}",
            234,
        )
        put(f"Blur={row.get('blur_score', '')} | {row.get('priority_reason', '')}", 254)
        put("LEFT: original     RIGHT: MediaPipe", 276, 0.39)
        return detected

    with hands_api.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
    ) as detector:
        for priority in (1, 2):
            group = [row for p, row in selected if p == priority]
            if not group:
                continue
            group_dir = outdir / f"priority_{priority}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(group), columns * rows_per_sheet):
                batch = group[start : start + columns * rows_per_sheet]
                sheet = np.full(
                    (
                        header_height + rows_per_sheet * tile_height,
                        columns * tile_width,
                        3,
                    ),
                    255,
                    dtype=np.uint8,
                )
                cv2.putText(
                    sheet,
                    f"PRIORITY {priority} | Page {start // 12 + 1}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
                for i, row in enumerate(batch):
                    new_detection = render_tile(sheet, i, row, detector)
                    sheet_rows.append(
                        {
                            "priority": priority,
                            "sample_id": row["sample_id"],
                            "relative_path": row["relative_path"],
                            "label": row["label"],
                            "participant_id": row["participant_id"],
                            "reason_codes": row["reason_codes"],
                            "audit_detection_status": row.get("detection_status", ""),
                            "overlay_detection": new_detection,
                            "review_decision": row.get("review_decision", ""),
                            "review_note": row.get("review_note", ""),
                        }
                    )
                sheet_path = group_dir / f"sheet_{start // 12 + 1:02d}.jpg"
                write_jpg(sheet_path, sheet)
            print(
                f"Priority {priority}: tạo ảnh cho {len(group)} mẫu, "
                f"{(len(group) + 11) // 12} contact sheet"
            )

    fields = [
        "priority",
        "sample_id",
        "relative_path",
        "label",
        "participant_id",
        "reason_codes",
        "audit_detection_status",
        "overlay_detection",
        "review_decision",
        "review_note",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sheet_rows)
    print(f"Đã xuất {len(sheet_rows)} ảnh ưu tiên vào {outdir}")
    if skipped_resolved:
        print(
            f"Bỏ qua {skipped_resolved} ảnh đã hết trạng thái review_needed từ lúc audit."
        )
    print("Không thay đổi Raw, manifest, review_queue.csv hay quyết định keep/reject.")
    print("Lưu ý: overlay chỉ minh họa landmarks 2D; không xác thực đúng ký hiệu.")


# --- Task 2.2: Landmark-assisted contact sheets, including threshold-border samples ---
THRESHOLD_RANGES = {
    "blur_0_20": ("blur_score", 0, 20),
    "contrast_0_15": ("contrast_std", 0, 15),
}


def create_paired_sheet(batch, raw_dir, group, detector, draw_api, hands_api):
    """Render original and MediaPipe overlays side-by-side; detect at full resolution."""
    columns, rows = 3, 4
    tile_width, tile_height, header_height = 470, 300, 44
    image_size = 212
    sheet = np.full(
        (header_height + rows * tile_height, columns * tile_width, 3),
        255,
        dtype=np.uint8,
    )
    cv2.putText(
        sheet, group[:65], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2
    )
    result_rows = []
    for i, row in enumerate(batch):
        x = (i % columns) * tile_width
        y = header_height + (i // columns) * tile_height
        original = read_image(raw_dir / row["relative_path"])
        if original is None:
            status, n_hands = "unreadable", 0
            cv2.putText(
                sheet,
                "UNREADABLE",
                (x + 8, y + 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 190),
                2,
            )
        else:
            rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            hands = result.multi_hand_landmarks or []
            n_hands = len(hands)
            status = "detected" if hands else "no_hand"
            overlay = original.copy()
            for landmarks in hands:
                draw_api.draw_landmarks(overlay, landmarks, hands_api.HAND_CONNECTIONS)
            a = cv2.resize(
                original, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            b = cv2.resize(
                overlay, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            sheet[y : y + image_size, x + 5 : x + 5 + image_size] = a
            sheet[y : y + image_size, x + 231 : x + 231 + image_size] = b
        info = [
            f"{row['sample_id']} | {row['label']}/{row['participant_id']} | hands={n_hands}",
            f"blur={row.get('blur_score', '')} | contrast={row.get('contrast_std', '')}",
            f"brightness={row.get('brightness_mean', '')} | {status}",
            "LEFT: original     RIGHT: landmarks (2D)",
        ]
        for j, line in enumerate(info):
            cv2.putText(
                sheet,
                line,
                (x + 8, y + 231 + 17 * j),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (0, 0, 0),
                1,
            )
        result_rows.append(
            {
                "review_group": group,
                "sample_id": row["sample_id"],
                "relative_path": row["relative_path"],
                "label": row["label"],
                "participant_id": row["participant_id"],
                "blur_score": row.get("blur_score", ""),
                "contrast_std": row.get("contrast_std", ""),
                "brightness_mean": row.get("brightness_mean", ""),
                "detected_hands": n_hands,
                "detection_status": status,
                "review_decision": "",
                "review_note": "",
            }
        )
    return sheet, result_rows


def create_threshold_landmark_sheets(args):
    """Sample near proposed thresholds and display original vs landmark overlay."""
    try:
        import mediapipe as mp

        hands_api = mp.solutions.hands  # type: ignore
        draw_api = mp.solutions.drawing_utils  # type: ignore
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cần MediaPipe hỗ trợ mp.solutions.hands/drawing_utils"
        ) from exc

    pending = load_pending_records(args.manifest)
    output_dir = args.threshold_output_dir
    sample_csv = output_dir / "threshold_landmark_sample.csv"
    if sample_csv.exists() and not args.overwrite_threshold_sheets:
        raise FileExistsError(
            f"{sample_csv} đã tồn tại. Sao lưu ghi chú trước khi dùng "
            "--overwrite-threshold-sheets để tạo lại."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    all_results = []
    with hands_api.Hands(
        static_image_mode=True,
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detection_confidence,
    ) as detector:
        for group, (metric, lower, upper) in THRESHOLD_RANGES.items():
            candidates = []
            for record in pending:
                value = record.get(metric)
                if value is None:
                    continue
                try:
                    if lower <= float(value) < upper:
                        candidates.append(record)
                except (TypeError, ValueError):
                    continue
            selected = choose_samples(candidates, args.per_group, rng)
            group_dir = output_dir / group
            group_dir.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(selected), 12):
                sheet, result_rows = create_paired_sheet(
                    selected[start : start + 12],
                    args.raw_dir,
                    group,
                    detector,
                    draw_api,
                    hands_api,
                )
                write_jpg(group_dir / f"sheet_{start // 12 + 1:02d}.jpg", sheet)
                all_results.extend(result_rows)
            print(
                f"{group}: {len(selected)}/{len(candidates)} ảnh; "
                f"phát hiện {sum(r['detection_status'] == 'detected' for r in all_results if r['review_group'] == group)}",
                flush=True,
            )
    fields = [
        "review_group",
        "sample_id",
        "relative_path",
        "label",
        "participant_id",
        "blur_score",
        "contrast_std",
        "brightness_mean",
        "detected_hands",
        "detection_status",
        "review_decision",
        "review_note",
    ]
    with sample_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Đã xuất {len(all_results)} lượt ảnh: {output_dir}")
    print("Ảnh gốc bên trái, landmarks bên phải. Không tạo file 63D.")
    print("KHÔNG thay đổi Raw, manifest_final.jsonl hoặc review_queue.csv.")
    print("Mẫu đã phát hiện tay vẫn cần xem độ chính xác landmarks và nhãn bằng mắt.")


def main():
    parser = argparse.ArgumentParser(
        description="Contact sheets hoặc MediaPipe audit cho Task 2.2"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--reviews", type=Path, default=Path("manifests/review_queue.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manifests/review_sheets")
    )
    parser.add_argument("--per-group", type=int, default=40)
    parser.add_argument(
        "--threshold-sheets",
        action="store_true",
        help="Xem ảnh ở khoảng 40-50 blur, 25-35 contrast, 210-220 brightness kèm landmarks",
    )
    parser.add_argument(
        "--threshold-output-dir",
        type=Path,
        default=Path("manifests/threshold_review_mediapipe"),
    )
    parser.add_argument(
        "--overwrite-threshold-sheets",
        action="store_true",
        help="Cho phép ghi đè sample CSV và ảnh ngưỡng cũ",
    )
    parser.add_argument(
        "--audit", action="store_true", help="Chạy MediaPipe cho mọi ảnh review_needed"
    )
    parser.add_argument(
        "--priority-sheets",
        action="store_true",
        help="Tạo contact sheet ảnh Priority 1/2 từ báo cáo Audit",
    )
    parser.add_argument(
        "--priority-output-dir",
        type=Path,
        default=Path("manifests/priority_review_sheets"),
    )
    parser.add_argument(
        "--overwrite-priority-sheets",
        action="store_true",
        help="Cho phép tạo lại báo cáo review ảnh ưu tiên",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/manifest_final.jsonl")
    )
    parser.add_argument(
        "--audit-output-dir", type=Path, default=Path("manifests/pending_audit")
    )
    parser.add_argument(
        "--overwrite-audit", action="store_true", help="Ghi đè các báo cáo audit cũ"
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    args = parser.parse_args()
    if args.per_group < 1:
        parser.error("--per-group phải >= 1")
    if args.max_hands < 1:
        parser.error("--max-hands phải >= 1")
    if not (0 < args.min_detection_confidence <= 1):
        parser.error("--min-detection-confidence phải thuộc (0, 1]")
    if sum([args.audit, args.priority_sheets, args.threshold_sheets]) > 1:
        parser.error(
            "Chỉ chọn một chế độ: --audit, --priority-sheets hoặc --threshold-sheets"
        )
    if args.audit:
        audit_pending_images(args)
    elif args.priority_sheets:
        create_priority_sheets(args)
    elif args.threshold_sheets:
        create_threshold_landmark_sheets(args)
    else:
        create_contact_sheets(args)


if __name__ == "__main__":
    main()
