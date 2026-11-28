"""Kiểm chứng file ONNX trên CPU local và đóng gói đầy đủ bàn giao cho Thành viên C.

Chạy từ thư mục gốc dự án sau khi đặt file static_gesture_mlp.onnx vào models/:
    .\.venv311\Scripts\python.exe scripts/package_onnx_bundle.py

Script này chỉ dùng numpy + onnxruntime + matplotlib (KHÔNG cần TensorFlow),
nên chạy ổn định ngay trên máy Windows có bật Smart App Control.

Đầu ra tự động sinh ra:
    - models/classes.json                 (bản sao khóa thứ tự 34 nhãn)
    - models/model_metadata.json          (hợp đồng input/output, SHA-256, modes cho 03_app.py)
    - results/evaluation_summary.json     (chỉ số Val/Test: 34 nhãn, Số, Chữ, từng người, Coverage, Precision/Recall)
    - results/error_gallery.json          (danh sách sample_id bị đoán sai để phân tích lỗi)
    - results/confusion_matrix_34.png     (hình Ma trận nhầm lẫn 34x34 trên tập Test)
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

DIGIT_COUNT = 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_index(manifest_path: Path) -> tuple[dict, dict]:
    """Đọc manifests/split_manifest.jsonl -> ánh xạ sample_id và đếm tổng mẫu gốc mỗi split."""
    by_id = {}
    total_per_split = {"train": 0, "val": 0, "test": 0}
    if not manifest_path.exists():
        return by_id, total_per_split
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[rec["sample_id"]] = {
                "participant_id": rec.get("participant_id", "UNKNOWN"),
                "relative_path": rec.get("relative_path", ""),
                "split": rec.get("split", ""),
            }
            sp = rec.get("split")
            if sp in total_per_split:
                total_per_split[sp] += 1
    return by_id, total_per_split


def predict_with_mode(probs: np.ndarray, mode: str) -> np.ndarray:
    if mode == "digit":
        return np.argmax(probs[:, :DIGIT_COUNT], axis=1)
    if mode == "letter":
        return DIGIT_COUNT + np.argmax(probs[:, DIGIT_COUNT:], axis=1)
    return np.argmax(probs, axis=1)


def per_class_metrics(cm: np.ndarray, classes: list[str]) -> tuple[dict, float]:
    per_cls = {}
    f1s = []
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        per_cls[name] = {
            "support": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
    return per_cls, float(np.mean(f1s))


def evaluate_split(
    probs: np.ndarray,
    y_true: np.ndarray,
    sample_ids: np.ndarray,
    classes: list[str],
    manifest_idx: dict,
    total_before_extract: int,
):
    n_cls = len(classes)
    preds_all = predict_with_mode(probs, "all")
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, preds_all):
        cm[int(t), int(p)] += 1

    per_cls, macro_f1 = per_class_metrics(cm, classes)
    acc_all = float((preds_all == y_true).mean())

    digit_mask = y_true < DIGIT_COUNT
    letter_mask = y_true >= DIGIT_COUNT
    preds_digit = predict_with_mode(probs[digit_mask], "digit")
    preds_letter = predict_with_mode(probs[letter_mask], "letter")
    acc_digit = float((preds_digit == y_true[digit_mask]).mean())
    acc_letter = float((preds_letter == y_true[letter_mask]).mean())

    # Coverage và độ chính xác toàn pipeline (tính cả ảnh MediaPipe thất bại)
    extracted_n = int(len(y_true))
    denom = total_before_extract if total_before_extract > 0 else extracted_n
    coverage = extracted_n / denom
    end_to_end_acc = float((preds_all == y_true).sum() / denom)

    # Chỉ số theo từng người tham gia (Participant)
    by_person = {}
    for idx, sid in enumerate(sample_ids):
        pid = manifest_idx.get(str(sid), {}).get("participant_id", "UNKNOWN")
        by_person.setdefault(pid, []).append(idx)

    per_participant = {}
    for pid, idxs in sorted(by_person.items()):
        idxs_arr = np.asarray(idxs, dtype=int)
        yt = y_true[idxs_arr]
        pr = probs[idxs_arr]
        pa = predict_with_mode(pr, "all")
        dm = yt < DIGIT_COUNT
        lm = yt >= DIGIT_COUNT
        per_participant[pid] = {
            "samples": int(len(idxs)),
            "all_34_accuracy": round(float((pa == yt).mean()), 4),
            "digit_mode_accuracy": round(float((predict_with_mode(pr[dm], "digit") == yt[dm]).mean()), 4) if dm.any() else None,
            "letter_mode_accuracy": round(float((predict_with_mode(pr[lm], "letter") == yt[lm]).mean()), 4) if lm.any() else None,
        }

    # Top cặp nhầm lẫn
    def top_pairs(yt, yp, limit=15):
        counts = {}
        for t, p in zip(yt, yp):
            if t != p:
                key = (classes[int(t)], classes[int(p)])
                counts[key] = counts.get(key, 0) + 1
        return [
            {"true": k[0], "pred": k[1], "count": int(v)}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])[:limit]
        ]

    # Error gallery (tối đa 200 mẫu lỗi kèm confidence)
    errors = []
    for idx in range(len(y_true)):
        t, p = int(y_true[idx]), int(preds_all[idx])
        if t != p:
            sid = str(sample_ids[idx])
            info = manifest_idx.get(sid, {})
            errors.append({
                "sample_id": sid,
                "participant_id": info.get("participant_id", "UNKNOWN"),
                "relative_path": info.get("relative_path", ""),
                "true_label": classes[t],
                "pred_label": classes[p],
                "pred_confidence": round(float(probs[idx, p]), 4),
                "true_confidence": round(float(probs[idx, t]), 4),
            })

    summary = {
        "samples_extracted": extracted_n,
        "samples_total_in_split": int(denom),
        "coverage": round(coverage, 4),
        "all_34_accuracy": round(acc_all, 4),
        "all_34_end_to_end_accuracy": round(end_to_end_acc, 4),
        "all_34_macro_f1": round(macro_f1, 4),
        "digit_mode_accuracy": round(acc_digit, 4),
        "letter_mode_accuracy": round(acc_letter, 4),
        "per_participant": per_participant,
        "top_confusions_all_34": top_pairs(y_true, preds_all, 15),
        "top_confusions_letter_mode": top_pairs(y_true[letter_mask], preds_letter, 10),
        "per_class": per_cls,
    }
    return summary, cm, errors


def benchmark_latency_ms(session: ort.InferenceSession, input_name: str, sample: np.ndarray, runs: int = 300) -> dict:
    single = sample[:1].astype(np.float32)
    for _ in range(30):
        session.run(None, {input_name: single})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: single})
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "runs": runs,
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "mean_ms": round(float(np.mean(arr)), 4),
    }


def save_confusion_matrix_plot(cm: np.ndarray, classes: list[str], out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bỏ qua vẽ hình vì chưa cài matplotlib.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Nhãn dự đoán (Predicted)")
    ax.set_ylabel("Nhãn thật (True)")
    ax.set_title("Confusion Matrix — Tập Test (34 nhãn, ONNX CPU)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Kiểm chứng ONNX local và đóng gói bundle cho Thành viên C.")
    parser.add_argument("--onnx-path", type=Path, default=Path("models/static_gesture_mlp.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/split_manifest.jsonl"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.onnx_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file ONNX tại {args.onnx_path}.\n"
            f"Hãy tải static_gesture_mlp.onnx từ Google Colab và chép vào thư mục models/ trước."
        )

    classes = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))
    data_meta = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest_idx, total_per_split = load_manifest_index(args.manifest)

    X_val = np.load(args.data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(args.data_dir / "y_val.npy").astype(np.int64)
    sid_val = np.load(args.data_dir / "sample_ids_val.npy")

    X_test = np.load(args.data_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(args.data_dir / "y_test.npy").astype(np.int64)
    sid_test = np.load(args.data_dir / "sample_ids_test.npy")

    # 1. Nạp ONNX trên CPU và kiểm tra hợp đồng batch=1 & batch=N
    session = ort.InferenceSession(str(args.onnx_path), providers=["CPUExecutionProvider"])
    inp_meta = session.get_inputs()[0]
    out_meta = session.get_outputs()[0]
    input_name, output_name = inp_meta.name, out_meta.name

    one_out = session.run(None, {input_name: X_val[:1]})[0]
    if one_out.shape != (1, len(classes)) or not np.isfinite(one_out).all():
        raise ValueError(f"Đầu ra batch=1 không hợp lệ: shape={one_out.shape}")

    val_probs = session.run(None, {input_name: X_val})[0]
    test_probs = session.run(None, {input_name: X_test})[0]
    if not (np.isfinite(val_probs).all() and np.isfinite(test_probs).all()):
        raise ValueError("Đầu ra ONNX chứa NaN hoặc Infinity!")

    # Kiểm tra nhất quán batch=1 so với batch=N
    batch1_diff = float(np.max(np.abs(one_out[0] - val_probs[0])))
    latency = benchmark_latency_ms(session, input_name, X_test)

    # 2. Đánh giá chi tiết Val & Test
    val_summary, _, _ = evaluate_split(
        val_probs, y_val, sid_val, classes, manifest_idx, total_per_split["val"]
    )
    test_summary, cm_test, test_errors = evaluate_split(
        test_probs, y_test, sid_test, classes, manifest_idx, total_per_split["test"]
    )

    # 3. Đóng gói models/ cho Thành viên C
    models_dir = args.onnx_path.parent
    classes_dst = models_dir / "classes.json"
    shutil.copyfile(args.data_dir / "classes.json", classes_dst)

    model_meta = {
        "model_version": "static_mlp_v2_handgeom",
        "model_file": args.onnx_path.name,
        "model_sha256": sha256_file(args.onnx_path),
        "classes_file": classes_dst.name,
        "classes_sha256": sha256_file(classes_dst),
        "input_name": input_name,
        "output_name": output_name,
        "input_dtype": "float32",
        "input_shape": ["batch", int(X_val.shape[1])],
        "output_shape": ["batch", len(classes)],
        "input_dim": int(X_val.shape[1]),
        "num_classes": len(classes),
        "feature_version": data_meta.get("feature_version", "hand63_aspect_wrist09_v1"),
        "normalization": data_meta.get("normalization", ""),
        "classes": classes,
        "modes": {
            "all": {"start": 0, "end": len(classes)},
            "digit": {"start": 0, "end": DIGIT_COUNT},
            "letter": {"start": DIGIT_COUNT, "end": len(classes)},
        },
        "opset": 13,
        "onnxruntime_verified_version": ort.__version__,
        "batch1_vs_batchN_max_diff": batch1_diff,
    }
    (models_dir / "model_metadata.json").write_text(
        json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4. Lưu báo cáo & hình vào results/
    args.results_dir.mkdir(parents=True, exist_ok=True)
    eval_report = {
        "model_file": str(args.onnx_path),
        "model_sha256": model_meta["model_sha256"],
        "onnxruntime_version": ort.__version__,
        "cpu_single_frame_latency_ms": latency,
        "validation": val_summary,
        "test": test_summary,
    }
    (args.results_dir / "evaluation_summary.json").write_text(
        json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.results_dir / "error_gallery.json").write_text(
        json.dumps({"total_errors_test": len(test_errors), "samples": test_errors[:200]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_confusion_matrix_plot(cm_test, classes, args.results_dir / "confusion_matrix_34.png")

    print("=== KIỂM CHỨNG ONNX TRÊN CPU THÀNH CÔNG ===")
    print(f"Input : {input_name} {inp_meta.shape} | Output: {output_name} {out_meta.shape}")
    print(f"SHA256: {model_meta['model_sha256']}")
    print(f"Tốc độ CPU (1 khung hình): median={latency['median_ms']} ms | p95={latency['p95_ms']} ms")
    print(
        f"Validation | 34 nhãn: {val_summary['all_34_accuracy']*100:.2f}% (F1={val_summary['all_34_macro_f1']*100:.2f}%) "
        f"| Số: {val_summary['digit_mode_accuracy']*100:.2f}% | Chữ: {val_summary['letter_mode_accuracy']*100:.2f}%"
    )
    print(
        f"Test       | 34 nhãn: {test_summary['all_34_accuracy']*100:.2f}% (F1={test_summary['all_34_macro_f1']*100:.2f}%) "
        f"| Toàn pipeline: {test_summary['all_34_end_to_end_accuracy']*100:.2f}% "
        f"| Số: {test_summary['digit_mode_accuracy']*100:.2f}% | Chữ: {test_summary['letter_mode_accuracy']*100:.2f}%"
    )
    print("\nĐã đóng gói xong cho Thành viên C:")
    print(f"  - {args.onnx_path}")
    print(f"  - {classes_dst}")
    print(f"  - {models_dir / 'model_metadata.json'}")
    print(f"  - {args.results_dir / 'evaluation_summary.json'}")
    print(f"  - {args.results_dir / 'error_gallery.json'}")
    print(f"  - {args.results_dir / 'confusion_matrix_34.png'}")


if __name__ == "__main__":
    main()
