r"""Huấn luyện và đánh giá Random Forest cho 34 ký hiệu tay (đặc trưng 63D).

Chạy từ thư mục gốc Hand-Tracking:
    .\.venv311\Scripts\python.exe scripts/train_random_forest.py

Dùng Train để fit; Validation để chọn cấu hình; chỉ đánh giá Test sau khi chọn.
Không sửa/chia lại dữ liệu; không tích hợp vào webcam.
"""

import csv
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Xuất PNG mà không cần cửa sổ đồ họa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "random_forest"
SEED = 42

# Cấu hình thử nghiệm (có thể bổ sung TRƯỚC KHI xem kết quả Test)
CANDIDATES = [
    {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
]


def read_data():
    required = ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
                "X_test.npy", "y_test.npy", "classes.json"]
    missing = [name for name in required if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu file trong {DATA_DIR}: {', '.join(missing)}. "
            "Hãy kiểm tra thư mục data/processed/ của Thành viên A."
        )

    classes = json.loads((DATA_DIR / "classes.json").read_text(encoding="utf-8"))
    if not isinstance(classes, list) or len(classes) != 34:
        raise ValueError("classes.json phải là danh sách 34 tên nhãn, đúng thứ tự của y_*.npy")

    datasets = {}
    for part in ("train", "val", "test"):
        X = np.load(DATA_DIR / f"X_{part}.npy", allow_pickle=False).astype(np.float32)
        y = np.load(DATA_DIR / f"y_{part}.npy", allow_pickle=False).astype(np.int64).reshape(-1)
        if X.ndim != 2 or X.shape[1] != 63 or len(X) != len(y):
            raise ValueError(f"{part}: kỳ vọng X có 63 chiều và số hàng X/y bằng nhau; nhận {X.shape} và {y.shape}")
        if not np.isfinite(X).all() or (y < 0).any() or (y >= len(classes)).any():
            raise ValueError(f"{part}: đặc trưng không hữu hạn hoặc nhãn nằm ngoài 0..33")
        datasets[part] = (X, y)
    return classes, datasets


def basic_metrics(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
    }


def save_confusion(cm, classes, filename):
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)), labels=classes, fontsize=8)
    ax.set_yticks(range(len(classes)), labels=classes, fontsize=8)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_title("Random Forest — Confusion Matrix (Test, 34 nhãn)")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def write_classification_csv(y_true, y_pred, classes, filename):
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0,
    )
    with filename.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "precision", "recall", "f1_score", "support"])
        for name in classes + ["macro avg", "weighted avg"]:
            row = report[name]
            writer.writerow([name, row["precision"], row["recall"], row["f1-score"], row["support"]])
        writer.writerow(["accuracy", report["accuracy"], "", "", len(y_true)])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classes, ds = read_data()
    X_train, y_train = ds["train"]
    X_val, y_val = ds["val"]
    X_test, y_test = ds["test"]

    print("=== RANDOM FOREST 63D / 34 NHÃN ===")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print("Chọn cấu hình bằng Validation Macro-F1; không dùng Test để chọn.")

    trials = []
    models = []
    for index, params in enumerate(CANDIDATES, start=1):
        model = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start
        val_pred = model.predict(X_val)
        metric = basic_metrics(y_val, val_pred)
        trial = {"candidate": index, "parameters": params,
                 "train_seconds": round(train_seconds, 4), "validation": metric}
        trials.append(trial)
        models.append(model)
        print(f"[{index}/{len(CANDIDATES)}] {params} | "
              f"Val Accuracy={metric['accuracy']:.4%} | Macro-F1={metric['f1_macro']:.4%} | "
              f"Train={train_seconds:.2f}s")

    best_i = max(range(len(trials)), key=lambda i: (trials[i]["validation"]["f1_macro"],
                                                       trials[i]["validation"]["accuracy"]))
    model = models[best_i]
    best = trials[best_i]
    print(f"\nChọn cấu hình #{best['candidate']} theo Validation Macro-F1.")

    # Đánh giá một lần trên Test sau khi chọn cấu hình.
    start = time.perf_counter()
    test_pred = model.predict(X_test)
    test_predict_seconds = time.perf_counter() - start
    test_metrics = basic_metrics(y_test, test_pred)
    val_pred = model.predict(X_val)
    val_metrics = basic_metrics(y_val, val_pred)

    labels = list(range(len(classes)))
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    save_confusion(cm, classes, OUTPUT_DIR / "confusion_matrix_test.png")
    write_classification_csv(y_test, test_pred, classes, OUTPUT_DIR / "classification_report_test.csv")
    with (OUTPUT_DIR / "confusion_matrix_test.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + classes)
        for name, row in zip(classes, cm):
            writer.writerow([name] + row.tolist())

    # Dự đoán kèm sample_id nếu file được bàn giao.
    ids_file = DATA_DIR / "sample_ids_test.npy"
    ids = np.load(ids_file, allow_pickle=False).astype(str) if ids_file.exists() else None
    with (OUTPUT_DIR / "test_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "true_label", "pred_label", "correct"])
        for i, (truth, pred) in enumerate(zip(y_test, test_pred)):
            writer.writerow([ids[i] if ids is not None and i < len(ids) else i,
                             classes[int(truth)], classes[int(pred)], int(truth == pred)])

    joblib.dump(model, OUTPUT_DIR / "random_forest.joblib")
    summary = {
        "model": "RandomForestClassifier",
        "role": "Thành viên A - mô hình so sánh, không triển khai webcam",
        "feature_dim": 63,
        "classes": classes,
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "seed": SEED,
        "selection_rule": "Highest validation macro-F1; tie: validation accuracy",
        "candidates": trials,
        "chosen_candidate": best["candidate"],
        "chosen_parameters": best["parameters"],
        "validation": val_metrics,
        "test": test_metrics,
        "test_predict_total_seconds": round(test_predict_seconds, 6),
        "test_predict_ms_per_sample": round(1000 * test_predict_seconds / len(y_test), 6),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scikit_learn": sklearn.__version__},
        "note": "Tốc độ batch Test không tương đương độ trễ webcam 1 frame; không đánh giá unknown gestures.",
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== KẾT QUẢ ĐỂ ĐƯA VÀO SLIDE ===")
    for name, metric in [("Validation", val_metrics), ("Test", test_metrics)]:
        print(f"{name}: Accuracy={metric['accuracy']:.2%} | "
              f"Precision(macro)={metric['precision_macro']:.2%} | "
              f"Recall(macro)={metric['recall_macro']:.2%} | "
              f"F1(macro)={metric['f1_macro']:.2%}")
    print(f"\nĐã xuất kết quả tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
