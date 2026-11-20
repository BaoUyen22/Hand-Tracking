"""Huấn luyện mô hình MLP 34 ký hiệu tĩnh, đánh giá đa chế độ và xuất ONNX.

Chạy từ thư mục gốc dự án:
    python 02_train.py

Đầu vào (do Thành viên A bàn giao trong data/processed/):
    - X_train.npy, y_train.npy
    - X_val.npy,   y_val.npy
    - X_test.npy,  y_test.npy
    - classes.json, metadata.json

Đầu ra:
    - models/static_gesture_mlp.onnx      (mô hình suy luận trên CPU cho 03_app.py)
    - models/model_metadata.json          (hợp đồng input/output & thứ tự nhãn)
    - results/evaluation_summary.json     (độ chính xác 34 nhãn, chế độ Số, chế độ Chữ)
    - results/training_curves.png         (biểu đồ Loss / Accuracy nếu có matplotlib)
    - results/confusion_matrix_34.png     (ma trận nhầm lẫn nếu có matplotlib)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow.keras import callbacks

from src.model import build_mlp_static

DIGIT_COUNT = 10  # 0..9 nằm ở 10 vị trí đầu trong classes.json


def load_dataset(data_dir: Path):
    """Đọc và kiểm tra hợp đồng dữ liệu từ thư mục data/processed/."""
    X_train = np.load(data_dir / "X_train.npy").astype(np.float32)
    y_train = np.load(data_dir / "y_train.npy").astype(np.int64)
    X_val = np.load(data_dir / "X_val.npy").astype(np.float32)
    y_val = np.load(data_dir / "y_val.npy").astype(np.int64)
    X_test = np.load(data_dir / "X_test.npy").astype(np.float32)
    y_test = np.load(data_dir / "y_test.npy").astype(np.int64)

    with open(data_dir / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    expected_dim = metadata.get("feature_dimensions", 63)
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        if X.ndim != 2 or X.shape[1] != expected_dim:
            raise ValueError(f"{name}: kỳ vọng shape (*, {expected_dim}), nhận {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"{name}: số dòng X ({len(X)}) khác y ({len(y)})")
        if not np.isfinite(X).all():
            raise ValueError(f"{name}: dữ liệu chứa NaN hoặc Infinity")

    if len(classes) != 34:
        raise ValueError(f"Kỳ vọng 34 nhãn trong classes.json, nhận {len(classes)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), classes, metadata


def augment_train(X: np.ndarray, y: np.ndarray, copies: int = 3, seed: int = 42):
    """Tăng cường dữ liệu chỉ trên tập Train: xoay nhẹ +-12 độ, co giãn và nhiễu nhỏ."""
    if copies <= 0:
        return X, y

    rng = np.random.default_rng(seed)
    X_list, y_list = [X], [y]
    pts = X.reshape(-1, 21, 3)

    for _ in range(copies):
        angles = np.radians(rng.uniform(-12.0, 12.0, size=(len(X), 1))).astype(np.float32)
        c, s = np.cos(angles), np.sin(angles)
        x_rot = pts[:, :, 0] * c - pts[:, :, 1] * s
        y_rot = pts[:, :, 0] * s + pts[:, :, 1] * c

        scale = rng.uniform(0.93, 1.07, size=(len(X), 1)).astype(np.float32)
        x_new = x_rot * scale + rng.normal(0.0, 0.015, size=x_rot.shape).astype(np.float32)
        y_new = y_rot * scale + rng.normal(0.0, 0.015, size=y_rot.shape).astype(np.float32)
        z_new = pts[:, :, 2] * scale + rng.normal(0.0, 0.03, size=x_rot.shape).astype(np.float32)

        aug = np.stack([x_new, y_new, z_new], axis=-1)
        aug[:, 0, :] = 0.0  # Giữ điểm cổ tay (landmark 0) cố định tại gốc tọa độ
        X_list.append(aug.reshape(-1, 63).astype(np.float32))
        y_list.append(y)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def predict_with_mode(probs: np.ndarray, mode: str = "all") -> np.ndarray:
    """Lọc xác suất theo chế độ sử dụng từ cùng 1 mô hình 34 nhãn.

    Args:
        probs: Mảng xác suất shape (N, 34).
        mode: "all" (34 nhãn), "digit" (0-9), hoặc "letter" (A-Y).
    """
    if mode == "digit":
        return np.argmax(probs[:, :DIGIT_COUNT], axis=1)
    if mode == "letter":
        return DIGIT_COUNT + np.argmax(probs[:, DIGIT_COUNT:], axis=1)
    if mode == "all":
        return np.argmax(probs, axis=1)
    raise ValueError(f"Chế độ không hợp lệ: {mode}")


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def evaluate_predictions(probs: np.ndarray, y_true: np.ndarray, classes: list[str]) -> dict:
    """Đánh giá đầy đủ cả 3 chế độ (Chung 34 nhãn, Số 0-9, Chữ A-Y) và top cặp nhầm."""
    num_classes = len(classes)
    preds_all = predict_with_mode(probs, "all")
    cm_all = compute_confusion_matrix(y_true, preds_all, num_classes)

    digit_mask = y_true < DIGIT_COUNT
    letter_mask = y_true >= DIGIT_COUNT

    preds_digit = predict_with_mode(probs[digit_mask], "digit")
    preds_letter = predict_with_mode(probs[letter_mask], "letter")

    def top_confusions(yt, yp, limit=15):
        counts = {}
        for t, p in zip(yt, yp):
            if t != p:
                key = (classes[int(t)], classes[int(p)])
                counts[key] = counts.get(key, 0) + 1
        return [
            {"true": k[0], "pred": k[1], "count": int(v)}
            for k, v in sorted(counts.items(), key=lambda item: -item[1])[:limit]
        ]

    return {
        "all_34_accuracy": float((preds_all == y_true).mean()),
        "all_34_macro_f1": compute_macro_f1(cm_all),
        "digit_mode_accuracy": float((preds_digit == y_true[digit_mask]).mean()),
        "letter_mode_accuracy": float((preds_letter == y_true[letter_mask]).mean()),
        "top_confusions_all_34": top_confusions(y_true, preds_all, limit=15),
        "top_confusions_letter_mode": top_confusions(y_true[letter_mask], preds_letter, limit=10),
        "confusion_matrix_all_34": cm_all,
    }


def export_to_onnx(model: tf.keras.Model, input_dim: int, output_path: Path, opset: int = 13):
    """Xuất mô hình Keras sang ONNX bằng from_function (tương thích tốt Keras 2 & Keras 3)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    spec = [tf.TensorSpec((None, input_dim), tf.float32, name="landmarks")]

    @tf.function(input_signature=spec)
    def serving_fn(x):
        return {"probs": model(x, training=False)}

    tf2onnx.convert.from_function(
        serving_fn,
        input_signature=spec,
        opset=opset,
        output_path=str(output_path),
    )


def verify_onnx(onnx_path: Path, X_test: np.ndarray, keras_probs: np.ndarray) -> float:
    """Chạy suy luận thử bằng ONNX Runtime trên CPU và đo độ lệch tối đa so với Keras."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_probs = session.run(None, {input_name: X_test.astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(onnx_probs - keras_probs)))
    return max_diff


def save_plots(history, cm: np.ndarray, classes: list[str], results_dir: Path):
    """Lưu biểu đồ Loss/Accuracy và Ma trận nhầm lẫn vào thư mục results/."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bỏ qua bước vẽ biểu đồ vì chưa cài matplotlib.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đường cong huấn luyện
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss theo Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy theo Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # 2. Ma trận nhầm lẫn 34 nhãn
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Nhãn dự đoán (Predicted)")
    ax.set_ylabel("Nhãn thật (True)")
    ax.set_title("Confusion Matrix — Tập Test (34 nhãn)")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix_34.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện MLP 34 ký hiệu tĩnh và xuất ONNX.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--aug-copies", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), classes, data_meta = load_dataset(args.data_dir)
    input_dim = X_train.shape[1]
    num_classes = len(classes)

    print(f"Dữ liệu: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}, Nhãn={num_classes}")

    X_train_aug, y_train_aug = augment_train(X_train, y_train, copies=args.aug_copies, seed=args.seed)
    print(f"Sau tăng cường dữ liệu: Train={X_train_aug.shape}")

    model = build_mlp_static(input_dim=input_dim, num_classes=num_classes)
    cbs = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
    )

    val_probs = model.predict(X_val, verbose=0)
    test_probs = model.predict(X_test, verbose=0)
    val_eval = evaluate_predictions(val_probs, y_val, classes)
    test_eval = evaluate_predictions(test_probs, y_test, classes)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(
        f"Validation | 34 nhãn: {val_eval['all_34_accuracy']*100:.2f}% "
        f"| Số (0-9): {val_eval['digit_mode_accuracy']*100:.2f}% "
        f"| Chữ (A-Y): {val_eval['letter_mode_accuracy']*100:.2f}%"
    )
    print(
        f"Test       | 34 nhãn: {test_eval['all_34_accuracy']*100:.2f}% "
        f"| Số (0-9): {test_eval['digit_mode_accuracy']*100:.2f}% "
        f"| Chữ (A-Y): {test_eval['letter_mode_accuracy']*100:.2f}%"
    )

    onnx_path = args.models_dir / "static_gesture_mlp.onnx"
    export_to_onnx(model, input_dim=input_dim, output_path=onnx_path, opset=args.opset)
    max_diff = verify_onnx(onnx_path, X_test, test_probs)
    print(f"\nĐã xuất ONNX: {onnx_path} (độ lệch tối đa so với Keras: {max_diff:.2e})")

    # Lưu metadata mô hình để 03_app.py kiểm tra hợp đồng
    model_meta = {
        "model_file": onnx_path.name,
        "input_name": "landmarks",
        "output_name": "probs",
        "input_dim": input_dim,
        "num_classes": num_classes,
        "feature_version": data_meta.get("feature_version", "hand63_aspect_wrist09_v1"),
        "classes": classes,
        "modes": {
            "all": {"start": 0, "end": num_classes},
            "digit": {"start": 0, "end": DIGIT_COUNT},
            "letter": {"start": DIGIT_COUNT, "end": num_classes},
        },
        "opset": args.opset,
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    (args.models_dir / "model_metadata.json").write_text(
        json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Lưu báo cáo số liệu (bỏ ma trận numpy trước khi ghi JSON)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cm_test = test_eval.pop("confusion_matrix_all_34")
    val_eval.pop("confusion_matrix_all_34")
    summary = {
        "seed": args.seed,
        "epochs_trained": len(history.history["loss"]),
        "aug_copies": args.aug_copies,
        "onnx_max_abs_diff": max_diff,
        "validation": val_eval,
        "test": test_eval,
    }
    (args.results_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(history, cm_test, classes, args.results_dir)
    print(f"Đã lưu báo cáo & biểu đồ tại: {args.results_dir}")


if __name__ == "__main__":
    main()
