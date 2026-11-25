"""Kiến trúc mô hình MLP phân loại 34 ký hiệu bàn tay tĩnh (Thành viên B).

Đầu vào tuân thủ hợp đồng 63 chiều từ ``src/features.py`` (21 landmarks x 3).
Tầng ``HandGeometryLayer`` được nhúng trực tiếp trong đồ thị TensorFlow/ONNX
để chuẩn hóa góc xoay bàn tay, tính vector đốt ngón và khoảng cách khớp mà
không làm thay đổi hợp đồng đầu vào 63 chiều.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# 20 đoạn xương nối 21 điểm mốc MediaPipe Hands
BONE_A = [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
BONE_B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# 14 điểm mốc trọng tâm (cổ tay, 5 đầu ngón, 4 gốc ngón, 4 khớp giữa) -> 91 cặp khoảng cách
KEY_PTS = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17, 6, 10, 14, 18]
PAIR_I = [KEY_PTS[i] for i in range(len(KEY_PTS)) for j in range(i + 1, len(KEY_PTS))]
PAIR_J = [KEY_PTS[j] for i in range(len(KEY_PTS)) for j in range(i + 1, len(KEY_PTS))]

GEOMETRY_FEATURE_DIM = 42 + 40 + len(PAIR_I) + 2 + 21  # 196 đặc trưng


class HandGeometryLayer(layers.Layer):
    """Biến đổi vector 63D (21x3) thành 196 đặc trưng hình học bất biến góc nghiêng."""

    def call(self, x):
        pts = tf.reshape(x, (-1, 21, 3))
        xy = pts[:, :, :2]
        z = pts[:, :, 2]

        # 1. Trục chuẩn cổ tay (0) -> gốc ngón giữa (9) để xoay thẳng bàn tay
        v = xy[:, 9, :] - xy[:, 0, :]
        v_norm = tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True) + 1e-8)
        u = v / v_norm
        ux, uy = u[:, 0:1], u[:, 1:2]

        xr = -uy * xy[:, :, 0] + ux * xy[:, :, 1]
        yr = -ux * xy[:, :, 0] - uy * xy[:, :, 1]
        rot_xy = tf.stack([xr, yr], axis=-1)              # (batch, 21, 2)
        rot_flat = tf.reshape(rot_xy, (-1, 42))           # 42 chiều

        # 2. 20 vector hướng đơn vị của các đốt ngón tay (40 chiều)
        bones = tf.gather(rot_xy, BONE_B, axis=1) - tf.gather(rot_xy, BONE_A, axis=1)
        bdir = bones / tf.sqrt(tf.reduce_sum(tf.square(bones), axis=-1, keepdims=True) + 1e-8)
        bdir_flat = tf.reshape(bdir, (-1, 40))

        # 3. 91 khoảng cách 2D giữa các đầu ngón và khớp quan trọng
        diffs = tf.gather(rot_xy, PAIR_J, axis=1) - tf.gather(rot_xy, PAIR_I, axis=1)
        pdist = tf.sqrt(tf.reduce_sum(tf.square(diffs), axis=-1) + 1e-8)

        # Ghép tất cả: 42 + 40 + 91 + 1 + 1 + 21 = 196 đặc trưng
        return tf.concat([rot_flat, bdir_flat, pdist, ux, uy, z], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], GEOMETRY_FEATURE_DIM)

    def get_config(self):
        return super().get_config()


def build_mlp_static(
    input_dim=63,
    num_classes=34,
    learning_rate=1e-3,
    l2_weight=1e-4,
    use_geometry_layer=True,
):
    """Xây dựng và biên dịch mô hình MLP cho nhận diện ký hiệu bàn tay tĩnh.

    Args:
        input_dim: Số chiều đặc trưng đầu vào (mặc định 63 theo hợp đồng).
        num_classes: Số nhãn phân loại (mặc định 34: 0-9 và A-Y trừ J, Z).
        learning_rate: Tốc độ học ban đầu cho bộ tối ưu Adam.
        l2_weight: Hệ số điều chuẩn L2 giúp hạn chế overfitting giữa các Participant.
        use_geometry_layer: Có bật tầng HandGeometryLayer (yêu cầu input_dim == 63) hay không.
    """
    inp = layers.Input(shape=(input_dim,), dtype="float32", name="landmarks")
    if use_geometry_layer:
        if input_dim != 63:
            raise ValueError(f"HandGeometryLayer yêu cầu input_dim=63, nhận {input_dim}")
        x = HandGeometryLayer(name="hand_geometry")(inp)
    else:
        x = inp

    reg = regularizers.l2(l2_weight)

    x = layers.Dense(256, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)

    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)

    model = models.Model(inputs=inp, outputs=out, name="static_gesture_mlp_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
