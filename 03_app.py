"""
03_app.py - Nhận diện ngôn ngữ ký hiệu, ghép câu bền vững và đọc giọng nói (TTS)
- KHÔNG BAO GIỜ TỰ XÓA CÂU
- Chỉ xóa khi người dùng tự bấm phím Backspace
- Enter: Đọc cả câu

Phiên bản giao diện: toàn bộ logic giữ nguyên, chỉ thiết kế lại phần hiển thị.
Hướng dẫn phím điều khiển được hiển thị trực tiếp trên cửa sổ camera
(thanh bên phải), người dùng không cần nhìn terminal.
"""

import json
import sys
from pathlib import Path
import time
from collections import deque
import threading
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pyttsx3

# Cho phép chạy trực tiếp `python src/03_app.py` mà vẫn import được `src.features`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import landmarks_to_features

# ================= CẤU HÌNH & LOAD MODEL =================
MODELS_DIR = Path("models")
ONNX_MODEL_PATH = MODELS_DIR / "static_gesture_mlp.onnx"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

if not ONNX_MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy model tại {MODELS_DIR}!")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

CLASSES = metadata["classes"]
INPUT_NAME = metadata.get("input_name", "landmarks")
OUTPUT_NAME = metadata.get("output_name", "probs")
DIGIT_COUNT = metadata.get("modes", {}).get("digit", {}).get("end", 10)

session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

SMOOTHING_WINDOW = 5
prob_buffer = deque(maxlen=SMOOTHING_WINDOW)

# ================= ĐỌC GIỌNG NÓI CHẠY NGẦM (NON-BLOCKING) =================
is_speaking = False

def speak_text_worker(text_to_speak):
    global is_speaking
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        print(f"Lỗi TTS: {e}")
    finally:
        is_speaking = False

def speak_async(text):
    global is_speaking
    if not is_speaking and text.strip():
        # [QUAN TRỌNG]: Chuyển toàn bộ chữ HOA thành chữ thường để loa đọc thành từ ngữ
        # Ví dụ: "HELLO" -> "hello", "THANK" -> "thank", "1 2 3" -> "1 2 3"
        readable_text = text.strip().lower()

        is_speaking = True
        t = threading.Thread(target=speak_text_worker, args=(readable_text,), daemon=True)
        t.start()

def predict_mode(probs, mode="all"):
    if mode == "digit":
        sub_probs = probs[:DIGIT_COUNT]
        idx = int(np.argmax(sub_probs))
        return CLASSES[idx], float(sub_probs[idx])
    elif mode == "letter":
        sub_probs = probs[DIGIT_COUNT:]
        idx = int(DIGIT_COUNT + np.argmax(sub_probs))
        return CLASSES[idx], float(probs[idx])
    else:
        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])

# ================= GIAO DIỆN (CHỈ PHẦN HIỂN THỊ) =================
# Toàn bộ giao diện được vẽ lại theo kích thước cửa sổ thực tế ở mỗi khung hình,
# nên có thể kéo to/nhỏ, phóng to toàn màn hình mà chữ không bị cắt hoặc đè nhau.
# Các nút trên màn hình chỉ "giả lập" phím bấm, đi qua đúng logic phím có sẵn.

WIN_NAME = "Hand Tracking - Sign Language Recognition"
DEFAULT_WIN_SIZE = (1280, 720)
MIN_CANVAS = (720, 420)     # dưới mức này canvas được vẽ lớn hơn rồi OpenCV tự thu nhỏ

# Màu theo hệ BGR của OpenCV
C_VIDEO_BG = (14, 13, 11)
C_BG = (26, 24, 20)
C_CARD = (38, 35, 30)
C_CARD_2 = (54, 50, 44)
C_HOVER = (66, 62, 54)
C_BORDER = (72, 68, 60)
C_TEXT = (245, 245, 245)
C_MUTED = (172, 167, 157)
C_DIM = (128, 123, 113)
C_ACCENT = (112, 153, 13)
C_ACCENT_DARK = (72, 106, 8)
C_ACCENT_LIGHT = (166, 213, 80)
C_AMBER = (40, 190, 255)
C_RED = (100, 100, 240)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

KEY_SPACE, KEY_BACKSPACE, KEY_ENTER, KEY_ESC = 32, 8, 13, 27

MODE_CHIPS = [
    (ord("1"), "all", "TAT CA", "34 ky tu"),
    (ord("2"), "digit", "SO", "0 - 9"),
    (ord("3"), "letter", "CHU", "A - Y"),
]
CONTROLS = [
    ("hold", "GIU TAY", "Giu yen ~0.5s de go chu"),
    (KEY_SPACE, "SPACE", "Them dau cach"),
    (KEY_BACKSPACE, "BACKSPACE", "Xoa 1 ky tu cuoi"),
    (KEY_ENTER, "ENTER", "Doc to ca cau"),
    (ord("q"), "Q / ESC", "Thoat ung dung"),
]


def I(v):
    return int(round(v))


class UIState:
    """Lưu vị trí chuột và các vùng nút có thể nhấn trên màn hình."""

    def __init__(self):
        self.mouse = (-1, -1)
        self.clicked = None
        self.buttons = []
        self._next_buttons = []

    def begin(self):
        self._next_buttons = []

    def end(self):
        self.buttons = self._next_buttons

    def add_button(self, rect, code):
        self._next_buttons.append((rect, code))

    def hover(self, rect):
        x, y = self.mouse
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse = (x, y)
            for rect, code in self.buttons:
                if self.hover(rect):
                    self.clicked = code
                    break

    def pop_click(self):
        code, self.clicked = self.clicked, None
        return code


# ---------- Vẽ chữ: tự co chữ / thêm "..." để không bao giờ tràn khung ----------
def _thickness(scale, bold):
    return max(1, I(scale * (2.1 if bold else 1.4)))


def measure(text, scale, bold=False):
    (tw, th), _ = cv2.getTextSize(text, FONT_BOLD if bold else FONT, scale, _thickness(scale, bold))
    return tw, th


def fit_text(text, scale, bold, max_w, shrink=0.75, tail=False):
    if max_w is None:
        return text, scale
    if max_w <= 4:
        return "", scale
    if measure(text, scale, bold)[0] <= max_w:
        return text, scale
    min_scale = scale * shrink
    while scale > min_scale and measure(text, scale, bold)[0] > max_w:
        scale -= 0.02
    if measure(text, scale, bold)[0] <= max_w:
        return text, scale
    body = text
    while len(body) > 1:
        body = body[1:] if tail else body[:-1]
        candidate = "..." + body if tail else body.rstrip() + "..."
        if measure(candidate, scale, bold)[0] <= max_w:
            return candidate, scale
    return "", scale


def draw_text(img, text, x, y, scale, color, bold=False, max_w=None,
              align="left", valign="base", shrink=0.75, tail=False):
    """Vẽ chữ, trả về độ rộng thực tế. y là baseline, hoặc tâm nếu valign='middle'."""
    if not text:
        return 0
    text, scale = fit_text(text, scale, bold, max_w, shrink, tail)
    if not text:
        return 0
    tw, th = measure(text, scale, bold)
    if align == "center":
        x -= tw / 2
    elif align == "right":
        x -= tw
    if valign == "middle":
        y += th / 2
    cv2.putText(img, text, (I(x), I(y)), FONT_BOLD if bold else FONT, scale, color,
                _thickness(scale, bold), cv2.LINE_AA)
    return tw


# ---------- Hình khối ----------
def rounded_rect(img, p1, p2, color, radius=8, thickness=-1):
    x1, y1 = I(p1[0]), I(p1[1])
    x2, y2 = I(p2[0]), I(p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    r = max(0, min(I(radius), (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
        if r > 0:
            for c in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
                cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        if r > 0:
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thickness, cv2.LINE_AA)


def glass_rect(img, p1, p2, color, alpha=0.8, radius=10):
    """Khối bo góc bán trong suốt đặt trên video."""
    x1, y1 = max(0, I(p1[0])), max(0, I(p1[1]))
    x2, y2 = min(img.shape[1], I(p2[0])), min(img.shape[0], I(p2[1]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    rounded_rect(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), color, radius)
    img[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def progress_bar(img, x, y, w, h, value, color, bg=C_CARD_2):
    rounded_rect(img, (x, y), (x + w, y + h), bg, radius=h / 2)
    fill = w * max(0.0, min(1.0, value))
    if fill >= h:
        rounded_rect(img, (x, y), (x + fill, y + h), color, radius=h / 2)
    elif fill > 0:
        cv2.circle(img, (I(x + h / 2), I(y + h / 2)), max(1, I(h / 2)), color, -1, cv2.LINE_AA)


def section_title(img, title, x, y, w, s):
    return draw_text(img, title, x, y + 16 * s, 0.4 * s, C_MUTED, True, max_w=w)


# ---------- Các khối của bảng điều khiển (kích thước gốc khi s = 1) ----------
BASE_W = 300
FOOTER_H = 40


def sec_header(img, x, y, w, s, st, ui):
    logo = 40 * s
    ly = y + 6 * s
    rounded_rect(img, (x, ly), (x + logo, ly + logo), C_ACCENT, 11 * s)
    draw_text(img, "SL", x + logo / 2, ly + logo / 2, 0.55 * s, C_TEXT, True,
              align="center", valign="middle")

    fps_txt = f"{int(st['fps'])} FPS"
    fs = 0.4 * s
    tw, _ = measure(fps_txt, fs, True)
    bw, bh = tw + 18 * s, 24 * s
    bx, by = x + w - bw, ly + logo / 2 - bh / 2
    rounded_rect(img, (bx, by), (bx + bw, by + bh), C_CARD_2, bh / 2)
    draw_text(img, fps_txt, bx + bw / 2, by + bh / 2, fs, C_ACCENT_LIGHT, True,
              align="center", valign="middle")

    tx = x + logo + 12 * s
    tmax = bx - tx - 8 * s
    draw_text(img, "SIGN LANGUAGE", tx, ly + logo * 0.46, 0.56 * s, C_TEXT, True, max_w=tmax)
    draw_text(img, "Hand Tracking Recognition", tx, ly + logo * 0.92, 0.37 * s, C_MUTED, max_w=tmax)
    cv2.line(img, (I(x), I(y + 62 * s)), (I(x + w), I(y + 62 * s)), C_BORDER, 1, cv2.LINE_AA)


def sec_result(img, x, y, w, s, st, ui):
    section_title(img, "KET QUA NHAN DIEN", x, y, w, s)
    cy, ch = y + 26 * s, 106 * s
    rounded_rect(img, (x, cy), (x + w, cy + ch), C_CARD, 12 * s)

    box = ch - 24 * s
    bx, by = x + 12 * s, cy + 12 * s
    rounded_rect(img, (bx, by), (bx + box, by + box), C_CARD_2, 10 * s)
    waiting = st["label"] == "Waiting..."
    big = "?" if waiting else str(st["label"])
    draw_text(img, big, bx + box / 2, by + box / 2, 1.7 * s, C_DIM if waiting else C_ACCENT_LIGHT,
              True, max_w=box - 14 * s, align="center", valign="middle", shrink=0.3)

    rx = bx + box + 14 * s
    rw = x + w - 14 * s - rx
    conf, thr = st["conf"], st["threshold"]
    pct = f"{conf * 100:.0f}%"
    pw = draw_text(img, pct, x + w - 14 * s, cy + 30 * s, 0.5 * s, C_TEXT, True, align="right")
    draw_text(img, "Do tin cay", rx, cy + 30 * s, 0.4 * s, C_MUTED, max_w=rw - pw - 8 * s)
    ok = conf >= thr
    progress_bar(img, rx, cy + 42 * s, rw, 8 * s, conf, C_ACCENT_LIGHT if ok else C_AMBER)
    if waiting:
        status, color = "Dang cho ban tay...", C_DIM
    elif ok:
        status, color = "Dat nguong - dang go chu", C_ACCENT_LIGHT
    else:
        status, color = "Chua dat nguong", C_AMBER
    draw_text(img, status, rx, cy + 70 * s, 0.38 * s, color, max_w=rw)
    draw_text(img, f"Nguong go chu: {int(thr * 100)}%", rx, cy + 90 * s, 0.34 * s, C_DIM, max_w=rw)


def sec_mode(img, x, y, w, s, st, ui):
    tw = section_title(img, "CHE DO", x, y, w, s)
    draw_text(img, st["mode_text"][st["mode"]], x + tw + 10 * s, y + 16 * s, 0.4 * s, C_AMBER,
              max_w=w - tw - 10 * s)

    cy, chh, gap = y + 26 * s, 50 * s, 8 * s
    cw = (w - 2 * gap) / 3
    for i, (code, mode, name, sub) in enumerate(MODE_CHIPS):
        cx = x + i * (cw + gap)
        rect = (I(cx), I(cy), I(cx + cw), I(cy + chh))
        active = st["mode"] == mode
        hover = ui.hover(rect)
        fill = C_ACCENT if active else (C_HOVER if hover else C_CARD)
        rounded_rect(img, rect[:2], rect[2:], fill, 10 * s)
        if not active:
            rounded_rect(img, rect[:2], rect[2:], C_ACCENT_LIGHT if hover else C_BORDER, 10 * s, 1)

        kb = 22 * s
        kx, ky = cx + 8 * s, cy + chh / 2 - kb / 2
        rounded_rect(img, (kx, ky), (kx + kb, ky + kb), C_ACCENT_DARK if active else C_CARD_2, 6 * s)
        draw_text(img, chr(code), kx + kb / 2, ky + kb / 2, 0.42 * s, C_TEXT, True,
                  align="center", valign="middle")
        tx = kx + kb + 7 * s
        tmax = cx + cw - tx - 6 * s
        draw_text(img, name, tx, cy + chh * 0.47, 0.4 * s, C_TEXT if active else C_MUTED, True, max_w=tmax)
        draw_text(img, sub, tx, cy + chh * 0.8, 0.32 * s, C_TEXT if active else C_DIM, max_w=tmax)
        ui.add_button(rect, code)

    draw_text(img, "Nhan nut hoac go phim 1 / 2 / 3", x, cy + chh + 22 * s, 0.35 * s, C_DIM, max_w=w)


def sec_controls(img, x, y, w, s, st, ui):
    tw = section_title(img, "DIEU KHIEN", x, y, w, s)
    draw_text(img, "nhan nut / go phim", x + w, y + 16 * s, 0.34 * s, C_DIM, align="right",
              max_w=w - tw - 12 * s)

    ry, pitch, rh = y + 26 * s, 41 * s, 35 * s
    for code, cap_label, desc in CONTROLS:
        rect = (I(x), I(ry), I(x + w), I(ry + rh))
        clickable = code != "hold"
        hover = clickable and ui.hover(rect)
        flash = st["flash"] == code or (code == ord("q") and st["flash"] == KEY_ESC)
        is_quit = code == ord("q")

        fill = C_ACCENT_DARK if flash else (C_HOVER if hover else C_CARD)
        rounded_rect(img, rect[:2], rect[2:], fill, 9 * s)
        if hover:
            rounded_rect(img, rect[:2], rect[2:], C_RED if is_quit else C_ACCENT_LIGHT, 9 * s, 1)

        kw, kh = min(104 * s, w * 0.4), 25 * s
        kx, ky = x + 5 * s, ry + (rh - kh) / 2
        if clickable:
            rounded_rect(img, (kx, ky + 2 * s), (kx + kw, ky + kh + 2 * s), (18, 16, 14), 6 * s)
            rounded_rect(img, (kx, ky), (kx + kw, ky + kh),
                         C_ACCENT if flash else C_CARD_2, 6 * s)
            rounded_rect(img, (kx, ky), (kx + kw, ky + kh),
                         C_ACCENT_LIGHT if (flash or hover) else C_BORDER, 6 * s, 1)
        else:
            # Thao tác bằng cử chỉ tay, không phải phím: vẽ dạng viền
            rounded_rect(img, (kx, ky), (kx + kw, ky + kh), C_ACCENT_LIGHT if flash else C_ACCENT, 6 * s, 1)
        draw_text(img, cap_label, kx + kw / 2, ky + kh / 2, 0.38 * s,
                  C_TEXT if clickable else C_ACCENT_LIGHT, True,
                  max_w=kw - 10 * s, align="center", valign="middle", shrink=0.65)

        dx = kx + kw + 11 * s
        color = C_ACCENT_LIGHT if flash else (C_RED if is_quit else C_TEXT)
        draw_text(img, desc, dx, ry + rh / 2, 0.4 * s, color, max_w=x + w - 8 * s - dx, valign="middle")
        if clickable:
            ui.add_button(rect, code)
        ry += pitch


def sec_footer(img, x, y, w, s, st, ui):
    cv2.line(img, (I(x), I(y)), (I(x + w), I(y)), C_BORDER, 1, cv2.LINE_AA)
    speaking = st["speaking"]
    cv2.circle(img, (I(x + 5 * s), I(y + 22 * s)), max(2, I(5 * s)),
               C_ACCENT_LIGHT if speaking else C_DIM, -1, cv2.LINE_AA)
    text = "Loa: DANG DOC CAU..." if speaking else "Loa san sang - nhan ENTER de doc"
    draw_text(img, text, x + 18 * s, y + 22 * s, 0.38 * s,
              C_ACCENT_LIGHT if speaking else C_MUTED, max_w=w - 18 * s, valign="middle")


SECTIONS = {
    "header": (70, sec_header),
    "result": (142, sec_result),
    "mode": (124, sec_mode),
    "controls": (26 + 41 * len(CONTROLS), sec_controls),
}


def stack_scale(w, h, names, footer):
    base = sum(SECTIONS[n][0] for n in names) + (FOOTER_H if footer else 0)
    return min(w / BASE_W, h / base)


def draw_stack(img, x, y, w, h, names, footer, s, st, ui):
    """Xếp các khối theo chiều dọc với hệ số s đảm bảo vừa khung, không chồng lên nhau."""
    base = sum(SECTIONS[n][0] for n in names) + (FOOTER_H if footer else 0)
    cw = min(w, BASE_W * s * 1.5)
    cx = x + (w - cw) / 2
    n_gaps = len(names) - 1 + (1 if footer else 0)
    gap = min(max(0.0, h - base * s) / n_gaps, 22 * s) if n_gaps else 0
    cy = y
    for n in names:
        hh, fn = SECTIONS[n]
        fn(img, cx, cy, cw, s, st, ui)
        cy += hh * s + gap
    if footer:
        sec_footer(img, cx, y + h - FOOTER_H * s, cw, s, st, ui)


# ---------- Lớp phủ trên video ----------
def draw_hand_box(frame, x_min, y_min, x_max, y_max, label, conf, threshold):
    h, w = frame.shape[:2]
    color = C_ACCENT_LIGHT if conf >= threshold else C_AMBER
    L = max(14, min(x_max - x_min, y_max - y_min) // 5)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 1, cv2.LINE_AA)
    for (cx, cy, dx, dy) in ((x_min, y_min, 1, 1), (x_max, y_min, -1, 1),
                             (x_min, y_max, 1, -1), (x_max, y_max, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), color, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), color, 4, cv2.LINE_AA)

    tag = f"{label}  {conf * 100:.1f}%"
    scale = max(0.45, w / 1100)
    tw, th = measure(tag, scale, True)
    pad = th * 0.55
    bw, bh = tw + pad * 2, th + pad * 2
    tx = min(max(0, x_min), w - bw - 2)          # không tràn ra mép phải
    ty = y_min - bh - 6
    if ty < 2:                                   # không tràn lên mép trên
        ty = y_min + 6
    rounded_rect(frame, (tx, ty), (tx + bw, ty + bh), color, bh * 0.3)
    draw_text(frame, tag, tx + pad, ty + bh / 2, scale, (20, 20, 20), True, valign="middle")


def draw_live_chip(img, dx, dy, vs, st):
    text = st["mode_text"][st["mode"]]
    fs = 0.38 * vs
    tw, _ = measure(text, fs, True)
    h = 26 * vs
    x, y = dx + 14 * vs, dy + 14 * vs
    w = tw + 40 * vs
    glass_rect(img, (x, y), (x + w, y + h), C_BG, 0.78, h / 2)
    cv2.circle(img, (I(x + 13 * vs), I(y + h / 2)), max(2, I(4 * vs)), (80, 80, 255), -1, cv2.LINE_AA)
    draw_text(img, text, x + 24 * vs, y + h / 2, fs, C_TEXT, True, valign="middle")
    return w


def draw_hold_bar(img, rect, vs, st):
    dx, dy, dw, dh = rect
    bw = min(dw - 28 * vs, 380 * vs)
    bh = 64 * vs
    x, y = dx + (dw - bw) / 2, dy + 14 * vs
    if dw > 860 * vs:
        x = dx + dw - bw - 14 * vs   # màn rộng: đặt góc phải để chừa chỗ cho chip chế độ
    glass_rect(img, (x, y), (x + bw, y + bh), C_BG, 0.8, 12 * vs)
    rounded_rect(img, (x, y), (x + bw, y + bh), C_BORDER, 12 * vs, 1)

    px, inner = x + 16 * vs, bw - 32 * vs
    t_y, s_y = y + 25 * vs, y + 49 * vs
    if st["cooldown"] > 0:
        draw_text(img, "DA GO THANH CONG!", px, t_y, 0.5 * vs, C_ACCENT_LIGHT, True, max_w=inner * 0.62)
        progress_bar(img, x + bw - 16 * vs - inner * 0.3, y + 17 * vs, inner * 0.3, 8 * vs, 1.0, C_ACCENT_LIGHT)
        draw_text(img, "Doi chut roi lam ky hieu tiep theo", px, s_y, 0.4 * vs, C_MUTED, max_w=inner)
    elif not st["hand_visible"]:
        draw_text(img, "DUA TAY VAO KHUNG HINH", px, t_y, 0.5 * vs, C_TEXT, True, max_w=inner)
        draw_text(img, "Giu yen ky hieu ~0.5s de go chu", px, s_y, 0.4 * vs, C_MUTED, max_w=inner)
    else:
        progress = min(1.0, st["hold"] / st["required"])
        pct = f"{int(progress * 100)}%"
        pw = draw_text(img, pct, x + bw - 16 * vs, t_y, 0.46 * vs, C_TEXT, True, align="right")
        cand = st["candidate"] if st["candidate"] else "-"
        draw_text(img, f"GIU TAY DE GO:  {cand}", px, t_y, 0.5 * vs, C_AMBER, True,
                  max_w=inner - pw - 10 * vs)
        progress_bar(img, px, y + 38 * vs, inner, 11 * vs, progress, C_AMBER)


def draw_sentence_box(img, rect, vs, st):
    dx, dy, dw, dh = rect
    m, bh = 14 * vs, 90 * vs
    x1, x2 = dx + m, dx + dw - m
    y2 = dy + dh - m
    y1 = y2 - bh
    speaking = st["speaking"]
    glass_rect(img, (x1, y1), (x2, y2), C_BG, 0.85, 12 * vs)
    rounded_rect(img, (x1, y1), (x2, y2), C_ACCENT_LIGHT if speaking else C_AMBER, 12 * vs, 2)

    px = x1 + 18 * vs
    right = x2 - 18 * vs
    tw = draw_text(img, "CAU DA GHEP", px, y1 + 26 * vs, 0.4 * vs, C_MUTED, True)
    avail = right - (px + tw + 20 * vs)
    if speaking:
        badge = "DANG DOC..."
        fs = 0.38 * vs
        bw = measure(badge, fs, True)[0] + 20 * vs
        if bw <= avail:
            rounded_rect(img, (right - bw, y1 + 12 * vs), (right, y1 + 34 * vs), C_ACCENT, 11 * vs)
            draw_text(img, badge, right - bw / 2, y1 + 23 * vs, fs, C_TEXT, True,
                      align="center", valign="middle")
    elif avail > 60 * vs:
        draw_text(img, "ENTER: doc   |   BACKSPACE: xoa", right, y1 + 26 * vs, 0.36 * vs, C_DIM,
                  align="right", max_w=avail)

    sentence = st["sentence"]
    base_y = y2 - 24 * vs
    if sentence:
        stw = draw_text(img, sentence, px, base_y, 0.9 * vs, C_TEXT, True,
                        max_w=right - px - 14 * vs, tail=True, shrink=0.8)
        if int(time.time() * 2) % 2 == 0:
            cx = px + stw + 6 * vs
            cv2.line(img, (I(cx), I(base_y - 26 * vs)), (I(cx), I(base_y + 4 * vs)),
                     C_AMBER, max(1, I(2 * vs)), cv2.LINE_AA)
    else:
        draw_text(img, "Giu tay truoc camera de ghep tung chu cai...", px, base_y, 0.6 * vs, C_DIM,
                  max_w=right - px)


# ---------- Ghép toàn bộ màn hình theo kích thước cửa sổ ----------
def get_window_size(default=DEFAULT_WIN_SIZE):
    try:
        _, _, w, h = cv2.getWindowImageRect(WIN_NAME)
        if w > 50 and h > 50:
            return w, h
    except (cv2.error, AttributeError):
        pass
    return default


def render(frame, win_size, st, ui):
    W, H = win_size
    k = max(1.0, MIN_CANVAS[0] / W, MIN_CANVAS[1] / H)   # giữ tỉ lệ khung khi cửa sổ quá nhỏ
    W, H = I(W * k), I(H * k)
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = C_VIDEO_BG
    ui.begin()

    landscape = W >= H * 1.1
    if landscape:
        pw = min(I(min(max(W * 0.3, 290), 500)), W // 2)
        video, panel = (0, 0, W - pw, H), (W - pw, 0, pw, H)
    else:
        ph = min(I(min(max(H * 0.42, 250), 560)), H // 2)
        video, panel = (0, 0, W, H - ph), (0, H - ph, W, ph)

    # Video: giữ đúng tỉ lệ camera, căn giữa vùng hiển thị
    vx, vy, vw, vh = video
    fh, fw = frame.shape[:2]
    sc = min(vw / fw, vh / fh)
    dw, dh = max(1, I(fw * sc)), max(1, I(fh * sc))
    dx, dy = vx + (vw - dw) // 2, vy + (vh - dh) // 2
    canvas[dy:dy + dh, dx:dx + dw] = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)

    vs = min(max(min(dw / 760, dh / 540), 0.5), 1.8)
    drect = (dx, dy, dw, dh)
    if dw > 860 * vs:
        draw_live_chip(canvas, dx, dy, vs, st)
    draw_hold_bar(canvas, drect, vs, st)
    draw_sentence_box(canvas, drect, vs, st)

    # Bảng điều khiển
    px, py, pw_, ph_ = panel
    canvas[py:py + ph_, px:px + pw_] = C_BG
    pad = I(min(max(min(pw_, ph_) * 0.06, 12), 36))
    if landscape:
        cv2.line(canvas, (px, 0), (px, H), C_BORDER, 1)
        iw, ih = pw_ - 2 * pad, ph_ - 2 * pad
        names = ["header", "result", "mode", "controls"]
        s = stack_scale(iw, ih, names, True)
        draw_stack(canvas, px + pad, py + pad, iw, ih, names, True, s, st, ui)
    else:
        cv2.line(canvas, (0, py), (W, py), C_BORDER, 1)
        col_w, ih = (pw_ - 3 * pad) / 2, ph_ - 2 * pad
        left, right = ["header", "result", "mode"], ["controls"]
        s = min(stack_scale(col_w, ih, left, False), stack_scale(col_w, ih, right, True))
        draw_stack(canvas, px + pad, py + pad, col_w, ih, left, False, s, st, ui)
        draw_stack(canvas, px + 2 * pad + col_w, py + pad, col_w, ih, right, True, s, st, ui)

    ui.end()
    return canvas

# ================= CHƯƠNG TRÌNH WEBCAM =================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("LỖI: Không thể mở Webcam!")
        return

    current_mode = "all"
    mode_text = {"all": "34 Ky tu (ALL)", "digit": "So (0-9)", "letter": "Chu (A-Y)"}
    prev_time = 0

    # CÁC BIẾN GHÉP CÂU (LƯU TRỮ LIÊN TỤC KHÔNG BỊ XÓA)
    sentence = ""
    candidate_char = ""
    hold_counter = 0
    REQUIRED_FRAMES = 15       # Giữ tay ~0.5 giây để gõ
    CONFIDENCE_THRESHOLD = 0.75
    cooldown_counter = 0
    COOLDOWN_FRAMES = 20       # Nghỉ ~0.6 giây sau khi gõ

    # Chỉ dùng cho giao diện: cửa sổ co giãn, chuột, hiệu ứng sáng nút
    ui = UIState()
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, *DEFAULT_WIN_SIZE)
    cv2.setMouseCallback(WIN_NAME, ui.on_mouse)
    flash_code = None
    flash_until = 0.0

    print("\n================ HƯỚNG DẪN ĐIỀU KHIỂN ================")
    print(" [GIỮ TAY ~0.5s] : Gõ chữ nối tiếp vào câu")
    print(" [SPACE]         : Thêm dấu cách (khoảng trắng)")
    print(" [BACKSPACE]     : XÓA 1 CHỮ VỪA GÕ (DUY NHẤT CÁCH NÀY MỚI XÓA)")
    print(" [ENTER]         : Đọc to toàn bộ câu vừa ghép")
    print(" [1, 2, 3]       : Chuyển chế độ ALL / SỐ / CHỮ")
    print(" [Q / ESC]       : Thoát ứng dụng")
    print(" (Có thể nhấn chuột vào các nút trên màn hình thay cho phím)")
    print("=======================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        label_display = "Waiting..."
        conf_display = 0.0
        prev_sentence_len = len(sentence)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                try:
                    feat = landmarks_to_features([hand_landmarks], image_width=w, image_height=h)
                    feat = np.expand_dims(feat, axis=0)

                    raw_probs = session.run([OUTPUT_NAME], {INPUT_NAME: feat})[0][0]
                    prob_buffer.append(raw_probs)

                    smoothed_probs = np.mean(prob_buffer, axis=0)
                    label_display, conf_display = predict_mode(smoothed_probs, current_mode)

                    # LOGIC NỐI TỪ VÀO CÂU
                    if cooldown_counter > 0:
                        cooldown_counter -= 1
                        hold_counter = 0
                    else:
                        if conf_display >= CONFIDENCE_THRESHOLD:
                            if label_display == candidate_char:
                                hold_counter += 1
                                if hold_counter >= REQUIRED_FRAMES:
                                    sentence += candidate_char  # NỐI THÊM VÀO CÂU HIỆN CÓ
                                    print(f"-> Đã gõ thêm: '{candidate_char}' ===> Câu hiện tại: '{sentence}'")
                                    hold_counter = 0
                                    cooldown_counter = COOLDOWN_FRAMES
                            else:
                                candidate_char = label_display
                                hold_counter = 1
                        else:
                            if hold_counter > 0:
                                hold_counter -= 1

                    # Bounding Box
                    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                    x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                    y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)

                    draw_hand_box(frame, x_min, y_min, x_max, y_max,
                                  label_display, conf_display, CONFIDENCE_THRESHOLD)

                except Exception:
                    pass
        else:
            prob_buffer.clear()
            candidate_char = ""
            if hold_counter > 0:
                hold_counter -= 1

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # Hiệu ứng sáng dòng "GIU TAY" khi vừa gõ thêm 1 chữ
        if len(sentence) > prev_sentence_len:
            flash_code, flash_until = "hold", curr_time + 0.6
        if curr_time > flash_until:
            flash_code = None

        # ===== VẼ GIAO DIỆN THEO KÍCH THƯỚC CỬA SỔ =====
        ui_state = {
            "fps": fps,
            "mode": current_mode,
            "mode_text": mode_text,
            "label": label_display,
            "conf": conf_display,
            "threshold": CONFIDENCE_THRESHOLD,
            "speaking": is_speaking,
            "flash": flash_code,
            "sentence": sentence,
            "candidate": candidate_char,
            "hold": hold_counter,
            "required": REQUIRED_FRAMES,
            "cooldown": cooldown_counter,
            "hand_visible": bool(results.multi_hand_landmarks),
        }
        canvas = render(frame, get_window_size(), ui_state, ui)
        cv2.imshow(WIN_NAME, canvas)

        # BẮT PHÍM ĐIỀU KHIỂN (ĐÃ FIX: TUYỆT ĐỐI KHÔNG CÓ LỆNH NÀO TỰ XÓA)
        key = cv2.waitKey(1) & 0xFF
        clicked = ui.pop_click()
        if key == 255 and clicked is not None:
            key = clicked   # Nhấn nút trên màn hình = bấm phím tương ứng
        if key != 255:
            flash_code, flash_until = key, time.time() + 0.25

        # Đóng cửa sổ bằng nút X cũng được coi như thoát
        try:
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if key in (ord("q"), 27):
            break
        elif key == 32:  # PHÍM SPACE: Thêm khoảng trắng
            sentence += " "
            print(f"-> [Thêm khoảng cách] ===> Câu: '{sentence}'")
        elif key == 8:   # PHÍM BACKSPACE: DUY NHẤT PHÍM NÀY MỚI ĐƯỢC XÓA
            if len(sentence) > 0:
                sentence = sentence[:-1]
                print(f"-> [Xóa 1 ký tự] ===> Câu còn lại: '{sentence}'")
        elif key == 13:  # PHÍM ENTER: Đọc to toàn bộ câu vừa ghép
            if sentence.strip() and not is_speaking:
                print(f"==================================================")
                print(f"[LOA PHÁT TOÀN BỘ CÂU]: '{sentence}'")
                print(f"==================================================")
                speak_async(sentence)
        elif key == ord("1"):
            current_mode = "all"
            prob_buffer.clear()
        elif key == ord("2"):
            current_mode = "digit"
            prob_buffer.clear()
        elif key == ord("3"):
            current_mode = "letter"
            prob_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
