import cv2
import numpy as np
import math
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import mediapipe as mp
from objects import DrawingStroke

# ─── MediaPipe Tasks Setup ─────────────────────────────────────
BaseOptions = mp_python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
HandLandmarkerResult = vision.HandLandmarkerResult
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = HandLandmarker.create_from_options(options)

# ─── Canvas & Drawing ──────────────────────────────────────────
canvas = None
drawing_strokes = []  # List of DrawingStroke
current_stroke = None  # Stroke yang sedang digambar

grabbed_object = None

# ─── Helpers ───────────────────────────────────────────────────
def lm_pos(lm_list, idx, w, h):
    lm = lm_list[idx]
    return int(lm.x * w), int(lm.y * h)

def fingers_up(lm_list, w, h):
    f = []
    tx, ty = lm_pos(lm_list, 4, w, h)
    bx, by = lm_pos(lm_list, 3, w, h)
    wx, _  = lm_pos(lm_list, 0, w, h)
    f.append(1 if (tx < bx if wx < tx else tx > bx) else 0)
    for tip in [8, 12, 16, 20]:
        tip_y = lm_pos(lm_list, tip, w, h)[1]
        pip_y = lm_pos(lm_list, tip - 2, w, h)[1]
        f.append(1 if tip_y < pip_y else 0)
    return f

def pinch_dist(lm_list, w, h):
    tx, ty = lm_pos(lm_list, 4, w, h)
    ix, iy = lm_pos(lm_list, 8, w, h)
    return math.hypot(ix - tx, iy - ty), ((tx + ix) // 2, (ty + iy) // 2)

def detect_gesture(lm_list, w, h):
    f = fingers_up(lm_list, w, h)
    dist, mid = pinch_dist(lm_list, w, h)
    
    # Cek DRAW terlebih dahulu: hanya telunjuk yang terbuka
    if f == [0, 1, 0, 0, 0]:
        return "DRAW", lm_pos(lm_list, 8, w, h)
    
    # Cek PINCH: jempol & telunjuk dekat
    if dist < 40:
        return "PINCH", mid
    
    if f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0:
        return "SELECT", lm_pos(lm_list, 8, w, h)
    return "NONE", mid

def draw_landmarks(frame, lm_list, w, h):
    """Gambar titik & koneksi tangan secara manual"""
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [lm_pos(lm_list, i, w, h) for i in range(21)]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 180, 130), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 200), -1)

# ─── Main Loop ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("✋ Hand Gesture App started. Tekan 'C' untuk clear canvas, 'Q' untuk quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    if canvas is None:
        canvas = np.zeros_like(frame)

    # ── Deteksi tangan ───────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection_result = detector.detect(mp_image)

    all_landmarks = detection_result.hand_landmarks  # list of hands
    gesture_info = []

    for lm_list in all_landmarks:
        draw_landmarks(frame, lm_list, w, h)
        gesture, pos = detect_gesture(lm_list, w, h)
        gesture_info.append((gesture, pos, lm_list))

    # ── 1 Tangan ─────────────────────────────────────────────
    if len(gesture_info) == 1:
        gesture, pos, lm_list = gesture_info[0]
        px, py = pos

        if gesture == "DRAW":
            if current_stroke is None:
                current_stroke = DrawingStroke([(px, py)])
            else:
                current_stroke.points.append((px, py))
        else:
            # Simpan stroke jika selesai digambar
            if current_stroke is not None and len(current_stroke.points) >= 2:
                drawing_strokes.append(current_stroke)
            current_stroke = None

        if gesture == "PINCH":
            if grabbed_object is None:
                # Cek apakah grab stroke
                grabbed_stroke = None
                for stroke in drawing_strokes:
                    if stroke.is_point_inside(px, py):
                        grabbed_stroke = stroke
                        break
                
                if grabbed_stroke:
                    grabbed_object = grabbed_stroke
                    grabbed_stroke.grabbed = True
                    grabbed_stroke.offset = (px - grabbed_stroke.points[0][0], 
                                           py - grabbed_stroke.points[0][1])
            else:
                grabbed_object.move_to(px, py)
        else:
            if grabbed_object:
                grabbed_object.grabbed = False
                grabbed_object = None

    # ── 2 Tangan ─────────────────────────────────────────────
    elif len(gesture_info) == 2:
        if current_stroke is not None and len(current_stroke.points) >= 2:
            drawing_strokes.append(current_stroke)
        current_stroke = None
    else:
        if current_stroke is not None and len(current_stroke.points) >= 2:
            drawing_strokes.append(current_stroke)
        current_stroke = None

    # ── Gabungkan canvas ─────────────────────────────────────
    frame = cv2.addWeighted(frame, 1.0, canvas, 0.7, 0)

    # ── Gambar garis yang sudah tersimpan ────────────────────
    for stroke in drawing_strokes:
        stroke.draw(frame)
    
    # ── Gambar garis yang sedang digambar ────────────────────
    if current_stroke is not None:
        current_stroke.draw(frame)

    # ── HUD ──────────────────────────────────────────────────
    for i, (text, color) in enumerate([
        ("Index finger = Draw",    (0, 200, 255)),
        ("Thumb+Index = Move",     (0, 255, 150)),
    ]):
        cv2.putText(frame, text, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, "C=Clear  Q=Quit", (w - 180, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Hand Gesture Interaction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
        drawing_strokes.clear()
        current_stroke = None

cap.release()
cv2.destroyAllWindows()
