import cv2
import os
import time
import json
import subprocess
import sys
import signal
from datetime import datetime, time as dt_time

# main.py nam trong standalone/, con package imou_ai/ nam o thu muc goc
# du an (1 cap tren) - can them thu muc goc vao sys.path TRUOC khi
# import imou_ai.*, neu khong Python se bao "ModuleNotFoundError:
# No module named 'imou_ai'" khi chay truc tiep "python standalone/main.py".
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from imou_ai.face.face_engine import FaceEngine
from imou_ai.detection.frame_grabber import FrameGrabber
from imou_ai.core.zone_utils import box_in_zone
import imou_ai.core.events as events_log
import imou_ai.plate.plate_engine_yolo_anpr as plate_engine

# ==================================================
# LOGGING WITH TIMESTAMP
# ==================================================

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level}] {msg}")

# ==================================================
# CONFIG
# ==================================================

# QUAN TRONG: BASE phai la THU MUC GOC DU AN (noi co config.json,
# models/, data/, assets/), KHONG PHAI thu muc chua main.py (standalone/)
# - vi tat ca duong dan trong config.json (vd "models/yolo11n.pt",
# "data/detections") deu tinh tuong doi theo thu muc goc.
BASE = PROJECT_ROOT

with open(os.path.join(BASE, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

cam = cfg["camera"]
yolo_cfg = cfg["yolo"]
alert = cfg["alert"]
detect = cfg["detection"]
display = cfg["display"]
face_cfg = cfg.get("face", {"enabled": False})

# ---- Vung canh bao (ROI) ----
ZONE_ENABLED = alert.get("zone_enabled", False)
ZONE_POINTS = alert.get("zone_points", []) if ZONE_ENABLED else []

# ---- Loai su kien duoc phep kich hoat canh bao ----
TRIGGER_PERSON = alert.get("trigger_person", True)
TRIGGER_VEHICLE = alert.get("trigger_vehicle", False)

# ==================================================
# RTSP
# ==================================================

RTSP_URL = (
    f'rtsp://{cam["username"]}:{cam["password"]}'
    f'@{cam["ip"]}:{cam["rtsp_port"]}'
    f'/cam/realmonitor?channel=1&subtype={cam["rtsp_subtype"]}'
)

# ==================================================
# YOLO
# ==================================================

log("Loading YOLO...")
model = YOLO(yolo_cfg["model"])
log("YOLO loaded")

# ==================================================
# FACE ENGINE
# ==================================================

FACE_ENABLED = face_cfg.get("enabled", False)
face_engine = None

if FACE_ENABLED:
    log("Loading face recognition engine...")
    try:
        face_engine = FaceEngine(
            known_faces_dir=os.path.join(BASE, face_cfg.get("known_faces_dir", "data/known_faces")),
            db_cache_path=os.path.join(BASE, face_cfg.get("db_cache", "data/face_db.pkl")),
            similarity_threshold=face_cfg.get("similarity_threshold", 0.72),
            margin=face_cfg.get("margin", 0.05),
            top_k=face_cfg.get("top_k", 5),
            min_face_size=face_cfg.get("min_face_size", 40),
            debug=face_cfg.get("debug", False),
        )
        log("Face engine loaded")
    except Exception as e:
        log(f"Khong tai duoc face model: {e}", "ERROR")
        FACE_ENABLED = False
        face_engine = None

FACE_RECOGNIZE_FPS = face_cfg.get("recognize_fps", 2)
FACE_INTERVAL = 1 / FACE_RECOGNIZE_FPS if FACE_RECOGNIZE_FPS > 0 else 0
ALERT_ONLY_UNKNOWN = face_cfg.get("alert_only_unknown", True)
TRIGGER_UNKNOWN_FACE = alert.get("trigger_unknown_face", ALERT_ONLY_UNKNOWN)
SAVE_UNKNOWN_FACES = face_cfg.get("save_unknown_faces", True)
UNKNOWN_SAVE_DIR = os.path.join(BASE, face_cfg.get("unknown_save_directory", "data/unknown_faces"))

# ==================================================
# PLATE (YOLOv8 + EasyOCR - xem plate_engine_yolo_anpr.py)
# ==================================================

plate_cfg = cfg.get("plate", {})
PLATE_ENABLED = plate_cfg.get("enabled", False)
PLATE_SAVE_DIR = os.path.join(BASE, plate_cfg.get("save_directory", "detections/plates"))

plate_engine.configure(
    enabled=PLATE_ENABLED,
    detector_model=os.path.join(BASE, plate_cfg.get("detector_model", "models/license_plate_detector.pt")),
    detector_confidence=plate_cfg.get("detector_confidence", 0.35),
    ocr_min_confidence=plate_cfg.get("ocr_min_confidence", 0.35),
)

# ==================================================
# CAMERA
# ==================================================

log("Connecting to camera...")
grabber = FrameGrabber(RTSP_URL, reconnect_delay=2.0, stale_timeout=8.0)
grabber.start()

_wait_start = time.time()
while grabber.get_latest() is None and time.time() - _wait_start < 8:
    time.sleep(0.2)

if grabber.get_latest() is None:
    log("Cannot connect to camera!", "ERROR")
    raise SystemExit

log("Camera connected")

# ==================================================
# ALERT TIME
# ==================================================

def parse_time(value):
    h, m = map(int, value.split(":"))
    return dt_time(h, m)

ALERT_START = parse_time(alert["start"])
ALERT_END = parse_time(alert["end"])

def in_alert_time():
    now = datetime.now().time()
    if ALERT_START > ALERT_END:
        return now >= ALERT_START or now <= ALERT_END
    return ALERT_START <= now <= ALERT_END

# ==================================================
# SAVE IMAGE
# ==================================================

def save_image(frame):
    if not detect["save_images"]:
        return None
    directory = os.path.join(BASE, detect["save_directory"])
    os.makedirs(directory, exist_ok=True)
    filename = datetime.now().strftime("person_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(directory, filename)
    cv2.imwrite(path, frame)
    log(f"Saved: {path}")
    return os.path.join(detect["save_directory"], filename)

def save_unknown_face(face_box, frame_w, frame_h, low_res_frame):
    if not SAVE_UNKNOWN_FACES:
        return
    os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)
    x1, y1, x2, y2 = face_box
    h, w = low_res_frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    crop = low_res_frame[y1:y2, x1:x2]
    filename = datetime.now().strftime("unknown_%Y%m%d_%H%M%S_%f.jpg")
    path = os.path.join(UNKNOWN_SAVE_DIR, filename)
    cv2.imwrite(path, crop)
    log(f"Saved unknown face: {path}")

def save_plate_image(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return None
    os.makedirs(PLATE_SAVE_DIR, exist_ok=True)
    filename = datetime.now().strftime("plate_%Y%m%d_%H%M%S_%f.jpg")
    path = os.path.join(PLATE_SAVE_DIR, filename)
    cv2.imwrite(path, plate_crop)
    log(f"Saved plate image: {path}")
    return os.path.join(plate_cfg.get("save_directory", "detections/plates"), filename)

# ==================================================
# AUDIO
# ==================================================

audio_process = None

def start_audio():
    global audio_process
    if audio_process is not None and audio_process.poll() is None:
        return False
    log("Starting audio...")
    speaker = os.path.join(BASE, "imou_ai", "audio", "speaker.py")
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW
    popen_kwargs = {"creationflags": creation_flags}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = os.setsid
    audio_process = subprocess.Popen([sys.executable, speaker], **popen_kwargs)
    return True

def stop_audio():
    global audio_process
    if audio_process is None or audio_process.poll() is not None:
        audio_process = None
        return
    log("Stopping audio...")
    try:
        if sys.platform != "win32":
            os.killpg(os.getpgid(audio_process.pid), signal.SIGTERM)
        else:
            audio_process.terminate()
        audio_process.wait(timeout=3)
    except Exception:
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(audio_process.pid), signal.SIGKILL)
            else:
                audio_process.kill()
        except Exception:
            pass
    audio_process = None

# ==================================================
# EVENT TRACKER (CHỐNG SPAM SỰ KIỆN)
# ==================================================

class EventTracker:
    """
    Theo dõi các đối tượng đã ghi sự kiện để tránh spam.
    Mỗi đối tượng được định danh bằng class_id và vị trí bounding box.
    """
    def __init__(self, iou_thresh=0.5, idle_seconds=30):
        self.iou_thresh = iou_thresh
        self.idle_seconds = idle_seconds
        self.tracked = []  # [{class_id, box, last_seen, logged}]

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _prune(self, now):
        self.tracked = [t for t in self.tracked if now - t["last_seen"] <= self.idle_seconds]

    def should_log(self, class_id, box, now):
        """
        Kiểm tra có nên ghi sự kiện cho đối tượng này không.
        Trả về True nếu đối tượng mới hoặc đã rời đi rồi quay lại.
        """
        self._prune(now)

        # Tìm đối tượng trùng
        for t in self.tracked:
            if t["class_id"] == class_id and self._iou(box, t["box"]) >= self.iou_thresh:
                t["box"] = box
                t["last_seen"] = now
                if t["logged"]:
                    return False
                t["logged"] = True
                return True

        # Đối tượng mới
        self.tracked.append({
            "class_id": class_id,
            "box": box,
            "last_seen": now,
            "logged": True,
        })
        return True

event_tracker = EventTracker(iou_thresh=0.5, idle_seconds=30)

# ==================================================
# DETECTION
# ==================================================

person_since = None
last_alert = 0
last_detection = 0
last_trigger_seen = 0
last_event_types = []
last_event_log = 0

RETENTION_DAYS = detect.get("retention_days", 15)
RETENTION_CHECK_INTERVAL = 6 * 3600
MEDIA_RETENTION_DIRS = [
    detect["save_directory"],
    face_cfg.get("unknown_save_directory", "data/unknown_faces"),
    plate_cfg.get("save_directory", "detections/plates"),
]
last_retention_cleanup = 0

last_boxes = []
confirm_seconds = alert.get("confirm_seconds", 3)
GRACE_SECONDS = alert.get("grace_seconds", 1.5)

detect_fps = detect.get("detect_fps", 8)
detect_interval = 1 / detect_fps

last_face_check = 0
last_faces = []

# ==================================================
# CLASS CONFIG
# ==================================================

PERSON_CLASS = yolo_cfg.get("person_class", 0)
VEHICLE_CLASSES = yolo_cfg.get("vehicle_classes", [2, 3, 5, 7])
DETECT_VEHICLES = yolo_cfg.get("detect_vehicles", True)
DETECT_PERSON = yolo_cfg.get("detect_person", True)

detect_classes = []
if DETECT_PERSON:
    detect_classes.append(PERSON_CLASS)
if DETECT_VEHICLES:
    for class_id in VEHICLE_CLASSES:
        if class_id not in detect_classes:
            detect_classes.append(class_id)

log(f"Detect classes: {detect_classes}")

# ==================================================
# MAIN LOOP
# ==================================================

try:
    while True:
        frame = grabber.get_latest(max_age=5)
        if frame is None:
            time.sleep(0.2)
            continue

        now = time.time()

        # ---- Don du lieu cu ----
        if now - last_retention_cleanup >= RETENTION_CHECK_INTERVAL:
            last_retention_cleanup = now
            events_log.prune_old(BASE, MEDIA_RETENTION_DIRS, max_age_days=RETENTION_DAYS)

        # ==================================================
        # YOLO
        # ==================================================
        if now - last_detection >= detect_interval:
            last_detection = now
            results = model.predict(
                frame,
                classes=detect_classes,
                conf=yolo_cfg["confidence"],
                imgsz=yolo_cfg["imgsz"],
                device="cpu",
                verbose=False,
                max_det=yolo_cfg.get("max_det", 10)
            )
            boxes = results[0].boxes
            last_boxes = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1 = int(box.xyxy[0][0])
                    y1 = int(box.xyxy[0][1])
                    x2 = int(box.xyxy[0][2])
                    y2 = int(box.xyxy[0][3])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    last_boxes.append((x1, y1, x2, y2, confidence, class_id))

        # ==================================================
        # FACE RECOGNITION
        # ==================================================
        if FACE_ENABLED and now - last_face_check >= FACE_INTERVAL:
            last_face_check = now
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_faces = face_engine.recognize(rgb_frame)

        # ==================================================
        # LOC THEO VUNG CANH BAO
        # ==================================================
        frame_h, frame_w = frame.shape[:2]
        boxes_in_zone = [
            b for b in last_boxes
            if box_in_zone((b[0], b[1], b[2], b[3]), ZONE_POINTS, frame_w, frame_h)
        ]
        faces_in_zone = [
            f for f in last_faces
            if box_in_zone(f["box"], ZONE_POINTS, frame_w, frame_h)
        ]

        person_found = any(box[5] == PERSON_CLASS and box in boxes_in_zone for box in last_boxes)
        vehicle_found = any(box[5] in VEHICLE_CLASSES and box in boxes_in_zone for box in last_boxes)
        unknown_face_found = any(f["name"] == "Unknown" for f in faces_in_zone)

        event_types_firing = []
        if TRIGGER_PERSON and person_found:
            event_types_firing.append("person")
        if TRIGGER_VEHICLE and vehicle_found:
            event_types_firing.append("vehicle")
        if FACE_ENABLED and TRIGGER_UNKNOWN_FACE and unknown_face_found:
            event_types_firing.append("unknown_face")

        alert_trigger = len(event_types_firing) > 0

        # ==================================================
        # DRAW: vung canh bao
        # ==================================================
        if ZONE_POINTS and len(ZONE_POINTS) >= 3:
            poly_px = [(int(px * frame_w), int(py * frame_h)) for px, py in ZONE_POINTS]
            for i in range(len(poly_px)):
                p1 = poly_px[i]
                p2 = poly_px[(i + 1) % len(poly_px)]
                cv2.line(frame, p1, p2, (0, 200, 255), 2)

        # ==================================================
        # DRAW: boxes
        # ==================================================
        for x1, y1, x2, y2, confidence, class_id in last_boxes:
            in_zone = (x1, y1, x2, y2, confidence, class_id) in boxes_in_zone
            label = model.names.get(class_id, f"CLASS {class_id}").upper()
            if not in_zone:
                box_color = (120, 120, 120)
            elif class_id == PERSON_CLASS:
                box_color = (0, 255, 0)
            else:
                box_color = (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

        # ==================================================
        # DRAW: faces
        # ==================================================
        for face in last_faces:
            fx1, fy1, fx2, fy2 = face["box"]
            name = face["name"]
            similarity = face["similarity"]
            in_zone = face in faces_in_zone
            if not in_zone:
                face_color = (120, 120, 120)
            else:
                face_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            cv2.putText(frame, f"{name} ({similarity:.2f})", (fx1, max(fy1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

        # ==================================================
        # CONFIRM / ALERT (BỎ CONFIRM 3s - KÍCH HOẠT NGAY)
        # ==================================================
        if alert_trigger:
            last_trigger_seen = now
            last_event_types = event_types_firing

        within_grace = (now - last_trigger_seen) <= GRACE_SECONDS
        active = alert_trigger or (person_since is not None and within_grace)

        if active:
            if person_since is None:
                person_since = now
                log("Detection triggered - ALERT ACTIVE")

            # Hiển thị trạng thái (không còn đếm confirm)
            cv2.putText(frame, "ALERT ACTIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # ==================================================
            # GHI SỰ KIỆN NGAY LẬP TỨC (không chờ confirm)
            # ==================================================
            logged_any = False
            plate_text = None
            plate_image_path = None

            # Kiểm tra từng xe/người trong vùng
            for box in boxes_in_zone:
                x1, y1, x2, y2, conf, class_id = box
                if class_id in VEHICLE_CLASSES or class_id == PERSON_CLASS:
                    if event_tracker.should_log(class_id, (x1, y1, x2, y2), now):
                        logged_any = True

                        if class_id in VEHICLE_CLASSES and PLATE_ENABLED:
                            vx1, vy1 = max(0, x1), max(0, y1)
                            vx2, vy2 = min(frame_w, x2), min(frame_h, y2)
                            if vx2 > vx1 and vy2 > vy1:
                                vehicle_crop = frame[vy1:vy2, vx1:vx2]
                                text, plate_crop = plate_engine.read_plate(vehicle_crop)
                                if text:
                                    plate_text = text
                                    log(f"Doc duoc bien so: {text}")
                                if plate_crop is not None:
                                    plate_image_path = save_plate_image(plate_crop)

            # Kiểm tra khuôn mặt lạ
            face_logged = False
            if FACE_ENABLED and "unknown_face" in (last_event_types or []):
                for face in last_faces:
                    if face["name"] == "Unknown":
                        fx1, fy1, fx2, fy2 = face["box"]
                        if event_tracker.should_log(-1, (fx1, fy1, fx2, fy2), now):
                            face_logged = True
                            save_unknown_face(face["box"], frame_w, frame_h, frame)

            # Ghi sự kiện nếu có đối tượng mới
            if logged_any or face_logged:
                # Chỉ ghi nếu chưa ghi gần đây (cooldown)
                if (now - last_event_log) >= alert["cooldown"]:
                    img_path = save_image(frame)
                    for etype in (last_event_types or ["person"]):
                        extra = None
                        if etype == "vehicle" and (plate_text or plate_image_path):
                            extra = {}
                            if plate_text:
                                extra["plate"] = plate_text
                            if plate_image_path:
                                extra["plate_image"] = plate_image_path
                        events_log.log_event(BASE, etype, image_path=img_path, extra=extra)
                    last_event_log = now
                    log(f"Logged event: {last_event_types}")

            # ==================================================
            # CÒI HÚ KÍCH HOẠT NGAY (không chờ confirm)
            # ==================================================
            should_siren = alert["enabled"] and in_alert_time()
            if should_siren:
                siren_running = (audio_process is not None and audio_process.poll() is None)
                if not siren_running:
                    if start_audio():
                        log("Siren ON")
                        last_alert = now
                cv2.putText(frame, "SIREN ON", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                stop_audio()

        else:
            person_since = None
            stop_audio()

        # ==================================================
        # DISPLAY
        # ==================================================
        if display["show_camera"]:
            cv2.imshow(display["window_name"], frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    stop_audio()
    grabber.stop()
    cv2.destroyAllWindows()
    log("Main loop stopped")