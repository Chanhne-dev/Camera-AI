"""
web_app.py - Web server cho IMOU AI
Có EventTracker chống ghi sự kiện lặp cho cùng đối tượng
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, time as dt_time

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from ultralytics import YOLO

from imou_ai.face.face_engine import FaceEngine
from imou_ai.detection.frame_grabber import FrameGrabber
from imou_ai.core.zone_utils import box_in_zone
from imou_ai.core import events as events_log
from imou_ai.plate import plate_engine_yolo_anpr as plate_engine

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

# ==================================================
# LOGGING WITH TIMESTAMP
# ==================================================

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ts}] [{level}] {msg}")

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE, "config.json")
MODEL_DOWNLOAD_LOCK_FILE = os.path.join(BASE, ".model_download.lock")
PLAYBACK_DIR = os.path.join(BASE, "data", "playback_downloads")

@contextlib.contextmanager
def model_download_lock(timeout=None):
    if not HAS_FCNTL:
        yield
        return
    with open(MODEL_DOWNLOAD_LOCK_FILE, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

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

# ==================================================
# MOTION TRACKER (PHAN BIET VAT TINH VOI NGUOI/VAT DI CHUYEN)
# ==================================================

class MotionTracker:
    """
    Theo doi lich su vi tri cua tung doi tuong theo thoi gian de tinh
    do dich chuyen thuc te cua no.

    Dung de loc cac truong hop nhan dien nham vat tinh (VD: ao/quan
    vat tren xe may dung yen trong dieu kien thieu sang) thanh "nguoi":
    vat that su di chuyen se co do dich chuyen tam (centroid) lon theo
    thoi gian, trong khi vat tinh gan nhu khong doi vi tri.
    """
    def __init__(self, iou_thresh=0.3, window_seconds=5, max_idle=3):
        self.iou_thresh = iou_thresh
        self.window_seconds = window_seconds
        self.max_idle = max_idle
        self.tracks = []  # [{class_id, box, first_seen, last_seen, history:[(t,cx,cy)]}]

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
        self.tracks = [t for t in self.tracks if now - t["last_seen"] <= self.max_idle]

    def update(self, class_id, box, now):
        """
        Cap nhat (hoac khoi tao) track khop voi box nay.
        Tra ve (track_age_seconds, max_displacement_px) trong cua so
        window_seconds gan nhat.
        """
        self._prune(now)
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        matched = None
        for t in self.tracks:
            if t["class_id"] == class_id and self._iou(box, t["box"]) >= self.iou_thresh:
                matched = t
                break

        if matched is None:
            matched = {
                "class_id": class_id,
                "box": box,
                "first_seen": now,
                "last_seen": now,
                "history": [(now, cx, cy)],
            }
            self.tracks.append(matched)
        else:
            matched["box"] = box
            matched["last_seen"] = now
            matched["history"].append((now, cx, cy))
            matched["history"] = [h for h in matched["history"] if now - h[0] <= self.window_seconds]

        age = now - matched["first_seen"]
        xs = [h[1] for h in matched["history"]]
        ys = [h[2] for h in matched["history"]]
        max_disp = 0.0
        if len(xs) >= 2:
            x0, y0 = xs[0], ys[0]
            max_disp = max(((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 for x, y in zip(xs, ys))
        return age, max_disp

# ==================================================
# DETECTOR
# ==================================================

class Detector:
    def __init__(self):
        self.thread = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self.latest_jpeg = None
        self.status = "STOPPED"
        self.last_error = None
        self.warning = None
        self.audio_process = None
        self.last_faces_info = []
        self.siren_on = False
        self.last_frame_time = 0
        self.event_tracker = EventTracker(iou_thresh=0.5, idle_seconds=30)
        self.motion_tracker = MotionTracker(iou_thresh=0.3, window_seconds=5, max_idle=3)

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        with self.lifecycle_lock:
            if self.is_running():
                log("Da dang chay - bo qua Start trung lap", "WARN")
                return
            self.stop_flag.clear()
            self.last_error = None
            self.warning = None
            self.status = "STARTING"
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            log("Detector thread started")

    def stop(self):
        with self.lifecycle_lock:
            self.stop_flag.set()
            log("Stopping detector...")
            if self.thread is not None:
                self.thread.join(timeout=10)
                if self.thread.is_alive():
                    log("Thread chua dung han (co the dang tai model)", "WARN")
            self._stop_audio()
            if not (self.thread is not None and self.thread.is_alive()):
                self.status = "STOPPED"
                log("Detector stopped")

    def restart(self):
        with self.lifecycle_lock:
            log("Restarting detector...")
            self.stop()
            time.sleep(0.3)
            self.start()

    # ==================================================
    # AUDIO
    # ==================================================

    def _start_audio(self):
        if self.audio_process is not None and self.audio_process.poll() is None:
            return False
        speaker = os.path.join(BASE, "imou_ai", "audio", "speaker.py")
        popen_kwargs = {}
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = os.setsid
        self.audio_process = subprocess.Popen([sys.executable, speaker], **popen_kwargs)
        log("Audio started")
        return True

    def _stop_audio(self):
        if self.audio_process is None or self.audio_process.poll() is not None:
            self.audio_process = None
            self.siren_on = False
            return
        log("Stopping audio...")
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(self.audio_process.pid), 15)
            else:
                self.audio_process.terminate()
            self.audio_process.wait(timeout=3)
        except Exception:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.audio_process.pid), 9)
                else:
                    self.audio_process.kill()
            except Exception:
                pass
        self.audio_process = None
        self.siren_on = False
        log("Audio stopped")

    # ==================================================
    # MAIN LOOP
    # ==================================================

    def _run(self):
        try:
            log("Detector _run started")
            cfg = load_config()
            cam = cfg["camera"]
            yolo_cfg = cfg["yolo"]
            alert = cfg["alert"]
            detect = cfg["detection"]
            face_cfg = cfg.get("face", {"enabled": False})

            zone_enabled = alert.get("zone_enabled", False)
            zone_points = alert.get("zone_points", []) if zone_enabled else []

            trigger_person = alert.get("trigger_person", True)
            trigger_vehicle = alert.get("trigger_vehicle", False)
            trigger_unknown_face = alert.get(
                "trigger_unknown_face",
                face_cfg.get("alert_only_unknown", True)
            )

            rtsp_url = (
                f'rtsp://{cam["username"]}:{cam["password"]}'
                f'@{cam["ip"]}:{cam["rtsp_port"]}'
                f'/cam/realmonitor?channel=1&subtype={cam["rtsp_subtype"]}'
            )

            log("Loading YOLO model...")
            model = YOLO(yolo_cfg["model"])
            log("YOLO model loaded")

            face_enabled = face_cfg.get("enabled", False)
            face_engine = None
            if face_enabled:
                log("Loading face engine...")
                try:
                    with model_download_lock():
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
                    face_enabled = False
                    face_engine = None
                    self.warning = f"Khong tai duoc face model: {e}"
                    log(self.warning, "ERROR")

            alert_only_unknown = face_cfg.get("alert_only_unknown", True)
            save_unknown_faces = face_cfg.get("save_unknown_faces", True)
            unknown_save_dir = os.path.join(BASE, face_cfg.get("unknown_save_directory", "data/unknown_faces"))
            recognize_fps = face_cfg.get("recognize_fps", 2)
            face_interval = 1 / recognize_fps if recognize_fps > 0 else 0

            plate_cfg = cfg.get("plate", {})
            plate_enabled = plate_cfg.get("enabled", False)
            plate_save_dir = os.path.join(BASE, plate_cfg.get("save_directory", "data/detections/plates"))

            plate_engine.configure(
                enabled=plate_enabled,
                detector_model=os.path.join(BASE, plate_cfg.get("detector_model", "models/license_plate_detector.pt")),
                detector_confidence=plate_cfg.get("detector_confidence", 0.35),
                ocr_min_confidence=plate_cfg.get("ocr_min_confidence", 0.35),
            )

            log("Connecting to camera...")
            grabber = FrameGrabber(rtsp_url, reconnect_delay=2.0, stale_timeout=8.0)
            grabber.start()

            wait_start = time.time()
            while grabber.get_latest() is None and time.time() - wait_start < 8:
                if self.stop_flag.is_set():
                    grabber.stop()
                    return
                time.sleep(0.2)

            if grabber.get_latest() is None:
                self.status = "ERROR"
                self.last_error = "Khong ket noi duoc camera"
                log(self.last_error, "ERROR")
                grabber.stop()
                return

            self.status = "RUNNING"
            log("Camera connected, detection running")

            def parse_time(value):
                h, m = map(int, value.split(":"))
                return dt_time(h, m)

            alert_start = parse_time(alert["start"])
            alert_end = parse_time(alert["end"])

            def in_alert_time():
                now_t = datetime.now().time()
                if alert_start > alert_end:
                    return now_t >= alert_start or now_t <= alert_end
                return alert_start <= now_t <= alert_end

            def save_image(frame):
                if not detect["save_images"]:
                    return None
                directory = os.path.join(BASE, detect["save_directory"])
                os.makedirs(directory, exist_ok=True)
                filename = datetime.now().strftime("person_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(os.path.join(directory, filename), frame)
                return os.path.join(detect["save_directory"], filename)

            def save_unknown_face(face_box, frame):
                if not save_unknown_faces:
                    return
                os.makedirs(unknown_save_dir, exist_ok=True)
                x1, y1, x2, y2 = face_box
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    return
                crop = frame[y1:y2, x1:x2]
                filename = datetime.now().strftime("unknown_%Y%m%d_%H%M%S_%f.jpg")
                cv2.imwrite(os.path.join(unknown_save_dir, filename), crop)

            def save_plate_image(plate_crop):
                if plate_crop is None or plate_crop.size == 0:
                    return None
                os.makedirs(plate_save_dir, exist_ok=True)
                filename = datetime.now().strftime("plate_%Y%m%d_%H%M%S_%f.jpg")
                cv2.imwrite(os.path.join(plate_save_dir, filename), plate_crop)
                return os.path.join(plate_cfg.get("save_directory", "data/detections/plates"), filename)

            retention_days = detect.get("retention_days", 15)
            retention_check_interval = 6 * 3600
            media_retention_dirs = [
                detect["save_directory"],
                face_cfg.get("unknown_save_directory", "data/unknown_faces"),
                plate_cfg.get("save_directory", "data/detections/plates"),
            ]
            last_retention_cleanup = 0

            person_class = yolo_cfg.get("person_class", 0)
            vehicle_classes = yolo_cfg.get("vehicle_classes", [2, 3, 5, 7])
            detect_vehicles = yolo_cfg.get("detect_vehicles", True)
            detect_person = yolo_cfg.get("detect_person", True)

            detect_classes = []
            if detect_person:
                detect_classes.append(person_class)
            if detect_vehicles:
                for c in vehicle_classes:
                    if c not in detect_classes:
                        detect_classes.append(c)

            detect_fps = detect.get("detect_fps", 8)
            detect_interval = 1 / detect_fps

            confirm_seconds = alert.get("confirm_seconds", 3)
            grace_seconds = alert.get("grace_seconds", 1.5)

            # ---- Chong nham vat tinh (VD: ao vat tren xe) thanh nguoi khi thieu sang ----
            low_light_brightness_threshold = detect.get("low_light_brightness_threshold", 70)
            low_light_person_min_conf = detect.get("low_light_person_min_confidence", 0.55)
            movement_check_enabled = detect.get("person_movement_check_enabled", True)
            movement_window_seconds = detect.get("person_movement_window_seconds", 5)
            movement_min_pixels = detect.get("person_movement_min_pixels", 18)
            movement_min_track_age = detect.get("person_movement_min_track_age", 1.5)

            # ---- Chong canh bao (coi hu) keu lien tuc khong ngung ----
            siren_max_duration = alert.get("siren_max_duration", 60)
            siren_rearm_gap = alert.get("siren_rearm_gap", 20)

            person_since = None
            last_alert = 0
            last_detection = 0
            last_trigger_seen = 0
            last_face_check = 0
            last_event_log = 0
            siren_started_at = None
            siren_forced_off_at = None

            last_boxes = []
            last_faces = []
            last_event_types = []
            last_static_suspects = set()

            def box_key(b):
                return (b[0], b[1], b[2], b[3], b[5])

            # Tracker cho su kien (chong spam ghi log) va chuyen dong (chong nham vat tinh)
            self.event_tracker = EventTracker(iou_thresh=0.5, idle_seconds=30)
            self.motion_tracker = MotionTracker(
                iou_thresh=0.3, window_seconds=movement_window_seconds, max_idle=3
            )

            while not self.stop_flag.is_set():
                frame = grabber.get_latest(max_age=5)
                if frame is None:
                    if not grabber.is_healthy():
                        self.status = "RECONNECTING"
                    time.sleep(0.2)
                    continue

                if self.status != "RUNNING":
                    self.status = "RUNNING"

                now = time.time()
                self.last_frame_time = now

                if now - last_retention_cleanup >= retention_check_interval:
                    last_retention_cleanup = now
                    events_log.prune_old(BASE, media_retention_dirs, max_age_days=retention_days)

                # Do do sang trung binh khung hinh de phat hien dieu kien thieu sang
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_brightness = float(gray_frame.mean())
                is_low_light = frame_brightness < low_light_brightness_threshold

                if now - last_detection >= detect_interval:
                    last_detection = now
                    try:
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
                    except Exception as e:
                        log(f"YOLO predict error: {e}", "ERROR")

                    # ====== LOC NHAN NHAM VAT TINH LA NGUOI KHI THIEU SANG ======
                    # Truong hop dien hinh: xe may dung yen co ao/quan vat tren xe
                    # bi nhan nham thanh "nguoi". Vat that di chuyen (nguoi luot
                    # qua) se co do dich chuyen tam lon nen khong bi loai.
                    last_static_suspects = set()
                    if movement_check_enabled:
                        for bx1, by1, bx2, by2, bconf, bclass in last_boxes:
                            if bclass != person_class:
                                continue
                            track_age, max_disp = self.motion_tracker.update(
                                bclass, (bx1, by1, bx2, by2), now
                            )
                            low_conf_in_dark = is_low_light and bconf < low_light_person_min_conf
                            stationary_in_dark = (
                                is_low_light
                                and track_age >= movement_min_track_age
                                and max_disp < movement_min_pixels
                            )
                            if low_conf_in_dark or stationary_in_dark:
                                last_static_suspects.add(box_key((bx1, by1, bx2, by2, bconf, bclass)))

                if face_enabled and now - last_face_check >= face_interval:
                    last_face_check = now
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        last_faces = face_engine.recognize(rgb_frame)
                    except Exception as e:
                        log(f"Face recognition error: {e}", "ERROR")

                frame_h, frame_w = frame.shape[:2]
                boxes_in_zone = [b for b in last_boxes if box_in_zone((b[0], b[1], b[2], b[3]), zone_points, frame_w, frame_h)]
                faces_in_zone = [f for f in last_faces if box_in_zone(f["box"], zone_points, frame_w, frame_h)]

                person_found = any(
                    box[5] == person_class
                    and box in boxes_in_zone
                    and box_key(box) not in last_static_suspects
                    for box in last_boxes
                )
                vehicle_found = any(box[5] in vehicle_classes and box in boxes_in_zone for box in last_boxes)
                unknown_face_found = any(f["name"] == "Unknown" for f in faces_in_zone)

                event_types_firing = []
                if trigger_person and person_found:
                    event_types_firing.append("person")
                if trigger_vehicle and vehicle_found:
                    event_types_firing.append("vehicle")
                if face_enabled and trigger_unknown_face and unknown_face_found:
                    event_types_firing.append("unknown_face")

                alert_trigger = len(event_types_firing) > 0

                # ====== DRAW: zone ======
                if zone_points and len(zone_points) >= 3:
                    poly_px = [(int(px * frame_w), int(py * frame_h)) for px, py in zone_points]
                    for i in range(len(poly_px)):
                        p1 = poly_px[i]
                        p2 = poly_px[(i + 1) % len(poly_px)]
                        cv2.line(frame, p1, p2, (0, 200, 255), 2)

                # ====== DRAW: boxes ======
                for x1, y1, x2, y2, confidence, class_id in last_boxes:
                    in_zone = (x1, y1, x2, y2, confidence, class_id) in boxes_in_zone
                    is_suspect = box_key((x1, y1, x2, y2, confidence, class_id)) in last_static_suspects
                    label = model.names.get(class_id, f"CLASS {class_id}").upper()
                    if is_suspect:
                        box_color = (0, 165, 255)  # cam: nghi vat tinh, da bo qua canh bao
                        label = f"{label} (TINH?)"
                    elif not in_zone:
                        box_color = (120, 120, 120)
                    else:
                        box_color = (0, 255, 0) if class_id == person_class else (255, 165, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

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

                self.last_faces_info = [{"name": f["name"], "similarity": round(f["similarity"], 3)} for f in last_faces]

                # ====== CONFIRM / ALERT (BỎ CONFIRM 3s - KÍCH HOẠT NGAY) ======
                if alert_trigger:
                    last_trigger_seen = now
                    last_event_types = event_types_firing

                within_grace = (now - last_trigger_seen) <= grace_seconds
                active = alert_trigger or (person_since is not None and within_grace)

                if active:
                    if person_since is None:
                        person_since = now
                        log("Detection triggered - ALERT ACTIVE")

                    cv2.putText(frame, "", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # ====== GHI SỰ KIỆN NGAY ======
                    logged_any = False
                    plate_text = None
                    plate_image_path = None
                    for box in boxes_in_zone:
                        x1, y1, x2, y2, conf, class_id = box
                        # Bo qua vat bi nghi la tinh (VD: ao vat tren xe nhan nham la nguoi)
                        if class_id == person_class and box_key(box) in last_static_suspects:
                            continue
                        if class_id in vehicle_classes or class_id == person_class:
                            if self.event_tracker.should_log(class_id, (x1, y1, x2, y2), now):
                                logged_any = True

                                if class_id in vehicle_classes and plate_enabled:
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

                    face_logged = False
                    if face_enabled and "unknown_face" in (last_event_types or []):
                        for face in last_faces:
                            if face["name"] == "Unknown":
                                fx1, fy1, fx2, fy2 = face["box"]
                                if self.event_tracker.should_log(-1, (fx1, fy1, fx2, fy2), now):
                                    face_logged = True
                                    save_unknown_face(face["box"], frame)

                    if logged_any or face_logged:
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

                    # ====== CÒI HÚ NGAY (có giới hạn thời lượng để tránh kêu liên tục vô hạn) ======
                    should_siren = alert["enabled"] and in_alert_time()
                    if should_siren:
                        if siren_started_at is not None and (now - siren_started_at) >= siren_max_duration:
                            # Da keu du lau lien tuc cho cung 1 dot canh bao -> tu tat,
                            # chi keu lai neu doi tuong roi di (het active) va cham
                            # tiep xuc lai, hoac sau siren_rearm_gap giay.
                            if self.siren_on:
                                self._stop_audio()
                                log(f"Siren tu dong TAT (da keu lien tuc {siren_max_duration}s)", "WARN")
                            siren_started_at = None
                            siren_forced_off_at = now
                        elif siren_forced_off_at is not None and (now - siren_forced_off_at) < siren_rearm_gap:
                            # Van trong khoang "nghi" sau khi bi tu tat, chua keu lai
                            pass
                        else:
                            siren_forced_off_at = None
                            siren_running = self.audio_process is not None and self.audio_process.poll() is None
                            if not siren_running:
                                if self._start_audio():
                                    self.siren_on = True
                                    if siren_started_at is None:
                                        siren_started_at = now
                                    log("Siren ON")
                                    last_alert = now
                            cv2.putText(frame, "SIREN ON", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        self._stop_audio()
                        siren_started_at = None
                        siren_forced_off_at = None
                else:
                    person_since = None
                    self._stop_audio()
                    siren_started_at = None
                    siren_forced_off_at = None

                # ====== ENCODE ======
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self.lock:
                        self.latest_jpeg = jpeg.tobytes()

            grabber.stop()
            self._stop_audio()
            self.status = "STOPPED"
            log("Detector _run ended")

        except Exception as e:
            self.status = "ERROR"
            self.last_error = str(e)
            log(f"Detector _run error: {e}", "ERROR")
            import traceback
            log(traceback.format_exc(), "ERROR")
            self._stop_audio()
            try:
                if "grabber" in locals():
                    grabber.stop()
            except Exception:
                pass


detector = Detector()


# ==================================================
# FLASK APP
# ==================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, "web", "templates"),
    static_folder=os.path.join(BASE, "web", "static"),
)

@app.route("/")
def index():
    cfg = load_config()
    return render_template(
        "index.html",
        status=detector.status,
        error=detector.last_error,
        warning=detector.warning,
        siren_on=detector.siren_on,
        faces=detector.last_faces_info,
        face_enabled=cfg.get("face", {}).get("enabled", False),
    )

def mjpeg_generator():
    while True:
        with detector.lock:
            frame = detector.latest_jpeg
        if frame is None:
            time.sleep(0.2)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.05)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/snapshot.jpg")
def snapshot():
    with detector.lock:
        frame = detector.latest_jpeg
    if frame is None:
        return "Chua co hinh anh", 503
    return Response(frame, mimetype="image/jpeg")

@app.route("/media/<path:subpath>")
def media(subpath):
    full_path = os.path.normpath(os.path.join(BASE, subpath))
    if not full_path.startswith(os.path.normpath(BASE) + os.sep):
        return "Khong hop le", 400
    if not os.path.isfile(full_path):
        return "Khong tim thay file", 404
    directory, filename = os.path.split(full_path)
    return send_from_directory(directory, filename)

@app.route("/api/status")
def api_status():
    return jsonify({
        "status": detector.status,
        "error": detector.last_error,
        "warning": detector.warning,
        "siren_on": detector.siren_on,
        "faces": detector.last_faces_info,
        "last_frame_time": detector.last_frame_time,
    })

@app.route("/start", methods=["POST"])
def start_route():
    log("Start requested via web")
    detector.start()
    return redirect(url_for("index"))

@app.route("/stop", methods=["POST"])
def stop_route():
    log("Stop requested via web")
    detector.stop()
    return redirect(url_for("index"))

@app.route("/restart", methods=["POST"])
def restart_route():
    log("Restart requested via web")
    detector.restart()
    return redirect(url_for("index"))

# ==================================================
# CONFIG PAGE
# ==================================================

@app.route("/config", methods=["GET"])
def config_page():
    cfg = load_config()
    return render_template("config.html", cfg=cfg)

@app.route("/config", methods=["POST"])
def config_save():
    log("Config save requested")
    cfg = load_config()
    form = request.form

    cfg["camera"]["ip"] = form.get("camera_ip", cfg["camera"]["ip"])
    cfg["camera"]["rtsp_port"] = int(form.get("camera_rtsp_port", cfg["camera"]["rtsp_port"]))
    cfg["camera"]["username"] = form.get("camera_username", cfg["camera"]["username"])
    new_password = form.get("camera_password", "")
    if new_password:
        cfg["camera"]["password"] = new_password
    cfg["camera"]["serial"] = form.get("camera_serial", cfg["camera"].get("serial", ""))
    cfg["camera"]["rtsp_subtype"] = int(form.get("camera_rtsp_subtype", cfg["camera"]["rtsp_subtype"]))
    cfg["camera"]["talk_port"] = int(form.get("camera_talk_port", cfg["camera"].get("talk_port", 8086)))

    cfg["yolo"]["model"] = form.get("yolo_model", cfg["yolo"]["model"])
    cfg["yolo"]["confidence"] = float(form.get("yolo_confidence", cfg["yolo"]["confidence"]))
    cfg["yolo"]["imgsz"] = int(form.get("yolo_imgsz", cfg["yolo"]["imgsz"]))
    cfg["yolo"]["max_det"] = int(form.get("yolo_max_det", cfg["yolo"].get("max_det", 10)))
    cfg["yolo"]["detect_person"] = "yolo_detect_person" in form
    cfg["yolo"]["detect_vehicles"] = "yolo_detect_vehicles" in form
    vehicle_classes = []
    if "vehicle_car" in form: vehicle_classes.append(2)
    if "vehicle_motorcycle" in form: vehicle_classes.append(3)
    if "vehicle_bus" in form: vehicle_classes.append(5)
    if "vehicle_truck" in form: vehicle_classes.append(7)
    cfg["yolo"]["vehicle_classes"] = vehicle_classes

    cfg["alert"]["enabled"] = "alert_enabled" in form
    cfg["alert"]["start"] = form.get("alert_start", cfg["alert"]["start"])
    cfg["alert"]["end"] = form.get("alert_end", cfg["alert"]["end"])
    cfg["alert"]["confirm_seconds"] = float(form.get("alert_confirm_seconds", cfg["alert"]["confirm_seconds"]))
    cfg["alert"]["cooldown"] = float(form.get("alert_cooldown", cfg["alert"]["cooldown"]))
    cfg["alert"]["grace_seconds"] = float(form.get("alert_grace_seconds", cfg["alert"].get("grace_seconds", 1.5)))
    cfg["alert"]["sound"] = form.get("alert_sound", cfg["alert"].get("sound", "assets/sound.wav"))
    cfg["alert"]["trigger_person"] = "alert_trigger_person" in form
    cfg["alert"]["trigger_vehicle"] = "alert_trigger_vehicle" in form
    cfg["alert"]["trigger_unknown_face"] = "alert_trigger_unknown_face" in form

    cfg["detection"]["save_images"] = "detection_save_images" in form
    cfg["detection"]["detect_fps"] = float(form.get("detection_detect_fps", cfg["detection"]["detect_fps"]))
    cfg["detection"]["save_directory"] = form.get("detection_save_directory", cfg["detection"]["save_directory"])

    cfg["display"]["show_camera"] = "display_show_camera" in form
    cfg["display"]["window_name"] = form.get("display_window_name", cfg["display"]["window_name"])

    if "face" not in cfg:
        cfg["face"] = {}
    cfg["face"]["enabled"] = "face_enabled" in form
    cfg["face"]["alert_only_unknown"] = "face_alert_only_unknown" in form
    cfg["face"]["save_unknown_faces"] = "face_save_unknown_faces" in form
    cfg["face"]["known_faces_dir"] = form.get("face_known_faces_dir", cfg["face"].get("known_faces_dir", "data/known_faces"))
    cfg["face"]["unknown_save_directory"] = form.get("face_unknown_save_directory", cfg["face"].get("unknown_save_directory", "data/unknown_faces"))
    cfg["face"]["similarity_threshold"] = float(form.get("face_similarity_threshold", cfg["face"].get("similarity_threshold", 0.72)))
    cfg["face"]["margin"] = float(form.get("face_margin", cfg["face"].get("margin", 0.05)))
    cfg["face"]["top_k"] = int(form.get("face_top_k", cfg["face"].get("top_k", 5)))
    cfg["face"]["min_face_size"] = int(form.get("face_min_face_size", cfg["face"].get("min_face_size", 40)))
    cfg["face"]["recognize_fps"] = float(form.get("face_recognize_fps", cfg["face"].get("recognize_fps", 2)))
    cfg["face"]["debug"] = "face_debug" in form

    save_config(cfg)
    log("Config saved, restarting detector")
    detector.restart()
    return redirect(url_for("index"))

# ==================================================
# FACE MANAGEMENT
# ==================================================

@app.route("/faces", methods=["GET"])
def faces_page():
    cfg = load_config()
    known_dir = os.path.join(BASE, cfg.get("face", {}).get("known_faces_dir", "data/known_faces"))
    people = []
    if os.path.isdir(known_dir):
        for name in sorted(os.listdir(known_dir)):
            person_dir = os.path.join(known_dir, name)
            if os.path.isdir(person_dir):
                count = len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                people.append({"name": name, "count": count})
    return render_template("faces.html", people=people)

@app.route("/faces/upload", methods=["POST"])
def faces_upload():
    cfg = load_config()
    known_dir = os.path.join(BASE, cfg.get("face", {}).get("known_faces_dir", "data/known_faces"))
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("faces_page"))
    person_dir = os.path.join(known_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    files = request.files.getlist("photos")
    existing = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    next_idx = len(existing) + 1
    for f in files:
        if not f or not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            continue
        dest = os.path.join(person_dir, f"{next_idx}{ext}")
        f.save(dest)
        next_idx += 1
    log(f"Uploaded {len(files)} photos for {name}")
    return redirect(url_for("faces_page"))

@app.route("/faces/rebuild", methods=["POST"])
def faces_rebuild():
    cfg = load_config()
    face_cfg = cfg.get("face", {})
    engine = FaceEngine(
        known_faces_dir=os.path.join(BASE, face_cfg.get("known_faces_dir", "data/known_faces")),
        db_cache_path=os.path.join(BASE, face_cfg.get("db_cache", "data/face_db.pkl")),
        similarity_threshold=face_cfg.get("similarity_threshold", 0.72),
        margin=face_cfg.get("margin", 0.05),
        top_k=face_cfg.get("top_k", 5),
        min_face_size=face_cfg.get("min_face_size", 40),
    )
    engine.build_db()
    log("Face database rebuilt")
    detector.restart()
    return redirect(url_for("faces_page"))

# ==================================================
# EVENTS
# ==================================================

EVENT_TYPE_LABELS = {
    "person": "Người",
    "vehicle": "Xe cộ",
    "unknown_face": "Người lạ (khuôn mặt)",
}
EVENTS_PER_PAGE = 24

@app.route("/events")
def events_page():
    event_type = request.args.get("type") or None
    page = max(1, int(request.args.get("page", 1)))
    offset = (page - 1) * EVENTS_PER_PAGE
    items, total = events_log.load_events(BASE, limit=EVENTS_PER_PAGE, event_type=event_type, offset=offset)
    for e in items:
        e["ts_str"] = datetime.fromtimestamp(e["ts"]).strftime("%d/%m/%Y %H:%M:%S")
        e["type_label"] = EVENT_TYPE_LABELS.get(e.get("type"), e.get("type", "?"))
    total_pages = max(1, (total + EVENTS_PER_PAGE - 1) // EVENTS_PER_PAGE)
    return render_template(
        "events.html",
        events=items,
        total=total,
        page=page,
        total_pages=total_pages,
        current_type=event_type,
        type_labels=EVENT_TYPE_LABELS,
    )

# ==================================================
# ZONE
# ==================================================

@app.route("/zone", methods=["GET"])
def zone_page():
    cfg = load_config()
    zone_cfg = cfg.get("alert", {})
    return render_template(
        "zone.html",
        zone_enabled=zone_cfg.get("zone_enabled", False),
        zone_points=json.dumps(zone_cfg.get("zone_points", [])),
    )

@app.route("/zone", methods=["POST"])
def zone_save():
    cfg = load_config()
    cfg["alert"]["zone_enabled"] = "zone_enabled" in request.form
    points_raw = request.form.get("zone_points", "[]")
    try:
        points = json.loads(points_raw)
        points = [[float(p[0]), float(p[1])] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    except Exception:
        points = []
    cfg["alert"]["zone_points"] = points
    save_config(cfg)
    log(f"Zone saved: {len(points)} points")
    detector.restart()
    return redirect(url_for("zone_page"))

# ==================================================
# PLAYBACK
# ==================================================

playback_jobs = {}

def build_playback_url(cam, start_dt, end_dt):
    def fmt(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    return (
        f'rtsp://{cam["username"]}:{cam["password"]}'
        f'@{cam["ip"]}:{cam["rtsp_port"]}'
        f'/cam/playback?channel=1&subtype={cam["rtsp_subtype"]}'
        f'&starttime={fmt(start_dt)}&endtime={fmt(end_dt)}'
    )

def run_ffmpeg_playback(job_id, playback_url, duration_seconds, output_path):
    playback_jobs[job_id]["status"] = "running"
    timeout_option_candidates = [
        ["-rw_timeout", "8000000"],
        ["-timeout", "8000000"],
        ["-stimeout", "8000000"],
        [],
    ]
    last_result = None
    for timeout_opts in timeout_option_candidates:
        cmd = ["ffmpeg", "-y", "-rtsp_transport", "tcp", *timeout_opts, "-i", playback_url, "-t", str(int(duration_seconds)), "-c", "copy", "-movflags", "+faststart", output_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 60)
        except FileNotFoundError:
            playback_jobs[job_id]["status"] = "error"
            playback_jobs[job_id]["error"] = "Khong tim thay ffmpeg"
            return
        except subprocess.TimeoutExpired:
            playback_jobs[job_id]["status"] = "error"
            playback_jobs[job_id]["error"] = "Timeout"
            return
        last_result = result
        stderr_lower = (result.stderr or "").lower()
        option_rejected = "unrecognized option" in stderr_lower or "option not found" in stderr_lower or ("invalid argument" in stderr_lower and "timeout" in stderr_lower)
        if option_rejected and timeout_opts:
            log(f"ffmpeg rejected {timeout_opts[0]}, trying next...", "WARN")
            continue
        break
    result = last_result
    if result is not None and result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10000:
        playback_jobs[job_id]["status"] = "done"
        log(f"Playback done: {output_path}")
    else:
        playback_jobs[job_id]["status"] = "error"
        playback_jobs[job_id]["error"] = "Camera khong phan hoi hoac khong co du lieu"
        log(f"Playback error: {playback_jobs[job_id]['error']}", "ERROR")

@app.route("/playback", methods=["GET"])
def playback_page():
    os.makedirs(PLAYBACK_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(PLAYBACK_DIR) if f.lower().endswith(".mp4")], reverse=True)
    return render_template("playback.html", files=files, jobs=playback_jobs)

@app.route("/playback/start", methods=["POST"])
def playback_start():
    cfg = load_config()
    cam = cfg["camera"]
    date_str = request.form.get("date", "")
    start_time_str = request.form.get("start_time", "")
    end_time_str = request.form.get("end_time", "")
    try:
        start_dt = datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M")
    except Exception:
        return "Ngay/gio khong hop le", 400
    if end_dt <= start_dt:
        return "Gio ket thuc phai sau gio bat dau", 400
    duration = (end_dt - start_dt).total_seconds()
    if duration > 30 * 60:
        return "Chi ho tro toi da 30 phut", 400
    os.makedirs(PLAYBACK_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    filename = f"playback_{start_dt.strftime('%Y%m%d_%H%M%S')}_{job_id}.mp4"
    output_path = os.path.join(PLAYBACK_DIR, filename)
    playback_url = build_playback_url(cam, start_dt, end_dt)
    playback_jobs[job_id] = {"status": "starting", "file": filename, "error": None}
    thread = threading.Thread(target=run_ffmpeg_playback, args=(job_id, playback_url, duration, output_path), daemon=True)
    thread.start()
    log(f"Playback started: {filename}")
    return redirect(url_for("playback_page"))

@app.route("/api/playback_jobs")
def api_playback_jobs():
    return jsonify(playback_jobs)

@app.route("/playback/delete", methods=["POST"])
def playback_delete():
    filename = request.form.get("file", "")
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Khong hop le", 400
    path = os.path.join(PLAYBACK_DIR, filename)
    if os.path.isfile(path):
        os.remove(path)
        log(f"Deleted playback file: {filename}")
    return redirect(url_for("playback_page"))

# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    log("Starting web_app.py")
    detector.start()
    HOST = "0.0.0.0"
    PORT = 5000
    try:
        from waitress import serve
        log(f"Running with waitress at http://{HOST}:{PORT}")
        serve(app, host=HOST, port=PORT, threads=8)
    except ImportError:
        log("Waitress not found, using Flask dev server", "WARN")
        app.run(host=HOST, port=PORT, debug=False, threaded=True)