"""
web_app.py

Ban thay the cho gui.py (tkinter) - dung cho moi truong khong co man
hinh desktop / X11, dac biet la chay tren dien thoai qua Termux.

Chay:
    python3 web_app.py

Roi mo trinh duyet vao:
    http://127.0.0.1:5000        (ngay tren dien thoai dang chay Termux)
    http://<IP-LAN-cua-dien-thoai>:5000   (tu thiet bi khac cung wifi)

Khong can tkinter, khong can cv2.imshow/X11 - moi thu chay headless va
stream hinh anh qua MJPEG.
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

from face_engine import FaceEngine
from frame_grabber import FrameGrabber
from zone_utils import box_in_zone
import events as events_log
import plate_engine
from vehicle_tracker import VehicleTracker

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows khong co fcntl - bo qua file-lock, van dung threading lock
    HAS_FCNTL = False


BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE, "config.json")
MODEL_DOWNLOAD_LOCK_FILE = os.path.join(BASE, ".model_download.lock")
PLAYBACK_DIR = os.path.join(BASE, "playback_downloads")


@contextlib.contextmanager
def model_download_lock(timeout=None):
    """
    File-lock cap he dieu hanh: dam bao chi 1 TIEN TRINH (process) duy
    nhat duoc tai model cung mot luc, ke ca khi co ai vo tinh chay 2 lan
    'python3 web_app.py' o 2 cua so Termux/tmux khac nhau. threading.Lock
    ben trong Detector chi bao ve trong cung 1 tien trinh, khong du - can
    them lop nay de an toan tuyet doi tren dien thoai (de vo tinh mo
    trung phien lam viec).
    """
    if not HAS_FCNTL:
        yield
        return

    with open(MODEL_DOWNLOAD_LOCK_FILE, "w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# ==================================================
# CONFIG HELPERS
# ==================================================

def load_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)


# ==================================================
# DETECTOR (chay nen trong 1 thread rieng)
# ==================================================

class Detector:
    """
    Gom toan bo logic YOLO + nhan dien khuon mat + coi hu, giong het
    main.py, nhung khong dung cv2.imshow ma luu frame moi nhat (JPEG
    bytes) de web_app stream qua MJPEG.
    """

    def __init__(self):
        self.thread = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()          # bao ve self.latest_jpeg
        self.lifecycle_lock = threading.RLock()  # bao ve start/stop/restart

        self.latest_jpeg = None
        self.status = "STOPPED"
        self.last_error = None
        self.warning = None

        self.audio_process = None

        # De hien thi tren dashboard
        self.last_faces_info = []
        self.siren_on = False

    # --------------------------------------------------
    # START / STOP
    # --------------------------------------------------

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        # lifecycle_lock dam bao 2 request Start/Restart bam gan nhau
        # (vd double-tap tren dien thoai, hoac trinh duyet tu dong gui
        # lai request khi mang cham) KHONG BAO GIO tao ra 2 luong chay
        # song song cung tai model -> day chinh la nguyen nhan gay loi
        # "No such file" va toc do tai bi chia doi bang thong truoc day.
        with self.lifecycle_lock:
            if self.is_running():
                print("[WEB] Da dang chay (hoac dang tai model) - bo qua yeu cau Start trung lap")
                self.warning = (
                    "Dang tai model / da dang chay roi - vui long doi, "
                    "khong bam Start/Restart lien tuc (se lam cham them "
                    "toc do tai do bi chia bang thong)."
                )
                return

            self.stop_flag.clear()
            self.last_error = None
            self.warning = None
            self.status = "STARTING"

            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lifecycle_lock:
            self.stop_flag.set()

            if self.thread is not None:
                self.thread.join(timeout=10)

                if self.thread.is_alive():
                    # Luong van chua thoat duoc (thuong vi dang ket dinh
                    # trong 1 loi goi mang dong bo nhu tai model - Python
                    # khong the "kill" thread nay). Bao trang thai ro rang
                    # thay vi im lang coi nhu da dung, de tranh nham lan.
                    print(
                        "[WEB][WARNING] Luong xu ly cu van chua dung han "
                        "(co the dang tai model do dang) - se tu dung khi "
                        "xong. Chua the Start lai cho toi luc do."
                    )

            self._stop_audio()

            if not (self.thread is not None and self.thread.is_alive()):
                self.status = "STOPPED"

    def restart(self):
        with self.lifecycle_lock:
            self.stop()
            time.sleep(0.3)
            self.start()

    # --------------------------------------------------
    # AUDIO / SIREN (giong main.py)
    # --------------------------------------------------

    def _start_audio(self):
        if self.audio_process is not None and self.audio_process.poll() is None:
            return False

        speaker = os.path.join(BASE, "speaker.py")

        popen_kwargs = {}

        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = os.setsid

        self.audio_process = subprocess.Popen(
            [sys.executable, speaker],
            **popen_kwargs
        )

        return True

    def _stop_audio(self):
        if self.audio_process is None or self.audio_process.poll() is not None:
            self.audio_process = None
            self.siren_on = False
            return

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

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------

    def _run(self):
        try:
            cfg = load_config()

            cam = cfg["camera"]
            yolo_cfg = cfg["yolo"]
            alert = cfg["alert"]
            detect = cfg["detection"]
            face_cfg = cfg.get("face", {"enabled": False})
            plate_cfg = cfg.get("plate", {"enabled": False})

            # ---- Vung canh bao (ROI) - chong nhieu ----
            zone_enabled = alert.get("zone_enabled", False)
            zone_points = alert.get("zone_points", []) if zone_enabled else []

            # ---- Loai su kien duoc phep kich hoat canh bao ----
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

            # Luong "chinh" (subtype=0 mac dinh) - do phan giai GOC cua
            # camera, dung RIENG de chup anh xe/bien so cho ro net.
            # Luong o tren (rtsp_url, thuong subtype=1 - luong phu, do
            # phan giai thap) van dung cho detect/xem truc tiep nhu cu.
            snapshot_enabled = cam.get("snapshot_enabled", True)
            snapshot_subtype = cam.get("snapshot_subtype", 0)
            snapshot_rtsp_url = (
                f'rtsp://{cam["username"]}:{cam["password"]}'
                f'@{cam["ip"]}:{cam["rtsp_port"]}'
                f'/cam/realmonitor?channel=1&subtype={snapshot_subtype}'
            )

            print("[WEB] Loading YOLO...")
            model = YOLO(yolo_cfg["model"])

            face_enabled = face_cfg.get("enabled", False)
            face_engine = None

            if face_enabled:
                print("[WEB] Loading face engine...")
                try:
                    with model_download_lock():
                        face_engine = FaceEngine(
                            known_faces_dir=os.path.join(BASE, face_cfg.get("known_faces_dir", "known_faces")),
                            db_cache_path=os.path.join(BASE, face_cfg.get("db_cache", "face_db.pkl")),
                            similarity_threshold=face_cfg.get("similarity_threshold", 0.72),
                            margin=face_cfg.get("margin", 0.05),
                            top_k=face_cfg.get("top_k", 5),
                            min_face_size=face_cfg.get("min_face_size", 40),
                            debug=face_cfg.get("debug", False),
                        )
                except Exception as e:
                    # KHONG de loi tai model nhan dien khuon mat lam chet
                    # toan bo luong (camera + YOLO + live view van phai
                    # chay duoc binh thuong). Thuong gap khi file trong so
                    # pretrained bi tai do dang (mang chap chon) - xem
                    # TERMUX_SETUP.md muc "Loi tai model nhan dien mat".
                    face_enabled = False
                    face_engine = None
                    self.warning = (
                        "Khong tai duoc model nhan dien khuon mat (da TAT tinh "
                        f"nang nay, cac phan khac van chay binh thuong): {e}"
                    )
                    print("[WEB][ERROR][FACE]", self.warning)

            alert_only_unknown = face_cfg.get("alert_only_unknown", True)
            save_unknown_faces = face_cfg.get("save_unknown_faces", True)
            unknown_save_dir = os.path.join(BASE, face_cfg.get("unknown_save_directory", "unknown_faces"))

            recognize_fps = face_cfg.get("recognize_fps", 2)
            face_interval = 1 / recognize_fps if recognize_fps > 0 else 0

            print("[WEB] Connecting to camera...")
            grabber = FrameGrabber(rtsp_url, reconnect_delay=2.0, stale_timeout=8.0)
            grabber.start()

            # Luong chat luong cao (chi de chup anh xe/bien so) - khong
            # bat buoc phai ket noi thanh cong ngay, cac tinh nang khac
            # van chay binh thuong va se tu dung duoc luong nay ngay
            # khi no san sang.
            snapshot_grabber = None

            if snapshot_enabled:
                print("[WEB] Connecting to camera (luong chat luong cao - chup xe/bien so)...")
                snapshot_grabber = FrameGrabber(snapshot_rtsp_url, reconnect_delay=2.0, stale_timeout=8.0)
                snapshot_grabber.start()

            def get_snapshot_frame():
                if snapshot_grabber is None:
                    return None
                return snapshot_grabber.get_latest(max_age=3)

            def scale_box(box, src_w, src_h, dst_w, dst_h):
                x1, y1, x2, y2 = box
                sx = dst_w / src_w
                sy = dst_h / src_h
                return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

            # Doi toi da 8s cho lan ket noi/frame dau tien truoc khi bao loi.
            # Sau buoc nay, FrameGrabber tu chay nen va tu ket noi lai neu
            # rot mang - KHONG can kiem tra "isOpened" 1 lan roi thoi nhu
            # truoc (day chinh la nguyen nhan gay "dung hinh": cv2.VideoCapture
            # goc co the treo vo han khi mang chap chon, khong co co che
            # tu phuc hoi).
            wait_start = time.time()
            while grabber.get_latest() is None and time.time() - wait_start < 8:
                if self.stop_flag.is_set():
                    grabber.stop()
                    if snapshot_grabber is not None:
                        snapshot_grabber.stop()
                    return
                time.sleep(0.2)

            if grabber.get_latest() is None:
                self.status = "ERROR"
                self.last_error = "Khong ket noi duoc camera (kiem tra IP/mat khau/mang)"
                print("[WEB][ERROR]", self.last_error)
                grabber.stop()
                if snapshot_grabber is not None:
                    snapshot_grabber.stop()
                return

            self.status = "RUNNING"
            print("[WEB] Camera connected, detection running")

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

            def save_unknown_face(frame, box):
                if not save_unknown_faces:
                    return
                os.makedirs(unknown_save_dir, exist_ok=True)
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    return
                crop = frame[y1:y2, x1:x2]
                filename = datetime.now().strftime("unknown_%Y%m%d_%H%M%S_%f.jpg")
                cv2.imwrite(os.path.join(unknown_save_dir, filename), crop)

            # ---- Xe + bien so (crop rieng anh xe / anh bien so) ----
            plate_enabled = plate_cfg.get("enabled", False)
            vehicle_save_dir = os.path.join(
                BASE, plate_cfg.get("vehicle_save_directory", "detections/vehicles")
            )
            plate_save_dir = os.path.join(
                BASE, plate_cfg.get("plate_save_directory", "detections/plates")
            )
            vehicle_tracker = VehicleTracker(
                idle_seconds=plate_cfg.get("dedup_seconds", 300),
                iou_thresh=plate_cfg.get("iou_thresh", 0.3),
                confirm_seconds=plate_cfg.get("confirm_seconds", 2.0),
            )
            vehicle_min_confidence = plate_cfg.get("min_confidence", 0.5)

            def crop_box(frame, box):
                x1, y1, x2, y2 = box
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    return None
                return frame[y1:y2, x1:x2]

            def save_vehicle_image(crop):
                if not detect["save_images"] or crop is None or crop.size == 0:
                    return None
                os.makedirs(vehicle_save_dir, exist_ok=True)
                filename = datetime.now().strftime("vehicle_%Y%m%d_%H%M%S_%f.jpg")
                path = os.path.join(vehicle_save_dir, filename)
                cv2.imwrite(path, crop)
                return os.path.relpath(path, BASE)

            def save_plate_image(crop):
                if not detect["save_images"] or crop is None or crop.size == 0:
                    return None
                os.makedirs(plate_save_dir, exist_ok=True)
                filename = datetime.now().strftime("plate_%Y%m%d_%H%M%S_%f.jpg")
                path = os.path.join(plate_save_dir, filename)
                cv2.imwrite(path, crop)
                return os.path.relpath(path, BASE)

            # ---- Don du lieu cu (anh + su kien) qua han luu tru ----
            retention_days = detect.get("retention_days", 15)
            retention_check_interval = 6 * 3600  # kiem tra moi 6 tieng
            media_retention_dirs = [
                detect["save_directory"],
                plate_cfg.get("vehicle_save_directory", "detections/vehicles"),
                plate_cfg.get("plate_save_directory", "detections/plates"),
                face_cfg.get("unknown_save_directory", "unknown_faces"),
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

            person_since = None
            last_alert = 0
            last_detection = 0
            last_trigger_seen = 0
            last_face_check = 0
            last_event_log = 0  # thoi diem gan nhat GHI SU KIEN (doc lap voi coi hu)

            last_boxes = []
            last_faces = []
            last_event_types = []

            while not self.stop_flag.is_set():

                frame = grabber.get_latest(max_age=5)

                if frame is None:
                    # Chua co frame moi (dang ket noi lai) - khong "dung
                    # hinh" ca chuong trinh, chi cho ngan roi kiem tra lai.
                    # self.status van la RUNNING nhung khung hinh se dung
                    # tam thoi tren trinh duyet cho toi khi co frame moi.
                    if not grabber.is_healthy():
                        self.status = "RECONNECTING"
                    time.sleep(0.2)
                    continue

                if self.status != "RUNNING":
                    self.status = "RUNNING"

                now = time.time()

                # -------- DON DU LIEU CU (anh + su kien qua han) --------
                if now - last_retention_cleanup >= retention_check_interval:
                    last_retention_cleanup = now
                    events_log.prune_old(BASE, media_retention_dirs, max_age_days=retention_days)

                # -------- YOLO --------
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

                # -------- FACE --------
                if face_enabled and now - last_face_check >= face_interval:
                    last_face_check = now
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_faces = face_engine.recognize(rgb_frame)

                frame_h, frame_w = frame.shape[:2]

                # -------- LOC THEO VUNG CANH BAO (chong nhieu) --------
                # Neu co dinh nghia vung, chi tinh cac phat hien co diem
                # "chan" nam TRONG vung; ben ngoai van hien thi (mau xam)
                # nhung KHONG tinh vao alert_trigger.
                boxes_in_zone = [
                    b for b in last_boxes
                    if box_in_zone((b[0], b[1], b[2], b[3]), zone_points, frame_w, frame_h)
                ]

                faces_in_zone = [
                    f for f in last_faces
                    if box_in_zone(f["box"], zone_points, frame_w, frame_h)
                ]

                person_found = any(b[5] == person_class and b in boxes_in_zone for b in last_boxes)
                vehicle_found = any(
                    b[5] in vehicle_classes and b in boxes_in_zone for b in last_boxes
                )

                # -------- XE + BIEN SO (DOC LAP voi trigger_vehicle / coi
                # hu - luon nhan dien bien so moi khi thay xe trong vung
                # canh bao, ke ca khi da tat "xe lam keu coi"). --------
                if plate_enabled:
                    for box in boxes_in_zone:
                        x1, y1, x2, y2, box_conf, class_id = box

                        if class_id not in vehicle_classes:
                            continue

                        if box_conf < vehicle_min_confidence:
                            continue

                        vehicle_box = (x1, y1, x2, y2)

                        if not vehicle_tracker.should_log(vehicle_box, None, now=now):
                            continue

                        snap_frame = get_snapshot_frame()

                        if snap_frame is not None:
                            snap_h, snap_w = snap_frame.shape[:2]
                            snap_box = scale_box(vehicle_box, frame_w, frame_h, snap_w, snap_h)
                            vehicle_crop = crop_box(snap_frame, snap_box)
                        else:
                            vehicle_crop = crop_box(frame, vehicle_box)

                        if vehicle_crop is None:
                            continue

                        plate_text, plate_crop = plate_engine.read_plate(vehicle_crop)

                        vehicle_img_path = save_vehicle_image(vehicle_crop)
                        plate_img_path = save_plate_image(plate_crop)

                        events_log.log_event(
                            BASE,
                            "vehicle",
                            image_path=vehicle_img_path,
                            extra={"plate": plate_text, "plate_image": plate_img_path},
                        )

                unknown_face_found = any(f["name"] == "Unknown" for f in faces_in_zone)

                event_types_firing = []

                if trigger_person and person_found:
                    event_types_firing.append("person")
                if trigger_vehicle and vehicle_found:
                    event_types_firing.append("vehicle")
                if face_enabled and trigger_unknown_face and unknown_face_found:
                    event_types_firing.append("unknown_face")

                alert_trigger = len(event_types_firing) > 0

                # -------- DRAW: vung canh bao --------
                if zone_points and len(zone_points) >= 3:
                    poly_px = [
                        (int(px * frame_w), int(py * frame_h))
                        for px, py in zone_points
                    ]
                    for i in range(len(poly_px)):
                        p1 = poly_px[i]
                        p2 = poly_px[(i + 1) % len(poly_px)]
                        cv2.line(frame, p1, p2, (0, 200, 255), 2)

                # -------- DRAW: boxes (xam neu ngoai vung, mau binh
                # thuong neu trong vung hoac khong dung vung) --------
                for x1, y1, x2, y2, confidence, class_id in last_boxes:
                    in_zone = (x1, y1, x2, y2, confidence, class_id) in boxes_in_zone
                    label = model.names.get(class_id, f"CLASS {class_id}").upper()

                    if not in_zone:
                        box_color = (120, 120, 120)
                    else:
                        box_color = (0, 255, 0) if class_id == person_class else (255, 165, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(
                        frame, f"{label} {confidence:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2
                    )

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
                    cv2.putText(
                        frame, f"{name} ({similarity:.2f})", (fx1, max(fy1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2
                    )

                self.last_faces_info = [
                    {"name": f["name"], "similarity": round(f["similarity"], 3)}
                    for f in last_faces
                ]

                # -------- CONFIRM / ALERT (grace period) --------
                if alert_trigger:
                    last_trigger_seen = now
                    last_event_types = event_types_firing

                within_grace = (now - last_trigger_seen) <= grace_seconds
                active = alert_trigger or (person_since is not None and within_grace)

                if active:
                    if person_since is None:
                        person_since = now

                    duration = now - person_since

                    cv2.putText(
                        frame, f"Confirm: {duration:.1f}/{confirm_seconds}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                    )

                    # -------- GHI SU KIEN VAO LICH SU --------
                    # Luon ghi khi da xac nhan du thoi gian, KHONG phu
                    # thuoc coi hu bat/tat hay dang trong khung gio bao
                    # dong hay khong - neu khong Lich su su kien se bo
                    # sot cac lan phat hien ngoai gio bao dong / khi tat coi.
                    confirmed = duration >= confirm_seconds

                    if confirmed and (now - last_event_log >= alert["cooldown"]):
                        img_path = save_image(frame)

                        for etype in (last_event_types or ["person"]):
                            if etype == "vehicle":
                                continue  # da duoc ghi rieng o tren (doc lap voi trigger_vehicle)
                            events_log.log_event(BASE, etype, image_path=img_path)

                        if face_enabled and "unknown_face" in (last_event_types or []):
                            for face in last_faces:
                                if face["name"] == "Unknown":
                                    save_unknown_face(frame, face["box"])

                        last_event_log = now

                    should_siren = confirmed and alert["enabled"] and in_alert_time()

                    if should_siren:
                        siren_running = (
                            self.audio_process is not None
                            and self.audio_process.poll() is None
                        )

                        if not siren_running:
                            if self._start_audio():
                                self.siren_on = True
                                print("[WEB][ALERT] Siren ON")
                                last_alert = now

                        cv2.putText(
                            frame, "SIREN ON", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                        )
                    else:
                        self._stop_audio()

                else:
                    person_since = None
                    self._stop_audio()

                # -------- ENCODE FOR STREAM --------
                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                if ok:
                    with self.lock:
                        self.latest_jpeg = jpeg.tobytes()

            grabber.stop()

            if snapshot_grabber is not None:
                snapshot_grabber.stop()

            self._stop_audio()
            self.status = "STOPPED"
            print("[WEB] Detector stopped")

        except Exception as e:
            self.status = "ERROR"
            self.last_error = str(e)
            print("[WEB][ERROR]", e)
            self._stop_audio()

            # Dam bao khong bo quen thread grabber dang chay ngam neu loi
            # xay ra giua chung (tranh ri thread/ket noi camera).
            try:
                if "grabber" in locals():
                    grabber.stop()
                if "snapshot_grabber" in locals() and snapshot_grabber is not None:
                    snapshot_grabber.stop()
            except Exception:
                pass


detector = Detector()


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)


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

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/snapshot.jpg")
def snapshot():
    """1 anh JPEG tinh - dung lam nen cho trang ve vung canh bao."""
    with detector.lock:
        frame = detector.latest_jpeg

    if frame is None:
        return "Chua co hinh anh (camera co the chua ket noi)", 503

    return Response(frame, mimetype="image/jpeg")


@app.route("/media/<path:subpath>")
def media(subpath):
    """
    Phuc vu file anh/video da luu (detections/, unknown_faces/,
    playback_downloads/) de hien thi tren trang Su kien / Phat lai.
    Chan path traversal (khong cho '..' thoat ra ngoai BASE).
    """
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
    })


@app.route("/start", methods=["POST"])
def start_route():
    detector.start()
    return redirect(url_for("index"))


@app.route("/stop", methods=["POST"])
def stop_route():
    detector.stop()
    return redirect(url_for("index"))


@app.route("/restart", methods=["POST"])
def restart_route():
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
    cfg = load_config()
    form = request.form

    # ---- Camera ----
    cfg["camera"]["ip"] = form.get("camera_ip", cfg["camera"]["ip"])
    cfg["camera"]["rtsp_port"] = int(form.get("camera_rtsp_port", cfg["camera"]["rtsp_port"]))
    cfg["camera"]["username"] = form.get("camera_username", cfg["camera"]["username"])

    new_password = form.get("camera_password", "")
    if new_password:
        cfg["camera"]["password"] = new_password

    cfg["camera"]["serial"] = form.get("camera_serial", cfg["camera"].get("serial", ""))
    cfg["camera"]["rtsp_subtype"] = int(form.get("camera_rtsp_subtype", cfg["camera"]["rtsp_subtype"]))
    cfg["camera"]["snapshot_enabled"] = "camera_snapshot_enabled" in form
    cfg["camera"]["snapshot_subtype"] = int(
        form.get("camera_snapshot_subtype", cfg["camera"].get("snapshot_subtype", 0))
    )
    cfg["camera"]["talk_port"] = int(form.get("camera_talk_port", cfg["camera"].get("talk_port", 8086)))

    # ---- YOLO ----
    cfg["yolo"]["model"] = form.get("yolo_model", cfg["yolo"]["model"])
    cfg["yolo"]["confidence"] = float(form.get("yolo_confidence", cfg["yolo"]["confidence"]))
    cfg["yolo"]["imgsz"] = int(form.get("yolo_imgsz", cfg["yolo"]["imgsz"]))
    cfg["yolo"]["max_det"] = int(form.get("yolo_max_det", cfg["yolo"].get("max_det", 10)))
    cfg["yolo"]["detect_person"] = "yolo_detect_person" in form
    cfg["yolo"]["detect_vehicles"] = "yolo_detect_vehicles" in form

    vehicle_classes = []
    if "vehicle_car" in form:
        vehicle_classes.append(2)
    if "vehicle_motorcycle" in form:
        vehicle_classes.append(3)
    if "vehicle_bus" in form:
        vehicle_classes.append(5)
    if "vehicle_truck" in form:
        vehicle_classes.append(7)
    cfg["yolo"]["vehicle_classes"] = vehicle_classes

    # ---- Alert ----
    cfg["alert"]["enabled"] = "alert_enabled" in form
    cfg["alert"]["start"] = form.get("alert_start", cfg["alert"]["start"])
    cfg["alert"]["end"] = form.get("alert_end", cfg["alert"]["end"])
    cfg["alert"]["confirm_seconds"] = float(form.get("alert_confirm_seconds", cfg["alert"]["confirm_seconds"]))
    cfg["alert"]["cooldown"] = float(form.get("alert_cooldown", cfg["alert"]["cooldown"]))
    cfg["alert"]["grace_seconds"] = float(form.get("alert_grace_seconds", cfg["alert"].get("grace_seconds", 1.5)))
    cfg["alert"]["sound"] = form.get("alert_sound", cfg["alert"].get("sound", "sound.wav"))
    cfg["alert"]["trigger_person"] = "alert_trigger_person" in form
    cfg["alert"]["trigger_vehicle"] = "alert_trigger_vehicle" in form
    cfg["alert"]["trigger_unknown_face"] = "alert_trigger_unknown_face" in form

    # ---- Detection ----
    cfg["detection"]["save_images"] = "detection_save_images" in form
    cfg["detection"]["detect_fps"] = float(form.get("detection_detect_fps", cfg["detection"]["detect_fps"]))
    cfg["detection"]["save_directory"] = form.get("detection_save_directory", cfg["detection"]["save_directory"])

    # ---- Display (giu lai field cho tuong thich voi main.py ban desktop) ----
    cfg["display"]["show_camera"] = "display_show_camera" in form
    cfg["display"]["window_name"] = form.get("display_window_name", cfg["display"]["window_name"])

    # ---- Face ----
    if "face" not in cfg:
        cfg["face"] = {}

    cfg["face"]["enabled"] = "face_enabled" in form
    cfg["face"]["alert_only_unknown"] = "face_alert_only_unknown" in form
    cfg["face"]["save_unknown_faces"] = "face_save_unknown_faces" in form
    cfg["face"]["known_faces_dir"] = form.get("face_known_faces_dir", cfg["face"].get("known_faces_dir", "known_faces"))
    cfg["face"]["unknown_save_directory"] = form.get(
        "face_unknown_save_directory", cfg["face"].get("unknown_save_directory", "unknown_faces")
    )
    cfg["face"]["similarity_threshold"] = float(
        form.get("face_similarity_threshold", cfg["face"].get("similarity_threshold", 0.72))
    )
    cfg["face"]["margin"] = float(form.get("face_margin", cfg["face"].get("margin", 0.05)))
    cfg["face"]["top_k"] = int(form.get("face_top_k", cfg["face"].get("top_k", 5)))
    cfg["face"]["min_face_size"] = int(form.get("face_min_face_size", cfg["face"].get("min_face_size", 40)))
    cfg["face"]["recognize_fps"] = float(form.get("face_recognize_fps", cfg["face"].get("recognize_fps", 2)))
    cfg["face"]["debug"] = "face_debug" in form

    save_config(cfg)

    detector.restart()

    return redirect(url_for("index"))


# ==================================================
# FACE MANAGEMENT (dang ky / xay lai database qua web)
# ==================================================

@app.route("/faces", methods=["GET"])
def faces_page():
    cfg = load_config()
    known_dir = os.path.join(BASE, cfg.get("face", {}).get("known_faces_dir", "known_faces"))

    people = []

    if os.path.isdir(known_dir):
        for name in sorted(os.listdir(known_dir)):
            person_dir = os.path.join(known_dir, name)
            if os.path.isdir(person_dir):
                count = len([
                    f for f in os.listdir(person_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                people.append({"name": name, "count": count})

    return render_template("faces.html", people=people)


@app.route("/faces/upload", methods=["POST"])
def faces_upload():
    cfg = load_config()
    known_dir = os.path.join(BASE, cfg.get("face", {}).get("known_faces_dir", "known_faces"))

    name = request.form.get("name", "").strip()

    if not name:
        return redirect(url_for("faces_page"))

    person_dir = os.path.join(known_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    files = request.files.getlist("photos")

    existing = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
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

    return redirect(url_for("faces_page"))


@app.route("/faces/rebuild", methods=["POST"])
def faces_rebuild():
    cfg = load_config()
    face_cfg = cfg.get("face", {})

    engine = FaceEngine(
        known_faces_dir=os.path.join(BASE, face_cfg.get("known_faces_dir", "known_faces")),
        db_cache_path=os.path.join(BASE, face_cfg.get("db_cache", "face_db.pkl")),
        similarity_threshold=face_cfg.get("similarity_threshold", 0.72),
        margin=face_cfg.get("margin", 0.05),
        top_k=face_cfg.get("top_k", 5),
        min_face_size=face_cfg.get("min_face_size", 40),
    )
    engine.build_db()

    detector.restart()

    return redirect(url_for("faces_page"))


# ==================================================
# SU KIEN (EVENT HISTORY) - xem lai cac lan phat hien
# nguoi/xe co/nguoi la vao vung canh bao
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

    items, total = events_log.load_events(
        BASE, limit=EVENTS_PER_PAGE, event_type=event_type, offset=offset
    )

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
# VUNG CANH BAO (ZONE EDITOR) - chong nhieu
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
        # Chi chap nhan danh sach cac cap so [x,y] trong khoang 0..1
        points = [
            [float(p[0]), float(p[1])]
            for p in points
            if isinstance(p, (list, tuple)) and len(p) == 2
        ]
    except Exception:
        points = []

    cfg["alert"]["zone_points"] = points

    save_config(cfg)

    detector.restart()

    return redirect(url_for("zone_page"))


# ==================================================
# PHAT LAI (SD CARD PLAYBACK) - THU NGHIEM
# ==================================================
#
# Dua tren tai lieu da reverse-engineer trong imou-life/docs/p2p-media-
# flow.md: camera dong Dahua/Imou ho tro URL RTSP phat lai chuan:
#   rtsp://user:pass@ip:554/cam/playback?channel=1&subtype=0
#         &starttime=YYYY-MM-DDTHH:MM:SS&endtime=YYYY-MM-DDTHH:MM:SS
#
# CANH BAO: day la tinh nang THU NGHIEM - phu thuoc dong camera/firmware
# co ho tro URL nay hay khong, va camera co du lieu tren the nho trong
# khoang thoi gian yeu cau hay khong. Neu khong hoat dong, day la gioi
# han cua firmware camera, khong phai loi cua ung dung - dung app IMOU
# chinh hang de xem lai chac chan hoat dong duoc.

playback_jobs = {}  # job_id -> {"status", "file", "error"}


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

    # Cac ffmpeg build khac nhau dung ten tuy chon timeout khac nhau
    # (-stimeout bi loai bo o cac ban ffmpeg moi, doi thanh -timeout hoac
    # -rw_timeout tuy phien ban). Thu lan luot, chi chuyen sang phuong an
    # tiep theo NEU loi la do "khong nhan dien duoc tuy chon" (loi cu
    # phap dong lenh) - neu loi la do camera/mang that su thi dung ngay,
    # khong thu lai vo ich.
    timeout_option_candidates = [
        ["-rw_timeout", "8000000"],
        ["-timeout", "8000000"],
        ["-stimeout", "8000000"],
        [],  # phuong an cuoi: khong dung tuy chon timeout nao ca
    ]

    last_result = None

    for timeout_opts in timeout_option_candidates:
        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            *timeout_opts,
            "-i", playback_url,
            "-t", str(int(duration_seconds)),
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=duration_seconds + 60
            )
        except FileNotFoundError:
            playback_jobs[job_id]["status"] = "error"
            playback_jobs[job_id]["error"] = (
                "Khong tim thay ffmpeg. Cai dat: apt install ffmpeg (Ubuntu "
                "proot) hoac pkg install ffmpeg (Termux), hoac tai ve tu "
                "ffmpeg.org (Windows)."
            )
            return
        except subprocess.TimeoutExpired:
            playback_jobs[job_id]["status"] = "error"
            playback_jobs[job_id]["error"] = "Qua thoi gian cho phep - camera khong phan hoi."
            return

        last_result = result

        stderr_lower = (result.stderr or "").lower()
        option_rejected = (
            "unrecognized option" in stderr_lower
            or "option not found" in stderr_lower
            or "invalid argument" in stderr_lower and "timeout" in stderr_lower
        )

        if option_rejected and timeout_opts:
            print(
                f"[PLAYBACK] ffmpeg khong nhan tuy chon {timeout_opts[0]} - "
                f"thu phuong an khac..."
            )
            continue

        # Khong phai loi cu phap dong lenh -> dung lai o day, du thanh
        # cong hay that bai that su (do camera/mang).
        break

    result = last_result

    if (
        result is not None
        and result.returncode == 0
        and os.path.exists(output_path)
        and os.path.getsize(output_path) > 10_000
    ):
        playback_jobs[job_id]["status"] = "done"
        print(f"[PLAYBACK] Xong: {output_path}")
    else:
        playback_jobs[job_id]["status"] = "error"
        playback_jobs[job_id]["error"] = (
            "Camera khong phan hoi URL phat lai, hoac khong co du "
            "lieu trong khoang thoi gian nay. Chi tiet ffmpeg: "
            + ((result.stderr if result else "") or "")[-1500:]
        )
        print("[PLAYBACK][ERROR]", playback_jobs[job_id]["error"])
    return


@app.route("/playback", methods=["GET"])
def playback_page():
    os.makedirs(PLAYBACK_DIR, exist_ok=True)

    files = sorted(
        (f for f in os.listdir(PLAYBACK_DIR) if f.lower().endswith(".mp4")),
        reverse=True
    )

    return render_template(
        "playback.html",
        files=files,
        jobs=playback_jobs,
    )


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
        return "Ngày/giờ không hợp lệ", 400

    if end_dt <= start_dt:
        return "Giờ kết thúc phải sau giờ bắt đầu", 400

    duration = (end_dt - start_dt).total_seconds()

    if duration > 30 * 60:
        return "Chỉ hỗ trợ tối đa 30 phút mỗi lần tải (tránh treo quá lâu)", 400

    os.makedirs(PLAYBACK_DIR, exist_ok=True)

    job_id = uuid.uuid4().hex[:10]
    filename = f"playback_{start_dt.strftime('%Y%m%d_%H%M%S')}_{job_id}.mp4"
    output_path = os.path.join(PLAYBACK_DIR, filename)

    playback_url = build_playback_url(cam, start_dt, end_dt)

    playback_jobs[job_id] = {"status": "starting", "file": filename, "error": None}

    thread = threading.Thread(
        target=run_ffmpeg_playback,
        args=(job_id, playback_url, duration, output_path),
        daemon=True
    )
    thread.start()

    return redirect(url_for("playback_page"))


@app.route("/api/playback_jobs")
def api_playback_jobs():
    return jsonify(playback_jobs)


@app.route("/playback/delete", methods=["POST"])
def playback_delete():
    filename = request.form.get("file", "")

    # Chi cho phep xoa file NGAY TRONG thu muc playback_downloads, khong
    # cho duong dan phu (chong path traversal).
    if "/" in filename or "\\" in filename or ".." in filename:
        return "Khong hop le", 400

    path = os.path.join(PLAYBACK_DIR, filename)

    if os.path.isfile(path):
        os.remove(path)

    return redirect(url_for("playback_page"))


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    detector.start()

    HOST = "0.0.0.0"
    PORT = 5000

    try:
        # waitress: WSGI server production-grade, thuan Python (khong can
        # bien dich, chay tot tren ca Windows/Linux/Termux), loai bo canh
        # bao "This is a development server..." cua Flask dev server, va
        # xu ly nhieu ket noi dong thoi on dinh hon (quan trong vi con co
        # luong MJPEG /video_feed giu ket noi mo lien tuc).
        from waitress import serve
        print(f"[WEB] Chay bang waitress (production server) tai http://{HOST}:{PORT}")
        serve(app, host=HOST, port=PORT, threads=8)

    except ImportError:
        print(
            "[WEB] Khong tim thay 'waitress' - dang chay bang Flask dev "
            "server (van hoat dong binh thuong cho dung ca nhan, nhung se "
            "hien canh bao 'development server'). De het canh bao va on "
            "dinh hon, cai: pip install waitress"
        )
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            threaded=True,
        )
