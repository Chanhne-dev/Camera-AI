import cv2
import os
import time
import json
import subprocess
import sys
import signal
from datetime import datetime, time as dt_time
from ultralytics import YOLO
from face_engine import FaceEngine
from frame_grabber import FrameGrabber
from zone_utils import box_in_zone
import events as events_log
import plate_engine
from vehicle_tracker import VehicleTracker


# ==================================================
# CONFIG
# ==================================================

BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)

cam = cfg["camera"]
yolo_cfg = cfg["yolo"]
alert = cfg["alert"]
detect = cfg["detection"]
display = cfg["display"]
face_cfg = cfg.get("face", {"enabled": False})
plate_cfg = cfg.get("plate", {"enabled": False})

# ---- Vung canh bao (ROI) - chong nhieu ----
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

# Luong "chinh" (subtype=0 mac dinh) - do phan giai GOC cua camera
# (vd 2688x1664), dung RIENG de chup anh xe/bien so cho ro net, doc
# bien so duoc. Luong o tren (RTSP_URL, thuong subtype=1 - luong phu,
# do phan giai thap vd 640x352) van dung cho detect/xem truc tiep de
# nhe CPU/bang thong nhu cu - khong doi hanh vi hien tai.
SNAPSHOT_ENABLED = cam.get("snapshot_enabled", True)
SNAPSHOT_SUBTYPE = cam.get("snapshot_subtype", 0)

SNAPSHOT_RTSP_URL = (
    f'rtsp://{cam["username"]}:{cam["password"]}'
    f'@{cam["ip"]}:{cam["rtsp_port"]}'
    f'/cam/realmonitor?channel=1&subtype={SNAPSHOT_SUBTYPE}'
)


# ==================================================
# YOLO
# ==================================================

print("[INFO] Loading YOLO...")
model = YOLO(yolo_cfg["model"])


# ==================================================
# FACE ENGINE
# ==================================================

FACE_ENABLED = face_cfg.get("enabled", False)
face_engine = None

if FACE_ENABLED:
    print("[INFO] Loading face recognition engine...")

    try:
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
        # Khong de loi tai model nhan dien khuon mat lam chet toan bo
        # chuong trinh - tat tinh nang nay va chay tiep phan con lai
        # (YOLO + camera). Thuong gap khi file trong so pretrained bi
        # tai do dang (mang chap chon luc tai lan dau).
        print(f"[ERROR][FACE] Khong tai duoc model nhan dien khuon mat: {e}")
        print("[ERROR][FACE] Da TAT tinh nang nhan dien khuon mat, cac phan khac van chay binh thuong.")
        FACE_ENABLED = False
        face_engine = None

FACE_RECOGNIZE_FPS = face_cfg.get("recognize_fps", 2)
FACE_INTERVAL = 1 / FACE_RECOGNIZE_FPS if FACE_RECOGNIZE_FPS > 0 else 0
ALERT_ONLY_UNKNOWN = face_cfg.get("alert_only_unknown", True)
TRIGGER_UNKNOWN_FACE = alert.get("trigger_unknown_face", ALERT_ONLY_UNKNOWN)
SAVE_UNKNOWN_FACES = face_cfg.get("save_unknown_faces", True)
UNKNOWN_SAVE_DIR = os.path.join(BASE, face_cfg.get("unknown_save_directory", "unknown_faces"))


# ==================================================
# CAMERA
# ==================================================

print("[INFO] Connecting to IMOU...")

# FrameGrabber chay rieng 1 thread, tu dong ket noi lai khi mat mang -
# tranh loi "dung hinh" kinh dien cua cv2.VideoCapture khi RTSP bi treo.
grabber = FrameGrabber(RTSP_URL, reconnect_delay=2.0, stale_timeout=8.0)
grabber.start()

# Luong chat luong cao (chi de chup anh xe/bien so) - khong bat buoc
# phai ket noi thanh cong ngay, cac tinh nang khac van chay binh
# thuong va se tu dong dung duoc luong nay ngay khi no san sang.
snapshot_grabber = None

if SNAPSHOT_ENABLED:
    print("[INFO] Connecting to IMOU (luong chat luong cao - chup xe/bien so)...")
    snapshot_grabber = FrameGrabber(SNAPSHOT_RTSP_URL, reconnect_delay=2.0, stale_timeout=8.0)
    snapshot_grabber.start()

_wait_start = time.time()
while grabber.get_latest() is None and time.time() - _wait_start < 8:
    time.sleep(0.2)

if grabber.get_latest() is None:
    print("[ERROR] Cannot connect to camera!")
    raise SystemExit

print("[INFO] Camera connected")


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

    print("[SAVE]", path)

    return os.path.join(detect["save_directory"], filename)


def save_unknown_face(frame, box):
    if not SAVE_UNKNOWN_FACES:
        return

    os.makedirs(UNKNOWN_SAVE_DIR, exist_ok=True)

    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return

    crop = frame[y1:y2, x1:x2]

    filename = datetime.now().strftime("unknown_%Y%m%d_%H%M%S_%f.jpg")
    path = os.path.join(UNKNOWN_SAVE_DIR, filename)

    cv2.imwrite(path, crop)

    print("[SAVE][FACE]", path)


# ==================================================
# VEHICLE + BIEN SO (crop rieng anh xe / anh bien so)
# ==================================================

PLATE_ENABLED = plate_cfg.get("enabled", False)
VEHICLE_SAVE_DIR = os.path.join(BASE, plate_cfg.get("vehicle_save_directory", "detections/vehicles"))
PLATE_SAVE_DIR = os.path.join(BASE, plate_cfg.get("plate_save_directory", "detections/plates"))
VEHICLE_DEDUP_SECONDS = plate_cfg.get("dedup_seconds", 300)
VEHICLE_IOU_THRESH = plate_cfg.get("iou_thresh", 0.3)

vehicle_tracker = VehicleTracker(
    idle_seconds=VEHICLE_DEDUP_SECONDS,
    iou_thresh=VEHICLE_IOU_THRESH,
    confirm_seconds=plate_cfg.get("confirm_seconds", 2.0),
)
VEHICLE_MIN_CONFIDENCE = plate_cfg.get("min_confidence", 0.5)


def crop_box(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def get_snapshot_frame():
    """
    Tra ve frame chat luong cao (luong chinh cua camera, do phan giai
    goc) neu luong nay dang san sang, de chup anh xe/bien so ro net
    hon. Tra ve None neu tinh nang nay bi tat hoac luong chua ket noi
    duoc / frame qua cu - noi goi se tu dong fallback ve frame do phan
    giai thap dang dung de detect (van chay duoc, chi anh mo hon).
    """

    if snapshot_grabber is None:
        return None

    return snapshot_grabber.get_latest(max_age=3)


def scale_box(box, src_w, src_h, dst_w, dst_h):
    """
    Quy doi toa do 1 hop gioi han tu he truc cua frame co kich thuoc
    (src_w, src_h) sang he truc cua frame co kich thuoc (dst_w, dst_h)
    - dung de anh xa vi tri xe phat hien duoc tren luong do phan giai
    thap (dung de detect) sang dung vi tri tren luong do phan giai cao
    (dung de chup anh ro net).
    """

    x1, y1, x2, y2 = box
    sx = dst_w / src_w
    sy = dst_h / src_h

    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def save_vehicle_image(crop):
    if not detect["save_images"] or crop is None or crop.size == 0:
        return None

    os.makedirs(VEHICLE_SAVE_DIR, exist_ok=True)

    filename = datetime.now().strftime("vehicle_%Y%m%d_%H%M%S_%f.jpg")
    path = os.path.join(VEHICLE_SAVE_DIR, filename)

    cv2.imwrite(path, crop)

    print("[SAVE][VEHICLE]", path)

    return os.path.relpath(path, BASE)


def save_plate_image(crop):
    if not detect["save_images"] or crop is None or crop.size == 0:
        return None

    os.makedirs(PLATE_SAVE_DIR, exist_ok=True)

    filename = datetime.now().strftime("plate_%Y%m%d_%H%M%S_%f.jpg")
    path = os.path.join(PLATE_SAVE_DIR, filename)

    cv2.imwrite(path, crop)

    print("[SAVE][PLATE]", path)

    return os.path.relpath(path, BASE)


# ==================================================
# AUDIO
# ==================================================

audio_process = None


def start_audio():
    global audio_process

    if audio_process is not None and audio_process.poll() is None:
        print("[AUDIO] Already playing")
        return False

    print("[ALERT] Starting camera speaker...")

    speaker = os.path.join(BASE, "speaker.py")

    creation_flags = 0

    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    popen_kwargs = {"creationflags": creation_flags}

    if sys.platform != "win32":
        # dua speaker.py (va ffmpeg/helper con cua no) vao 1 process
        # group rieng de co the dung gon gang toan bo bang os.killpg
        popen_kwargs["preexec_fn"] = os.setsid

    audio_process = subprocess.Popen(
        [sys.executable, speaker],
        **popen_kwargs
    )

    return True


def stop_audio():
    """Dung coi hu ngay lap tuc (dung khi khong con thay mat la)."""

    global audio_process

    if audio_process is None or audio_process.poll() is not None:
        audio_process = None
        return

    print("[AUDIO] Stopping siren...")

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
# DETECTION
# ==================================================

person_since = None
last_alert = 0
last_detection = 0
last_trigger_seen = 0  # thoi diem gan nhat alert_trigger = True (cho grace period)
last_event_types = []  # loai su kien gan nhat kich hoat canh bao (cho log)
last_event_log = 0  # thoi diem gan nhat GHI SU KIEN vao lich su (doc lap voi coi hu)

# ---- Don du lieu cu (anh + su kien) qua han luu tru ----
RETENTION_DAYS = detect.get("retention_days", 15)
RETENTION_CHECK_INTERVAL = 6 * 3600  # kiem tra moi 6 tieng
MEDIA_RETENTION_DIRS = [
    detect["save_directory"],
    plate_cfg.get("vehicle_save_directory", "detections/vehicles"),
    plate_cfg.get("plate_save_directory", "detections/plates"),
    face_cfg.get("unknown_save_directory", "unknown_faces"),
]
last_retention_cleanup = 0  # =0 -> chay lan don dep dau tien ngay khi khoi dong

# x1, y1, x2, y2, confidence, class_id
last_boxes = []

confirm_seconds = alert.get("confirm_seconds", 3)

# Khoan dung: alert_trigger phai TAT lien tuc qua khoang thoi gian nay
# thi moi bi coi la "het nguoi la / het nguoi" va reset bo dem + tat coi.
# Muc dich: tranh 1 frame nhan nham (vd nguoi la bi nhan thanh quen trong
#1 khoanh khac, hoac mat khuat 1 nhip) lam gian doan/reset bo dem oan uong.
GRACE_SECONDS = alert.get("grace_seconds", 1.5)

detect_fps = detect.get("detect_fps", 8)
detect_interval = 1 / detect_fps

# Face recognition state
last_face_check = 0
last_faces = []  # list of {"box":(x1,y1,x2,y2), "name":str, "similarity":float}


# ==================================================
# CLASS CONFIG
# ==================================================

PERSON_CLASS = yolo_cfg.get("person_class", 0)

VEHICLE_CLASSES = yolo_cfg.get(
    "vehicle_classes",
    [2, 3, 5, 7]
)

DETECT_VEHICLES = yolo_cfg.get(
    "detect_vehicles",
    True
)

DETECT_PERSON = yolo_cfg.get(
    "detect_person",
    True
)


CLASS_NAMES = {
    PERSON_CLASS: "PERSON",
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}


# ==================================================
# BUILD YOLO CLASS LIST
# ==================================================

detect_classes = []

if DETECT_PERSON:
    detect_classes.append(PERSON_CLASS)

if DETECT_VEHICLES:
    for class_id in VEHICLE_CLASSES:
        if class_id not in detect_classes:
            detect_classes.append(class_id)

print("[INFO] Detect classes:", detect_classes)


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

    # ---- Don du lieu cu (anh + su kien qua han RETENTION_DAYS) ----
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

        if boxes is not None and len(boxes) > 0:

            last_boxes = []

            for box in boxes:

                x1 = int(box.xyxy[0][0])
                y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2])
                y2 = int(box.xyxy[0][3])

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                last_boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        class_id
                    )
                )

        else:
            last_boxes = []


    # ==================================================
    # FACE RECOGNITION
    # ==================================================

    if FACE_ENABLED and now - last_face_check >= FACE_INTERVAL:

        last_face_check = now

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        last_faces = face_engine.recognize(rgb_frame)


    # ==================================================
    # PERSON FOUND (co loc theo vung canh bao - chong nhieu)
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

    person_found = any(
        box[5] == PERSON_CLASS and box in boxes_in_zone
        for box in last_boxes
    )

    vehicle_found = any(
        box[5] in VEHICLE_CLASSES and box in boxes_in_zone
        for box in last_boxes
    )

    # ==================================================
    # XE + BIEN SO (DOC LAP voi TRIGGER_VEHICLE / coi hu - luon nhan
    # dien bien so moi khi thay xe trong vung canh bao, KE CA khi da
    # tat "xe lam keu coi". Chi rieng viec coi co keu hay khong moi
    # phu thuoc TRIGGER_VEHICLE, viec GHI NHAN xe + doc bien thi luon
    # chay neu tinh nang bien so dang bat.)
    # ==================================================

    if PLATE_ENABLED:
        for box in boxes_in_zone:

            x1, y1, x2, y2, box_conf, class_id = box

            if class_id not in VEHICLE_CLASSES:
                continue

            # Nguong tin cay RIENG cho xe (thuong cao hon nguong chung
            # dung cho canh bao) - han che nhan dien sai vat khac
            # thanh "xe" (vd do vat, bong, anh sang...) do model nho
            # (yolo11n) de nham lan hon tren anh chat luong thap/IR
            # ban dem.
            if box_conf < VEHICLE_MIN_CONFIDENCE:
                continue

            vehicle_box = (x1, y1, x2, y2)

            # Ghi nhan lien tuc + chi log DUY NHAT 1 LAN dung luc xac
            # nhan xong (should_log). Neu chi la nhan dien sai thoang
            # qua (xuat hien roi mat ngay, chua du confirm_seconds) se
            # KHONG BAO GIO duoc ghi thanh su kien.
            if not vehicle_tracker.should_log(vehicle_box, None, now=now):
                continue

            # Uu tien crop tu luong chat luong cao (do phan giai goc
            # cua camera) de doc bien so ro net hon - phai quy doi lai
            # toa do hop gioi han vi 2 luong co do phan giai khac nhau.
            # Neu luong nay chua san sang thi fallback ve frame do
            # phan giai thap dang dung de detect (van chay duoc, chi
            # anh mo hon / kho doc bien so hon).
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

            print(f"[VEHICLE] Xe moi - bien so: {plate_text or 'khong doc duoc'}")

    unknown_face_found = any(
        f["name"] == "Unknown"
        for f in faces_in_zone
    )

    known_face_found = any(
        f["name"] != "Unknown"
        for f in last_faces
    )

    # Loai su kien nao dang thuc su kich hoat canh bao luc nay (co the
    # nhieu loai cung luc, vd vua co nguoi vua co xe)
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
        poly_px = [
            (int(px * frame_w), int(py * frame_h))
            for px, py in ZONE_POINTS
        ]
        for i in range(len(poly_px)):
            p1 = poly_px[i]
            p2 = poly_px[(i + 1) % len(poly_px)]
            cv2.line(frame, p1, p2, (0, 200, 255), 2)


    # ==================================================
    # DRAW (xam neu ngoai vung, mau binh thuong neu trong vung)
    # ==================================================

    for x1, y1, x2, y2, confidence, class_id in last_boxes:

        label = model.names.get(
            class_id,
            f"CLASS {class_id}"
        ).upper()

        in_zone = (x1, y1, x2, y2, confidence, class_id) in boxes_in_zone

        if not in_zone:
            box_color = (120, 120, 120)
        # Người
        elif class_id == PERSON_CLASS:
            box_color = (0, 255, 0)
        # Xe
        else:
            box_color = (255, 165, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )


    # ==================================================
    # DRAW FACES
    # ==================================================

    for face in last_faces:

        fx1, fy1, fx2, fy2 = face["box"]
        name = face["name"]
        similarity = face["similarity"]

        in_zone = face in faces_in_zone

        if not in_zone:
            face_color = (120, 120, 120)
        else:
            # Xanh la = nguoi quen, Do = nguoi la
            face_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(
            frame,
            (fx1, fy1),
            (fx2, fy2),
            face_color,
            2
        )

        cv2.putText(
            frame,
            f"{name} ({similarity:.2f})",
            (fx1, max(fy1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            face_color,
            2
        )


    # ==================================================
    # PERSON CONFIRM (co grace period chong giat/reset oan)
    # ==================================================

    if alert_trigger:
        last_trigger_seen = now
        last_event_types = event_types_firing

    # Con trong "khoan dung" neu vua thay alert_trigger cach day chua qua
    # GRACE_SECONDS -> van coi nhu dang "co nguoi la / co nguoi"
    within_grace = (now - last_trigger_seen) <= GRACE_SECONDS
    active = alert_trigger or (person_since is not None and within_grace)

    if active:

        if person_since is None:
            person_since = now

            if FACE_ENABLED and ALERT_ONLY_UNKNOWN:
                print("[DETECT] Unknown face detected...")
            else:
                print("[DETECT] Person detected...")

        duration = now - person_since

        cv2.putText(
            frame,
            f"Confirm: {duration:.1f}/{confirm_seconds}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # ==================================================
        # GHI SU KIEN VAO LICH SU (luon ghi khi da xac nhan du thoi
        # gian, KHONG phu thuoc coi hu bat/tat hay co dang trong khung
        # gio bao dong hay khong - neu khong Lich su su kien se bo sot
        # cac lan phat hien xay ra ngoai gio bao dong / khi tat coi).
        # ==================================================

        confirmed = duration >= confirm_seconds

        if confirmed and (now - last_event_log >= alert["cooldown"]):

            img_path = save_image(frame)

            for etype in (last_event_types or ["person"]):
                if etype == "vehicle":
                    continue  # da duoc ghi rieng o tren (doc lap voi TRIGGER_VEHICLE)
                events_log.log_event(BASE, etype, image_path=img_path)

            if FACE_ENABLED and "unknown_face" in (last_event_types or []):
                for face in last_faces:
                    if face["name"] == "Unknown":
                        save_unknown_face(frame, face["box"])

            last_event_log = now

        # ==================================================
        # ALERT / COI HU (phat lien tuc trong khi con active)
        # ==================================================

        should_siren = confirmed and alert["enabled"] and in_alert_time()

        if should_siren:

            siren_running = (
                audio_process is not None
                and audio_process.poll() is None
            )

            if not siren_running:
                if start_audio():
                    print("[ALERT] CONFIRMED! Siren ON")
                    last_alert = now

            cv2.putText(
                frame,
                "SIREN ON",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        else:
            # Ngoai khung gio bao dong / chua du thoi gian xac nhan
            # -> dam bao coi khong keu.
            stop_audio()

    else:

        person_since = None
        stop_audio()


    # ==================================================
    # DISPLAY
    # ==================================================

    if display["show_camera"]:

        cv2.imshow(
            display["window_name"],
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    # ==================================================
    # CLEANUP
    # ==================================================

    stop_audio()
    grabber.stop()

    if snapshot_grabber is not None:
        snapshot_grabber.stop()

    cv2.destroyAllWindows()