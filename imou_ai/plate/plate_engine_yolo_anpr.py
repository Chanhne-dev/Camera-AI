"""
plate_engine_yolo_anpr.py

Doc bien so xe bang kien truc YOLOv8 + EasyOCR, phong theo huong tiep can
cua repo:
    https://github.com/computervisioneng/automatic-number-plate-recognition-python-yolov8

Kien truc gom 2 model rieng biet (dung dung tinh than cua repo tren):

    1. Model YOLO PHAT HIEN PHUONG TIEN: main.py/web_app.py DA CO SAN
       (model YOLO COCO thong dung dang dung de phat hien nguoi/xe) -
       module nay KHONG nap lai, chi nhan vao anh xe da duoc cat san
       (vehicle_crop) tu ket qua cua model do.

    2. Model YOLO PHAT HIEN BIEN SO (rieng, chuyen train cho bai toan
       "tim khung bien so ben trong anh xe") - module nay tu nap.

    3. EasyOCR: doc chu/so ben trong khung bien so da cat.

Khac voi repo goc (video giao thong chau Au, dung them SORT de theo dau
xe qua nhieu khung hinh, chi doc bien 1 lan cho ca doan video), du an
nay la camera an ninh gia dinh nen:
    - KHONG can SORT: viec "1 xe chi tao 1 su kien" da duoc xu ly rieng
      boi EventTracker/VehicleTracker (chong spam theo IoU + confirm
      thoi gian) trong main.py/web_app.py.
    - Doc bien MOI LAN 1 xe duoc XAC NHAN la su kien moi (khong phai
      moi khung hinh), de tiet kiem CPU.
    - Dinh dang bien so VN khac chau Au nen KHONG ep theo 1 khuon dang
      co dinh (repo goc kiem tra dung 7 ky tu kieu bien EU) - o day chi
      chuan hoa ky tu (chu hoa, bo ky tu la) roi tra ve nguyen van.


*** QUAN TRONG - BAN CAN TU CHUAN BI MODEL PHAT HIEN BIEN SO ***

Repo goc KHONG kem san file trong so (.pt) cho model phat hien bien so -
tac gia repo chi chia se file nay qua Patreon (tra phi), README chi noi
ro dataset + huong dan tu train:

    - Dataset (mien phi, Roboflow Universe):
      https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4
    - Huong dan tung buoc tu train YOLOv8 tren du lieu rieng (cung tac
      gia):
      https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide

Ban co 2 lua chon:
    (a) Tu train model theo huong dan tren (khuyen nghi neu muon do
        chinh xac cao voi bien so Viet Nam - nen tu chup/gan nhan them
        anh bien so VN thuc te, vi dataset goc la bien so quoc te).
    (b) Tim 1 model YOLO phat hien bien so co san khac ma ban tin
        tuong (vd tim tren Roboflow Universe voi tu khoa "license
        plate detection Vietnam").

Sau khi co file .pt, dat duong dan vao config.json:

    "plate": {
        "enabled": true,
        "detector_model": "models/license_plate_detector.pt",
        "detector_confidence": 0.35,
        "ocr_min_confidence": 0.35,
        "save_directory": "detections/plates"
    }

Neu khong tim thay file model (hoac chua cai easyocr), tinh nang tu
dong TAT va chi in 1 dong canh bao - KHONG lam crash toan bo chuong
trinh.
"""

import os
import re
import threading

import cv2

_lock = threading.Lock()

_detector = None
_detector_load_failed = False

_reader = None
_reader_load_failed = False

_cfg = {
    "enabled": True,
    "detector_model": "models/license_plate_detector.pt",
    "detector_confidence": 0.35,
    "ocr_min_confidence": 0.35,
}


def configure(**kwargs):
    """
    Goi 1 lan luc khoi dong (tu main.py / web_app.py) de truyen cau
    hinh doc tu config.json vao module nay, truoc khi goi read_plate().
    Cac key khong duoc truyen se giu gia tri mac dinh o tren.
    """
    _cfg.update({k: v for k, v in kwargs.items() if v is not None})


def _get_detector():
    global _detector, _detector_load_failed

    if _detector is not None or _detector_load_failed:
        return _detector

    with _lock:
        if _detector is not None or _detector_load_failed:
            return _detector

        model_path = _cfg["detector_model"]

        if not os.path.exists(model_path):
            print(
                f"[PLATE][WARNING] Khong tim thay model phat hien bien so: "
                f"'{model_path}'. Tinh nang doc bien so se TAT. Xem huong "
                f"dan chuan bi model o dau file plate_engine_yolo_anpr.py."
            )
            _detector_load_failed = True
            return None

        try:
            from ultralytics import YOLO
            _detector = YOLO(model_path)
            print(f"[PLATE] Da nap model phat hien bien so: {model_path}")
        except Exception as e:
            print(f"[PLATE][ERROR] Khong nap duoc model phat hien bien so: {e}")
            _detector_load_failed = True
            _detector = None

    return _detector


def _get_reader():
    global _reader, _reader_load_failed

    if _reader is not None or _reader_load_failed:
        return _reader

    with _lock:
        if _reader is not None or _reader_load_failed:
            return _reader

        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=False)
            print("[PLATE] Da nap EasyOCR")
        except Exception as e:
            print(
                f"[PLATE][ERROR] Khong nap duoc EasyOCR (chay "
                f"'pip install easyocr' neu chua cai): {e}"
            )
            _reader_load_failed = True
            _reader = None

    return _reader


def _detect_plate_box(vehicle_crop):
    """
    Dung model YOLO chuyen bien so de tim khung bien so trong 1 anh xe
    da cat san. Tra ve (x1, y1, x2, y2) cua khung co do tin cay cao
    nhat, hoac None neu khong tim thay / model chua san sang.
    """

    detector = _get_detector()

    if detector is None:
        return None

    try:
        results = detector.predict(
            vehicle_crop,
            conf=_cfg["detector_confidence"],
            device="cpu",
            verbose=False,
        )
    except Exception as e:
        print(f"[PLATE][ERROR] Loi khi chay model phat hien bien so: {e}")
        return None

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    best_box = None
    best_conf = -1.0

    for box in boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            best_box = (x1, y1, x2, y2)

    return best_box


def _preprocess_for_ocr(plate_crop):
    """
    Chuan hoa anh bien so truoc khi dua vao OCR: chuyen xam + threshold
    nhi phan hoa (Otsu) - giup EasyOCR doc on dinh hon voi anh bien so
    thuc te (phan chieu anh sang, tuong phan thap, nhieu buoi toi...).
    Cung phong to neu bien qua nho, vi OCR doc chu nho rat de sai.
    """

    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if h < 60:
        scale = 60 / h
        gray = cv2.resize(
            gray, (max(1, int(w * scale)), 60), interpolation=cv2.INTER_CUBIC
        )

    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return thresh


def _clean_plate_text(raw_text):
    """
    Chi giu lai chu cai + chu so, viet hoa toan bo. Bien so VN co the co
    dau '-' va '.' nhung EasyOCR thuong tra ve tung cum rieng le nen o
    day chi loc ky tu co y nghia - KHONG ep theo 1 dinh dang co dinh
    (khac cach kiem tra bien 7 ky tu chau Au trong repo goc).
    """

    return re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()


def read_plate(vehicle_crop):
    """
    Ham chinh, goi tu main.py / web_app.py.

    vehicle_crop: anh xe da cat (BGR, numpy array - tu ket qua model
        YOLO phat hien phuong tien co san).

    Tra ve tuple (plate_text, plate_crop_image):
        plate_text: chuoi da chuan hoa, hoac None neu khong doc duoc /
            tinh nang dang tat / chua co model.
        plate_crop_image: anh bien so da cat (BGR, numpy array) de luu
            lam bang chung, hoac None neu khong tim thay khung bien so.
    """

    if not _cfg.get("enabled", True):
        return None, None

    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, None

    box = _detect_plate_box(vehicle_crop)

    if box is None:
        return None, None

    x1, y1, x2, y2 = box
    h, w = vehicle_crop.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None, None

    plate_crop = vehicle_crop[y1:y2, x1:x2]

    reader = _get_reader()

    if reader is None:
        return None, plate_crop

    processed = _preprocess_for_ocr(plate_crop)

    try:
        ocr_results = reader.readtext(processed)
    except Exception as e:
        print(f"[PLATE][ERROR] Loi OCR: {e}")
        return None, plate_crop

    if not ocr_results:
        return None, plate_crop

    # Sap xep cac doan chu doc duoc theo vi tri tren-xuong-duoi, trai-
    # qua-phai (bien so xe may VN thuong co 2 dong) truoc khi ghep lai.
    ocr_results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

    parts = [
        _clean_plate_text(text)
        for _, text, conf in ocr_results
        if conf >= _cfg["ocr_min_confidence"]
    ]
    parts = [p for p in parts if p]

    plate_text = "".join(parts) if parts else None

    return plate_text, plate_crop
