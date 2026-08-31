"""
plate_engine.py

Nhan dien va doc bien so xe tu anh xe da cat rieng (khong dung API
ngoai / khong can model rieng cho bien so - chay hoan toan local, phu
hop may yeu / Termux).

Pipeline (OpenCV co dien + Tesseract OCR):

  1. Anh xe (BGR) -> grayscale -> khu nhieu -> Canny edge
  2. Tim contour, LOC BO cac vung KHONG GIONG bien so that:
       - Ty le khung (aspect ratio) phai gan giong bien so xe VN
         (bien 1 dong dai: ~3.0-5.5; bien vuong 2 dong cua xe may:
         ~0.9-1.6)
       - Kich thuoc vua phai so voi anh xe (khong qua nho/qua to)
       - "Do dac" (fill ratio = dien tich contour / dien tich khung
         bao) phai cao - LOAI BO cac vat MONG/CONG nhu chan chong,
         ong xa, day dien... vi nhung vat nay tinh co co the co ty le
         khung giong bien so nhung KHONG phai la 1 khoi dac hinh chu
         nhat nhu bien so that.
     Cac vung con lai duoc CHAM DIEM uu tien theo:
       - Vi tri CANG THAP trong anh xe cang tot (bien so xe may/o to
         VN thuong gan mat dat, o phan duoi cua xe - khac voi guong,
         tay lai, mu bao hiem... thuong o phan tren)
       - Do SANG trung binh CANG CAO cang tot (bien so VN nen trang/
         vang, phan quang manh, đặc biệt duoi anh sang IR ban dem -
         khac voi cac chi tiet kim loai/nhua toi mau khac cua xe)
  3. Crop vung diem cao nhat, phong to + threshold de OCR de doc hon
  4. Doc chu bang pytesseract (chi cho phep ky tu xuat hien tren bien
     so VN, cac ky tu khac se bi loai)

Vi khong dung model AI rieng cho bien so nen ty le nhan dung se KHONG
bang cac giai phap AI chuyen dung. Neu khong doc duoc chu (bien mo,
goc chup xau, thieu sang...), read_plate() van tra ve anh vung diem
cao nhat (neu tim thay) de xem lai bang mat thuong, chi rieng text se
la None.

Neu may khong cai pytesseract / tesseract-ocr, module van hoat dong
binh thuong (chi khong doc duoc chu, van cat/luu duoc anh xe + vung
nghi la bien so).
"""

import re

import cv2
import numpy as np

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


PLATE_CHAR_WHITELIST = "ABCDEFGHKLMNPSTUVXYZ0123456789.-"

# "Do dac" toi thieu (dien tich contour / dien tich khung bao). Bien so
# that la 1 khoi dac hinh chu nhat (~0.85-1.0). Vat mong/cong (chan
# chong, ong xa, day dien...) co dien tich contour rat nho so voi
# khung bao cua no (~0.1-0.3) du ty le khung co the trung hop giong
# bien so - day la bo loc quan trong nhat de tranh cat nham vi tri.
MIN_FILL_RATIO = 0.45


def _candidate_plate_boxes(vehicle_crop):
    """
    Tra ve danh sach (x1, y1, x2, y2) cac vung nghi la bien so trong
    anh xe da cat, sap xep uu tien vung diem cao nhat truoc (dac, to,
    o phia duoi anh, sang mau).
    """

    h, w = vehicle_crop.shape[:2]

    if h < 20 or w < 20:
        return []

    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    frame_area = h * w
    candidates = []

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)

        if cw == 0 or ch == 0:
            continue

        box_area = cw * ch
        aspect = cw / ch
        area_ratio = box_area / frame_area

        # Bien so VN: bien dai 1 dong (~3.0-5.5) hoac bien vuong 2 dong
        # cua xe may (~0.9-1.6). Khong qua nho / qua to so voi anh xe.
        looks_like_plate = (3.0 <= aspect <= 5.5) or (0.9 <= aspect <= 1.6)
        size_ok = 0.005 <= area_ratio <= 0.35

        if not (looks_like_plate and size_ok):
            continue

        # Loai bo vat mong/cong (chan chong, ong xa, day dien...) - chi
        # giu lai cac vung LA 1 KHOI DAC hinh chu nhat nhu bien so that.
        contour_area = cv2.contourArea(c)
        fill_ratio = contour_area / box_area if box_area > 0 else 0

        if fill_ratio < MIN_FILL_RATIO:
            continue

        # Diem uu tien vi tri: bien so xe thuong o phan DUOI cua anh xe
        # (gan mat dat) - cac chi tiet o phan tren (guong, tay lai...)
        # bi giam diem manh.
        center_y_ratio = (y + ch / 2) / h
        position_score = 0.3 + 0.7 * center_y_ratio  # 0.3 (tren cung) -> 1.0 (duoi cung)

        # Diem uu tien do sang: bien so VN (trang/vang) thuong SANG hon
        # cac chi tiet kim loai/nhua toi mau khac cua xe, dac biet ro
        # duoi anh sang hong ngoai (IR) ban dem.
        region_gray = gray[y:y + ch, x:x + cw]
        brightness = float(np.mean(region_gray)) / 255.0 if region_gray.size else 0.0
        brightness_score = 0.4 + 0.6 * brightness  # 0.4 (toi) -> 1.0 (sang)

        score = box_area * fill_ratio * position_score * brightness_score

        candidates.append((score, (x, y, x + cw, y + ch)))

    candidates.sort(key=lambda item: item[0], reverse=True)

    return [box for _, box in candidates[:5]]


def _clean_plate_text(raw):
    if not raw:
        return None

    text = re.sub(r"[^A-Z0-9]", "", raw.upper())

    # Bien so VN thuong dai 7-9 ky tu (vd 29A12345, 30F12345). Loai bo
    # ket qua qua ngan - thuong la nhieu OCR chu khong phai bien that.
    if len(text) < 6:
        return None

    return text


def read_plate(vehicle_crop):
    """
    Nhan anh xe da cat (BGR numpy array), tra ve:
        (plate_text_or_None, plate_crop_or_None)

    plate_text: chuoi bien so da doc duoc (vd "29A12345"), hoac None
                neu khong doc duoc / khong cai tesseract.
    plate_crop: anh (numpy array) vung diem cao nhat (nghi la bien so)
                de luu lai xem sau, hoac None neu khong tim thay vung
                nao phu hop.
    """

    if vehicle_crop is None or vehicle_crop.size == 0:
        return None, None

    boxes = _candidate_plate_boxes(vehicle_crop)

    if not boxes:
        return None, None

    best_crop = None

    for (x1, y1, x2, y2) in boxes:
        crop = vehicle_crop[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        if best_crop is None:
            # Luon co it nhat 1 anh vung diem cao nhat de luu lai, ke
            # ca khong doc duoc chu.
            best_crop = crop

        if not TESSERACT_AVAILABLE:
            continue

        # Phong to + nhi phan hoa de OCR de doc hon voi anh nho/mo.
        scale = max(1, 200 // max(crop.shape[0], 1))
        big = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        try:
            raw = pytesseract.image_to_string(
                thresh,
                config=(
                    "--psm 7 -c tessedit_char_whitelist=" + PLATE_CHAR_WHITELIST
                ),
            )
        except Exception:
            continue

        text = _clean_plate_text(raw)

        if text:
            return text, crop

    return None, best_crop
