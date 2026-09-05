"""
vehicle_tracker.py

Chong spam su kien "xe" VA chong bao dong gia do YOLO nhan nham 1
khung hinh don le (vd do vat/anh sang giong hinh xe trong 1-2 frame
roi mat ngay). Gom 2 co che:

  1. XAC NHAN (confirm_seconds): 1 "xe" chi duoc GHI SU KIEN khi da
     duoc thay LIEN TUC (khop vi tri/bien so qua nhieu lan quet) it
     nhat `confirm_seconds` giay. Neu chi xuat hien thoang qua 1-2
     frame roi bien mat (thuong la nhan dien sai / nhieu anh) thi
     KHONG BAO GIO duoc ghi thanh su kien - giong het co che
     "confirm_seconds" cua nguoi/khuon mat da co san.

  2. CHONG SPAM (idle_seconds): sau khi 1 xe DA duoc xac nhan + ghi su
     kien, no se KHONG duoc ghi lai them lan nao nua cho toi khi that
     su "vang mat" (khong con khop voi vi tri/bien so nao) qua
     idle_seconds - vd xe dau lien tuc truoc cong se chi tao 1 su
     kien duy nhat, khong spam moi frame/moi giay.

Nhan dang cung 1 xe dua tren 2 tin hieu (uu tien bien so neu doc
duoc, vi day la tin hieu chinh xac nhat):
  - Bien so da doc duoc trung nhau.
  - Vi tri hop gioi han (IoU) trung nhau kha nhieu - dung khi khong
    doc duoc bien (bien mo, goc chup xau, thieu sang...).
"""

import time


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class VehicleTracker:

    def __init__(self, idle_seconds=300, iou_thresh=0.3, confirm_seconds=2.0):
        # idle_seconds: xe phai "vang mat" qua khoang nay (khong con
        # khop voi ban ghi nao) thi lan sau xuat hien moi duoc coi la
        # xe moi (ghi su kien moi).
        self.idle_seconds = idle_seconds
        self.iou_thresh = iou_thresh
        # confirm_seconds: xe phai duoc thay LIEN TUC it nhat khoang
        # nay truoc khi duoc coi la that (khong phai nhan dien sai
        # thoang qua) va duoc phep ghi su kien.
        self.confirm_seconds = confirm_seconds
        self._tracked = []  # [{"box", "plate", "first_seen", "last_seen", "logged"}]

    def _prune(self, now):
        self._tracked = [
            v for v in self._tracked
            if now - v["last_seen"] <= self.idle_seconds
        ]

    def _find(self, box, plate):
        for v in self._tracked:
            if plate and v["plate"] and v["plate"] == plate:
                return v
            if _iou(box, v["box"]) >= self.iou_thresh:
                return v
        return None

    def peek(self, box, plate=None, now=None):
        """
        Kiem tra (khong ghi/cap nhat gi) xem xe nay da duoc GHI SU
        KIEN roi hay chua - dung de re nhanh truoc khi chay OCR ton
        CPU cho 1 xe da biet chac chan roi.
        """

        now = now if now is not None else time.time()
        self._prune(now)

        match = self._find(box, plate)

        return match is not None and match["logged"]

    def should_log(self, box, plate=None, now=None):
        """
        Goi ham nay MOI LAN thay 1 "xe" trong khung hinh (moi frame).
        Tra ve True DUY NHAT 1 LAN - dung luc xe do vua duoc XAC NHAN
        (da thay lien tuc du confirm_seconds) VA CHUA tung duoc ghi su
        kien - day la thoi diem nen chay OCR + ghi su kien.

        Tra ve False trong moi truong hop khac: xe qua moi/chua du
        thoi gian xac nhan (co the la nhan dien sai thoang qua), hoac
        xe da duoc ghi su kien roi (dang chong spam).

        Luon tu dong cap nhat vi tri + bien so + thoi diem thay gan
        nhat, du ket qua tra ve la gi.
        """

        now = now if now is not None else time.time()
        self._prune(now)

        match = self._find(box, plate)

        if match is None:
            self._tracked.append({
                "box": box,
                "plate": plate,
                "first_seen": now,
                "last_seen": now,
                "logged": self.confirm_seconds <= 0,
            })
            return self.confirm_seconds <= 0

        match["box"] = box
        match["last_seen"] = now

        if plate and not match["plate"]:
            match["plate"] = plate

        if match["logged"]:
            return False

        if now - match["first_seen"] >= self.confirm_seconds:
            match["logged"] = True
            return True

        return False
