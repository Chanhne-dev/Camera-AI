"""
frame_grabber.py

Sua loi kinh dien cua cv2.VideoCapture + RTSP tren mang yeu/khong on
dinh (dien thoai, wifi/4G chap chon): ham cap.read() co the TREO VO THOI
HAN khi mat goi tin / camera khong phan hoi, khong tu ket noi lai, lam
"dung hinh" toan bo chuong trinh (dung y nhu trieu chung "camera dung
yen" khi chay web_app.py).

Giai phap chuan cho van de nay:
  1. Chay viec doc frame (cap.read()) trong 1 THREAD RIENG, tach biet
     hoan toan voi vong lap xu ly AI (YOLO/nhan dien mat) - de xu ly
     nang tren dien thoai cham KHONG lam anh huong toi viec doc camera.
  2. Dat tuy chon FFMPEG (qua bien moi truong OPENCV_FFMPEG_CAPTURE_
     OPTIONS) de ep dung RTSP qua TCP (on dinh hon UDP tren mang yeu)
     va dat "stimeout" (timeout socket) de cap.read() KHONG treo vo han
     ma se tra ve loi sau vai giay, cho phep tu ket noi lai.
  3. Mot thread "watchdog" rieng theo doi: neu qua lau khong co frame
     moi (vd 8 giay), CHU DONG goi cap.release() de ngat ket noi dang
     bi treo (goi tu thread khac trong khi thread doc dang bi block
     thuong se lam ham read() tra ve loi ngay, thoat khoi trang thai
     treo) roi mo lai ket noi tu dau.

Su dung:

    grabber = FrameGrabber(rtsp_url)
    grabber.start()
    ...
    frame = grabber.get_latest()   # None neu chua co frame nao / qua cu
    ...
    grabber.stop()
"""

import os
import threading
import time

import cv2


# Ep RTSP dung TCP (on dinh hon UDP khi mat goi) + timeout socket 5s de
# cap.read()/VideoCapture khong bao gio treo vo han. Dung "rw_timeout"
# (tuy chon generic, on dinh qua nhieu phien ban FFmpeg) thay vi
# "stimeout" - tuy chon nay da bi loai bo/doi ten o cac ban FFmpeg moi
# (>= 6.x), gay loi "Unrecognized option" tren may co ffmpeg moi.
# Phai dat TRUOC khi tao cv2.VideoCapture dau tien trong tien trinh.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|rw_timeout;5000000|max_delay;500000"
)


class FrameGrabber:

    def __init__(self, url, reconnect_delay=2.0, stale_timeout=8.0):
        self.url = url
        self.reconnect_delay = reconnect_delay
        self.stale_timeout = stale_timeout

        self.cap = None
        self.cap_lock = threading.Lock()  # bao ve self.cap khoi watchdog

        self.frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0
        self.connected = False

        self.stop_flag = threading.Event()
        self.grab_thread = None
        self.watchdog_thread = None

    # --------------------------------------------------

    def start(self):
        self.stop_flag.clear()

        self.grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)

        self.grab_thread.start()
        self.watchdog_thread.start()

    def stop(self):
        self.stop_flag.set()

        if self.grab_thread is not None:
            self.grab_thread.join(timeout=5)

        if self.watchdog_thread is not None:
            self.watchdog_thread.join(timeout=5)

        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    # --------------------------------------------------

    def _open(self):
        with self.cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

            self.cap = cv2.VideoCapture(self.url)
            self.connected = self.cap.isOpened()

        if self.connected:
            print("[GRABBER] Ket noi camera thanh cong")
        else:
            print("[GRABBER] Khong ket noi duoc camera")

        return self.connected

    def _grab_loop(self):
        self._open()

        while not self.stop_flag.is_set():

            with self.cap_lock:
                cap = self.cap

            if cap is None or not cap.isOpened():
                time.sleep(self.reconnect_delay)
                self._open()
                continue

            # Day la lenh CO THE TREO neu mang chap chon va FFMPEG
            # khong ho tro stimeout tot - watchdog thread se cuu bang
            # cach goi cap.release() tu ben ngoai neu treo qua lau.
            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.2)
                continue

            with self.frame_lock:
                self.frame = frame
                self.last_frame_time = time.time()

    def _watchdog_loop(self):
        while not self.stop_flag.is_set():

            time.sleep(1.0)

            with self.frame_lock:
                last = self.last_frame_time

            if last == 0:
                # Chua tung nhan frame nao - cho _grab_loop tu xu ly
                # (co the dang trong lan ket noi dau tien).
                continue

            if time.time() - last > self.stale_timeout:
                print(
                    f"[GRABBER][WATCHDOG] Khong co frame moi trong "
                    f"{self.stale_timeout}s - nghi camera bi treo, "
                    f"buoc ket noi lai..."
                )

                with self.cap_lock:
                    if self.cap is not None:
                        try:
                            # Goi tu thread khac trong khi _grab_loop co
                            # the dang bi block trong cap.read() - day la
                            # cach chuan de "danh thuc" no (thuong se lam
                            # read() tra ve loi ngay lap tuc).
                            self.cap.release()
                        except Exception:
                            pass

                # Reset moc thoi gian de khong lien tuc trigger lai khi
                # dang trong qua trinh ket noi lai (_grab_loop se tu mo
                # lai cap trong vong lap cua no).
                with self.frame_lock:
                    self.last_frame_time = time.time()

    # --------------------------------------------------

    def get_latest(self, max_age=None):
        """
        Tra ve BAN SAO (copy) cua frame moi nhat, hoac None neu chua co
        frame nao / frame qua cu (qua max_age giay, neu duoc chi dinh).
        """

        with self.frame_lock:
            if self.frame is None:
                return None

            if max_age is not None and (time.time() - self.last_frame_time) > max_age:
                return None

            return self.frame.copy()

    def is_healthy(self):
        with self.frame_lock:
            if self.last_frame_time == 0:
                return False
            return (time.time() - self.last_frame_time) <= self.stale_timeout
