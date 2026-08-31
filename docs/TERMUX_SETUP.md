# Chạy project trên Termux (điện thoại Android)

Có 2 phần: (1) cài môi trường Linux thật bên trong Termux để cài được
`torch`/`opencv`, và (2) chạy `web_app.py` (giao diện web) thay cho
`gui.py` (tkinter) và `main.py` (cửa sổ desktop) — vì điện thoại không có
sẵn màn hình X11.

---

## 1. Vì sao không chạy trực tiếp bằng Python gốc của Termux?

Python cài qua `pkg install python` trong Termux dùng thư viện hệ thống
Android (Bionic libc), khác với Linux thường (glibc). Rất nhiều gói quan
trọng của project này — quan trọng nhất là **`torch`** (PyTorch, dùng
cho YOLO và nhận diện khuôn mặt) — **không có bản build sẵn cho Bionic**,
nên `pip install torch` sẽ báo lỗi không tìm thấy bản phù hợp.

Giải pháp: cài **Ubuntu ARM64 thật** bên trong Termux bằng
`proot-distro`. Lúc đó có glibc Linux như trên máy tính, `pip install`
mọi thứ bình thường.

---

## 2. Cài đặt

Mở Termux, chạy lần lượt:

```bash
# Cap nhat Termux va cai proot-distro
pkg update -y && pkg upgrade -y
pkg install -y proot-distro git wget

# Cai Ubuntu (ban 22.04, on dinh, nhieu goi ho tro)
proot-distro install ubuntu

# Vao trong moi truong Ubuntu
proot-distro login ubuntu
```

Từ đây, mọi lệnh bên dưới chạy **bên trong Ubuntu** (dấu nhắc lệnh sẽ đổi
khác đi, kiểu `root@localhost:~#`):

```bash
apt update && apt upgrade -y

# Cac thu vien he thong can cho opencv, PIL, camera, am thanh
apt install -y python3 python3-pip python3-venv \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    git wget unzip

# Tao thu muc lam viec va giai nen project vao day
mkdir -p ~/camera-ai
cd ~/camera-ai
# (chep file zip project vao day - xem muc 3 ben duoi)
```

### Đưa project vào Ubuntu (proot)

Cách dễ nhất: dùng Termux truy cập bộ nhớ điện thoại, tải file zip
project vào thư mục `Download`, rồi copy vào trong proot.

Ở **Termux gốc** (không phải trong Ubuntu, mở tab/session mới hoặc gõ
`exit` để thoát Ubuntu trước):

```bash
termux-setup-storage      # cho phep Termux truy cap bo nho may
# Neu file zip nam trong Download:
cp ~/storage/downloads/Camera_with_face_recognition.zip ~/
```

Sau đó copy file zip vào bên trong proot-distro Ubuntu (thư mục Ubuntu
nằm ở `~/.termux/proot-distro/installed-rootfs/ubuntu/root/` theo mặc
định của proot-distro):

```bash
proot-distro login ubuntu -- bash -c "mkdir -p /root/camera-ai"
cp ~/Camera_with_face_recognition.zip \
   ~/.termux/proot-distro/installed-rootfs/ubuntu/root/camera-ai/
```

Rồi vào lại Ubuntu và giải nén:

```bash
proot-distro login ubuntu
cd ~/camera-ai
unzip Camera_with_face_recognition.zip
cd Camera_extracted_v3   # hoac ten thu muc sau khi giai nen
```

### Cài thư viện Python

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Cai torch truoc (thu ban thuong, thuong co san wheel cho linux-aarch64)
pip install torch

# Neu lenh tren bao loi "khong tim thay ban phu hop", thu:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Cài `torch` + `ultralytics` + `facenet-pytorch` khá lâu (có thể 15-30
phút tuỳ điện thoại) và tải khá nhiều dữ liệu (vài trăm MB) — nên dùng
wifi, không dùng 4G.

---

## 3. Chạy

**Khuyến nghị: tải trước model nhận diện khuôn mặt bằng script tải
song song (nhanh hơn nhiều so với để chương trình tự tải)**

```bash
bash download_models.sh
```

Script này dùng `aria2c` (16 kết nối song song, tự tiếp tục nếu bị ngắt
giữa chừng) để tải file `20180402-114759-vggface2.pt` (~107MB) — nhanh
hơn rất nhiều so với 1 kết nối đơn mà `facenet-pytorch` tự dùng (từng
ghi nhận chỉ ~60kB/s → mất 30 phút trên mạng yếu, trong khi tải song
song thường chỉ mất 1-3 phút với cùng đường truyền). Nếu chưa có
`aria2c`:

```bash
apt install -y aria2      # trong Ubuntu proot
# hoac
pkg install aria2         # o Termux goc
```

Chạy script này **một lần duy nhất**, tốt nhất trên wifi ổn định. Sau đó
mọi lần chạy `web_app.py`/`main.py` sau này đều dùng file đã tải sẵn,
không cần tải lại.

Rồi chạy chương trình chính:

```bash
python3 web_app.py
```

Thấy dòng `Running on http://0.0.0.0:5000` là thành công. Mở trình
duyệt (Chrome/Firefox) **ngay trên điện thoại đang chạy Termux**, vào:

```
http://127.0.0.1:5000
```

Nếu muốn xem từ **thiết bị khác** (máy tính, điện thoại khác cùng wifi
nhà bạn), tìm địa chỉ IP LAN của điện thoại (Cài đặt → Wifi → chi tiết
mạng, hoặc gõ `ip addr` trong Termux) rồi vào:

```
http://<IP-điện-thoại>:5000
```

---

## 4. Giữ chương trình chạy nền lâu dài trên điện thoại

Android sẽ tự tắt các tiến trình chạy nền để tiết kiệm pin, kể cả
Termux, nếu không được cấu hình đúng:

1. **Tắt tối ưu hoá pin cho Termux**: Cài đặt → Ứng dụng → Termux →
   Pin → chọn "Không giới hạn" / "Không tối ưu hoá".
2. **Giữ Termux chạy nền**: trong Termux, chạy `termux-wake-lock` trước
   khi chạy `web_app.py`, để tránh CPU bị ngủ.
3. **Dùng `tmux`** để chương trình vẫn chạy kể cả khi bạn đóng app
   Termux (không tắt hẳn, chỉ background):
   ```bash
   pkg install tmux    # o Termux goc, ngoai proot
   tmux new -s camera
   # roi vao proot-distro + venv + chay web_app.py nhu binh thuong
   # Nhan Ctrl+B roi D de "thoat" ma khong tat tien trinh
   # Vao lai bang: tmux attach -t camera
   ```
4. Cân nhắc dùng **Termux:Boot** (app riêng trong F-Droid) nếu muốn tự
   động chạy lại sau khi khởi động lại điện thoại.

---

## 5. Hiệu năng — điều cần biết trước

- Chạy đồng thời YOLO + nhận diện khuôn mặt trên CPU điện thoại **chậm
  hơn nhiều** so với máy tính, và khá tốn pin/toả nhiệt nếu chạy liên
  tục 24/7.
- Vào trang **Cấu hình** trên web, giảm các giá trị sau nếu máy giật/lag:
  - `detection_detect_fps` (tần suất YOLO): thử 1
  - `face_recognize_fps` (tần suất nhận diện mặt): thử 1
- Camera IMOU và điện thoại phải **cùng mạng LAN/wifi** với nhau (dùng
  IP nội bộ như `192.168.x.x`, không phải qua internet).
- Nếu điện thoại đời cũ/yếu, cân nhắc dùng model YOLO nhỏ hơn nữa hoặc
  hạ `imgsz` (vd 480) trong tab Cấu hình.

---

## 6. Khác biệt so với bản desktop (`main.py` + `gui.py`)

| | `main.py` + `gui.py` (desktop) | `web_app.py` (Termux/phone) |
|---|---|---|
| Xem camera trực tiếp | Cửa sổ `cv2.imshow` | Trình duyệt, MJPEG stream |
| Cấu hình | Ứng dụng tkinter | Trang web (mobile-friendly) |
| Đăng ký khuôn mặt | `register_face.py --webcam` (cần webcam PC) | Trang "Khuôn mặt" trên web — tải ảnh lên hoặc chụp trực tiếp bằng camera điện thoại |
| Cần X11/màn hình | Có | Không (chạy headless hoàn toàn) |

Cả hai bản dùng chung `config.json`, `face_engine.py`, `speaker.py` —
đổi cấu hình ở bản nào cũng áp dụng cho bản kia.

---

## 7. Lỗi "unexpected EOF..." hoặc "No such file or directory" khi tải model khuôn mặt

Hai lỗi này đều liên quan tới việc tải file trọng số (weights) pretrained
của `facenet-pytorch` (~107MB) bị gián đoạn hoặc bị **tải trùng 2 lần
cùng lúc** (ví dụ bấm nút Bắt đầu/Khởi động lại nhiều lần liên tiếp
trong lúc đang tải — 2 lượt tải cùng tranh băng thông khiến tốc độ tụt
thê thảm, rồi đá nhau file tạm gây lỗi).

**Đã sửa trong bản này**:
- Thêm khoá (lock) chống bấm/gọi trùng Start/Restart — dù bấm nhiều lần
  liên tiếp cũng chỉ có **đúng 1 lượt tải** chạy tại một thời điểm.
- Nút Bắt đầu/Dừng/Khởi động lại tự vô hiệu hoá ngay khi bấm, tránh
  bấm đúp do mạng chậm làm trang phản hồi trễ.
- Nếu model tải lỗi, hệ thống **tự tắt tính năng nhận diện khuôn mặt và
  chạy tiếp** phần camera + YOLO + xem trực tiếp (không crash toàn bộ).

**Cách tránh gặp lại lỗi này hoàn toàn**: chạy `bash download_models.sh`
(mục 3 ở trên) **trước khi** chạy `web_app.py` lần đầu — tải xong 1 lần
bằng nhiều kết nối song song thì không bao giờ phải tải lúc chương trình
đang chạy nữa.

Nếu vẫn dính lỗi cũ từ trước (file dở dang còn sót lại trong cache), xoá
sạch rồi tải lại bằng script:

```bash
rm -rf ~/.cache/torch
bash download_models.sh
```

---

## 8. Live view không hiện / bị đứng

Nếu vào trang chủ mà khung hình camera không hiện (mãi loading):

1. Kiểm tra badge trạng thái ở đầu trang — nếu là `ERROR`, đọc dòng lỗi
   đỏ bên dưới (thường do không kết nối được camera: sai IP/mật khẩu,
   hoặc điện thoại không cùng mạng LAN với camera).
2. Nếu trạng thái là `RUNNING` nhưng ảnh vẫn không hiện, thử tải lại
   trang, hoặc bấm **"🔄 Khởi động lại"**.
3. Xem log trong cửa sổ Termux đang chạy `web_app.py` — mọi lỗi runtime
   đều được in ra đó với tiền tố `[WEB][ERROR]`.

