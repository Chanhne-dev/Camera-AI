# Tính năng nhận diện khuôn mặt

Đã thêm vào project: phát hiện + nhận diện khuôn mặt bằng `facenet-pytorch`
(MTCNN để phát hiện, InceptionResnetV1 pretrained trên VGGFace2 để trích
embedding). Dùng lại `torch` mà `ultralytics` đã cài sẵn nên không cần
framework nặng nào khác.

## 1. Cài thêm thư viện

```
pip install -r requirements.txt
```

(đã thêm `facenet-pytorch` và `Pillow` vào `requirements.txt`)

Lần đầu chạy, `facenet-pytorch` sẽ tự tải file trọng số pretrained
(~100MB) từ GitHub về, cần có mạng.

## 2. Đăng ký người quen

Tạo thư mục `known_faces/<Tên người>/` chứa ảnh, hoặc dùng script có sẵn:

```
# Chụp từ webcam máy tính
python register_face.py --name "Bo" --webcam

# Hoặc dùng ảnh có sẵn
python register_face.py --name "Bo" --images anh1.jpg anh2.jpg anh3.jpg
```

Nên có 3-10 ảnh/người, nhiều góc mặt khác nhau để nhận diện chính xác hơn.
Script tự động xây lại database (`face_db.pkl`) sau khi thêm ảnh. Database
cũng tự rebuild mỗi khi `main.py` khởi động nếu phát hiện ảnh mới/sửa
trong `known_faces/`.

## 3. Cấu hình (trong `config.json`, mục `"face"`)

| Trường | Ý nghĩa |
|---|---|
| `enabled` | Bật/tắt nhận diện khuôn mặt |
| `known_faces_dir` | Thư mục chứa ảnh người quen |
| `similarity_threshold` | Ngưỡng giống nhau để coi là "người quen" (mặc định 0.72). Tăng lên (vd 0.8) nếu vẫn bị nhận nhầm; giảm xuống nếu người quen hay bị báo "Unknown" |
| `margin` | Độ cách biệt tối thiểu giữa người khớp tốt nhất và người đứng thứ 2 (mặc định 0.05). Nếu 2 người bị nhận nhầm lẫn nhau, tăng giá trị này lên |
| `recognize_fps` | Tần suất chạy nhận diện khuôn mặt (tách riêng với `detect_fps` của YOLO, vì MTCNN nặng hơn) |
| `alert_only_unknown` | `true` = chỉ báo động khi thấy **người lạ** (không nhận diện được); `false` = báo động với bất kỳ ai (giữ hành vi cũ, chỉ dựa vào YOLO person) |
| `save_unknown_faces` | Lưu ảnh cận mặt người lạ khi có báo động |
| `unknown_save_directory` | Thư mục lưu ảnh người lạ |
| `top_k` | Số ảnh gần nhất dùng để "bình chọn" khi nhận diện (mặc định 5) |
| `min_face_size` | Bỏ qua khuôn mặt nhỏ hơn N pixel (mặc định 40) — mặt quá nhỏ/xa camera cho embedding không đáng tin |

Ngoài ra, trong mục `"alert"` có thêm:

| Trường | Ý nghĩa |
|---|---|
| `grace_seconds` | Thời gian "khoan dung" (mặc định 1.5s). Nếu người lạ tạm biến mất khỏi khung hình (quay đi, bị che, hoặc 1 nhịp nhận nhầm) chưa quá thời gian này thì bộ đếm xác nhận và còi hú **không bị reset** — tránh còi giật/ngắt quãng do lỗi 1 khung hình |

### Cách nhận diện hoạt động (đã cải tiến)

Thay vì chỉ so ảnh mới với **1 ảnh gần giống nhất** đã đăng ký (dễ bị lệch
nếu đúng ảnh đó chụp xấu), hệ thống giờ lấy **top-K ảnh gần giống nhất**
(mặc định K=5) đã vượt ngưỡng `similarity_threshold`, rồi cho các ảnh đó
"bình chọn" theo tên người — tên nào được nhiều ảnh ủng hộ nhất và có độ
giống trung bình cao nhất sẽ được chọn. Cách này ổn định hơn nhiều, nhất
là khi có vài ảnh đăng ký chất lượng không đều.

Khuôn mặt quá nhỏ (xa camera, dưới `min_face_size` px) sẽ bị bỏ qua thay
vì cố nhận diện — vì embedding từ ảnh nhỏ/mờ không đáng tin, dễ gây
nhận nhầm.

### Logic báo động / còi hú khi phát hiện người lạ

- Mỗi khi phát hiện khuôn mặt "Unknown" liên tục đủ `confirm_seconds`
  giây (trong khung giờ `alert.start`–`alert.end`), còi hú bật và **kêu
  liên tục** cho đến khi không còn thấy người lạ.
- Nhờ `grace_seconds`, hệ thống không bị "giật" còi (bật/tắt liên tục)
  nếu người lạ chỉ tạm khuất 1-2 giây.
- Trong lúc còi đang kêu, ảnh + ảnh cận mặt người lạ vẫn được lưu định kỳ
  theo `alert.cooldown` để làm bằng chứng, không lưu dồn dập gây đầy ổ đĩa.
- Có thể bật/tắt toàn bộ tính năng và chỉnh các thông số này trực tiếp
  trong GUI (`gui.py`), tab **"Khuôn mặt"** và tab **"Cảnh báo"** — sau
  khi lưu, `main.py` sẽ tự khởi động lại để áp dụng cấu hình mới. Tab
  "Khuôn mặt" còn có nút **"Xây lại Database khuôn mặt"** để build lại
  ngay mà không cần khởi động lại chương trình chính.

### Nếu thấy mặt rõ (có khung xanh/đỏ) nhưng KHÔNG nhận ra là ai (báo "Unknown" dù là người quen)

Bật `face.debug = true` (tab "Khuôn mặt" trong GUI, hoặc sửa tay trong
`config.json`), rồi xem log console khi chạy `main.py`. Log sẽ in ra
điểm giống nhau (similarity) thực tế với từng người đã đăng ký, ví dụ:

```
[FACE][DEBUG] Top-5 giong nhat: Bo=0.583, Me=0.401, ... (nguong hien tai: 0.72)
[FACE][DEBUG] -> Unknown: khong anh nao vuot nguong (cao nhat=0.583 < 0.72)
```

Nhìn vào đây để biết chính xác đang bị chặn ở đâu:

1. **Điểm cao nhất luôn thấp hơn ngưỡng một chút** (ví dụ trên: 0.58 vs
   0.72) → `similarity_threshold` đang đặt quá cao so với chất lượng
   ảnh thực tế từ camera (ảnh RTSP nén, ánh sáng yếu, hồng ngoại ban đêm,
   góc nghiêng... đều làm giảm độ giống so với ảnh đăng ký sạch đẹp).
   **Hạ `similarity_threshold` xuống 0.55-0.65** trong tab "Khuôn mặt"
   rồi thử lại — hạ dần từng 0.05 cho tới khi nhận đúng ổn định.
2. **Điểm cao nhất khá tốt nhưng bị báo Unknown vì "quá sát người thứ 2"**
   → giảm `margin` xuống (vd 0.02) nếu 2 người đó thực ra không dễ nhầm
   trong thực tế.
3. **Không thấy dòng "Top-K giong nhat" nào cả, chỉ thấy "Bo qua mat qua
   nho"** → camera đứng quá xa hoặc `min_face_size` đặt quá cao, giảm
   giá trị này xuống (vd 25-30px).
4. **Không thấy log gì từ face_engine** (không phát hiện được mặt) →
   khả năng do ảnh camera quá tối/nhiễu (hồng ngoại đen trắng ban đêm),
   MTCNN được huấn luyện chủ yếu trên ảnh màu ban ngày nên khó phát hiện
   mặt trong ảnh IR. Cân nhắc bật đèn hỗ trợ hoặc chỉ dùng tính năng này
   vào ban ngày/có đủ sáng.
5. Ảnh đăng ký khác quá nhiều so với điều kiện camera thực tế (chụp bằng
   webcam sáng rõ, trong khi camera CCTV nén ảnh + góc cao + xa) cũng là
   nguyên nhân phổ biến — tốt nhất nên **đăng ký bằng chính ảnh chụp từ
   camera IMOU** (dùng `register_face.py --images` với ảnh cắt ra từ
   `detections/` thay vì `--webcam`).

Nhớ tắt `debug` lại sau khi chỉnh xong để log không bị spam liên tục.

### Nếu vẫn bị nhận nhầm người này thành người khác

1. **Thêm ảnh đăng ký chất lượng hơn**: 5-10 ảnh/người, đủ sáng, các góc
   mặt khác nhau (chính diện, hơi nghiêng trái/phải), không đeo khẩu
   trang/kính râm. Ảnh mờ/quá tối là nguyên nhân phổ biến nhất gây nhận
   nhầm.
2. **Tăng `similarity_threshold`** lên 0.78-0.85 nếu vẫn nhầm — đổi lại
   người quen có thể thỉnh thoảng bị báo "Unknown" (tốt hơn là nhận nhầm
   thành người khác, vì hệ thống dùng để báo động).
3. **Tăng `margin`** lên 0.1 nếu 2 người cụ thể hay bị lẫn với nhau.
4. Xoá `face_db.pkl` để buộc xây lại database từ đầu sau khi đổi/thêm
   ảnh (bình thường hệ thống tự phát hiện ảnh mới và tự rebuild, nhưng
   nếu nghi ngờ cache bị lỗi thì xoá tay cho chắc).
5. Từ bản cập nhật này, mỗi ảnh đăng ký được lưu **riêng lẻ** (không lấy
   trung bình cộng thành 1 vector) và khi nhận diện sẽ so khớp với từng
   ảnh rồi lấy khớp gần nhất — chính xác hơn cách lấy trung bình cũ.


## 4. Cách hoạt động trong `main.py`

- Mỗi khung hình, YOLO vẫn phát hiện người/phương tiện như cũ.
- Song song, cứ mỗi `1/recognize_fps` giây, `FaceEngine.recognize()` chạy
  trên toàn khung hình, trả về danh sách khuôn mặt kèm tên (hoặc
  "Unknown") và độ tương đồng.
- Khung hình hiển thị: khung xanh + tên = người quen, khung đỏ +
  "Unknown" = người lạ.
- Nếu `alert_only_unknown = true`, bộ đếm xác nhận (`confirm_seconds`) và
  báo động chỉ kích hoạt khi có khuôn mặt "Unknown" xuất hiện liên tục,
  thay vì bất kỳ người nào — phù hợp với nhà có người thân ra vào thường
  xuyên, chỉ muốn được báo khi có người lạ.

## 5. Giới hạn cần biết

- Nếu người quay lưng / mặt bị che, hệ thống không thấy được mặt nên sẽ
  không tính là "unknown" (khi `alert_only_unknown=true`) — cân nhắc kết
  hợp thêm YOLO person nếu muốn chắc chắn không bỏ sót ai.
- MTCNN chạy trên CPU khá nặng hơn YOLO nano; nếu máy yếu, giảm
  `recognize_fps` xuống 1 hoặc thấp hơn.
- Vì đây là dữ liệu sinh trắc học (khuôn mặt), nếu camera hướng ra khu
  vực công cộng hoặc ghi hình người ngoài gia đình, nên cân nhắc quy định
  pháp luật về bảo vệ dữ liệu cá nhân tại nơi bạn lắp đặt.
