# Vùng cảnh báo, nhiều loại sự kiện, lịch sử & phát lại thẻ nhớ

Các tính năng này chỉ có trên `web_app.py` (giao diện web). `main.py`
(bản desktop) cũng đã được cập nhật để áp dụng **vùng cảnh báo** và
**nhiều loại sự kiện** khi chạy phát hiện (đọc từ `config.json`), nhưng
chưa có giao diện đồ hoạ để chỉnh 2 mục này trong `gui.py` — muốn đổi ở
bản desktop thì sửa `config.json` bằng tay, hoặc mở `web_app.py` để chỉnh
qua giao diện web rồi chạy lại `main.py` (cả 2 bản dùng chung file cấu
hình).

## 1. Vùng cảnh báo (chống nhiễu)

Trang **"Vùng cảnh báo"**: chạm trên ảnh chụp trực tiếp để vẽ 1 đa giác.
Hệ thống chỉ tính cảnh báo khi người/xe/khuôn mặt có **điểm chân** (giữa
cạnh dưới khung nhận diện) nằm **trong vùng này**. Đối tượng ngoài vùng
vẫn được vẽ (màu xám) để bạn biết hệ thống có thấy nhưng không tính —
hữu ích để chống nhiễu từ: xe cộ ngoài đường, sân nhà hàng xóm, cây cối
đung đưa ở rìa khung hình, v.v.

Để trống (0 điểm) hoặc tắt checkbox = quét toàn bộ khung hình như cũ.

Lưu ý: mỗi lần đổi vùng, hệ thống tự khởi động lại (mất vài giây để nạp
lại model).

## 2. Nhiều loại sự kiện cảnh báo

Trong tab **Cấu hình → Cảnh báo**, giờ có 3 công tắc độc lập:

- **Người** (`trigger_person`) — YOLO phát hiện người
- **Xe cộ** (`trigger_vehicle`) — YOLO phát hiện ô tô/xe máy/xe tải/xe buýt
- **Người lạ** (`trigger_unknown_face`) — khuôn mặt không nhận diện được
  (cần bật tính năng Khuôn mặt)

Bật loại nào thì loại đó mới kích hoạt còi hú + xác nhận thời gian. Có
thể bật nhiều loại cùng lúc.

## 3. Lịch sử sự kiện

Mỗi lần còi hú kích hoạt (kể cả các lần lưu lặp lại theo `cooldown` trong
lúc còi vẫn đang kêu), 1 sự kiện được ghi vào `events.jsonl` kèm loại
(người/xe/người lạ), thời điểm, và đường dẫn ảnh chụp. Trang **"Sự
kiện"** hiển thị dạng lưới ảnh, lọc theo loại, phân trang.

`events.jsonl` là file text thường (1 dòng JSON/sự kiện) — dễ đọc, dễ
sao lưu, không cần cài thêm database.

## 4. Phát lại video từ thẻ nhớ (THỬ NGHIỆM)

Trang **"Phát lại"** giờ có **thanh timeline kéo-thả** để chọn khoảng
thời gian (giống kiểu app IMOU) thay vì gõ tay giờ bắt đầu/kết thúc —
kéo 2 đầu thanh để chọn, tối đa 30 phút/lần. Vẫn có 2 ô giờ chính xác
bên dưới để gõ tay nếu cần, đồng bộ 2 chiều với thanh kéo.

Lưu ý: đây **không phải** timeline "biết trước đoạn nào có ghi hình" như
app chính hãng (app IMOU truy vấn trực tiếp từ cloud/camera để biết chính
xác khung giờ nào có dữ liệu) — thanh này chỉ giúp **chọn khoảng giờ
nhanh hơn**, còn có dữ liệu ghi hình trong khoảng đó hay không thì phải
thử tải mới biết (do tính năng truy vấn "khung giờ nào có ghi hình" cần
một giao thức riêng của camera chưa được xác nhận hoạt động ổn định qua
reverse-engineering, xem thêm ghi chú "THỬ NGHIỆM" bên dưới).

Hệ thống gọi `ffmpeg` tải đoạn ghi hình qua URL RTSP phát lại chuẩn của
dòng camera Dahua/Imou:

```
rtsp://user:pass@ip:554/cam/playback?channel=1&subtype=0
      &starttime=YYYY-MM-DDTHH:MM:SS&endtime=YYYY-MM-DDTHH:MM:SS
```

Video tải về lưu ở `playback_downloads/`, xem trực tiếp bằng trình phát
video HTML5 ngay trên trang (có tua/seek, không như live view MJPEG).

**Đây là tính năng thử nghiệm — mình KHÔNG thể đảm bảo hoạt động trên
camera của bạn**, vì:

- URL này được xác định qua reverse-engineering (xem
  `imou-life/docs/p2p-media-flow.md`), quan sát qua kênh P2P nội bộ của
  app IMOU, chưa được xác nhận hoạt động trực tiếp qua LAN IP:554 trên
  mọi đời camera/firmware.
- Camera phải có **thẻ nhớ đã lắp + đang ghi hình** trong khoảng thời
  gian bạn chọn.
- Một số model/firmware có thể trả lỗi "InterfaceNotFound" hoặc đóng kết
  nối ngay (đã ghi nhận trên 1 model khác trong quá trình reverse-
  engineering).

Nếu báo lỗi, thử: khoảng thời gian ngắn hơn (5-10 phút), kiểm tra thẻ
nhớ còn hoạt động qua app IMOU chính hãng, hoặc coi đây là giới hạn phần
cứng/firmware và dùng app IMOU để xem lại (luôn hoạt động vì dùng giao
thức chính hãng đầy đủ, không phải suy luận từ tài liệu công khai).

## Cần cài thêm gì không?

Không — `ffmpeg` đã có trong danh sách cài đặt của `TERMUX_SETUP.md`
(mục 2). Nếu chạy trên máy tính thường, cài `ffmpeg` qua trình quản lý
gói của hệ điều hành (vd `apt install ffmpeg`, hoặc tải từ
ffmpeg.org cho Windows) nếu chưa có sẵn.

Lưu ý: các bản `ffmpeg` mới (>= 6.x) đã đổi/loại bỏ tên tuỳ chọn
`-stimeout`. Hệ thống tự thử lần lượt `-rw_timeout`, `-timeout`,
`-stimeout`, rồi không dùng tuỳ chọn timeout nào cả — nên chạy được trên
mọi phiên bản `ffmpeg` mà không cần bạn chỉnh gì thêm.

## Server chạy web_app.py

Nếu có cài `pip install waitress` (đã có trong `requirements.txt`),
`web_app.py` tự dùng `waitress` thay cho server phát triển mặc định của
Flask — hết cảnh báo "This is a development server..." và ổn định hơn
khi có nhiều kết nối cùng lúc (vd vừa xem live view vừa tải video phát
lại). Nếu chưa cài `waitress`, chương trình vẫn chạy bình thường bằng
Flask dev server, chỉ là còn hiện cảnh báo đó (không ảnh hưởng gì tới
hoạt động thực tế, chỉ là khuyến cáo của Flask).
