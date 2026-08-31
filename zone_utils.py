"""
zone_utils.py

Tien ich dung chung cho "vung canh bao" (Region of Interest / ROI) - giup
CHONG NHIEU: chi tinh la co canh bao khi doi tuong (nguoi/xe/khuon mat)
nam TRONG vung duoc chi dinh, thay vi bat ky dau trong khung hinh (vd bo
qua chuyen dong ngoai duong, san nha hang xom, canh cay dong do gio...).

Vung duoc luu trong config.json duoi dang danh sach diem toa do CHUAN HOA
(0.0 - 1.0, khong phu thuoc do phan giai khung hinh):

    "zone_points": [[0.1, 0.3], [0.9, 0.3], [0.9, 0.95], [0.1, 0.95]]

Neu danh sach rong hoac zone_enabled=False, coi nhu KHONG loc vung - moi
noi trong khung hinh deu tinh (hanh vi mac dinh/cu).
"""


def point_in_polygon(x, y, polygon):
    """
    Ray-casting algorithm co dien, khong can them thu vien ngoai
    (khong dung shapely/matplotlib de giu project nhe).

    polygon: list cac (x, y) - co the la toa do pixel hoac chuan hoa,
             mien la x,y truyen vao dung don vi voi polygon.
    """

    if not polygon or len(polygon) < 3:
        return True  # khong co vung hop le -> khong loc gi ca

    inside = False
    n = len(polygon)
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def box_in_zone(box, zone_points_normalized, frame_width, frame_height):
    """
    box: (x1, y1, x2, y2) toa do pixel.
    zone_points_normalized: list [[x,y], ...] toa do chuan hoa 0..1, hoac
        rong/None -> luon tra ve True (khong loc).

    Dung diem "chan" (giua canh duoi bounding box) de kiem tra, vi day la
    vi tri thuc te doi tuong dang dung tren mat dat - chinh xac hon dung
    tam hinh chu nhat (tam hinh nguoi cao se nam lung chung, khong phan
    anh dung vi tri dung chan).
    """

    if not zone_points_normalized:
        return True

    x1, y1, x2, y2 = box

    foot_x = (x1 + x2) / 2
    foot_y = y2

    polygon_px = [
        (px * frame_width, py * frame_height)
        for px, py in zone_points_normalized
    ]

    return point_in_polygon(foot_x, foot_y, polygon_px)
