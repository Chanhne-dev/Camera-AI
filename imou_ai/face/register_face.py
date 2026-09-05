"""
register_face.py

Cong cu dang ky khuon mat nguoi quen vao thu muc known_faces/.

Cach dung (chay tu THU MUC GOC du an, khong phai tu ben trong imou_ai/):

    1) Chup tu webcam (may tinh, khong phai camera IMOU):
       python -m imou_ai.face.register_face --name "Nguyen Van A" --webcam

       -> Nhan phim SPACE de chup 1 anh, chup khoang 5-10 anh voi
          goc mat / bieu cam khac nhau, nhan ESC/Q khi xong.

    2) Dung anh co san:
       python -m imou_ai.face.register_face --name "Nguyen Van A" --images anh1.jpg anh2.jpg

    (Van chay duoc bang duong dan file truc tiep, vd
     "python imou_ai/face/register_face.py ...", vi file nay tu them
     thu muc goc du an vao sys.path - nhung khuyen nghi dung "-m" o tren
     cho chuan.)

Sau khi dang ky xong, chay lai main.py (hoac goi FaceEngine().build_db())
de he thong cap nhat database nhan dien.
"""

import argparse
import json
import os
import shutil
import sys

import cv2


# File nay nam trong imou_ai/face/, con config.json/data/ nam o thu muc
# goc du an (2 cap tren). Them thu muc goc vao sys.path TRUOC de:
#   (a) BASE tinh dung vi tri config.json/data/
#   (b) "from imou_ai.face.face_engine import FaceEngine" o duoi hoat
#       dong dung ke ca khi file nay duoc chay TRUC TIEP (khong qua -m)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

BASE = PROJECT_ROOT
KNOWN_FACES_DIR = os.path.join(BASE, "data", "known_faces")


def ensure_person_dir(name):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir


def next_index(person_dir):
    existing = [
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return len(existing) + 1


def capture_from_webcam(name, camera_index=0):
    person_dir = ensure_person_dir(name)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Khong mo duoc webcam")
        return

    print("[INFO] Nhan SPACE de chup anh, Q hoac ESC de thoat")
    print(f"[INFO] Anh se luu vao: {person_dir}")

    saved = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Khong doc duoc frame")
            break

        preview = frame.copy()

        cv2.putText(
            preview,
            f"{name} | Da chup: {saved} | SPACE=chup  Q=thoat",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Dang ky khuon mat", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            idx = next_index(person_dir)
            path = os.path.join(person_dir, f"{idx}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            print(f"[SAVE] {path}")

        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"[DONE] Da chup {saved} anh cho {name}")


def import_images(name, image_paths):
    person_dir = ensure_person_dir(name)

    imported = 0

    for src in image_paths:
        if not os.path.isfile(src):
            print(f"[WARNING] Khong tim thay file: {src}")
            continue

        idx = next_index(person_dir)
        ext = os.path.splitext(src)[1].lower() or ".jpg"
        dst = os.path.join(person_dir, f"{idx}{ext}")

        shutil.copy(src, dst)
        imported += 1
        print(f"[COPY] {src} -> {dst}")

    print(f"[DONE] Da import {imported} anh cho {name}")


def rebuild_database():
    # Import cham (lazy) vi facenet-pytorch mat vai giay de nap model
    from imou_ai.face.face_engine import FaceEngine

    print("[INFO] Dang xay lai database nhan dien...")

    config_path = os.path.join(BASE, "config.json")
    face_cfg = {}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            face_cfg = json.load(f).get("face", {})

    known_dir = os.path.join(BASE, face_cfg.get("known_faces_dir", "data/known_faces"))
    cache_path = os.path.join(BASE, face_cfg.get("db_cache", "data/face_db.pkl"))

    engine = FaceEngine(
        known_faces_dir=known_dir,
        db_cache_path=cache_path,
    )

    engine.build_db()

    print("[DONE] Database da duoc cap nhat.")


def main():
    parser = argparse.ArgumentParser(description="Dang ky khuon mat nguoi quen")

    parser.add_argument("--name", required=False, help="Ten nguoi can dang ky")
    parser.add_argument("--webcam", action="store_true", help="Chup anh tu webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="Chi so webcam (mac dinh 0)")
    parser.add_argument("--images", nargs="*", default=None, help="Danh sach duong dan anh co san")
    parser.add_argument("--no-rebuild", action="store_true", help="Khong tu dong xay lai database sau khi them anh")
    parser.add_argument("--rebuild-only", action="store_true", help="Chi xay lai database, khong them anh moi")

    args = parser.parse_args()

    if args.rebuild_only:
        rebuild_database()
        return

    if not args.name:
        parser.error("Can --name (hoac dung --rebuild-only de chi xay lai database)")

    if not args.webcam and not args.images:
        parser.error("Can chon --webcam hoac --images <file1> <file2> ...")

    if args.webcam:
        capture_from_webcam(args.name, args.camera_index)

    if args.images:
        import_images(args.name, args.images)

    if not args.no_rebuild:
        rebuild_database()


if __name__ == "__main__":
    main()
