"""
events.py

Ghi lai lich su cac su kien canh bao (nguoi / xe co / khuon mat la) vao 1
file JSON Lines (moi dong 1 su kien, de doc/ghi, khong can dependency
database ngoai). Dung chung cho main.py va web_app.py de xem lai lich su
("xem lai cac su kien phat hien nguoi/xe vao vung canh bao").
"""

import json
import os
import time


def _events_file(base_dir):
    return os.path.join(base_dir, "events.jsonl")


def log_event(base_dir, event_type, image_path=None, extra=None):
    """
    event_type: "person" | "vehicle" | "unknown_face" | "known_face"
    image_path: duong dan TUONG DOI tinh tu base_dir (vd "detections/xxx.jpg")
    extra: dict bo sung tuy y (vd {"name": "Bo", "similarity": 0.81})
    """

    event = {
        "ts": time.time(),
        "type": event_type,
        "image": image_path,
    }

    if extra:
        event.update(extra)

    try:
        with open(_events_file(base_dir), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[EVENTS][WARNING] Khong ghi duoc su kien: {e}")


def load_events(base_dir, limit=100, event_type=None, offset=0):
    """
    Doc su kien tu file JSONL, moi nhat truoc. Doc toan bo file vao bo
    nho - du dung cho quy mo camera gia dinh (vai nghin dong van nhe).
    Neu file lon dan qua nhieu (nhieu nam), co the can chuyen sang
    SQLite sau nay, nhung hien tai JSONL la du va don gian.
    """

    path = _events_file(base_dir)

    if not os.path.exists(path):
        return [], 0

    events = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[EVENTS][WARNING] Khong doc duoc file su kien: {e}")
        return [], 0

    events.reverse()  # moi nhat truoc

    if event_type:
        events = [e for e in events if e.get("type") == event_type]

    total = len(events)

    page = events[offset:offset + limit]

    return page, total


def prune_events(base_dir, keep_last=5000):
    """
    Cat bot file su kien neu qua dai (giu lai keep_last dong gan nhat).
    Goi ham nay dinh ky (vd 1 lan moi khi khoi dong) de file khong phinh
    to vo han qua nhieu nam su dung.
    """

    path = _events_file(base_dir)

    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) > keep_last:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines[-keep_last:])

    except Exception as e:
        print(f"[EVENTS][WARNING] Khong cat gon duoc file su kien: {e}")


def prune_old(base_dir, media_dirs=None, max_age_days=15):
    """
    Don du lieu qua han luu tru (mac dinh 15 ngay):

      1. Xoa cac DONG su kien cu hon max_age_days trong events.jsonl
         (dua theo "ts" cua tung su kien).
      2. Xoa cac FILE anh cu hon max_age_days (dua theo thoi gian sua
         doi file) trong cac thu muc trong media_dirs (vd detections/,
         detections/vehicles, detections/plates, unknown_faces/...).

    Goi dinh ky (vd moi vai tieng dong ho trong vong lap chinh) de dia
    khong bi day dan qua nhieu ngay su dung. An toan khi goi nhieu lan
    lien tuc / khong co gi de xoa.
    """

    cutoff_ts = time.time() - max_age_days * 86400

    # ---- 1) events.jsonl ----
    path = _events_file(base_dir)

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            kept = []

            for line in lines:
                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    ev = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                if ev.get("ts", 0) >= cutoff_ts:
                    kept.append(line)

            if len(kept) != len(lines):
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(kept)

        except Exception as e:
            print(f"[EVENTS][WARNING] Khong don duoc su kien cu: {e}")

    # ---- 2) anh trong cac thu muc media ----
    removed = 0

    for rel_dir in (media_dirs or []):
        directory = os.path.join(base_dir, rel_dir)

        if not os.path.isdir(directory):
            continue

        try:
            names = os.listdir(directory)
        except Exception:
            continue

        for name in names:
            fpath = os.path.join(directory, name)

            if not os.path.isfile(fpath):
                continue

            try:
                if os.path.getmtime(fpath) < cutoff_ts:
                    os.remove(fpath)
                    removed += 1
            except Exception:
                pass

    if removed:
        print(f"[EVENTS] Da xoa {removed} anh cu hon {max_age_days} ngay")
