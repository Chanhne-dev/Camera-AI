import json
import os
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# ==================================================
# CONFIG
# ==================================================

# gui.py nam trong standalone/, cung thu muc voi main.py - nhung
# config.json, register_face.py (trong imou_ai/face/) lai o vi tri khac
# nen can 2 bien rieng:
#   SCRIPT_DIR    = thu muc chua gui.py/main.py (standalone/)
#   PROJECT_ROOT  = thu muc goc du an (chua config.json, imou_ai/, data/...)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE = PROJECT_ROOT
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")
MAIN_FILE = os.path.join(SCRIPT_DIR, "main.py")
REGISTER_FACE_FILE = os.path.join(PROJECT_ROOT, "imou_ai", "face", "register_face.py")


def load_config():

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def save_config():

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cfg,
            f,
            ensure_ascii=False,
            indent=4
        )


cfg = load_config()


# ==================================================
# MAIN PROCESS
# ==================================================

main_process = None


def start_main():

    global main_process

    if main_process is not None:

        if main_process.poll() is None:
            return

    print("[GUI] Starting main.py...")

    creation_flags = 0

    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    main_process = subprocess.Popen(
        [
            sys.executable,
            MAIN_FILE
        ],
        cwd=BASE,
        creationflags=creation_flags
    )

    status_var.set("● AI RUNNING")


def stop_main():

    global main_process

    if main_process is None:
        return

    if main_process.poll() is None:

        print("[GUI] Stopping main.py...")

        try:

            if sys.platform == "win32":

                main_process.send_signal(
                    signal.CTRL_BREAK_EVENT
                )

                try:
                    main_process.wait(timeout=3)

                except subprocess.TimeoutExpired:
                    main_process.kill()

            else:

                main_process.terminate()

                try:
                    main_process.wait(timeout=3)

                except subprocess.TimeoutExpired:
                    main_process.kill()

        except Exception as e:

            print("[GUI] Stop error:", e)

            try:
                main_process.kill()
            except Exception:
                pass

    main_process = None

    status_var.set("● AI STOPPED")


def restart_main():

    print("[GUI] Restarting main.py...")

    stop_main()

    root.after(
        500,
        start_main
    )


# ==================================================
# WINDOW
# ==================================================

root = tk.Tk()

root.title(
    "IMOU AI - Configuration"
)

root.geometry(
    "700x640"
)

root.resizable(
    False,
    False
)


style = ttk.Style()

style.configure(
    "TButton",
    padding=6
)

style.configure(
    "TLabel",
    padding=3
)


# ==================================================
# VARIABLES
# ==================================================

camera_vars = {}
yolo_vars = {}
alert_vars = {}
detect_vars = {}
display_vars = {}
face_vars = {}


def make_var(value):

    return tk.StringVar(
        value=str(value)
    )


def add_field(
    parent,
    row,
    label,
    value,
    variables,
    key,
    show=None
):

    ttk.Label(
        parent,
        text=label
    ).grid(
        row=row,
        column=0,
        sticky="w",
        padx=15,
        pady=6
    )

    var = make_var(value)

    variables[key] = var

    ttk.Entry(
        parent,
        textvariable=var,
        width=45,
        show=show
    ).grid(
        row=row,
        column=1,
        padx=10,
        pady=6,
        sticky="w"
    )


# ==================================================
# NOTEBOOK
# ==================================================

notebook = ttk.Notebook(
    root
)

notebook.pack(
    fill="both",
    expand=True,
    padx=10,
    pady=10
)


# ==================================================
# CAMERA
# ==================================================

camera_tab = ttk.Frame(
    notebook
)

notebook.add(
    camera_tab,
    text="Camera"
)

add_field(
    camera_tab,
    0,
    "IP Camera",
    cfg["camera"]["ip"],
    camera_vars,
    "ip"
)

add_field(
    camera_tab,
    1,
    "RTSP Port",
    cfg["camera"]["rtsp_port"],
    camera_vars,
    "rtsp_port"
)

add_field(
    camera_tab,
    2,
    "Username",
    cfg["camera"]["username"],
    camera_vars,
    "username"
)

add_field(
    camera_tab,
    3,
    "Password",
    cfg["camera"]["password"],
    camera_vars,
    "password",
    "*"
)

add_field(
    camera_tab,
    4,
    "Serial",
    cfg["camera"].get(
        "serial",
        ""
    ),
    camera_vars,
    "serial"
)

add_field(
    camera_tab,
    5,
    "RTSP Stream",
    cfg["camera"]["rtsp_subtype"],
    camera_vars,
    "rtsp_subtype"
)

add_field(
    camera_tab,
    6,
    "Talk Port",
    cfg["camera"].get(
        "talk_port",
        8086
    ),
    camera_vars,
    "talk_port"
)


# ==================================================
# YOLO
# ==================================================

yolo_tab = ttk.Frame(
    notebook
)

notebook.add(
    yolo_tab,
    text="YOLO"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

add_field(
    yolo_tab,
    0,
    "Model",
    cfg["yolo"]["model"],
    yolo_vars,
    "model"
)

add_field(
    yolo_tab,
    1,
    "Confidence",
    cfg["yolo"]["confidence"],
    yolo_vars,
    "confidence"
)

add_field(
    yolo_tab,
    2,
    "Image Size",
    cfg["yolo"]["imgsz"],
    yolo_vars,
    "imgsz"
)

add_field(
    yolo_tab,
    3,
    "Max detections",
    cfg["yolo"].get(
        "max_det",
        10
    ),
    yolo_vars,
    "max_det"
)


# --------------------------------------------------
# PERSON
# --------------------------------------------------

person_enabled = tk.BooleanVar(
    value=cfg["yolo"].get(
        "detect_person",
        True
    )
)

ttk.Checkbutton(
    yolo_tab,
    text="Phát hiện người",
    variable=person_enabled
).grid(
    row=4,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=8
)

add_field(
    yolo_tab,
    5,
    "Person Class",
    cfg["yolo"].get(
        "person_class",
        0
    ),
    yolo_vars,
    "person_class"
)


# --------------------------------------------------
# VEHICLES
# --------------------------------------------------

vehicle_enabled = tk.BooleanVar(
    value=cfg["yolo"].get(
        "detect_vehicles",
        True
    )
)

ttk.Checkbutton(
    yolo_tab,
    text="Phát hiện phương tiện",
    variable=vehicle_enabled
).grid(
    row=6,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=8
)


# COCO class IDs
vehicle_classes = cfg["yolo"].get(
    "vehicle_classes",
    [2, 3, 5, 7]
)


def has_vehicle_class(class_id):

    try:
        return int(class_id) in vehicle_classes
    except (ValueError, TypeError):
        return False


car_enabled = tk.BooleanVar(
    value=has_vehicle_class(2)
)

motorcycle_enabled = tk.BooleanVar(
    value=has_vehicle_class(3)
)

bus_enabled = tk.BooleanVar(
    value=has_vehicle_class(5)
)

truck_enabled = tk.BooleanVar(
    value=has_vehicle_class(7)
)


vehicle_frame = ttk.Frame(
    yolo_tab
)

vehicle_frame.grid(
    row=7,
    column=0,
    columnspan=2,
    sticky="w",
    padx=30,
    pady=3
)


ttk.Checkbutton(
    vehicle_frame,
    text="Ô tô",
    variable=car_enabled
).grid(
    row=0,
    column=0,
    padx=10,
    pady=4
)

ttk.Checkbutton(
    vehicle_frame,
    text="Xe máy",
    variable=motorcycle_enabled
).grid(
    row=0,
    column=1,
    padx=10,
    pady=4
)

ttk.Checkbutton(
    vehicle_frame,
    text="Xe buýt",
    variable=bus_enabled
).grid(
    row=1,
    column=0,
    padx=10,
    pady=4
)

ttk.Checkbutton(
    vehicle_frame,
    text="Xe tải",
    variable=truck_enabled
).grid(
    row=1,
    column=1,
    padx=10,
    pady=4
)


# ==================================================
# ALERT
# ==================================================

alert_tab = ttk.Frame(
    notebook
)

notebook.add(
    alert_tab,
    text="Cảnh báo"
)

alert_enabled = tk.BooleanVar(
    value=cfg["alert"]["enabled"]
)

ttk.Checkbutton(
    alert_tab,
    text="Bật cảnh báo",
    variable=alert_enabled
).grid(
    row=0,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=10
)

add_field(
    alert_tab,
    1,
    "Bắt đầu",
    cfg["alert"]["start"],
    alert_vars,
    "start"
)

add_field(
    alert_tab,
    2,
    "Kết thúc",
    cfg["alert"]["end"],
    alert_vars,
    "end"
)

add_field(
    alert_tab,
    3,
    "Xác nhận (giây)",
    cfg["alert"].get(
        "confirm_seconds",
        3
    ),
    alert_vars,
    "confirm_seconds"
)

add_field(
    alert_tab,
    4,
    "Cooldown (giây)",
    cfg["alert"]["cooldown"],
    alert_vars,
    "cooldown"
)

add_field(
    alert_tab,
    5,
    "Grace / khoan dung (giây)",
    cfg["alert"].get(
        "grace_seconds",
        1.5
    ),
    alert_vars,
    "grace_seconds"
)

add_field(
    alert_tab,
    6,
    "Âm thanh",
    cfg["alert"].get(
        "sound",
        "assets/sound.wav"
    ),
    alert_vars,
    "sound"
)


def choose_sound():

    path = filedialog.askopenfilename(
        title="Chọn file âm thanh",
        filetypes=[
            ("WAV", "*.wav"),
            ("Audio", "*.wav *.mp3"),
            ("All files", "*.*")
        ]
    )

    if path:
        alert_vars["sound"].set(path)


ttk.Button(
    alert_tab,
    text="Chọn file",
    command=choose_sound
).grid(
    row=6,
    column=2,
    padx=5
)


# ==================================================
# DETECTION
# ==================================================

detect_tab = ttk.Frame(
    notebook
)

notebook.add(
    detect_tab,
    text="Phát hiện"
)

detect_save = tk.BooleanVar(
    value=cfg["detection"]["save_images"]
)

ttk.Checkbutton(
    detect_tab,
    text="Lưu ảnh khi phát hiện",
    variable=detect_save
).grid(
    row=0,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=10
)

add_field(
    detect_tab,
    1,
    "Detection FPS",
    cfg["detection"].get(
        "detect_fps",
        8
    ),
    detect_vars,
    "detect_fps"
)

add_field(
    detect_tab,
    2,
    "Thư mục lưu",
    cfg["detection"]["save_directory"],
    detect_vars,
    "save_directory"
)


# ==================================================
# FACE (KHUÔN MẶT)
# ==================================================

face_tab = ttk.Frame(
    notebook
)

notebook.add(
    face_tab,
    text="Khuôn mặt"
)

face_cfg_gui = cfg.get("face", {})

face_enabled = tk.BooleanVar(
    value=face_cfg_gui.get("enabled", True)
)

ttk.Checkbutton(
    face_tab,
    text="Bật nhận diện khuôn mặt",
    variable=face_enabled
).grid(
    row=0,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=8
)

alert_only_unknown_var = tk.BooleanVar(
    value=face_cfg_gui.get("alert_only_unknown", True)
)

ttk.Checkbutton(
    face_tab,
    text="Chỉ báo động khi gặp NGƯỜI LẠ (bỏ qua người quen)",
    variable=alert_only_unknown_var
).grid(
    row=1,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=8
)

save_unknown_var = tk.BooleanVar(
    value=face_cfg_gui.get("save_unknown_faces", True)
)

ttk.Checkbutton(
    face_tab,
    text="Lưu ảnh cận mặt người lạ khi báo động",
    variable=save_unknown_var
).grid(
    row=2,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=8
)

add_field(
    face_tab,
    3,
    "Thư mục người quen",
    face_cfg_gui.get("known_faces_dir", "data/known_faces"),
    face_vars,
    "known_faces_dir"
)


def choose_known_faces_dir():

    path = filedialog.askdirectory(
        title="Chọn thư mục chứa ảnh người quen"
    )

    if path:
        face_vars["known_faces_dir"].set(path)


ttk.Button(
    face_tab,
    text="Chọn thư mục",
    command=choose_known_faces_dir
).grid(
    row=3,
    column=2,
    padx=5
)

add_field(
    face_tab,
    4,
    "Thư mục lưu người lạ",
    face_cfg_gui.get("unknown_save_directory", "data/unknown_faces"),
    face_vars,
    "unknown_save_directory"
)

add_field(
    face_tab,
    5,
    "Ngưỡng giống nhau (0-1)",
    face_cfg_gui.get("similarity_threshold", 0.72),
    face_vars,
    "similarity_threshold"
)

add_field(
    face_tab,
    6,
    "Margin phân biệt 2 người",
    face_cfg_gui.get("margin", 0.05),
    face_vars,
    "margin"
)

add_field(
    face_tab,
    7,
    "Top-K ảnh so khớp",
    face_cfg_gui.get("top_k", 5),
    face_vars,
    "top_k"
)

add_field(
    face_tab,
    8,
    "Kích thước mặt tối thiểu (px)",
    face_cfg_gui.get("min_face_size", 40),
    face_vars,
    "min_face_size"
)

add_field(
    face_tab,
    9,
    "Tần suất nhận diện (FPS)",
    face_cfg_gui.get("recognize_fps", 2),
    face_vars,
    "recognize_fps"
)

face_debug_var = tk.BooleanVar(
    value=face_cfg_gui.get("debug", False)
)

ttk.Checkbutton(
    face_tab,
    text="Debug: in điểm số giống nhau (similarity) ra console",
    variable=face_debug_var
).grid(
    row=10,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=(8, 0)
)


face_hint = ttk.Label(
    face_tab,
    text=(
        "Ngưỡng cao hơn = chặt hơn (ít nhận nhầm, nhưng dễ báo Unknown\n"
        "nhầm cho người quen). Margin cao hơn = phân biệt 2 người giống\n"
        "nhau tốt hơn. Xoá face_db.pkl và bấm \"Xây lại Database\" sau khi\n"
        "đổi ảnh trong thư mục người quen."
    ),
    justify="left",
    foreground="#555555"
)

face_hint.grid(
    row=11,
    column=0,
    columnspan=3,
    sticky="w",
    padx=15,
    pady=(10, 5)
)


def rebuild_face_db():

    script = REGISTER_FACE_FILE

    try:

        result = subprocess.run(
            [sys.executable, script, "--rebuild-only"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode == 0:
            messagebox.showinfo(
                "Xây lại Database",
                "Đã xây lại database khuôn mặt thành công."
            )
        else:
            messagebox.showerror(
                "Xây lại Database - Lỗi",
                result.stderr[-1500:] if result.stderr else "Lỗi không xác định"
            )

    except Exception as e:
        messagebox.showerror(
            "Xây lại Database - Lỗi",
            str(e)
        )


ttk.Button(
    face_tab,
    text="Xây lại Database khuôn mặt",
    command=rebuild_face_db
).grid(
    row=12,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=10
)


# ==================================================
# DISPLAY
# ==================================================

display_tab = ttk.Frame(
    notebook
)

notebook.add(
    display_tab,
    text="Hiển thị"
)

display_show = tk.BooleanVar(
    value=cfg["display"]["show_camera"]
)

ttk.Checkbutton(
    display_tab,
    text="Hiển thị camera",
    variable=display_show
).grid(
    row=0,
    column=0,
    columnspan=2,
    sticky="w",
    padx=15,
    pady=10
)

add_field(
    display_tab,
    1,
    "Tên cửa sổ",
    cfg["display"]["window_name"],
    display_vars,
    "window_name"
)


# ==================================================
# SYSTEM
# ==================================================

system_tab = ttk.Frame(
    notebook
)

notebook.add(
    system_tab,
    text="Hệ thống"
)

status_var = tk.StringVar(
    value="● STARTING..."
)

ttk.Label(
    system_tab,
    textvariable=status_var,
    font=("Arial", 12)
).pack(
    pady=20
)


# ==================================================
# SAVE CONFIG
# ==================================================

def save_changes():

    global cfg

    try:

        # ==================================================
        # CAMERA
        # ==================================================

        cfg["camera"]["ip"] = (
            camera_vars["ip"].get()
        )

        cfg["camera"]["rtsp_port"] = int(
            camera_vars["rtsp_port"].get()
        )

        cfg["camera"]["username"] = (
            camera_vars["username"].get()
        )

        cfg["camera"]["password"] = (
            camera_vars["password"].get()
        )

        cfg["camera"]["serial"] = (
            camera_vars["serial"].get()
        )

        cfg["camera"]["rtsp_subtype"] = int(
            camera_vars["rtsp_subtype"].get()
        )

        cfg["camera"]["talk_port"] = int(
            camera_vars["talk_port"].get()
        )


        # ==================================================
        # YOLO
        # ==================================================

        cfg["yolo"]["model"] = (
            yolo_vars["model"].get()
        )

        cfg["yolo"]["confidence"] = float(
            yolo_vars["confidence"].get()
        )

        cfg["yolo"]["imgsz"] = int(
            yolo_vars["imgsz"].get()
        )

        cfg["yolo"]["max_det"] = int(
            yolo_vars["max_det"].get()
        )

        cfg["yolo"]["person_class"] = int(
            yolo_vars["person_class"].get()
        )

        cfg["yolo"]["detect_person"] = (
            person_enabled.get()
        )

        cfg["yolo"]["detect_vehicles"] = (
            vehicle_enabled.get()
        )


        # ==================================================
        # VEHICLE CLASSES
        # ==================================================

        new_vehicle_classes = []

        if car_enabled.get():
            new_vehicle_classes.append(2)

        if motorcycle_enabled.get():
            new_vehicle_classes.append(3)

        if bus_enabled.get():
            new_vehicle_classes.append(5)

        if truck_enabled.get():
            new_vehicle_classes.append(7)

        cfg["yolo"]["vehicle_classes"] = (
            new_vehicle_classes
        )


        # ==================================================
        # ALERT
        # ==================================================

        cfg["alert"]["enabled"] = (
            alert_enabled.get()
        )

        cfg["alert"]["start"] = (
            alert_vars["start"].get()
        )

        cfg["alert"]["end"] = (
            alert_vars["end"].get()
        )

        cfg["alert"]["confirm_seconds"] = float(
            alert_vars["confirm_seconds"].get()
        )

        cfg["alert"]["cooldown"] = float(
            alert_vars["cooldown"].get()
        )

        cfg["alert"]["grace_seconds"] = float(
            alert_vars["grace_seconds"].get()
        )

        cfg["alert"]["sound"] = (
            alert_vars["sound"].get()
        )


        # ==================================================
        # FACE (KHUÔN MẶT)
        # ==================================================

        if "face" not in cfg:
            cfg["face"] = {}

        cfg["face"]["enabled"] = (
            face_enabled.get()
        )

        cfg["face"]["alert_only_unknown"] = (
            alert_only_unknown_var.get()
        )

        cfg["face"]["save_unknown_faces"] = (
            save_unknown_var.get()
        )

        cfg["face"]["known_faces_dir"] = (
            face_vars["known_faces_dir"].get()
        )

        cfg["face"]["unknown_save_directory"] = (
            face_vars["unknown_save_directory"].get()
        )

        cfg["face"]["similarity_threshold"] = float(
            face_vars["similarity_threshold"].get()
        )

        cfg["face"]["margin"] = float(
            face_vars["margin"].get()
        )

        cfg["face"]["top_k"] = int(
            face_vars["top_k"].get()
        )

        cfg["face"]["min_face_size"] = int(
            face_vars["min_face_size"].get()
        )

        cfg["face"]["recognize_fps"] = float(
            face_vars["recognize_fps"].get()
        )

        cfg["face"]["debug"] = (
            face_debug_var.get()
        )


        # ==================================================
        # DETECTION
        # ==================================================

        cfg["detection"]["save_images"] = (
            detect_save.get()
        )

        cfg["detection"]["detect_fps"] = int(
            detect_vars["detect_fps"].get()
        )

        cfg["detection"]["save_directory"] = (
            detect_vars["save_directory"].get()
        )


        # ==================================================
        # DISPLAY
        # ==================================================

        cfg["display"]["show_camera"] = (
            display_show.get()
        )

        cfg["display"]["window_name"] = (
            display_vars["window_name"].get()
        )


        # ==================================================
        # SAVE
        # ==================================================

        save_config()

        print("[GUI] Config saved")

        status_var.set(
            "● CONFIG SAVED - RESTARTING AI..."
        )

        root.after(
            200,
            restart_main
        )


    except ValueError as e:

        messagebox.showerror(
            "Lỗi cấu hình",
            f"Giá trị không hợp lệ:\n{e}"
        )


# ==================================================
# BUTTONS
# ==================================================

button_frame = ttk.Frame(
    root
)

button_frame.pack(
    fill="x",
    padx=10,
    pady=5
)

ttk.Button(
    button_frame,
    text="LƯU CONFIG + RESTART",
    command=save_changes
).pack(
    side="right",
    padx=5
)


# ==================================================
# CLOSE
# ==================================================

def on_close():

    if main_process is not None:
        stop_main()

    root.destroy()


root.protocol(
    "WM_DELETE_WINDOW",
    on_close
)


# ==================================================
# START MAIN.PY
# ==================================================

root.after(
    500,
    start_main
)


# ==================================================
# RUN GUI
# ==================================================

root.mainloop()