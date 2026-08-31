import subprocess
import os
import sys
import time
import json
import signal


BASE = os.path.dirname(os.path.abspath(__file__))

with open(
    os.path.join(BASE, "config.json"),
    "r",
    encoding="utf-8"
) as f:
    cfg = json.load(f)


cam = cfg["camera"]
talk = cfg["talk"]

sound = os.path.join(
    BASE,
    cfg["alert"]["sound"]
)

helper = os.path.join(
    BASE,
    "imou-life",
    "deploy",
    "dockge",
    "imou-bridge",
    "frigate_imou_talk_exec.py"
)


# Coi hu: mac dinh phat lap lai lien tuc cho den khi tien trinh nay
# nhan tin hieu dung (SIGTERM/SIGINT) tu tien trinh cha (main.py).
# Truyen "--once" neu chi muon phat 1 lan roi thoat.
LOOP = "--once" not in sys.argv


print("[SPEAKER] Starting...", "(loop)" if LOOP else "(once)")


if not os.path.exists(sound):
    print("[ERROR] Sound file not found:", sound)
    sys.exit(1)


if not os.path.exists(helper):
    print("[ERROR] Talk helper not found:", helper)
    sys.exit(1)


stop_requested = False


def _handle_stop(signum, frame):
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGTERM, _handle_stop)
signal.signal(signal.SIGINT, _handle_stop)

if hasattr(signal, "SIGBREAK"):
    # Windows: gui khi dung CTRL_BREAK_EVENT
    signal.signal(signal.SIGBREAK, _handle_stop)


ffmpeg_cmd = [
    "ffmpeg",

    "-hide_banner",
    "-loglevel", "error",
]

if LOOP:
    # lap file coi vo han, tu ffmpeg tu noi lien mach cac lan lap
    ffmpeg_cmd += ["-stream_loop", "-1"]

ffmpeg_cmd += [
    "-i", sound,

    "-ac", "1",
    "-ar", "16000",
    "-sample_fmt", "s16",

    "-f", "s16le",
    "pipe:1"
]

ffmpeg = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE
)


helper_process = subprocess.Popen(
    [
        sys.executable,

        helper,

        "--direct",

        "--host",
        cam["ip"],

        "--port",
        str(cam["talk_port"]),

        "--serial",
        cam["serial"],

        "--username",
        cam["username"],

        "--password",
        cam["password"],

        "--input-codec",
        "s16le",

        "--input-sample-rate",
        "16000",

        "--output-codec",
        talk["output_codec"],

        "--sample-rate",
        str(talk["sample_rate"]),

        "--volume-gain",
        str(talk["volume_gain"]),

        "--frame-ms",
        str(talk["frame_ms"]),

        "--timeout",
        str(talk["timeout"])
    ],

    stdin=subprocess.PIPE
)


try:

    while not stop_requested:

        data = ffmpeg.stdout.read(640)

        if not data:
            # het du lieu that su (chi xay ra khi --once va het bai,
            # vi che do loop thi ffmpeg tu lap vo han)
            break

        try:
            helper_process.stdin.write(data)
            helper_process.stdin.flush()
        except (BrokenPipeError, OSError):
            # phia camera/helper da dong ket noi (vd het timeout)
            break

        time.sleep(0.018)


finally:

    print("[SPEAKER] Stopping...")

    # Chu dong dung ca 2 tien trinh con ngay lap tuc thay vi doi
    # ffmpeg lap vo han tu ket thuc.
    for proc in (ffmpeg, helper_process):
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        ffmpeg.stdout.close()
    except Exception:
        pass

    try:
        helper_process.stdin.close()
    except Exception:
        pass

    for proc in (ffmpeg, helper_process):
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


print("[SPEAKER] Finished")