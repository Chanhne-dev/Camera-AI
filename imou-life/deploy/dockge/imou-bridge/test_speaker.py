import subprocess
import os
import time

SERIAL = "NOMI-IPC-K2E-5H3W-09D6"
PASSWORD = "L29012FE"

BASE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(BASE, "frigate_imou_talk_exec.py")

WAV = r"F:\source-code-main\python\Camera\sound-1.wav"

# WAV -> PCM 16kHz mono
ffmpeg = subprocess.Popen(
    [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", WAV,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "s16le",
        "pipe:1"
    ],
    stdout=subprocess.PIPE
)

# IMOU talk
helper = subprocess.Popen(
    [
        "python",
        HELPER,

        "--direct",
        "--host", "192.168.1.7",
        "--port", "8086",

        "--serial", SERIAL,
        "--username", "admin",
        "--password", PASSWORD,

        "--input-codec", "s16le",
        "--input-sample-rate", "16000",

        "--output-codec", "aac-adts",
        "--sample-rate", "16000",

        "--volume-gain", "5.0",
        "--frame-ms", "20",

        "--timeout", "120",

        "--debug"
    ],
    stdin=subprocess.PIPE
)

print("[INFO] Streaming 78s audio...")

total = 0

while True:
    data = ffmpeg.stdout.read(640)

    if not data:
        break

    helper.stdin.write(data)
    helper.stdin.flush()

    total += len(data)

    # 640 bytes = 20ms @ 16kHz mono s16le
    time.sleep(0.018)

print(f"[INFO] Sent {total:,} bytes")

ffmpeg.stdout.close()
helper.stdin.close()

ffmpeg.wait()
helper.wait()

print("[INFO] Finished")