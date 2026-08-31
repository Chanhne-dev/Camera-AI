#!/usr/bin/env bash
#
# download_models.sh
#
# Tai truoc file trong so (weights) pretrained cua facenet-pytorch bang
# nhieu ket noi song song + co the tiep tuc neu bi ngat giua chung -
# nhanh hon RAT NHIEU so voi de facenet-pytorch tu tai bang 1 ket noi
# don (thuong la nguyen nhan gay toc do ~60kB/s tren mang di dong/wifi
# yeu hoac bi gioi han toc do).
#
# Chay 1 lan duy nhat TRUOC KHI chay web_app.py hoac main.py lan dau:
#
#     bash download_models.sh
#
# Sau khi chay xong, facenet-pytorch se thay file da co san trong cache
# va BO QUA buoc tu tai, vao thang qua.

set -e

URL="https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
DEST_DIR="${TORCH_HOME:-$HOME/.cache/torch}/checkpoints"
DEST_FILE="$DEST_DIR/20180402-114759-vggface2.pt"

mkdir -p "$DEST_DIR"

echo "[INFO] Dich: $DEST_FILE"

if [ -f "$DEST_FILE" ]; then
    SIZE=$(stat -c%s "$DEST_FILE" 2>/dev/null || stat -f%z "$DEST_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 100000000 ]; then
        echo "[OK] File da co san va co ve day du ($SIZE bytes). Khong can tai lai."
        exit 0
    else
        echo "[WARNING] File da co nhung co ve chua day du ($SIZE bytes) - se tai/tiep tuc lai."
    fi
fi

if command -v aria2c >/dev/null 2>&1; then
    echo "[INFO] Dung aria2c (16 ket noi song song, co the tiep tuc neu bi ngat)..."
    aria2c \
        -x 16 -s 16 -k 1M \
        --continue=true \
        --dir="$DEST_DIR" \
        --out="20180402-114759-vggface2.pt" \
        "$URL"

elif command -v wget >/dev/null 2>&1; then
    echo "[WARNING] Khong co aria2c (chi 1 ket noi, se cham hon). Cai aria2c de nhanh hon:"
    echo "          pkg install aria2   (Termux)   hoac   apt install aria2   (Ubuntu proot)"
    wget -c -O "$DEST_FILE" "$URL"

elif command -v curl >/dev/null 2>&1; then
    echo "[WARNING] Khong co aria2c/wget (chi 1 ket noi, se cham hon). Cai aria2c de nhanh hon."
    curl -L -C - -o "$DEST_FILE" "$URL"

else
    echo "[ERROR] Khong tim thay aria2c, wget, hay curl. Cai 1 trong 3 roi chay lai:"
    echo "        pkg install aria2 wget curl   (Termux)"
    echo "        apt install aria2 wget curl   (Ubuntu proot)"
    exit 1
fi

echo "[DONE] Da tai xong: $DEST_FILE"
echo "[INFO] Bay gio chay web_app.py hoac main.py, phan nhan dien khuon mat se KHONG can tai lai."
