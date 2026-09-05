"""
face_engine.py

Module phat hien + nhan dien khuon mat.

- Phat hien khuon mat: MTCNN (facenet-pytorch)
- Trich xuat embedding: InceptionResnetV1 (pretrained="vggface2")
- So khop voi database khuon mat da biet bang cosine similarity

Database khuon mat duoc xay tu thu muc:

    known_faces/
        Nguyen Van A/
            1.jpg
            2.jpg
        Tran Thi B/
            1.jpg

Embedding cua tung nguoi duoc cache vao file .pkl de lan sau khoi
phai tinh lai (chi tinh lai neu co anh moi/sua trong known_faces/).
"""

import os
import pickle

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceEngine:

    def __init__(
        self,
        known_faces_dir="known_faces",
        db_cache_path="face_db.pkl",
        similarity_threshold=0.72,
        margin=0.05,
        top_k=5,
        min_face_size=40,
        device="cpu",
        min_detect_prob=0.90,
        debug=False,
    ):
        self.known_faces_dir = known_faces_dir
        self.db_cache_path = db_cache_path
        self.similarity_threshold = similarity_threshold
        self.margin = margin
        self.top_k = top_k
        self.min_face_size = min_face_size
        self.device = device
        self.min_detect_prob = min_detect_prob
        self.debug = debug

        self.mtcnn = MTCNN(
            keep_all=True,
            device=device,
            # QUAN TRONG: phai la True. Day la buoc chuan hoa pixel anh
            # (0-255 -> khoang -1..1) truoc khi dua vao InceptionResnetV1.
            # Neu de False, anh mat duoc giu nguyen thang do 0-255, sai
            # hoan toan so voi luc mang duoc huan luyen (pretrained
            # vggface2), khien embedding cua MOI nguoi deu gan nhu giong
            # het nhau -> nhan dien ai cung ra "cung 1 nguoi". Day chinh
            # la nguyen nhan gay loi "nhan tat ca thanh 1 nguoi".
            post_process=True,
        )

        self.resnet = InceptionResnetV1(
            pretrained="vggface2"
        ).eval().to(device)

        self.names = []
        self.embeddings = None  # torch.Tensor (N, 512) hoac None

        self.load_or_build_db()

    # ==================================================
    # DATABASE
    # ==================================================

    # Tang version nay moi khi thay doi cach tinh/luu embedding, de cache
    # cu (kieu du lieu khac) tu dong bi coi la het han va build lai.
    CACHE_VERSION = 3

    def load_or_build_db(self):

        if os.path.exists(self.db_cache_path) and self._cache_is_fresh():
            self._load_cache()
            print(f"[FACE] Da nap {len(self.names)} anh tu cache")
        else:
            self.build_db()

    def _cache_is_fresh(self):
        """Cache het han neu co anh trong known_faces/ moi hon file cache,
        hoac neu cache duoc luu boi phien ban schema cu."""

        try:
            with open(self.db_cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") != self.CACHE_VERSION:
                return False
        except Exception:
            return False

        if not os.path.isdir(self.known_faces_dir):
            return True

        cache_mtime = os.path.getmtime(self.db_cache_path)

        for root, _, files in os.walk(self.known_faces_dir):
            for filename in files:
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, filename)
                    if os.path.getmtime(full_path) > cache_mtime:
                        return False

        return True

    def _load_cache(self):
        with open(self.db_cache_path, "rb") as f:
            data = pickle.load(f)

        self.names = data["names"]

        if len(data["embeddings"]) > 0:
            self.embeddings = torch.tensor(data["embeddings"])
        else:
            self.embeddings = None

    def build_db(self):
        print("[FACE] Dang xay dung database khuon mat...")

        names = []
        embeddings = []

        if not os.path.isdir(self.known_faces_dir):
            print(f"[FACE] Khong tim thay thu muc: {self.known_faces_dir}")
            self.names = []
            self.embeddings = None
            return

        for person_name in sorted(os.listdir(self.known_faces_dir)):

            person_dir = os.path.join(self.known_faces_dir, person_name)

            if not os.path.isdir(person_dir):
                continue

            count_for_person = 0

            for filename in sorted(os.listdir(person_dir)):

                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                path = os.path.join(person_dir, filename)

                try:
                    img = Image.open(path).convert("RGB")
                except Exception as e:
                    print(f"[FACE] Bo qua {path}: {e}")
                    continue

                face_tensor = self.mtcnn(img)

                if face_tensor is None:
                    print(f"[FACE] Khong tim thay khuon mat trong {path}")
                    continue

                # keep_all=True co the tra ve nhieu mat -> lay mat dau tien
                # (mat lon nhat, vi select_largest=True mac dinh)
                if face_tensor.ndim == 4:
                    face_tensor = face_tensor[0]

                with torch.no_grad():
                    emb = self.resnet(
                        face_tensor.unsqueeze(0).to(self.device)
                    )

                # QUAN TRONG: luu tung embedding rieng le, KHONG lay trung
                # binh cong nhieu anh thanh 1 vector duy nhat. Lay trung
                # binh se lam "nhoe" dac trung khuon mat va de gay nham
                # giua nhung nguoi co net tuong dong.
                names.append(person_name)
                embeddings.append(emb.squeeze(0).cpu())
                count_for_person += 1

            if count_for_person > 0:
                print(f"[FACE] {person_name}: {count_for_person} anh")
            else:
                print(f"[FACE] Khong co anh hop le cho: {person_name}")

        self.names = names
        self.embeddings = torch.stack(embeddings) if embeddings else None

        self._save_cache()

    def _save_cache(self):
        data = {
            "version": self.CACHE_VERSION,
            "names": self.names,
            "embeddings": (
                self.embeddings.numpy()
                if self.embeddings is not None
                else np.zeros((0, 512), dtype=np.float32)
            ),
        }

        with open(self.db_cache_path, "wb") as f:
            pickle.dump(data, f)

    # ==================================================
    # RECOGNITION
    # ==================================================

    def recognize(self, frame_rgb):
        """
        frame_rgb: numpy array HxWx3, thu tu mau RGB.

        Tra ve list cac dict:
            {
                "box": (x1, y1, x2, y2),
                "name": str,          # ten nguoi hoac "Unknown"
                "similarity": float,  # do tuong dong cosine [0..1]
            }
        """

        img = Image.fromarray(frame_rgb)

        boxes, probs = self.mtcnn.detect(img)

        results = []

        if boxes is None:
            if self.debug:
                print("[FACE][DEBUG] Khong phat hien duoc khuon mat nao trong khung hinh")
            return results

        faces = self.mtcnn.extract(img, boxes, save_path=None)

        if faces is None:
            if self.debug:
                print("[FACE][DEBUG] MTCNN.extract khong tra ve duoc anh mat nao")
            return results

        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.resnet(faces.to(self.device)).cpu()

        for box, prob, emb in zip(boxes, probs, embeddings):

            if prob is None or prob < self.min_detect_prob:
                if self.debug:
                    print(f"[FACE][DEBUG] Bo qua mat: do tin cay phat hien thap ({prob})")
                continue

            x1, y1, x2, y2 = [int(v) for v in box]

            # Bo qua mat qua nho / qua xa camera: embedding tinh tu anh
            # mat nho, mo se khong dang tin cay va de gay nhan nham.
            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                if self.debug:
                    print(
                        f"[FACE][DEBUG] Bo qua mat qua nho: "
                        f"{x2-x1}x{y2-y1}px < min_face_size={self.min_face_size}px"
                    )
                continue

            name, similarity = self._match(emb)

            results.append(
                {
                    "box": (x1, y1, x2, y2),
                    "name": name,
                    "similarity": similarity,
                }
            )

        return results

    def _match(self, embedding):
        """
        So khop 1 embedding voi toan bo database bang k-NN (thay vi chi
        so voi 1 anh gan nhat). Lay top_k anh giong nhat da vuot nguong
        similarity_threshold, sau do cho cac anh do "bo phieu" theo ten
        nguoi. Ten nao co nhieu phieu nhat va do giong trung binh cao
        nhat se duoc chon.

        Cach nay on dinh hon nhieu so voi chi dua vao 1 anh gan nhat:
        neu vi 1 ly do nao do (anh chup xau, goc la) mot anh dang ky bi
        "lech" gan giong nguoi khac, no se bi cac anh con lai (bo phieu
        dung) at di, thay vi lam sai lech ket qua ca he thong.

        De giam nham lan giua 2 nguoi trong nhau: neu ten thang cuoc va
        ten a quan quan (khac ten) co diem trung binh cach nhau qua it
        (< margin), tra ve Unknown cho an toan.
        """

        if self.embeddings is None or len(self.names) == 0:
            if self.debug:
                print("[FACE][DEBUG] Database rong - chua co ai duoc dang ky")
            return "Unknown", 0.0

        emb = embedding.unsqueeze(0)

        sims = torch.nn.functional.cosine_similarity(emb, self.embeddings)

        k = min(self.top_k, sims.shape[0])

        top_sims, top_idx = torch.topk(sims, k)

        if self.debug:
            top_str = ", ".join(
                f"{self.names[int(idx)]}={float(sim):.3f}"
                for sim, idx in zip(top_sims, top_idx)
            )
            print(
                f"[FACE][DEBUG] Top-{k} giong nhat: {top_str} "
                f"(nguong hien tai: {self.similarity_threshold})"
            )

        # Chi giu lai nhung ung vien vuot nguong
        candidates = [
            (self.names[int(idx)], float(sim))
            for sim, idx in zip(top_sims, top_idx)
            if float(sim) >= self.similarity_threshold
        ]

        if not candidates:
            # Khong co ung vien nao dat nguong -> tra ve do giong cao
            # nhat tim duoc (de debug/hien thi), nhung van la Unknown
            best_sim = float(top_sims[0]) if k > 0 else 0.0

            if self.debug:
                print(
                    f"[FACE][DEBUG] -> Unknown: khong anh nao vuot nguong "
                    f"(cao nhat={best_sim:.3f} < {self.similarity_threshold})"
                )

            return "Unknown", best_sim

        # Gop diem theo ten: so phieu + diem trung binh
        scores = {}

        for name, sim in candidates:
            entry = scores.setdefault(name, {"count": 0, "sum": 0.0})
            entry["count"] += 1
            entry["sum"] += sim

        # Sap xep: nhieu phieu hon truoc, hoa thi diem trung binh cao hon truoc
        ranked = sorted(
            scores.items(),
            key=lambda item: (item[1]["count"], item[1]["sum"] / item[1]["count"]),
            reverse=True
        )

        best_name, best_stats = ranked[0]
        best_avg = best_stats["sum"] / best_stats["count"]

        second_avg = None

        if len(ranked) > 1:
            _, second_stats = ranked[1]
            second_avg = second_stats["sum"] / second_stats["count"]

        if second_avg is not None and (best_avg - second_avg) < self.margin:
            # 2 nguoi qua sat nhau -> khong chac chan
            if self.debug:
                print(
                    f"[FACE][DEBUG] -> Unknown: '{best_name}' ({best_avg:.3f}) "
                    f"qua sat voi nguoi dung thu 2 ({second_avg:.3f}), "
                    f"cach biet < margin={self.margin}"
                )
            return "Unknown", best_avg

        if self.debug:
            print(f"[FACE][DEBUG] -> Nhan dien: '{best_name}' (diem TB {best_avg:.3f})")

        return best_name, best_avg
