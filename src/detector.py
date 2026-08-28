"""Pemuatan model dan inference.

Berkas ini satu-satunya yang menyentuh Ultralytics. Semua bagian lain bekerja
pada daftar dict biasa, supaya logikanya bisa diuji tanpa GPU dan tanpa
menjalankan aplikasi.
"""

import os
import tempfile
from pathlib import Path

# Harus dijalankan SEBELUM ultralytics diimpor. Di lingkungan hosting yang sistem
# berkasnya dibatasi, Ultralytics gagal menulis berkas setelan ke direktori home
# dan bisa mencoba mengunduh font pada pemakaian pertama.
#
# Direktorinya dibuat dulu. Ultralytics tidak membuatnya sendiri, ia hanya
# memberi peringatan lalu diam-diam memakai lokasi lain.
_CFG = Path(tempfile.gettempdir()) / "ultralytics_cfg"
_CFG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_CFG))

# Mematikan pemeriksaan versi dan analytics. Ini TIDAK memblokir unduhan bobot,
# sudah diuji, sehingga model bawaan tetap bisa diambil saat pertama dijalankan.
os.environ.setdefault("YOLO_OFFLINE", "1")

from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DIR = Path(__file__).resolve().parent.parent
MODELS = DIR / "models"

# Batas sisi terpanjang gambar sebelum masuk model. Dataset latih saja memuat
# gambar 30,4 megapiksel, dan di server gratis yang memorinya sekitar 1 GB
# gambar sebesar itu bisa mematikan aplikasi.
MAKS_SISI = 1600


def daftar_model():
    """Bobot yang ada di folder models, terbaru lebih dulu."""
    if not MODELS.is_dir():
        return []
    return sorted((p.name for p in MODELS.glob("*.pt")), reverse=True)


def muat_model(nama: str) -> YOLO:
    """Muat bobot dari folder models, atau bobot bawaan Ultralytics.

    Path diturunkan dari lokasi berkas ini, bukan dari direktori kerja, karena
    direktori kerja berbeda antara komputer sendiri dan server hosting.

    Bobot bawaan diunduh ke direktori sementara yang pasti bisa ditulis, bukan
    ke direktori kerja yang belum tentu punya izin tulis di server hosting.

    Args:
        nama: nama berkas di folder models, atau nama bobot bawaan seperti
            "yolo11n.pt".
    """
    lokal = MODELS / nama
    if lokal.exists():
        return YOLO(str(lokal))

    unduhan = _CFG / nama
    if unduhan.exists():
        return YOLO(str(unduhan))

    sebelumnya = os.getcwd()
    try:
        os.chdir(_CFG)
        return YOLO(nama)
    finally:
        os.chdir(sebelumnya)


def siapkan_gambar(sumber) -> Image.Image:
    """Buka gambar dan batasi ukurannya sebelum masuk model."""
    gambar = Image.open(sumber).convert("RGB")
    gambar.thumbnail((MAKS_SISI, MAKS_SISI))
    return gambar


def deteksi(model, gambar: Image.Image, conf: float, iou: float, imgsz: int):
    """Jalankan inference dan kembalikan (daftar deteksi, objek Results).

    Objek PIL diberikan langsung ke model, BUKAN np.array(gambar).
    Ultralytics menafsirkan np.ndarray sebagai BGR, sedangkan np.array dari PIL
    menghasilkan RGB. Channel merah dan biru akan tertukar terhadap apa yang
    dilihat model saat training, tanpa memunculkan error apa pun.
    """
    hasil = model(gambar, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    return ke_deteksi(hasil), hasil


def ke_deteksi(hasil):
    """Ubah Results Ultralytics menjadi daftar dict yang netral.

    Returns:
        Daftar dict berisi kunci cls, conf, dan box dalam format xyxy.
        Daftar kosong kalau tidak ada objek terdeteksi.
    """
    out = []
    # Penjagaan nol deteksi. Kasus ini pasti terjadi di aplikasi nyata, misalnya
    # saat confidence threshold dinaikkan, dan hampir selalu lupa ditangani.
    if hasil.boxes is None or len(hasil.boxes) == 0:
        return out
    for b in hasil.boxes:
        out.append(
            {
                "cls": hasil.names[int(b.cls.item())],
                "conf": round(float(b.conf.item()), 3),
                "box": tuple(round(float(v), 1) for v in b.xyxy[0].tolist()),
            }
        )
    return out
