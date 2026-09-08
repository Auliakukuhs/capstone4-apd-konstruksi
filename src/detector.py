"""Pemuatan model dan inference.

Berkas ini satu-satunya yang menyentuh Ultralytics. Semua bagian lain bekerja
pada daftar dict biasa, supaya logikanya bisa diuji tanpa GPU dan tanpa
menjalankan aplikasi.
"""

import io
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
    """Buka gambar dan batasi ukurannya sebelum masuk model.

    Menerima jalur berkas, objek mirip berkas, maupun bytes mentah. Bentuk
    bytes dibutuhkan aplikasi, karena isi unggahan itulah yang dipakai sebagai
    kunci cache inference, dan objek unggahan Streamlit sendiri tidak bisa
    dipakai sebagai kunci.
    """
    if isinstance(sumber, (bytes, bytearray)):
        sumber = io.BytesIO(sumber)
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


def luminansi(gambar: Image.Image) -> float:
    """Rata-rata kecerahan 0 sampai 255, dipakai memeriksa kualitas unggahan.

    Dihitung lewat konversi ke mode L, yang memakai pembobotan luma ITU-R 601
    bawaan Pillow, bukan rata-rata tiga kanal RGB yang menyesatkan karena mata
    manusia jauh lebih peka pada hijau daripada biru.
    """
    kelabu = gambar.convert("L")
    histogram = kelabu.histogram()
    jumlah_piksel = sum(histogram)
    if jumlah_piksel == 0:
        return 0.0
    return sum(i * n for i, n in enumerate(histogram)) / jumlah_piksel


def setarakan_kontras(gambar: Image.Image, klip: float = 2.0, petak: int = 8):
    """CLAHE pada kanal luminansi saja, warnanya dibiarkan.

    TIDAK dipakai sebagai preprocessing tetap. Alasannya dua. Kondisi
    pencahayaan dataset ini bervariasi, bukan seragam gelap, sehingga
    memaksakan penyetaraan pada semua gambar juga merusak yang sudah baik.
    Dan preprocessing tetap harus diterapkan konsisten saat inference, yang
    menambah satu langkah yang bisa lupa dilakukan.

    Disediakan sebagai pembanding yang dijalankan pengguna secara sadar, supaya
    efeknya bisa dilihat berdampingan, bukan diam-diam mengubah masukan model.

    Perhatikan bahwa model dilatih TANPA ini, jadi hasil yang lebih banyak
    belum tentu hasil yang lebih benar.
    """
    import cv2
    import numpy as np

    lab = cv2.cvtColor(np.asarray(gambar.convert("RGB")), cv2.COLOR_RGB2LAB)
    alat = cv2.createCLAHE(clipLimit=klip, tileGridSize=(petak, petak))
    lab[:, :, 0] = alat.apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
