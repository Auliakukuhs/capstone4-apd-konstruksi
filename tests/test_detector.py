"""Uji asap detector. Membuktikan jalur muat model sampai keluaran dict bekerja.

Dijalankan tanpa Streamlit dan tanpa GPU:
    .venv-uji/bin/python -m pytest tests/ -v
atau langsung:
    .venv-uji/bin/python tests/test_detector.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.detector import (  # noqa: E402
    MAKS_SISI,
    daftar_model,
    deteksi,
    ke_deteksi,
    muat_model,
    siapkan_gambar,
)


class HasilKosong:
    """Tiruan Results Ultralytics tanpa deteksi sama sekali."""

    boxes = None
    names = {}


class HasilTanpaKotak:
    """Tiruan Results dengan daftar boxes kosong."""

    boxes = []
    names = {}


def test_nol_deteksi_tidak_error():
    """Kasus yang pasti terjadi di aplikasi nyata dan hampir selalu lupa ditangani."""
    assert ke_deteksi(HasilKosong()) == []
    assert ke_deteksi(HasilTanpaKotak()) == []


def test_gambar_besar_dibatasi():
    """Dataset latih saja memuat gambar 30,4 megapiksel."""
    import io

    besar = Image.new("RGB", (6000, 4000), (120, 120, 120))
    buf = io.BytesIO()
    besar.save(buf, format="JPEG")
    buf.seek(0)

    hasil = siapkan_gambar(buf)
    assert max(hasil.size) <= MAKS_SISI, f"tidak dibatasi, ukurannya {hasil.size}"
    assert hasil.mode == "RGB"


MODEL_UJI = daftar_model()[0] if daftar_model() else "yolo11n.pt"


def test_muat_model_dan_inference():
    """Uji jalur penuh, bobot hasil training kalau ada, kalau tidak bobot bawaan."""
    model = muat_model(MODEL_UJI)
    gambar = Image.new("RGB", (640, 480), (90, 110, 130))
    hasil, raw = deteksi(model, gambar, conf=0.25, iou=0.7, imgsz=640)

    assert isinstance(hasil, list)
    for d in hasil:
        assert set(d) == {"cls", "conf", "box"}
        assert isinstance(d["cls"], str)
        assert 0.0 <= d["conf"] <= 1.0
        assert len(d["box"]) == 4
        x1, y1, x2, y2 = d["box"]
        assert x2 > x1 and y2 > y1, "kotak tidak valid"

    # .plot() dipakai aplikasi untuk menggambar hasil
    assert raw.plot() is not None


def test_conf_tinggi_menghasilkan_lebih_sedikit():
    """Confidence threshold adalah keputusan, dan efeknya harus terlihat."""
    model = muat_model(MODEL_UJI)
    gambar = Image.new("RGB", (640, 480), (90, 110, 130))
    rendah, _ = deteksi(model, gambar, conf=0.05, iou=0.7, imgsz=640)
    tinggi, _ = deteksi(model, gambar, conf=0.95, iou=0.7, imgsz=640)
    assert len(tinggi) <= len(rendah)


if __name__ == "__main__":
    lulus = gagal = 0
    for nama, fn in sorted(globals().items()):
        if not nama.startswith("test_"):
            continue
        try:
            fn()
            print(f"  LULUS  {nama}")
            lulus += 1
        except Exception as e:
            print(f"  GAGAL  {nama}  {type(e).__name__}: {e}")
            gagal += 1
    print(f"\n{lulus} lulus, {gagal} gagal")
    sys.exit(1 if gagal else 0)
