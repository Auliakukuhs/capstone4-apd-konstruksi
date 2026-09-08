"""Uji lingkungan. Menjaga satu keputusan deploy yang tidak terlihat dari kode.

Aplikasi ini tidak lagi memakai `packages.txt`. Pustaka grafis sistem yang
dibutuhkan cv2 datang dari wheel `opencv-python-headless`, bukan dari apt, dan
`opencv-python` asli dicegah terpasang lewat paket pengalih di
`vendor/opencv-python-stub`.

Susunan itu tidak terlihat dari kode aplikasi. Kalau ada yang merapikannya,
build tetap berhasil di laptop dan baru mati di Streamlit Community Cloud,
tempat pustaka sistemnya tidak ada. Uji ini menangkapnya lebih awal.

Dijalankan tanpa Streamlit dan tanpa GPU:
    .venv-uji/bin/python -m pytest tests/ -v
atau langsung:
    .venv-uji/bin/python tests/test_lingkungan.py
"""

import importlib.metadata as metadata
import sys
import tomllib
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
PENGALIH = AKAR / "vendor" / "opencv-python-stub" / "pyproject.toml"
sys.path.insert(0, str(AKAR))


def terpasang():
    """Nama distribusi yang terpasang, huruf kecil semua."""
    nama = set()
    for d in metadata.distributions():
        n = d.metadata["Name"]
        if n:
            nama.add(n.lower())
    return nama


def test_cv2_bisa_diimpor():
    """Ini invarian yang sebenarnya. Ultralytics mengimpor cv2 saat diimpor.

    Kalau gagal, aplikasinya mati sebelum satu baris UI pun tampil.
    """
    import cv2

    assert cv2.__version__, "cv2 terimpor tapi tanpa versi"


def test_headless_terpasang():
    """Varian headless memuat pustaka grafisnya di dalam wheel sendiri.

    Varian biasa mengandalkan libGL.so.1, libxcb.so.1, dan
    libgthread-2.0.so.0 dari sistem, yang tidak ada di image Streamlit Cloud.
    """
    ada = terpasang()
    assert "opencv-python-headless" in ada, (
        "opencv-python-headless tidak terpasang. Di Streamlit Cloud ini "
        "berarti import cv2 gagal dengan libGL.so.1 tidak ditemukan."
    )


def test_opencv_python_asli_tidak_terpasang():
    """Yang ini bukti bahwa pengalihnya benar-benar bekerja.

    Kalau versinya bukan versi pengalih, berarti `opencv-python` asli terunduh
    dari PyPI dan sudah menimpa berkas cv2 milik headless. Di Streamlit Cloud
    itu berarti aplikasi mati saat dijalankan, bukan saat dipasang.
    """
    dipakai = metadata.version("opencv-python")
    seharusnya = tomllib.loads(PENGALIH.read_text(encoding="utf-8"))["project"]["version"]
    assert dipakai == seharusnya, (
        f"opencv-python terpasang versi {dipakai}, bukan pengalih {seharusnya}. "
        "Berarti paket asli terunduh dan menimpa headless."
    )


def test_requirements_masih_memakai_pengalih():
    """Menjaga dua barisnya tidak hilang saat requirements dirapikan.

    Menulis headless sendirian tidak cukup, dan menulis keduanya berurutan juga
    tidak. Tabel hasil ujinya ada di vendor/opencv-python-stub/README.md.
    """
    isi = (AKAR / "requirements.txt").read_text(encoding="utf-8")
    baris = [b.strip() for b in isi.splitlines() if b.strip() and not b.startswith("#")]
    assert "./vendor/opencv-python-stub" in baris, (
        "baris ./vendor/opencv-python-stub hilang dari requirements.txt"
    )
    assert any(b.startswith("opencv-python-headless") for b in baris), (
        "baris opencv-python-headless hilang dari requirements.txt"
    )


def test_pengalih_tidak_membawa_kode():
    """Pengalih harus tetap kosong. Kalau ia mulai mengirim berkas, ia bisa
    bertabrakan dengan cv2 milik headless dan justru menciptakan masalah baru.
    """
    cfg = tomllib.loads(PENGALIH.read_text(encoding="utf-8"))
    assert cfg["project"]["name"] == "opencv-python"
    assert cfg["tool"]["setuptools"]["packages"] == []
    assert any(
        d.startswith("opencv-python-headless") for d in cfg["project"]["dependencies"]
    ), "pengalih tidak lagi menunjuk ke headless"


def test_packages_txt_tidak_dihidupkan_lagi():
    """Sengaja dibuang, dan sengaja dijaga tetap terbuang.

    `apt-get update` di image Streamlit Cloud mengembalikan exit code non-nol
    karena sumber bullseye yang tertinggal sudah kedaluwarsa, dan itu
    menggagalkan seluruh build sebelum satu paket pun dicoba dipasang.
    Selama cv2 datang dari wheel, berkas ini tidak dibutuhkan.
    """
    assert not (AKAR / "packages.txt").exists(), (
        "packages.txt muncul lagi. Keberadaannya memicu apt-get update, "
        "yang saat ini menggagalkan build. Lihat bagian Catatan versi di README."
    )


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
