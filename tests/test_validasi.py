"""Uji perkakas validasi. Fungsi bersama yang dipakai tiga tempat sekaligus.

`nilai_vonis` dipakai oleh `skrip/validasi_analitik.py` untuk laporan, oleh
`notebooks/03_eksperimen.ipynb` untuk memilih model, dan oleh
`notebooks/04_evaluasi_final.ipynb` untuk sapuan confidence. Kalau ia salah,
ketiganya salah bersamaan dengan cara yang sama, sehingga tidak ada yang
saling mengoreksi.

Dijalankan tanpa GPU dan tanpa dataset asli:
    .venv-uji/bin/python tests/test_validasi.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skrip.validasi_analitik import (  # noqa: E402
    iou,
    muat_label,
    nilai_vonis,
    pasangkan,
)
from src.analitik import (  # noqa: E402
    LENGKAP,
    PERLU_DIPERIKSA,
    TIDAK_LENGKAP,
    asosiasi,
)


def d(cls, box, conf=0.9):
    return {"cls": cls, "conf": conf, "box": box}


def pekerja(kotak, helm=None, rompi=None):
    """Satu HasilGambar berisi satu pekerja dengan atribut yang diminta."""
    x1, y1, x2, y2 = kotak
    det = [d("person", kotak)]
    if helm:
        det.append(d(helm, (x1 + 0.3, y1 + 0.02, x1 + 0.6, y1 + 0.12)))
    if rompi:
        det.append(d(rompi, (x1 + 0.2, y1 + 0.3, x1 + 0.7, y1 + 0.6)))
    return asosiasi(det)


LENGKAP_1 = (0.0, 0.0, 1.0, 1.0)


# ------------------------------------------------------------- iou


def test_iou_dasar():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


# -------------------------------------------------------- pasangkan


def test_pasangkan_rakus_dari_iou_tertinggi():
    """Yang paling cocok harus menang, bukan yang kebetulan diperiksa duluan.

    Satu kotak prediksi di sini memenuhi ambang terhadap dua kotak acuan
    sekaligus. Kalau pasangan diambil begitu ambang terpenuhi, acuan yang
    kebetulan lebih dulu di daftar akan merebutnya meski cocoknya lebih buruk.

    Kedua acuan sengaja diberi x1 yang sama, sebab `asosiasi` mengurutkan
    pekerja dari kiri ke kanan dan pengurutannya stabil, jadi urutan daftar
    di sini benar-benar dipertahankan.
    """
    prediksi = asosiasi([d("person", (0.0, 0.0, 0.50, 1.0))]).pekerja
    acuan = asosiasi([
        d("person", (0.0, 0.0, 0.90, 1.0)),   # indeks 0, IoU 0,56, cocok buruk
        d("person", (0.0, 0.0, 0.55, 1.0)),   # indeks 1, IoU 0,91, cocok baik
    ]).pekerja

    assert iou(prediksi[0].box, acuan[0].box) < iou(prediksi[0].box, acuan[1].box)

    hasil = dict(pasangkan(prediksi, acuan))
    assert hasil == {0: 1}, f"seharusnya berpasangan dengan acuan 1, dapat {hasil}"


def test_pasangkan_menolak_di_bawah_ambang():
    prediksi = asosiasi([d("person", (0.0, 0.0, 0.2, 0.2))]).pekerja
    acuan = asosiasi([d("person", (0.8, 0.8, 1.0, 1.0))]).pekerja
    assert pasangkan(prediksi, acuan) == []


def test_pasangkan_satu_kotak_dipakai_sekali():
    prediksi = asosiasi([d("person", (0.0, 0.0, 1.0, 1.0)),
                         d("person", (0.01, 0.01, 0.99, 0.99))]).pekerja
    acuan = asosiasi([d("person", (0.0, 0.0, 1.0, 1.0))]).pekerja
    assert len(pasangkan(prediksi, acuan)) == 1


# ------------------------------------------------------- nilai_vonis


def test_semua_vonis_benar():
    p = pekerja(LENGKAP_1, "helmet", "vest")
    m = nilai_vonis([(p, p)])
    assert m["pekerja_terpasangkan"] == 1
    assert m["vonis_benar"] == 1
    assert m["akurasi_vonis"] == 1.0
    assert m["cakupan_pekerja"] == 1.0
    assert m["laju_pembebasan_keliru"] is None, "tidak ada pelanggar, jadi tidak ada lajunya"


def test_pembebasan_keliru_terhitung():
    """Pelanggar dinyatakan lengkap. Kesalahan termahal di sistem ini."""
    acuan = pekerja(LENGKAP_1, "helmet", "no-vest")
    prediksi = pekerja(LENGKAP_1, "helmet", "vest")
    assert acuan.pekerja[0].vonis == TIDAK_LENGKAP
    assert prediksi.pekerja[0].vonis == LENGKAP

    m = nilai_vonis([(prediksi, acuan)])
    assert m["pembebasan_keliru"] == 1
    assert m["laju_pembebasan_keliru"] == 1.0
    assert m["tuduhan_palsu"] == 0


def test_tuduhan_palsu_terhitung():
    acuan = pekerja(LENGKAP_1, "helmet", "vest")
    prediksi = pekerja(LENGKAP_1, "helmet", "no-vest")
    m = nilai_vonis([(prediksi, acuan)])
    assert m["tuduhan_palsu"] == 1
    assert m["laju_tuduhan_palsu"] == 1.0
    assert m["pembebasan_keliru"] == 0


def test_simulasi_dua_status_selalu_lebih_buruk_atau_sama():
    """Inti argumen status ketiga, dan ia harus benar secara aritmetika.

    Tuduhan palsu versi dua status adalah tuduhan palsu ditambah pekerja patuh
    yang dilempar ke pemeriksaan manusia, jadi ia tidak mungkin lebih kecil.
    """
    acuan = pekerja(LENGKAP_1, "helmet", "vest")
    prediksi = pekerja(LENGKAP_1, "helmet")       # rompi tidak terdeteksi
    assert prediksi.pekerja[0].vonis == PERLU_DIPERIKSA

    m = nilai_vonis([(prediksi, acuan)])
    assert m["tuduhan_palsu"] == 0
    assert m["tuduhan_palsu_kalau_hanya_dua_status"] == 1
    assert m["laju_tuduhan_palsu_dua_status"] >= m["laju_tuduhan_palsu"]


def test_perlu_pemeriksaan_manusia_dihitung():
    """Ongkos kehati-hatian harus ikut terlihat, bukan hanya manfaatnya."""
    acuan = pekerja(LENGKAP_1, "helmet", "vest")
    prediksi = pekerja(LENGKAP_1, "helmet")
    m = nilai_vonis([(prediksi, acuan)])
    assert m["perlu_pemeriksaan_manusia"] == 1
    assert m["porsi_perlu_pemeriksaan_manusia"] == 1.0


def test_pekerja_terlewat_dan_palsu():
    acuan = asosiasi([d("person", (0.0, 0.0, 0.4, 1.0)), d("person", (0.6, 0.0, 1.0, 1.0))])
    prediksi = asosiasi([d("person", (0.0, 0.0, 0.4, 1.0)), d("person", (0.42, 0.0, 0.5, 0.3))])
    m = nilai_vonis([(prediksi, acuan)])
    assert m["pekerja_terpasangkan"] == 1
    assert m["pekerja_terlewat"] == 1
    assert m["pekerja_palsu"] == 1
    assert m["cakupan_pekerja"] == 0.5


def test_tanpa_gambar_sama_sekali():
    """Nol pasangan tidak boleh melempar ZeroDivisionError."""
    m = nilai_vonis([])
    assert m["pekerja_terpasangkan"] == 0
    assert m["akurasi_vonis"] is None
    assert m["cakupan_pekerja"] is None
    assert m["porsi_perlu_pemeriksaan_manusia"] is None


# -------------------------------------------------------- muat_label


def test_muat_label_membaca_format_yolo():
    """Indeks 3 adalah person, bukan 0. Ini yang paling sering salah."""
    with tempfile.TemporaryDirectory() as t:
        p = Path(t, "a.txt")
        p.write_text("3 0.5 0.5 0.2 0.4\n1 0.25 0.1 0.05 0.05\n", encoding="utf-8")

        hasil = muat_label(p)

        assert [h["cls"] for h in hasil] == ["person", "no-helmet"]
        assert all(h["conf"] == 1.0 for h in hasil), "anotasi manusia bukan dugaan"
        x1, y1, x2, y2 = hasil[0]["box"]
        assert abs(x1 - 0.4) < 1e-9 and abs(x2 - 0.6) < 1e-9
        assert abs(y1 - 0.3) < 1e-9 and abs(y2 - 0.7) < 1e-9


def test_muat_label_melewati_baris_rusak():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t, "a.txt")
        p.write_text("3 0.5 0.5 0.2 0.4\n\n  \n0 0.1\n", encoding="utf-8")
        assert len(muat_label(p)) == 1


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
