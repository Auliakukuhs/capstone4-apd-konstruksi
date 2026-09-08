"""Uji lapisan analisis. Seluruhnya memakai kotak buatan, tanpa model.

Dijalankan tanpa Streamlit dan tanpa GPU:
    .venv-uji/bin/python -m pytest tests/ -v
atau langsung:
    .venv-uji/bin/python tests/test_analitik.py
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analitik import (  # noqa: E402
    BELUM_PASTI,
    LENGKAP,
    MEMAKAI,
    PERLU_DIPERIKSA,
    TIDAK_LENGKAP,
    TIDAK_MEMAKAI,
    Sesi,
    asosiasi,
    baris_csv,
    hitung_kelas,
    ioa,
    laporan_teks,
)


def d(cls, box, conf=0.9):
    """Satu deteksi, seperti keluaran src.detector.ke_deteksi."""
    return {"cls": cls, "conf": conf, "box": box}


def iou(a, b):
    """Hanya dipakai di dalam uji, untuk menunjukkan bedanya dengan IoA."""
    lebar = min(a[2], b[2]) - max(a[0], b[0])
    tinggi = min(a[3], b[3]) - max(a[1], b[1])
    if lebar <= 0 or tinggi <= 0:
        return 0.0
    irisan = lebar * tinggi
    luas_a = (a[2] - a[0]) * (a[3] - a[1])
    luas_b = (b[2] - b[0]) * (b[3] - b[1])
    return irisan / (luas_a + luas_b - irisan)


# ---------------------------------------------------------------- IoA


def test_ioa_bukan_iou():
    """Inti seluruh rancangan. Helmet termuat penuh, tapi IoU tetap kecil.

    Kalau ambang 0,5 diterapkan pada IoU, helmet yang jelas-jelas milik
    pekerja ini justru ditolak. Itu sebabnya IoA yang dipakai.
    """
    orang = (0, 0, 100, 300)
    helm = (35, 10, 65, 45)

    assert ioa(helm, orang) == 1.0, "helmet termuat penuh seharusnya bernilai 1,0"
    assert iou(helm, orang) < 0.10, "IoU tertekan oleh luas kotak person"


def test_ioa_setengah_di_luar():
    """Separuh keluar berarti separuh nilainya, bukan nol dan bukan satu."""
    assert ioa((0, 0, 10, 10), (5, 0, 100, 10)) == 0.5


def test_ioa_tanpa_irisan_dan_luas_nol():
    """Dua kasus tepi yang gampang melempar ZeroDivisionError."""
    assert ioa((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0
    assert ioa((5, 5, 5, 5), (0, 0, 100, 100)) == 0.0


# ---------------------------------------------------- tiga vonis dasar


def test_pekerja_lengkap():
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45)),
        d("vest", (20, 90, 80, 190)),
    ])
    p = hasil.pekerja[0]
    assert p.helmet == MEMAKAI
    assert p.vest == MEMAKAI
    assert p.vonis == LENGKAP
    assert hasil.ringkasan.kepatuhan == 1.0


def test_pekerja_melanggar():
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45)),
        d("no-vest", (20, 90, 80, 190)),
    ])
    p = hasil.pekerja[0]
    assert p.vest == TIDAK_MEMAKAI
    assert p.vonis == TIDAK_LENGKAP
    assert "rompi" in p.alasan


def test_atribut_tidak_terdeteksi_bukan_pelanggaran():
    """Kasus yang membedakan sistem ini dari aritmetika hitungan.

    Pada ground truth, 11,8 persen person tidak punya kotak helmet sama
    sekali. Tanpa status ketiga, seluruh pekerja itu dituduh melanggar.
    """
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("vest", (20, 90, 80, 190)),
    ])
    p = hasil.pekerja[0]
    assert p.helmet == BELUM_PASTI
    assert p.vonis == PERLU_DIPERIKSA, "tidak terdeteksi tidak boleh sama dengan melanggar"
    assert "perlu diperiksa manusia" in p.alasan


def test_pelanggaran_menang_atas_ketidakpastian():
    """Helm tidak terlihat, tapi rompi jelas tidak dipakai. Tetap melanggar."""
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("no-vest", (20, 90, 80, 190)),
    ])
    p = hasil.pekerja[0]
    assert p.helmet == BELUM_PASTI
    assert p.vonis == TIDAK_LENGKAP


# ------------------------------------------------------- kasus tepi


def test_nol_deteksi():
    """Gambar tanpa apa pun tidak boleh melempar exception."""
    hasil = asosiasi([])
    assert hasil.pekerja == []
    assert hasil.ringkasan.pekerja == 0
    assert hasil.ringkasan.kepatuhan is None, "0 dari 0 bukan nol persen"
    assert hasil.ringkasan.kepatuhan_terperiksa is None
    assert "0" in laporan_teks(hasil)


def test_atribut_tanpa_induk():
    """Helmet melayang tanpa person adalah petunjuk, bukan sampah.

    Artinya besar kemungkinan ada pekerja yang tidak terdeteksi, dan pengawas
    perlu tahu itu.
    """
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45)),
        d("helmet", (500, 500, 530, 535)),
    ])
    assert len(hasil.tanpa_induk) == 1
    assert hasil.tanpa_induk[0].cls == "helmet"
    assert hasil.ringkasan.tanpa_induk == {"helmet": 1}
    assert hasil.pekerja[0].vonis == PERLU_DIPERIKSA  # rompinya tetap tidak terlihat


def test_hanya_atribut_tanpa_satu_pun_person():
    hasil = asosiasi([d("helmet", (35, 10, 65, 45)), d("vest", (20, 90, 80, 190))])
    assert hasil.pekerja == []
    assert len(hasil.tanpa_induk) == 2
    assert hasil.ringkasan.kepatuhan is None


def test_hitung_kelas_menampilkan_yang_nol():
    hitung = hitung_kelas([d("person", (0, 0, 10, 10))])
    assert hitung["person"] == 1
    assert hitung["no-helmet"] == 0, "kelas nol harus tetap muncul, bukan hilang"
    assert len(hitung) == 5


# --------------------------------------------------- konflik dan urutan


def test_konflik_dimenangkan_confidence_tertinggi():
    """Model bisa mengeluarkan helmet dan no-helmet pada orang yang sama."""
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("no-helmet", (35, 10, 65, 45), conf=0.44),
        d("helmet", (36, 11, 66, 46), conf=0.81),
    ])
    p = hasil.pekerja[0]
    assert p.helmet == MEMAKAI, "0,81 harus mengalahkan 0,44"
    assert len(p.konflik) == 1
    assert "0.81" in p.konflik[0] and "0.44" in p.konflik[0]


def test_urutan_deteksi_tidak_mengubah_hasil():
    """Jaminan yang mudah hilang kalau konflik diselesaikan dengan menimpa.

    Ultralytics mengembalikan kotak terurut confidence, tapi urutan itu bukan
    sesuatu yang boleh diandalkan. Seluruh permutasi harus memberi vonis sama.
    """
    deteksi = [
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45), conf=0.81),
        d("no-helmet", (36, 11, 66, 46), conf=0.44),
        d("no-vest", (20, 90, 80, 190), conf=0.70),
    ]
    vonis = {
        asosiasi(list(urutan)).pekerja[0].vonis
        for urutan in itertools.permutations(deteksi)
    }
    assert vonis == {TIDAK_LENGKAP}, f"hasil berbeda antar urutan: {vonis}"


def test_tiebreak_posisi_vertikal():
    """Saat IoA seri, posisi vertikal yang memutuskan.

    Helm ini termuat penuh di dua kotak person sekaligus, jadi IoA keduanya
    1,0 dan tidak bisa dipakai memilih. Yang benar adalah pekerja yang
    kepalanya berada di situ, bukan yang perutnya kebetulan tertimpa.
    """
    kepala = (0, 95, 100, 295)   # kepalanya tepat di helm
    perut = (0, 0, 100, 200)     # helm jatuh di tengah badannya
    helm = (40, 100, 60, 120)

    assert ioa(helm, kepala) == 1.0
    assert ioa(helm, perut) == 1.0, "prasyarat uji, IoA memang tidak bisa memutuskan"

    hasil = asosiasi([d("person", perut), d("person", kepala), d("helmet", helm)])
    pemilik = [p for p in hasil.pekerja if p.helmet == MEMAKAI]
    assert len(pemilik) == 1
    assert pemilik[0].box == kepala, "helm diberikan ke pekerja yang salah"


# --------------------------------------------------------- keluaran


def test_dua_pekerja_diberi_nomor_kiri_ke_kanan():
    hasil = asosiasi([
        d("person", (200, 0, 300, 300)),
        d("person", (0, 0, 100, 300)),
    ])
    assert [p.nomor for p in hasil.pekerja] == [1, 2]
    assert hasil.pekerja[0].box[0] == 0, "nomor 1 seharusnya yang paling kiri"


def test_dua_angka_kepatuhan_berbeda():
    """Yang belum dapat dipastikan tidak boleh diam-diam dihitung melanggar."""
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45)),
        d("vest", (20, 90, 80, 190)),
        d("person", (200, 0, 300, 300)),
        d("no-vest", (220, 90, 280, 190)),
        d("person", (400, 0, 500, 300)),
    ])
    r = hasil.ringkasan
    assert (r.lengkap, r.tidak_lengkap, r.belum_pasti) == (1, 1, 1)
    assert abs(r.kepatuhan - 1 / 3) < 1e-9
    assert abs(r.kepatuhan_terperiksa - 1 / 2) < 1e-9


def test_baris_csv_memuat_bukti():
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45), conf=0.77),
    ])
    baris = baris_csv(hasil, "uji.jpg")
    assert len(baris) == 1
    b = baris[0]
    assert b["gambar"] == "uji.jpg"
    assert b["vonis"] == PERLU_DIPERIKSA
    assert "helmet conf 0.77 ioa 1.00" in b["bukti"], "angka pembentuk vonis harus ikut"


def test_sesi_menjumlahkan_lintas_gambar():
    sesi = Sesi()
    sesi.tambah("a.jpg", asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45)),
        d("vest", (20, 90, 80, 190)),
    ]))
    sesi.tambah("b.jpg", asosiasi([
        d("person", (0, 0, 100, 300)),
        d("no-vest", (20, 90, 80, 190)),
        d("helmet", (900, 900, 930, 935)),
    ]))
    total = sesi.total()
    assert len(sesi) == 2
    assert total.pekerja == 2
    assert total.lengkap == 1
    assert total.tidak_lengkap == 1
    assert total.tanpa_induk == {"helmet": 1}
    assert {b["gambar"] for b in sesi.semua_baris()} == {"a.jpg", "b.jpg"}


def test_laporan_teks_menyebut_ketiga_status():
    hasil = asosiasi([
        d("person", (0, 0, 100, 300)),
        d("helmet", (35, 10, 65, 45)),
    ])
    teks = laporan_teks(hasil)
    assert "Belum dapat dipastikan" in teks
    assert "Rincian per pekerja" in teks
    assert "#1" in teks


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
