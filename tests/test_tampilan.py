"""Uji komponen antarmuka, bagian yang murni.

Yang diuji di sini adalah fungsi yang memutuskan dan menyusun, bukan yang
memanggil Streamlit. `pesan_banner` diuji, `banner` tidak, karena yang kedua
hanya satu baris pemetaan ke `st.error` dan sejenisnya.

Dijalankan tanpa GPU dan tanpa server Streamlit:
    .venv-uji/bin/python -m pytest tests/ -v
atau langsung:
    .venv-uji/bin/python tests/test_tampilan.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.analitik import (  # noqa: E402
    LENGKAP,
    PERLU_DIPERIKSA,
    TIDAK_LENGKAP,
    asosiasi,
)
from src.tampilan import (  # noqa: E402
    LUMA_MAKS,
    LUMA_MIN,
    SINGKATAN,
    WARNA_VONIS,
    gambar_vonis,
    peringatan_kelas_terlemah,
    peringatan_pencahayaan,
    pesan_banner,
    sebagai_csv,
)


def d(cls, box, conf=0.9):
    return {"cls": cls, "conf": conf, "box": box}


LENGKAP_SATU = [
    d("person", (10, 10, 110, 310)),
    d("helmet", (45, 20, 75, 55)),
    d("vest", (30, 100, 90, 200)),
]


# ------------------------------------------------------------- banner


def test_banner_aman_hanya_kalau_benar_benar_bersih():
    tingkat, kalimat = pesan_banner(asosiasi(LENGKAP_SATU).ringkasan)
    assert tingkat == "aman"
    assert "1 pekerja" in kalimat


def test_banner_bahaya_saat_ada_pelanggaran():
    hasil = asosiasi([
        d("person", (10, 10, 110, 310)),
        d("helmet", (45, 20, 75, 55)),
        d("no-vest", (30, 100, 90, 200)),
    ])
    tingkat, kalimat = pesan_banner(hasil.ringkasan)
    assert tingkat == "bahaya"
    assert "Hentikan pekerjaan" in kalimat


def test_banner_hati_hati_saat_hanya_belum_pasti():
    hasil = asosiasi([d("person", (10, 10, 110, 310))])
    tingkat, kalimat = pesan_banner(hasil.ringkasan)
    assert tingkat == "hati-hati"
    assert "jangan dianggap sudah aman" in kalimat


def test_banner_bahaya_menang_tapi_tetap_menyebut_yang_belum_pasti():
    """Pelanggaran diperiksa lebih dulu, tapi ketidakpastian tidak dibuang.

    Kalau kalimatnya hanya menyebut pelanggar, pekerja yang belum dapat
    dipastikan akan luput dari pemeriksaan manusia justru saat perhatian
    pengawas sedang tertuju ke gambar itu.
    """
    hasil = asosiasi([
        d("person", (10, 10, 110, 310)),
        d("no-vest", (30, 100, 90, 200)),
        d("person", (300, 10, 400, 310)),
    ])
    tingkat, kalimat = pesan_banner(hasil.ringkasan)
    assert tingkat == "bahaya"
    assert "1 pekerja belum dapat dipastikan" in kalimat


def test_banner_info_saat_tidak_ada_pekerja():
    tingkat, kalimat = pesan_banner(asosiasi([]).ringkasan)
    assert tingkat == "info"
    assert "confidence" in kalimat


def test_setiap_vonis_punya_warna():
    """Vonis tanpa warna akan melempar KeyError saat menggambar."""
    assert set(WARNA_VONIS) == {LENGKAP, TIDAK_LENGKAP, PERLU_DIPERIKSA}


# ----------------------------------------------------------- menggambar


def test_gambar_vonis_tidak_mengubah_aslinya():
    """Gambar asli dipakai lagi di kolom kiri, jadi tidak boleh ternoda."""
    asli = Image.new("RGB", (200, 400), (120, 120, 120))
    sebelum = asli.tobytes()

    keluaran = gambar_vonis(asli, asosiasi(LENGKAP_SATU))

    assert asli.tobytes() == sebelum, "gambar asli ikut tergambari"
    assert keluaran.size == asli.size
    assert keluaran.tobytes() != sebelum, "tidak ada yang tergambar sama sekali"


def test_gambar_vonis_nol_deteksi():
    asli = Image.new("RGB", (120, 90), (200, 200, 200))
    keluaran = gambar_vonis(asli, asosiasi([]))
    assert keluaran.size == asli.size
    assert keluaran.tobytes() == asli.tobytes(), "seharusnya tidak ada yang digambar"


def test_gambar_vonis_kotak_menempel_tepi_atas():
    """Label pekerja di baris teratas tidak boleh keluar bidang gambar.

    Kasus ini nyata, karena pekerja yang jauh sering terpotong tepi atas foto.
    Tanpa penanganan, teks vonisnya digambar di koordinat negatif dan hilang.
    """
    asli = Image.new("RGB", (200, 300), (120, 120, 120))
    hasil = asosiasi([d("person", (0, 0, 100, 300)), d("helmet", (30, 0, 60, 30))])
    keluaran = gambar_vonis(asli, hasil)

    baris_atas = [keluaran.getpixel((x, 3)) for x in range(0, 100)]
    assert any(p != (120, 120, 120) for p in baris_atas), "label tidak terlihat"


def test_gambar_vonis_bisa_menyembunyikan_atribut():
    asli = Image.new("RGB", (200, 400), (120, 120, 120))
    hasil = asosiasi(LENGKAP_SATU)
    dengan = gambar_vonis(asli, hasil, tampilkan_atribut=True)
    tanpa = gambar_vonis(asli, hasil, tampilkan_atribut=False)
    assert dengan.tobytes() != tanpa.tobytes()


def test_gambar_vonis_menandai_atribut_tanpa_induk():
    asli = Image.new("RGB", (400, 400), (120, 120, 120))
    hasil = asosiasi([d("helmet", (300, 300, 340, 340))])
    assert len(hasil.tanpa_induk) == 1
    keluaran = gambar_vonis(asli, hasil)
    assert keluaran.tobytes() != asli.tobytes(), "atribut tanpa induk tidak digambar"


def test_label_menyusut_pada_kotak_sempit():
    """Foto padat adalah kasus normal di dataset ini, bukan kasus tepi.

    Rata-rata 6,41 objek per gambar dengan maksimum 39. Kalau label vonis
    selalu ditulis penuh, sembilan pekerja berdampingan menghasilkan tumpukan
    teks yang tidak satu pun terbaca. Ujinya begini, kotak sempit harus
    menghasilkan gambar yang berbeda dari kotak lebar, karena labelnya
    memendek, bukan melimpah.
    """
    kanvas = Image.new("RGB", (600, 400), (120, 120, 120))

    lebar = asosiasi([d("person", (10, 20, 260, 380)), d("no-vest", (40, 120, 220, 260))])
    sempit = asosiasi([d("person", (10, 20, 55, 380)), d("no-vest", (15, 120, 50, 260))])

    a = gambar_vonis(kanvas, lebar)
    b = gambar_vonis(kanvas, sempit)
    assert a.tobytes() != b.tobytes()

    # Keduanya tetap menggambar sesuatu. Kotak sempit tidak boleh menjadi
    # kotak tanpa penanda apa pun.
    assert a.tobytes() != kanvas.tobytes()
    assert b.tobytes() != kanvas.tobytes()


def test_setiap_vonis_punya_singkatan():
    assert set(SINGKATAN) == set(WARNA_VONIS)
    assert all(len(v) <= 8 for v in SINGKATAN.values()), "singkatan harus pendek"


# ------------------------------------------------------- pencahayaan


def test_peringatan_pencahayaan():
    assert peringatan_pencahayaan(LUMA_MIN - 20).startswith("Gambar ini gelap")
    assert "sangat terang" in peringatan_pencahayaan(LUMA_MAKS + 20)
    assert peringatan_pencahayaan((LUMA_MIN + LUMA_MAKS) / 2) == ""


# ------------------------------------------- keterbatasan model


def catatan_uji(**recall):
    return {"per_kelas": {k: {"R": v} for k, v in recall.items()}}


def test_kelas_terlemah_dicari_dari_angkanya():
    """Bug nyata yang pernah ada di sini.

    Kalimat ini dulu menulis recall 0,333 langsung di dalam teks. Begitu model
    berganti, tabel di panel yang sama menampilkan 0,458 sementara kalimat di
    bawahnya masih menyebut 0,333, dan tidak ada yang memberi tahu.
    """
    pesan = peringatan_kelas_terlemah(catatan_uji(person=0.85, helmet=0.88, **{"no-vest": 0.41}))
    assert "no-vest" in pesan
    assert "0,41" in pesan, "desimal harus koma, mengikuti prosa repo"
    assert "no-helmet" not in pesan, "kelas terlemah tidak boleh diasumsikan"


def test_kelas_terlemah_ikut_berubah_saat_modelnya_berubah():
    lemah = peringatan_kelas_terlemah(catatan_uji(**{"no-helmet": 0.333, "person": 0.9}))
    baik = peringatan_kelas_terlemah(catatan_uji(**{"no-helmet": 0.458, "person": 0.9}))
    assert "0,333" in lemah and "0,458" in baik
    assert lemah != baik


def test_kalimat_menyesuaikan_seberapa_buruk():
    assert "lebih dari separuh" in peringatan_kelas_terlemah(catatan_uji(a=0.40))
    assert "sebagian" in peringatan_kelas_terlemah(catatan_uji(a=0.70))


def test_jumlah_contoh_latih_disebut_kalau_tercatat():
    c = catatan_uji(**{"no-helmet": 0.458})
    c["oversample"] = {"instance_sesudah": {"no-helmet": 376}}
    assert "376 contoh latih" in peringatan_kelas_terlemah(c)


def test_catatan_kosong_tidak_menghasilkan_kalimat():
    """Tanpa catatan versi, lebih baik diam daripada mengarang angka."""
    assert peringatan_kelas_terlemah({}) == ""
    assert peringatan_kelas_terlemah({"per_kelas": {}}) == ""
    assert peringatan_kelas_terlemah({"per_kelas": {"a": "bukan dict"}}) == ""


# --------------------------------------------------------------- csv


def test_csv_mengutip_koma_dan_tanda_petik():
    """Kolom bukti dan alasan memang mengandung koma, jadi ini bukan kasus tepi."""
    isi = sebagai_csv([
        {"a": "helmet 0.81, vest 0.70", "b": 'kata "penting"', "c": 3},
        {"a": "biasa", "b": "", "c": 4},
    ])
    baris = isi.splitlines()
    assert baris[0] == "a,b,c"
    assert baris[1] == '"helmet 0.81, vest 0.70","kata ""penting""",3'
    assert baris[2] == "biasa,,4"


def test_csv_kosong():
    assert sebagai_csv([]) == ""


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
