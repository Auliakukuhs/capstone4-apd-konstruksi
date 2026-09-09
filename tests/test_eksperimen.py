"""Uji perkakas eksperimen. Aturan pemilihan model diuji, bukan dipercaya.

Bobot mana yang akhirnya dipakai aplikasi ditentukan oleh `pilih_model`. Kalau
aturan itu salah, model yang lebih buruk bisa terpilih tanpa ada yang
menyadarinya, karena angkanya tetap terlihat rapi di tabel.

Dijalankan tanpa GPU dan tanpa dataset asli:
    .venv-uji/bin/python -m pytest tests/ -v
atau langsung:
    .venv-uji/bin/python tests/test_eksperimen.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skrip.eksperimen import (  # noqa: E402
    bangun_oversample,
    pilih_model,
    tabel_perbandingan,
)


def buat_dataset(akar: Path, isi: dict):
    """Bangun split train tiruan. `isi` memetakan nama ke daftar indeks kelas."""
    (akar / "images").mkdir(parents=True)
    (akar / "labels").mkdir(parents=True)
    for nama, kelas in isi.items():
        (akar / "images" / f"{nama}.jpg").write_bytes(b"bukan-gambar-sungguhan")
        baris = "\n".join(f"{k} 0.5 0.5 0.2 0.3" for k in kelas)
        (akar / "labels" / f"{nama}.txt").write_text(baris, encoding="utf-8")


def catatan(run, **kw):
    """Catatan run tiruan dengan bentuk yang sama seperti yang ditulis notebook."""
    dasar = {
        "run": run,
        "kandidat": True,
        "model_awal": "yolo11n.pt",
        "imgsz": 640,
        "epochs_berjalan": 40,
        "epoch_terbaik": 30,
        "map50": 0.7,
        "map50_95": 0.37,
        "per_kelas": {"no-helmet": {"R": 0.33}},
        "vonis_test": {
            "akurasi_vonis": 0.70,
            "laju_pembebasan_keliru": 0.07,
            "laju_tuduhan_palsu": 0.07,
            "cakupan_pekerja": 0.87,
        },
    }
    for k, v in kw.items():
        if k in ("akurasi_vonis", "laju_pembebasan_keliru", "laju_tuduhan_palsu", "cakupan_pekerja"):
            dasar["vonis_test"][k] = v
        elif k == "R_no_helmet":
            dasar["per_kelas"]["no-helmet"]["R"] = v
        else:
            dasar[k] = v
    return dasar


# ------------------------------------------------------- oversample


def test_oversample_menduplikasi_hanya_yang_memuat_target():
    with tempfile.TemporaryDirectory() as tmp:
        asal, tujuan = Path(tmp) / "asal", Path(tmp) / "tujuan"
        buat_dataset(asal, {
            "a": [3, 0],        # person, helmet
            "b": [3, 1],        # person, NO-HELMET
            "c": [3, 4],        # person, vest
            "d": [1, 1, 3],     # NO-HELMET dua, person
        })

        stat = bangun_oversample(asal, tujuan, kali=4)

        assert stat["gambar_asal"] == 4
        assert stat["gambar_memuat_target"] == 2
        assert stat["gambar_hasil"] == 4 + 2 * 3
        assert len(list((tujuan / "images").glob("*"))) == 10
        assert len(list((tujuan / "labels").glob("*.txt"))) == 10


def test_oversample_menaikkan_porsi_kelas_langka():
    with tempfile.TemporaryDirectory() as tmp:
        asal, tujuan = Path(tmp) / "asal", Path(tmp) / "tujuan"
        buat_dataset(asal, {f"biasa{i}": [3, 0, 4] for i in range(20)} | {"langka": [1, 3]})

        stat = bangun_oversample(asal, tujuan, kali=4)

        assert stat["instance_sebelum"]["no-helmet"] == 1
        assert stat["instance_sesudah"]["no-helmet"] == 4
        assert stat["porsi_target_sesudah"] > stat["porsi_target_sebelum"] * 3


def test_oversample_kali_satu_tidak_mengubah_apa_pun():
    """kali=1 harus menjadi salinan biasa, berguna sebagai pembanding jujur."""
    with tempfile.TemporaryDirectory() as tmp:
        asal, tujuan = Path(tmp) / "asal", Path(tmp) / "tujuan"
        buat_dataset(asal, {"a": [3, 1], "b": [3, 0]})

        stat = bangun_oversample(asal, tujuan, kali=1)

        assert stat["gambar_hasil"] == 2
        assert stat["instance_sebelum"] == stat["instance_sesudah"]


def test_oversample_label_ikut_tersalin_utuh():
    """Duplikat tanpa labelnya akan menjadi gambar tanpa anotasi, dan YOLO
    memperlakukannya sebagai background, yang justru merusak training."""
    with tempfile.TemporaryDirectory() as tmp:
        asal, tujuan = Path(tmp) / "asal", Path(tmp) / "tujuan"
        buat_dataset(asal, {"a": [1, 3]})

        bangun_oversample(asal, tujuan, kali=3)

        label = sorted((tujuan / "labels").glob("*.txt"))
        assert len(label) == 3
        isi = {p.read_text(encoding="utf-8") for p in label}
        assert len(isi) == 1, "isi label duplikat harus identik dengan aslinya"
        assert all(p.stat().st_size > 0 for p in label)


def test_oversample_menolak_masukan_keliru():
    with tempfile.TemporaryDirectory() as tmp:
        asal, tujuan = Path(tmp) / "asal", Path(tmp) / "tujuan"
        buat_dataset(asal, {"a": [3]})

        try:
            bangun_oversample(asal, tujuan, kelas_target="topi")
            raise AssertionError("kelas tak dikenal seharusnya ditolak")
        except ValueError:
            pass

        shutil.rmtree(tujuan, ignore_errors=True)
        try:
            bangun_oversample(asal, tujuan, kali=0)
            raise AssertionError("kali nol seharusnya ditolak")
        except ValueError:
            pass


# ----------------------------------------------------- pemilihan


def test_pilih_yang_pembebasan_kelirunya_terendah():
    terpilih, alasan, _ = pilih_model([
        catatan("v2", laju_pembebasan_keliru=0.07),
        catatan("v3", laju_pembebasan_keliru=0.03),
        catatan("v4", laju_pembebasan_keliru=0.11),
    ])
    assert terpilih == "v3"
    assert "0.03" in alasan


def test_map_tinggi_tidak_otomatis_menang():
    """Ini inti aturannya. mAP mengukur kotak, vonis yang dipakai pengawas."""
    terpilih, _, _ = pilih_model([
        catatan("v2_map_tinggi", map50_95=0.55, laju_pembebasan_keliru=0.20),
        catatan("v3_vonis_baik", map50_95=0.30, laju_pembebasan_keliru=0.04),
    ])
    assert terpilih == "v3_vonis_baik", "mAP tidak boleh mengalahkan biaya kesalahan"


def test_penjaga_cakupan_menyingkirkan_model_yang_buta():
    """Model yang hampir tidak mendeteksi pekerja punya laju kesalahan bagus
    hanya karena penyebutnya menyusut. Ia harus tersingkir, bukan menang."""
    terpilih, alasan, _ = pilih_model([
        catatan("v2_normal", cakupan_pekerja=0.87, laju_pembebasan_keliru=0.07),
        catatan("v9_buta", cakupan_pekerja=0.20, laju_pembebasan_keliru=0.00),
    ])
    assert terpilih == "v2_normal"
    assert "v9_buta" in alasan
    assert "penjaga cakupan" in alasan


def test_run_pembanding_dikeluarkan():
    """v5 menyalakan mosaic, melanggar SOAL. Ia diukur, bukan dipakai."""
    terpilih, _, _ = pilih_model([
        catatan("v2", laju_pembebasan_keliru=0.07),
        catatan("v5_dengan_mosaic", kandidat=False, laju_pembebasan_keliru=0.01),
    ])
    assert terpilih == "v2"


def test_seri_diputus_akurasi_lalu_map():
    terpilih, _, _ = pilih_model([
        catatan("a", laju_pembebasan_keliru=0.05, akurasi_vonis=0.70, map50_95=0.40),
        catatan("b", laju_pembebasan_keliru=0.05, akurasi_vonis=0.75, map50_95=0.30),
    ])
    assert terpilih == "b", "akurasi vonis memutus lebih dulu daripada mAP"

    terpilih, _, _ = pilih_model([
        catatan("c", laju_pembebasan_keliru=0.05, akurasi_vonis=0.70, map50_95=0.40),
        catatan("d", laju_pembebasan_keliru=0.05, akurasi_vonis=0.70, map50_95=0.44),
    ])
    assert terpilih == "d"


def test_tanpa_angka_vonis_tidak_memaksa_memilih():
    """Lebih baik mengaku tidak bisa memilih daripada memilih secara acak."""
    tanpa = catatan("v2")
    tanpa["vonis_test"] = {}
    terpilih, alasan, _ = pilih_model([tanpa])
    assert terpilih is None
    assert "tidak ada run kandidat" in alasan


def test_run_tanpa_angka_vonis_tersingkir_tanpa_memblokir_yang_lain():
    """Celah yang sempat nyata di repo ini.

    Catatan v1 dibuat sebelum penilaian vonis ada, jadi ia tidak punya kolom
    itu dan otomatis tersingkir dari pemilihan. Akibatnya "pertahankan
    baseline" bukan hasil yang mungkin, dan eksperimen hanya bisa membaik
    menurut konstruksinya sendiri. Catatan v1 sudah dilengkapi, dan uji ini
    yang menjaga perilakunya tetap benar kalau kasus serupa muncul lagi.
    """
    lengkap = catatan("punya_angka", laju_pembebasan_keliru=0.09)
    kosong = catatan("tanpa_angka")
    kosong["vonis_test"] = {}

    terpilih, _, tabel = pilih_model([kosong, lengkap])

    assert terpilih == "punya_angka", "yang punya angka harus tetap bisa menang"
    assert any(b["run"] == "tanpa_angka" for b in tabel), (
        "run tanpa angka tetap harus muncul di tabel, bukan hilang diam-diam"
    )


def test_tabel_bertahan_pada_catatan_tidak_lengkap():
    """Run yang mati di tengah jalan meninggalkan catatan setengah. Tabelnya
    harus tetap tersusun, bukan melempar KeyError."""
    baris = tabel_perbandingan([{"run": "setengah"}])
    assert baris[0]["run"] == "setengah"
    assert baris[0]["map50"] is None
    assert baris[0]["R_no_helmet"] is None
    assert baris[0]["cakupan_pekerja"] is None


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
