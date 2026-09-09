"""Perkakas eksperimen training hari 8 sampai 10.

Isi berkas ini sengaja tidak berada di dalam sel notebook. Aturan pemilihan
model menentukan bobot mana yang akhirnya dipakai aplikasi, dan aturan seperti
itu harus bisa diuji, bukan dipercaya begitu saja karena kelihatan masuk akal.

Tiga hal di sini.

`bangun_oversample` menyusun ulang split train dengan menduplikasi gambar yang
memuat kelas minoritas. Ini penanganan tingkat data, bukan augmentasi, jadi ia
tidak menyentuh larangan augmentasi geometris di SOAL.

`tabel_perbandingan` menyusun catatan seluruh run menjadi satu tabel.

`pilih_model` menerapkan aturan pemilihan yang **dinyatakan di muka**, sebelum
angkanya terlihat. Itu bedanya memilih dengan kriteria dan memilih sesudah tahu
siapa yang menang.
"""

import shutil
from pathlib import Path

KELAS = ("helmet", "no-helmet", "no-vest", "person", "vest")


def _kelas_di_label(jalur: Path) -> set:
    """Indeks kelas yang muncul di satu berkas label YOLO."""
    ada = set()
    for baris in jalur.read_text(encoding="utf-8").splitlines():
        bagian = baris.split()
        if bagian:
            ada.add(int(bagian[0]))
    return ada


def bangun_oversample(
    asal: Path,
    tujuan: Path,
    kelas_target: str = "no-helmet",
    kali: int = 4,
) -> dict:
    """Salin split train, lalu duplikasi gambar yang memuat `kelas_target`.

    Kenapa duplikasi gambar utuh dan bukan augmentasi. `copy_paste` milik
    Ultralytics tidak dapat dipakai di sini, karena ia keluar tanpa berbuat
    apa pun kalau label tidak punya segmentasi, dan seluruh 7.724 baris label
    dataset ini hanya berisi lima kolom, yaitu bbox saja. Kalau dipakai, run
    itu akan identik dengan run sebelumnya sementara namanya menjanjikan hal
    lain.

    Diukur pada dataset ini, `no-helmet` hadir di 49 dari 997 gambar train.
    Dengan `kali=4`, porsinya naik dari 1,5 menjadi 4,7 persen sementara
    kelas lain hampir tidak bergerak, dan gambar train hanya bertambah 14,7
    persen.

    Risikonya dinyatakan terbuka. Yang diduplikasi hanya 49 foto berbeda, dan
    model melihat masing-masing empat kali per epoch, jadi peluang overfitting
    pada foto itu memang naik. Kalau `no-helmet` membaik tapi kelas lain
    memburuk, inilah tersangka pertamanya.

    Returns:
        Dict berisi angka sebelum dan sesudah, untuk dicatat di laporan.
    """
    if kelas_target not in KELAS:
        raise ValueError(f"kelas {kelas_target} tidak ada di {KELAS}")
    if kali < 1:
        raise ValueError("kali minimal 1")

    indeks_target = KELAS.index(kelas_target)
    gambar_asal = asal / "images"
    label_asal = asal / "labels"
    if not label_asal.is_dir():
        raise FileNotFoundError(f"tidak ada folder label di {label_asal}")

    gambar_tujuan = tujuan / "images"
    label_tujuan = tujuan / "labels"
    for d in (gambar_tujuan, label_tujuan):
        d.mkdir(parents=True, exist_ok=True)

    semua = sorted(label_asal.glob("*.txt"))
    disalin = diduplikasi = 0
    instance_sebelum = {k: 0 for k in KELAS}
    instance_sesudah = {k: 0 for k in KELAS}

    for label in semua:
        gambar = next(
            (g for e in (".jpg", ".jpeg", ".png", ".webp") if (g := gambar_asal / (label.stem + e)).exists()),
            None,
        )
        if gambar is None:
            continue

        isi = label.read_text(encoding="utf-8")
        hitung = {k: 0 for k in KELAS}
        for baris in isi.splitlines():
            bagian = baris.split()
            if bagian:
                hitung[KELAS[int(bagian[0])]] += 1
        for k, n in hitung.items():
            instance_sebelum[k] += n

        salinan = kali if indeks_target in _kelas_di_label(label) else 1
        for i in range(salinan):
            akhiran = "" if i == 0 else f"_dup{i}"
            shutil.copy(gambar, gambar_tujuan / f"{label.stem}{akhiran}{gambar.suffix}")
            (label_tujuan / f"{label.stem}{akhiran}.txt").write_text(isi, encoding="utf-8")
            for k, n in hitung.items():
                instance_sesudah[k] += n

        disalin += 1
        if salinan > 1:
            diduplikasi += 1

    total_sebelum = sum(instance_sebelum.values()) or 1
    total_sesudah = sum(instance_sesudah.values()) or 1
    return {
        "kelas_target": kelas_target,
        "kali": kali,
        "gambar_asal": disalin,
        "gambar_memuat_target": diduplikasi,
        "gambar_hasil": disalin + diduplikasi * (kali - 1),
        "instance_sebelum": instance_sebelum,
        "instance_sesudah": instance_sesudah,
        "porsi_target_sebelum": round(instance_sebelum[kelas_target] / total_sebelum, 4),
        "porsi_target_sesudah": round(instance_sesudah[kelas_target] / total_sesudah, 4),
    }


def _ambil(catatan: dict, *jalur, bawaan=None):
    """Baca nilai bersarang dengan aman, karena catatan run bisa tidak lengkap."""
    kini = catatan
    for k in jalur:
        if not isinstance(kini, dict) or k not in kini:
            return bawaan
        kini = kini[k]
    return kini


def tabel_perbandingan(daftar: list) -> list:
    """Satu baris per run, kolom yang benar-benar dipakai memutuskan.

    mAP ikut ditampilkan, tapi ia bukan kolom yang menentukan. Yang menentukan
    tiga kolom terakhir, karena aplikasi ini menjatuhkan vonis tentang orang
    dan biaya ketiga jenis kesalahannya sangat berbeda.
    """
    baris = []
    for c in daftar:
        vonis = c.get("vonis_test", {}) or {}
        baris.append(
            {
                "run": c.get("run", "?"),
                "kandidat": bool(c.get("kandidat", True)),
                "model_awal": c.get("model_awal"),
                "imgsz": c.get("imgsz"),
                "epoch_berjalan": c.get("epochs_berjalan"),
                "epoch_terbaik": c.get("epoch_terbaik"),
                "map50": c.get("map50"),
                "map50_95": c.get("map50_95"),
                "R_no_helmet": _ambil(c, "per_kelas", "no-helmet", "R"),
                "akurasi_vonis": vonis.get("akurasi_vonis"),
                "laju_pembebasan_keliru": vonis.get("laju_pembebasan_keliru"),
                "laju_tuduhan_palsu": vonis.get("laju_tuduhan_palsu"),
                "cakupan_pekerja": vonis.get("cakupan_pekerja"),
            }
        )
    return baris


def pilih_model(daftar: list, toleransi_cakupan: float = 0.05):
    """Terapkan aturan pemilihan. Dinyatakan di muka, sebelum angkanya dilihat.

    Aturannya berurutan.

    1. Run yang ditandai bukan kandidat dikeluarkan. Ini untuk run pembanding
       yang sengaja melanggar batasan SOAL, misalnya yang menyalakan mosaic.
       Ia diukur untuk mengetahui berapa mAP yang dikorbankan demi kepatuhan,
       bukan untuk dipakai.
    2. **Penjaga cakupan.** Run yang cakupan pekerjanya lebih dari
       `toleransi_cakupan` di bawah cakupan terbaik dikeluarkan. Tanpa penjaga
       ini, model yang hampir tidak mendeteksi siapa pun akan terlihat unggul,
       sebab laju kesalahan dihitung dari pekerja yang berhasil dipasangkan
       dan penyebutnya menyusut.
    3. **Laju pembebasan keliru terendah menang.** Pelanggar yang dinyatakan
       lengkap adalah kesalahan termahal di sistem keselamatan, karena ia
       menghentikan pemeriksaan terhadap orang yang justru berisiko.
    4. Seri diputus akurasi vonis tertinggi, lalu mAP@0.5:0.95 tertinggi.

    Perhatikan bahwa mAP ada di urutan terakhir, bukan pertama. Itu disengaja.
    mAP mengukur kualitas kotak, sementara yang dipakai pengawas adalah vonis.

    Returns:
        Tuple (nama_terpilih, alasan, tabel_terurut). `nama_terpilih` bernilai
        None kalau tidak ada kandidat yang lolos penjaga.
    """
    baris = tabel_perbandingan(daftar)
    kandidat = [b for b in baris if b["kandidat"] and b["cakupan_pekerja"] is not None]
    if not kandidat:
        return None, "tidak ada run kandidat yang punya angka vonis", baris

    cakupan_terbaik = max(b["cakupan_pekerja"] for b in kandidat)
    batas = cakupan_terbaik - toleransi_cakupan
    lolos = [b for b in kandidat if b["cakupan_pekerja"] >= batas]
    tersingkir = [b["run"] for b in kandidat if b not in lolos]

    def kunci(b):
        return (
            b["laju_pembebasan_keliru"] if b["laju_pembebasan_keliru"] is not None else 9.9,
            -(b["akurasi_vonis"] or 0),
            -(b["map50_95"] or 0),
        )

    lolos.sort(key=kunci)
    menang = lolos[0]

    alasan = [
        f"Terpilih {menang['run']}.",
        f"Laju pembebasan keliru {menang['laju_pembebasan_keliru']}, "
        f"terendah di antara {len(lolos)} run yang lolos penjaga cakupan.",
        f"Akurasi vonis {menang['akurasi_vonis']}, "
        f"cakupan pekerja {menang['cakupan_pekerja']}, "
        f"mAP@0.5:0.95 {menang['map50_95']}.",
    ]
    if tersingkir:
        alasan.append(
            f"Tersingkir oleh penjaga cakupan, ambang {round(batas, 4)}: "
            + ", ".join(tersingkir)
            + ". Cakupan yang jatuh berarti banyak pekerja tidak terdeteksi, "
            "dan laju kesalahannya menjadi tidak sebanding."
        )

    return menang["run"], " ".join(alasan), lolos + [b for b in baris if b not in lolos]
