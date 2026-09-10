"""Validasi lapisan analisis pada test set, bukan pada contoh buatan.

Unit test di `tests/test_analitik.py` membuktikan logikanya benar pada kasus
yang saya rancang sendiri. Itu perlu, tapi tidak cukup. Yang belum terjawab
adalah seberapa sering vonis per pekerja benar pada gambar sungguhan.

Skrip ini menjawabnya dengan dua pengukuran terpisah.

**Bagian 1, tanpa model.** Jalankan asosiasi pada kotak ground truth test set.
Ini menguji algoritmanya sendiri, terlepas dari kualitas deteksi. Kalau di
sini pun banyak atribut jadi tanpa induk, berarti rancangannya yang salah,
bukan modelnya yang kurang.

**Bagian 2, dengan model.** Jalankan asosiasi pada prediksi model, lalu
bandingkan vonisnya dengan vonis yang lahir dari ground truth. Kotak person
prediksi dipasangkan ke kotak person ground truth lewat IoU 0,5. Angka inilah
yang layak dikutip di laporan dan video, karena ia mengukur sistem utuh dari
piksel sampai kesimpulan.

Jalankan:
    .venv-uji/bin/python skrip/validasi_analitik.py /jalur/ke/dataset

Hasilnya ditulis ke `laporan/validasi_analitik.json` supaya laporan, notebook,
dan README mengutip sumber yang sama.
"""

import json
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR))

from src.analitik import (  # noqa: E402
    HELMET,
    KELAS,
    LENGKAP,
    NO_HELMET,
    PERLU_DIPERIKSA,
    PERSON,
    TIDAK_LENGKAP,
    asosiasi,
    ioa,
)

IOU_PASANGAN = 0.5
CONF_MODEL = 0.25
IOU_MODEL = 0.7
IMGSZ = 640
VONIS = (LENGKAP, TIDAK_LENGKAP, PERLU_DIPERIKSA)


def muat_label(jalur: Path) -> list:
    """Satu berkas label YOLO menjadi daftar deteksi ternormalisasi.

    Koordinat dibiarkan dalam skala 0 sampai 1. IoA adalah perbandingan luas,
    dan kedua kotak mengalami peregangan aspek yang sama, jadi nilainya
    identik dengan hasil pada koordinat piksel. Confidence diisi 1,0 karena
    ini anotasi manusia, bukan dugaan.
    """
    keluar = []
    for baris in jalur.read_text(encoding="utf-8").splitlines():
        bagian = baris.split()
        if len(bagian) < 5:
            continue
        indeks = int(bagian[0])
        x, y, w, h = (float(v) for v in bagian[1:5])
        keluar.append(
            {
                "cls": KELAS[indeks],
                "conf": 1.0,
                "box": (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
            }
        )
    return keluar


def iou(a, b) -> float:
    lebar = min(a[2], b[2]) - max(a[0], b[0])
    tinggi = min(a[3], b[3]) - max(a[1], b[1])
    if lebar <= 0 or tinggi <= 0:
        return 0.0
    irisan = lebar * tinggi
    luas_a = (a[2] - a[0]) * (a[3] - a[1])
    luas_b = (b[2] - b[0]) * (b[3] - b[1])
    gabungan = luas_a + luas_b - irisan
    return irisan / gabungan if gabungan > 0 else 0.0


def pasangkan(prediksi, acuan, ambang=IOU_PASANGAN) -> list:
    """Pasangkan kotak pekerja prediksi ke kotak pekerja ground truth.

    Rakus dari IoU tertinggi, tiap kotak dipakai sekali. Mengembalikan daftar
    (indeks_prediksi, indeks_acuan).
    """
    calon = []
    for i, p in enumerate(prediksi):
        for j, a in enumerate(acuan):
            nilai = iou(p.box, a.box)
            if nilai >= ambang:
                calon.append((nilai, i, j))
    calon.sort(reverse=True)

    dipakai_pred, dipakai_acuan, pasangan = set(), set(), []
    for _, i, j in calon:
        if i in dipakai_pred or j in dipakai_acuan:
            continue
        dipakai_pred.add(i)
        dipakai_acuan.add(j)
        pasangan.append((i, j))
    return pasangan


def bagian_satu(berkas_label: list) -> dict:
    """Perilaku algoritma pada ground truth, tanpa melibatkan model."""
    sebaran = dict.fromkeys(VONIS, 0)
    yatim = {}
    total_atribut = cocok_satu = cocok_banyak = 0
    person_tanpa_helm = 0
    total_person = 0

    for jalur in berkas_label:
        deteksi = muat_label(jalur)
        hasil = asosiasi(deteksi)

        for p in hasil.pekerja:
            sebaran[p.vonis] += 1
            if not any(b.cls in (HELMET, NO_HELMET) for b in p.bukti):
                person_tanpa_helm += 1
        total_person += len(hasil.pekerja)

        for a in hasil.tanpa_induk:
            yatim[a.cls] = yatim.get(a.cls, 0) + 1

        # Berapa atribut yang sebenarnya ambigu, yaitu memenuhi ambang IoA
        # pada lebih dari satu kotak person sekaligus.
        kotak_person = [p.box for p in hasil.pekerja]
        for a in (d for d in deteksi if d["cls"] != PERSON):
            total_atribut += 1
            n = sum(1 for kp in kotak_person if ioa(a["box"], kp) >= 0.5)
            if n == 1:
                cocok_satu += 1
            elif n > 1:
                cocok_banyak += 1

    return {
        "gambar": len(berkas_label),
        "person": total_person,
        "atribut": total_atribut,
        "atribut_cocok_tepat_satu_person": cocok_satu,
        "atribut_cocok_lebih_dari_satu": cocok_banyak,
        "atribut_tanpa_induk": yatim,
        "porsi_cocok_tepat_satu": round(cocok_satu / total_atribut, 4) if total_atribut else None,
        "person_tanpa_kotak_helm": person_tanpa_helm,
        "porsi_person_tanpa_kotak_helm": round(person_tanpa_helm / total_person, 4)
        if total_person
        else None,
        "sebaran_vonis": sebaran,
    }


def nilai_vonis(pasangan_gambar) -> dict:
    """Bandingkan vonis prediksi terhadap vonis ground truth, per pekerja.

    Terpisah dari inference dengan sengaja. Sapuan confidence di
    `notebooks/04_evaluasi_final.ipynb` menjalankan model sekali lalu menyaring
    deteksinya di Python untuk belasan ambang, dan penilaian tiap ambang harus
    memakai kode yang sama persis dengan yang dipakai laporan. Kalau notebook
    menyalin logika ini, salinannya akan menyimpang cepat atau lambat dan
    angka di laporan menjadi angka dari dua sistem yang berbeda.

    Args:
        pasangan_gambar: iterable berisi pasangan (hasil prediksi, hasil acuan),
            keduanya HasilGambar keluaran `src.analitik.asosiasi`.
    """
    matriks = {a: dict.fromkeys(VONIS, 0) for a in VONIS}
    benar = total_pasangan = 0
    acuan_tak_terpasangkan = prediksi_tak_terpasangkan = 0
    total_acuan = total_prediksi = 0

    for hasil_prediksi, hasil_acuan in pasangan_gambar:
        total_acuan += len(hasil_acuan.pekerja)
        total_prediksi += len(hasil_prediksi.pekerja)

        pasangan = pasangkan(hasil_prediksi.pekerja, hasil_acuan.pekerja)
        for i, j in pasangan:
            v_pred = hasil_prediksi.pekerja[i].vonis
            v_acuan = hasil_acuan.pekerja[j].vonis
            matriks[v_acuan][v_pred] += 1
            total_pasangan += 1
            if v_pred == v_acuan:
                benar += 1

        acuan_tak_terpasangkan += len(hasil_acuan.pekerja) - len(pasangan)
        prediksi_tak_terpasangkan += len(hasil_prediksi.pekerja) - len(pasangan)

    # Tiga angka turunan yang lebih berguna daripada akurasi tunggal, karena
    # ketiga jenis kesalahan di sini biayanya sangat berbeda.
    gt_patuh = sum(matriks[LENGKAP].values())
    gt_langgar = sum(matriks[TIDAK_LENGKAP].values())

    tuduhan_palsu = matriks[LENGKAP][TIDAK_LENGKAP]
    # Kalau status ketiga dihapus, "tidak terdeteksi" terpaksa dibaca sebagai
    # melanggar. Inilah harga yang dibayar sistem dua status.
    tuduhan_palsu_dua_status = tuduhan_palsu + matriks[LENGKAP][PERLU_DIPERIKSA]
    pembebasan_keliru = matriks[TIDAK_LENGKAP][LENGKAP]

    # Berapa banyak pekerja yang dilempar ke pemeriksaan manusia. Ini ongkos
    # dari kehati-hatian, dan harus ikut terlihat supaya pertukarannya jujur.
    perlu_manusia = sum(matriks[a][PERLU_DIPERIKSA] for a in VONIS)

    return {
        "iou_pasangan": IOU_PASANGAN,
        "person_ground_truth": total_acuan,
        "person_prediksi": total_prediksi,
        "pekerja_terpasangkan": total_pasangan,
        "vonis_benar": benar,
        "akurasi_vonis": round(benar / total_pasangan, 4) if total_pasangan else None,
        # Penjaga. Model yang hampir tidak mendeteksi apa pun bisa terlihat
        # unggul pada laju kesalahan, karena penyebutnya menyusut. Cakupan
        # ini yang mencegah pembacaan itu.
        "cakupan_pekerja": round(total_pasangan / total_acuan, 4) if total_acuan else None,
        "pekerja_terlewat": acuan_tak_terpasangkan,
        "pekerja_palsu": prediksi_tak_terpasangkan,
        "matriks_vonis": matriks,
        "pekerja_patuh_terpasangkan": gt_patuh,
        "pekerja_melanggar_terpasangkan": gt_langgar,
        "tuduhan_palsu": tuduhan_palsu,
        "laju_tuduhan_palsu": round(tuduhan_palsu / gt_patuh, 4) if gt_patuh else None,
        "tuduhan_palsu_kalau_hanya_dua_status": tuduhan_palsu_dua_status,
        "laju_tuduhan_palsu_dua_status": round(tuduhan_palsu_dua_status / gt_patuh, 4)
        if gt_patuh
        else None,
        "pembebasan_keliru": pembebasan_keliru,
        "laju_pembebasan_keliru": round(pembebasan_keliru / gt_langgar, 4)
        if gt_langgar
        else None,
        "perlu_pemeriksaan_manusia": perlu_manusia,
        "porsi_perlu_pemeriksaan_manusia": round(perlu_manusia / total_pasangan, 4)
        if total_pasangan
        else None,
    }


def bagian_dua(
    berkas_label: list,
    nama_model: str,
    imgsz: int = IMGSZ,
    conf: float = CONF_MODEL,
    iou: float = IOU_MODEL,
) -> dict:
    """Vonis dari prediksi model dibandingkan vonis dari ground truth.

    Parameter deteksi bisa ditimpa, karena eksperimen hari 8 sampai 10
    melatih model pada resolusi berbeda dan inference harus memakai resolusi
    yang sama dengan saat training. `nama_model` boleh berupa jalur absolut,
    supaya bobot yang baru selesai dilatih di Colab bisa langsung dinilai
    tanpa harus disalin ke folder models lebih dulu.
    """
    from PIL import Image

    from src.detector import deteksi as jalankan, muat_model

    model = muat_model(nama_model)
    pasangan_gambar = []

    for jalur_label in berkas_label:
        jalur_gambar = None
        for ext in (".jpg", ".jpeg", ".png"):
            calon = jalur_label.parent.parent / "images" / (jalur_label.stem + ext)
            if calon.exists():
                jalur_gambar = calon
                break
        if jalur_gambar is None:
            continue

        with Image.open(jalur_gambar) as img:
            gambar = img.convert("RGB")
            lebar, tinggi = gambar.size
            mentah, _ = jalankan(model, gambar, conf=conf, iou=iou, imgsz=imgsz)

        # Prediksi diubah ke koordinat ternormalisasi supaya sebanding dengan
        # ground truth, yang memang disimpan Roboflow dalam bentuk itu.
        prediksi = [
            {
                "cls": d["cls"],
                "conf": d["conf"],
                "box": (
                    d["box"][0] / lebar,
                    d["box"][1] / tinggi,
                    d["box"][2] / lebar,
                    d["box"][3] / tinggi,
                ),
            }
            for d in mentah
        ]
        pasangan_gambar.append((asosiasi(prediksi), asosiasi(muat_label(jalur_label))))

    return {
        "model": nama_model,
        "conf": conf,
        "iou_nms": iou,
        "imgsz": imgsz,
        **nilai_vonis(pasangan_gambar),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("pemakaian: python skrip/validasi_analitik.py /jalur/ke/dataset [nama_bobot]")
        return 2

    akar_data = Path(sys.argv[1])
    label = sorted((akar_data / "test" / "labels").glob("*.txt"))
    if not label:
        print(f"tidak ada label di {akar_data / 'test' / 'labels'}")
        return 1

    laporan = {"split": "test"}

    print(f"Bagian 1, asosiasi pada ground truth {len(label)} gambar")
    laporan["ground_truth"] = bagian_satu(label)
    for k, v in laporan["ground_truth"].items():
        print(f"  {k:34} {v}")

    print("\nBagian 2, vonis prediksi model dibanding vonis ground truth")
    try:
        from src.detector import daftar_model

        # Bobot pertama dari daftar, yaitu yang juga menjadi pilihan bawaan
        # aplikasi, kecuali kalau namanya diberikan di argumen kedua.
        nama = sys.argv[2] if len(sys.argv) > 2 else (daftar_model() or [""])[0]
        assert nama, "tidak ada bobot di models/"

        # Resolusi mengikuti catatan versi bobot itu, bukan angka tetap, sebab
        # sejak hari 10 ada bobot 640 dan bobot 960 di repo yang sama.
        catatan = AKAR / "laporan" / f"{Path(nama).stem.removeprefix('apd_')}_catatan.json"
        ukuran = IMGSZ
        if catatan.exists():
            ukuran = int(json.loads(catatan.read_text(encoding="utf-8")).get("imgsz", IMGSZ))
        print(f"  bobot {nama}, imgsz {ukuran}")

        laporan["prediksi_model"] = bagian_dua(label, nama, imgsz=ukuran)
        for k, v in laporan["prediksi_model"].items():
            print(f"  {k:34} {v}")
    except ImportError as e:
        laporan["prediksi_model"] = {"dilewati": f"{type(e).__name__}: {e}"}
        print(f"  dilewati, {e}")

    keluaran = AKAR / "laporan" / "validasi_analitik.json"
    keluaran.write_text(json.dumps(laporan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nditulis ke {keluaran.relative_to(AKAR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
