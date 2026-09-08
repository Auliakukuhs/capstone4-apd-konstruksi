"""Sistem Pemeriksaan Kelengkapan APD Pekerja Konstruksi.

Aplikasi ini tidak melaporkan berapa helm yang terlihat. Ia melaporkan
**pekerja mana** yang kelengkapan alat pelindung dirinya belum penuh, dan
menyertakan angka yang melahirkan vonis itu supaya bisa diperiksa ulang.

Seluruh perhitungan ada di `src/analitik.py`, seluruh penggambaran di
`src/tampilan.py`. Berkas ini hanya menyusun urutannya dan mengurus keadaan
antarmuka.

Menjalankan di lokal:
    streamlit run app.py
"""

import io
import json
import sys
from pathlib import Path

import streamlit as st

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))

from src import tampilan  # noqa: E402
from src.analitik import Sesi, asosiasi  # noqa: E402
from src.detector import (  # noqa: E402
    daftar_model,
    deteksi,
    luminansi,
    muat_model,
    setarakan_kontras,
    siapkan_gambar,
)

MODEL_CADANGAN = "yolo11n.pt"
IMGSZ_LATIH = 640

st.set_page_config(
    page_title="Pemeriksaan APD Konstruksi",
    page_icon="🦺",
    layout="wide",
)


@st.cache_resource(show_spinner="Memuat model deteksi")
def model_ter_cache(nama: str):
    """Dimuat sekali per proses, bukan setiap rerun.

    Streamlit menjalankan ulang seluruh skrip setiap kali pengguna menggeser
    slider. Tanpa cache ini, bobot dibaca ulang dari disk setiap gerakan, dan
    di server gratis yang memorinya terbatas aplikasi jadi sangat lambat.
    """
    return muat_model(nama)


@st.cache_data(show_spinner=False, max_entries=64)
def deteksi_ter_cache(isi: bytes, nama_model: str, conf: float, iou: float, imgsz: int):
    """Hasil inference, di-cache berdasarkan isi berkas dan seluruh parameter.

    Yang disimpan hanya daftar dict, bukan objek Results Ultralytics, karena
    yang terakhir berat dan tidak dirancang untuk disimpan lintas rerun.
    Anotasi gambar digambar sendiri oleh `src.tampilan`, jadi Results memang
    tidak dibutuhkan lagi setelah inference selesai.

    Efeknya, menoleh ke tampilan CLAHE atau membuka expander tidak memicu
    inference ulang, sementara mengubah confidence tetap memicunya, karena
    conf ikut menjadi kunci cache.
    """
    gambar = siapkan_gambar(isi)
    model = model_ter_cache(nama_model)
    hasil, _ = deteksi(model, gambar, conf=conf, iou=iou, imgsz=imgsz)
    return hasil


def sebagai_png(gambar) -> bytes:
    """Ubah gambar PIL menjadi bytes PNG.

    Dibutuhkan karena `deteksi_ter_cache` memakai isi berkas sebagai kunci
    cache, sementara gambar hasil CLAHE tidak berasal dari berkas apa pun.
    PNG dipilih karena lossless, jadi masukan model tidak ikut berubah oleh
    kompresi.
    """
    penampung = io.BytesIO()
    gambar.save(penampung, format="PNG")
    return penampung.getvalue()


@st.cache_data(show_spinner=False)
def catatan_model(nama: str) -> dict:
    """Metrik model dari berkas catatan versinya, kalau ada."""
    jalur = DIR / "laporan" / f"{Path(nama).stem.removeprefix('apd_')}_catatan.json"
    if not jalur.exists():
        return {}
    return json.loads(jalur.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- kepala

st.title("Pemeriksaan Kelengkapan APD Pekerja Konstruksi")
st.caption(
    "Unggah satu atau beberapa foto lokasi kerja. Sistem mengenali setiap "
    "pekerja, mengaitkan helm dan rompi yang terdeteksi kepada pekerja yang "
    "memakainya, lalu menyatakan siapa yang kelengkapannya belum penuh."
)

tersedia = daftar_model()
if not tersedia:
    st.warning(
        f"Belum ada bobot hasil training di folder `models/`. Aplikasi memakai "
        f"model bawaan `{MODEL_CADANGAN}` yang dilatih di COCO, jadi ia "
        f"mengenali orang tapi belum mengenali helm dan rompi."
    )

# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Model")
    nama_model = st.selectbox("Bobot", tersedia or [MODEL_CADANGAN], index=0)

    st.subheader("Parameter deteksi")
    conf = st.slider(
        "Confidence threshold",
        0.05,
        0.90,
        0.25,
        0.05,
        help=(
            "Turunkan berarti lebih banyak objek tertangkap dan lebih banyak "
            "deteksi palsu. Untuk keselamatan, melewatkan pelanggaran lebih "
            "mahal daripada alarm palsu, jadi nilai rendah lebih dapat "
            "dipertahankan daripada nilai tinggi. Ini keputusan, bukan angka "
            "bawaan yang harus diterima."
        ),
    )
    iou = st.slider(
        "IoU untuk NMS",
        0.10,
        0.90,
        0.70,
        0.05,
        help="Turunkan kalau satu objek terdeteksi beberapa kali bertumpuk.",
    )
    imgsz = st.select_slider(
        "Ukuran masukan model",
        options=[640, 960],
        value=IMGSZ_LATIH,
        help=(
            "Samakan dengan nilai saat training. Nilai berbeda menggeser skala "
            "objek terhadap apa yang dipelajari model."
        ),
    )
    if imgsz != IMGSZ_LATIH:
        st.caption(
            f"Model ini dilatih pada {IMGSZ_LATIH}. Menaikkannya bisa menolong "
            f"objek kecil, tapi hasilnya belum tentu lebih baik karena skalanya "
            f"tidak lagi sama dengan saat training."
        )

    st.subheader("Tampilan")
    tampilkan_atribut = st.checkbox(
        "Gambar kotak helm dan rompi",
        value=True,
        help="Matikan kalau gambarnya padat dan kotak pekerja jadi sulit dilihat.",
    )
    banding_clahe = st.checkbox(
        "Bandingkan dengan CLAHE",
        value=False,
        help=(
            "Menjalankan deteksi kedua pada versi gambar yang kontrasnya "
            "disetarakan. Model dilatih TANPA ini, jadi hasil yang lebih "
            "banyak belum tentu lebih benar."
        ),
    )

# ----------------------------------------------------------------- masukan

berkas = st.file_uploader(
    "Unggah gambar",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not berkas:
    st.info(
        "Unggah sebuah gambar untuk memulai. Bisa beberapa sekaligus, dan "
        "totalnya akan muncul di panel kiri."
    )
    tampilan.panel_model(catatan_model(nama_model))
    st.stop()

# Sesi dibangun ulang dari seluruh berkas yang sedang terunggah, bukan
# ditambahkan sedikit demi sedikit ke st.session_state.
#
# Ini disengaja. Streamlit menjalankan ulang skrip setiap kali slider digeser,
# jadi riwayat yang ditumpuk akan menghitung gambar yang sama berkali-kali.
# Lebih buruk lagi, riwayat yang bertahan akan mencampur vonis dari confidence
# threshold yang berbeda ke dalam satu angka kepatuhan, dan angka itu tidak
# berarti apa-apa. Menghitung ulang pada ambang yang sedang aktif adalah
# perilaku yang benar, dan sifatnya idempoten.
sesi = Sesi()

# ----------------------------------------------------------------- per gambar

for i, satu in enumerate(berkas):
    isi = satu.getvalue()
    gambar = siapkan_gambar(isi)
    deteksi_gambar = deteksi_ter_cache(isi, nama_model, conf, iou, imgsz)
    hasil = asosiasi(deteksi_gambar)
    sesi.tambah(satu.name, hasil)

    if len(berkas) > 1:
        st.divider()
        st.subheader(f"{i + 1}. {satu.name}")

    catatan_cahaya = tampilan.peringatan_pencahayaan(luminansi(gambar))
    if catatan_cahaya:
        st.warning(catatan_cahaya)

    tampilan.sandingkan(gambar, tampilan.gambar_vonis(gambar, hasil, tampilkan_atribut))
    tampilan.legenda()

    tampilan.banner(hasil.ringkasan)
    tampilan.panel_ringkasan(hasil)

    if hasil.pekerja:
        st.markdown("**Rincian per pekerja**")
        tampilan.tabel_pekerja(hasil)

    tampilan.panel_bukti(hasil, deteksi_gambar, conf, iou)

    if banding_clahe:
        with st.expander("Perbandingan dengan CLAHE", expanded=True):
            disetarakan = setarakan_kontras(gambar)
            deteksi_clahe = deteksi_ter_cache(
                sebagai_png(disetarakan), nama_model, conf, iou, imgsz
            )
            hasil_clahe = asosiasi(deteksi_clahe)

            tampilan.sandingkan(
                tampilan.gambar_vonis(gambar, hasil, tampilkan_atribut),
                tampilan.gambar_vonis(disetarakan, hasil_clahe, tampilkan_atribut),
            )
            st.caption("Kiri gambar asli, kanan setelah kontras disetarakan.")

            st.dataframe(
                [
                    {
                        "ukuran": nama,
                        "gambar asli": asli,
                        "setelah CLAHE": sesudah,
                        "selisih": sesudah - asli,
                    }
                    for nama, asli, sesudah in (
                        ("objek terdeteksi", len(deteksi_gambar), len(deteksi_clahe)),
                        ("pekerja", hasil.ringkasan.pekerja, hasil_clahe.ringkasan.pekerja),
                        (
                            "tidak lengkap",
                            hasil.ringkasan.tidak_lengkap,
                            hasil_clahe.ringkasan.tidak_lengkap,
                        ),
                        (
                            "belum dapat dipastikan",
                            hasil.ringkasan.belum_pasti,
                            hasil_clahe.ringkasan.belum_pasti,
                        ),
                    )
                ],
                width="stretch",
                hide_index=True,
            )
            st.info(
                "Model ini dilatih tanpa CLAHE. Deteksi yang bertambah berarti "
                "penyetaraan kontras memunculkan objek yang tadinya tenggelam, "
                "tapi bisa juga berarti ia memunculkan tekstur yang keliru "
                "dikenali. Perbandingan ini disediakan untuk dilihat, bukan "
                "untuk dijadikan dasar vonis. Angka resmi tetap yang dari "
                "gambar asli."
            )

# ----------------------------------------------------------------- total sesi

with st.sidebar:
    st.divider()
    tampilan.dashboard_sesi(sesi)
    tampilan.unduh_csv(
        sesi.semua_baris(),
        "kepatuhan_apd.csv",
        "Unduh CSV per pekerja",
    )

if len(berkas) > 1:
    st.divider()
    st.subheader("Total seluruh gambar yang diunggah")
    total = sesi.total()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Gambar", len(sesi))
    k2.metric("Pekerja", total.pekerja)
    k3.metric("Tidak lengkap", total.tidak_lengkap)
    k4.metric("Belum dapat dipastikan", total.belum_pasti)

tampilan.panel_model(catatan_model(nama_model))
