"""Sistem Pemeriksaan Kelengkapan APD Pekerja Konstruksi.

Hari 3 dari 15. Versi ini SENGAJA paling sederhana. Unggah gambar, deteksi,
tampilkan hasil beranotasi dan hitungan per kelas. Tidak ada lagi.

Tujuannya membuktikan jalur dari kode sampai link publik tembus hari ini,
saat masih ada dua belas hari untuk memperbaiki. Lapisan analisis yang menjadi
inti penilaian dikerjakan hari 4 sampai 7.

Menjalankan di lokal:
    streamlit run app.py
"""

import collections
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detector import (  # noqa: E402
    daftar_model,
    deteksi,
    muat_model,
    siapkan_gambar,
)

MODEL_CADANGAN = "yolo11n.pt"

st.set_page_config(page_title="Pemeriksaan APD Konstruksi", layout="wide")


@st.cache_resource(show_spinner="Memuat model deteksi")
def model_ter_cache(nama: str):
    """Dimuat sekali per proses, bukan setiap rerun.

    Streamlit menjalankan ulang seluruh skrip setiap kali pengguna menggeser
    slider. Tanpa cache ini, bobot dibaca ulang dari disk setiap gerakan, dan
    di server gratis yang memorinya terbatas aplikasi jadi sangat lambat.
    """
    return muat_model(nama)


st.title("Pemeriksaan Kelengkapan APD Pekerja Konstruksi")
st.caption(
    "Unggah foto lokasi kerja, sistem mendeteksi pekerja beserta helm dan rompinya."
)

tersedia = daftar_model()
if not tersedia:
    st.warning(
        f"Belum ada bobot hasil training di folder `models/`. Aplikasi memakai "
        f"model bawaan `{MODEL_CADANGAN}` yang dilatih di COCO, jadi ia mengenali "
        f"orang tapi belum mengenali helm dan rompi. Ini kondisi sementara hari 3, "
        f"dipakai untuk membuktikan jalur deploy sudah tembus."
    )

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
            "deteksi palsu. Naikkan berarti sebaliknya. Ini keputusan, bukan "
            "angka bawaan yang harus diterima."
        ),
    )
    iou = st.slider(
        "IoU untuk NMS",
        0.10,
        0.90,
        0.70,
        0.05,
        help="Turunkan kalau satu objek terdeteksi beberapa kali.",
    )
    imgsz = st.select_slider(
        "Ukuran masukan model",
        options=[640, 960],
        value=640,
        help="Samakan dengan nilai yang dipakai saat training.",
    )

berkas = st.file_uploader(
    "Unggah gambar", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
)

if berkas is None:
    st.info("Unggah sebuah gambar untuk memulai.")
    st.stop()

gambar = siapkan_gambar(berkas)
model = model_ter_cache(nama_model)
hasil, raw = deteksi(model, gambar, conf=conf, iou=iou, imgsz=imgsz)

kiri, kanan = st.columns(2)
with kiri:
    st.image(gambar, caption="Gambar asli", use_container_width=True)
with kanan:
    # .plot() mengembalikan array BGR, dibalik supaya warnanya benar di Streamlit.
    st.image(
        raw.plot()[:, :, ::-1],
        caption="Hasil deteksi",
        use_container_width=True,
    )

if not hasil:
    st.warning(
        f"Tidak ada objek terdeteksi pada confidence {conf:.2f}. "
        f"Coba turunkan nilainya di panel kiri."
    )
    st.stop()

hitungan = collections.Counter(d["cls"] for d in hasil)

a, b = st.columns(2)
a.metric("Total objek", len(hasil))
b.metric("Jenis kelas terdeteksi", len(hitungan))

st.subheader("Hitungan per kelas")
st.caption(f"Pada confidence threshold {conf:.2f} dan IoU {iou:.2f}.")
st.dataframe(
    [{"kelas": k, "jumlah": v} for k, v in hitungan.most_common()],
    use_container_width=True,
    hide_index=True,
)

with st.expander("Rincian tiap deteksi"):
    st.dataframe(
        [
            {
                "kelas": d["cls"],
                "confidence": d["conf"],
                "x1": d["box"][0],
                "y1": d["box"][1],
                "x2": d["box"][2],
                "y2": d["box"][3],
            }
            for d in sorted(hasil, key=lambda x: -x["conf"])
        ],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Versi hari 3, sengaja minimal. Lapisan analisis yang menghubungkan helm dan "
    "rompi ke pekerja tertentu, beserta vonis kelengkapan APD, dikerjakan hari 4 "
    "sampai 7."
)
