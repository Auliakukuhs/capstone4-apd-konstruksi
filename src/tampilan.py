"""Komponen antarmuka. Menggambar dan merender, tidak menghitung apa pun.

Seluruh keputusan analisis sudah selesai di `src.analitik`. Modul ini hanya
menerjemahkannya menjadi gambar, banner, tabel, dan metrik.

Pemisahan itu disengaja. Fungsi yang memutuskan **apa** yang ditampilkan
dipisahkan dari fungsi yang **menampilkannya**, sehingga keputusannya bisa
diuji tanpa menjalankan Streamlit. `pesan_banner` mengembalikan tingkat dan
kalimat, `banner` yang memanggil `st.error`. Yang pertama diuji, yang kedua
cuma satu baris pemanggilan.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.analitik import (
    LENGKAP,
    PERLU_DIPERIKSA,
    TIDAK_LENGKAP,
    KELAS,
)

WARNA_VONIS = {
    LENGKAP: (34, 160, 70),
    TIDAK_LENGKAP: (205, 45, 45),
    PERLU_DIPERIKSA: (215, 150, 20),
}
# Vonis dalam bentuk pendek, dipakai kalau kotak pekerjanya terlalu sempit
# untuk memuat kalimat penuh.
SINGKATAN = {
    LENGKAP: "OK",
    TIDAK_LENGKAP: "TIDAK",
    PERLU_DIPERIKSA: "PERIKSA",
}

WARNA_ATRIBUT = (70, 130, 200)
WARNA_TANPA_INDUK = (150, 60, 190)
PUTIH = (255, 255, 255)

# Ambang pencahayaan. Diukur pada dataset, 7,6 persen gambar berada di luar
# rentang ini, dan model tidak pernah melihat banyak contoh seperti itu.
LUMA_MIN = 60
LUMA_MAKS = 200


def _font(ukuran: int):
    """Font yang pasti ada di mana pun aplikasi berjalan.

    `load_default(size=...)` tersedia sejak Pillow 10.1 dan mengembalikan font
    vektor yang bisa diskalakan, jadi tidak perlu menitipkan berkas ttf ke
    dalam repo maupun bergantung pada font sistem yang belum tentu ada di
    server hosting.
    """
    try:
        return ImageFont.load_default(size=ukuran)
    except TypeError:
        return ImageFont.load_default()


def _ukuran_pena(gambar):
    """Ketebalan garis dan besar huruf yang ikut ukuran gambar.

    Garis setebal 2 piksel yang pas di gambar 640 piksel akan nyaris tak
    terlihat di gambar 1600 piksel, dan sebaliknya huruf berukuran tetap akan
    menutupi wajah orang di gambar kecil.
    """
    sisi = min(gambar.size)
    return max(2, round(sisi * 0.005)), max(13, round(sisi * 0.022))


def _lebar_teks(pena, teks, font):
    kiri, atas, kanan, bawah = pena.textbbox((0, 0), teks, font=font)
    return kanan - kiri, bawah - atas, atas


def _label(pena, kotak, pilihan, warna, font):
    """Tulis label pertama dari `pilihan` yang muat di lebar kotak.

    Ini yang membuat gambar padat tetap terbaca. Pada foto berisi sembilan
    pekerja berdiri berdampingan, tulisan "TIDAK LENGKAP" milik tiap orang
    saling menimpa sampai tidak ada satu pun yang bisa dibaca. Nomor pekerja
    saja tetap terbaca, dan vonisnya sudah dibawa oleh warna kotak serta
    tabel di bawah gambar.

    `pilihan` diurutkan dari yang paling informatif ke yang paling pendek.
    Kalau tidak ada yang muat, tidak ada teks yang digambar, dan itu lebih
    baik daripada teks bertumpuk.
    """
    x1, y1, x2, y2 = kotak
    ruang = max(0, x2 - x1)

    terakhir = len(pilihan) - 1
    for i, teks in enumerate(pilihan):
        lebar, tinggi, atas = _lebar_teks(pena, teks, font)
        if lebar + 6 > ruang:
            if i < terakhir:
                continue
            # Bahkan bentuk terpendek tidak muat. Kotaknya terlalu sempit,
            # dan teks yang melimpah keluar kotak lebih mengganggu daripada
            # tidak ada teks sama sekali.
            return False

        # Kalau tidak muat di atas kotak, taruh di dalamnya, jangan sampai
        # terpotong keluar bidang gambar seperti yang terjadi pada pekerja
        # yang tergunting tepi atas foto.
        y = y1 - tinggi - 4 if y1 - tinggi - 4 >= 0 else y1 + 2
        pena.rectangle((x1, y, x1 + lebar + 6, y + tinggi + 4), fill=warna)
        pena.text((x1 + 3, y + 2 - atas), teks, fill=PUTIH, font=font)
        return True

    return False


def gambar_vonis(gambar, hasil, tampilkan_atribut: bool = True):
    """Salinan gambar dengan kotak berwarna mengikuti vonis tiap pekerja.

    Warna kotak yang mengikuti vonis adalah versi tingkat objek dari pola
    banner berwarna di materi. Pengawas bisa melihat siapa yang bermasalah
    tanpa membaca tabel apa pun.

    Gambar aslinya tidak diubah, yang dikembalikan salinan.
    """
    kanvas = gambar.copy().convert("RGB")
    pena = ImageDraw.Draw(kanvas)
    tebal, besar_huruf = _ukuran_pena(kanvas)
    font = _font(besar_huruf)
    font_kecil = _font(max(11, round(besar_huruf * 0.8)))

    if tampilkan_atribut:
        # Atribut digambar lebih dulu dan lebih tipis, supaya kotak pekerja
        # yang membawa vonis tetap yang paling menonjol.
        for p in hasil.pekerja:
            for b in p.bukti:
                pena.rectangle(b.box, outline=WARNA_ATRIBUT, width=max(1, tebal // 2))
                _label(
                    pena,
                    b.box,
                    [f"{b.cls} {b.conf:.2f}", b.cls],
                    WARNA_ATRIBUT,
                    font_kecil,
                )

    for a in hasil.tanpa_induk:
        pena.rectangle(a.box, outline=WARNA_TANPA_INDUK, width=max(1, tebal // 2))
        _label(
            pena,
            a.box,
            [f"{a.cls} tanpa pekerja", f"{a.cls} yatim", a.cls],
            WARNA_TANPA_INDUK,
            font_kecil,
        )

    for p in hasil.pekerja:
        warna = WARNA_VONIS[p.vonis]
        pena.rectangle(p.box, outline=warna, width=tebal)
        _label(pena, p.box, [f"#{p.nomor} {p.vonis}", f"#{p.nomor} {SINGKATAN[p.vonis]}",
                             f"#{p.nomor}"], warna, font)

    return kanvas


def pesan_banner(ringkasan):
    """Tentukan tingkat dan kalimat banner. Tanpa menyentuh Streamlit.

    Urutan pemeriksaannya penting. Pelanggaran lebih dulu, lalu ketidakpastian,
    baru aman. Pernyataan aman hanya boleh muncul kalau dua kondisi sebelumnya
    tidak ada satu pun, karena "aman" adalah klaim yang paling mahal kalau
    salah.
    """
    if ringkasan.pekerja == 0:
        return "info", (
            "Tidak ada pekerja terdeteksi pada gambar ini. Turunkan confidence "
            "threshold di panel kiri, atau periksa apakah gambarnya memang "
            "memuat orang."
        )

    if ringkasan.tidak_lengkap > 0:
        kalimat = (
            f"{ringkasan.tidak_lengkap} dari {ringkasan.pekerja} pekerja tidak "
            f"memakai APD lengkap. Hentikan pekerjaan di area itu dan lengkapi "
            f"alat pelindung diri sebelum melanjutkan."
        )
        if ringkasan.belum_pasti > 0:
            kalimat += (
                f" Selain itu {ringkasan.belum_pasti} pekerja belum dapat "
                f"dipastikan dan perlu diperiksa langsung."
            )
        return "bahaya", kalimat

    if ringkasan.belum_pasti > 0:
        return "hati-hati", (
            f"{ringkasan.belum_pasti} dari {ringkasan.pekerja} pekerja belum "
            f"dapat dipastikan kelengkapan APD-nya, karena atributnya tidak "
            f"terdeteksi. Periksa langsung, jangan dianggap sudah aman."
        )

    return "aman", (
        f"Seluruh {ringkasan.pekerja} pekerja terdeteksi memakai helm dan rompi. "
        f"Tidak ada tindakan yang diperlukan."
    )


def banner(ringkasan) -> None:
    tingkat, kalimat = pesan_banner(ringkasan)
    {"bahaya": st.error, "hati-hati": st.warning, "aman": st.success, "info": st.info}[
        tingkat
    ](kalimat)


def _persen(nilai):
    return "belum ada" if nilai is None else f"{nilai * 100:.1f} persen"


def panel_ringkasan(hasil) -> None:
    """Panel Summary. Empat angka yang menjawab pertanyaan pertama pengawas."""
    r = hasil.ringkasan
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pekerja terdeteksi", r.pekerja)
    k2.metric("APD lengkap", r.lengkap)
    k3.metric(
        "Tidak lengkap",
        r.tidak_lengkap,
        delta=None if r.tidak_lengkap == 0 else "perlu tindakan",
        delta_color="inverse",
    )
    k4.metric("Belum dapat dipastikan", r.belum_pasti)

    b1, b2 = st.columns(2)
    b1.metric(
        "Kepatuhan",
        _persen(r.kepatuhan),
        help=(
            "Lengkap dibagi seluruh pekerja. Angka pesimistis, karena pekerja "
            "yang belum dapat dipastikan ikut menjadi penyebut."
        ),
    )
    b2.metric(
        "Kepatuhan yang terbaca",
        _persen(r.kepatuhan_terperiksa),
        help=(
            "Lengkap dibagi pekerja yang statusnya benar-benar terbaca. Lebih "
            "adil dipakai membandingkan antar gambar, karena tidak menghukum "
            "gambar yang banyak pekerjanya terhalang."
        ),
    )

    if hasil.tanpa_induk:
        rincian = ", ".join(f"{n} {k}" for k, n in sorted(r.tanpa_induk.items()))
        st.caption(
            f"Ada {rincian} yang tidak dapat dikaitkan ke satu pekerja pun. "
            f"Kemungkinan besar ada pekerja yang tidak terdeteksi di gambar ini."
        )


def legenda() -> None:
    st.caption(
        "Warna kotak mengikuti vonis. "
        "Hijau lengkap, merah tidak lengkap, kuning belum dapat dipastikan. "
        "Biru atribut yang berhasil dikaitkan, ungu atribut tanpa pekerja."
    )


def tabel_pekerja(hasil) -> None:
    """Rincian per pekerja beserta bukti angkanya.

    Kolom bukti sengaja ikut. Vonis tanpa angka pembentuknya tidak bisa
    diperiksa ulang oleh siapa pun.
    """
    if not hasil.pekerja:
        return
    baris = []
    for p in hasil.pekerja:
        baris.append(
            {
                "pekerja": f"#{p.nomor}",
                "vonis": p.vonis,
                "helm": p.helmet,
                "rompi": p.vest,
                "alasan": p.alasan,
                "conf pekerja": round(p.conf, 3),
                "bukti": "; ".join(f"{b.cls} {b.conf:.2f} (ioa {b.ioa:.2f})" for b in p.bukti)
                or "tidak ada atribut terdeteksi",
                "konflik": "; ".join(p.konflik),
            }
        )
    st.dataframe(baris, width="stretch", hide_index=True)


def panel_bukti(hasil, deteksi_mentah, conf: float, iou: float) -> None:
    """Angka mentah, disembunyikan tapi tersedia.

    Ditaruh di dalam expander supaya tidak mengganggu pembacaan cepat, tapi
    tetap ada karena pertanyaan pertama saat vonis diragukan adalah
    "dari mana angkanya".
    """
    with st.expander("Angka mentah di balik vonis"):
        st.caption(f"Pada confidence {conf:.2f} dan IoU NMS {iou:.2f}.")

        st.markdown("**Hitungan per kelas**, termasuk yang nol")
        st.dataframe(
            [{"kelas": k, "jumlah": hasil.hitung_kelas.get(k, 0)} for k in KELAS],
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Seluruh deteksi**")
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
                for d in sorted(deteksi_mentah, key=lambda x: -x["conf"])
            ],
            width="stretch",
            hide_index=True,
        )


def dashboard_sesi(sesi, kunci_reset: str = "reset_sesi") -> None:
    """Total lintas gambar dalam satu sesi pemakaian.

    Perhitungannya ada di `src.analitik.Sesi`, bukan di sini, supaya ikut
    teruji dan tidak bergantung pada Streamlit.
    """
    st.subheader("Total sesi")
    if len(sesi) == 0:
        st.caption("Belum ada gambar yang diperiksa.")
        return

    t = sesi.total()
    st.metric("Gambar diperiksa", len(sesi))
    st.metric("Pekerja terdeteksi", t.pekerja)
    st.metric("Tidak lengkap", t.tidak_lengkap)
    st.metric("Belum dapat dipastikan", t.belum_pasti)
    st.metric("Kepatuhan", _persen(t.kepatuhan))
    st.metric("Kepatuhan yang terbaca", _persen(t.kepatuhan_terperiksa))

    if t.tanpa_induk:
        st.caption(
            "Atribut tanpa pekerja, "
            + ", ".join(f"{n} {k}" for k, n in sorted(t.tanpa_induk.items()))
        )


def panel_model(catatan: dict) -> None:
    """Metrik model, ditampilkan terbuka bukan disembunyikan.

    Aplikasi yang menyatakan vonis tentang orang wajib menyatakan juga
    seberapa bisa ia dipercaya. Angka terburuknya justru yang paling perlu
    terlihat.
    """
    if not catatan:
        return

    with st.expander("Tentang model yang dipakai"):
        k1, k2, k3 = st.columns(3)
        k1.metric("mAP@0.5", f"{catatan.get('map50', 0):.3f}")
        k2.metric("mAP@0.5:0.95", f"{catatan.get('map50_95', 0):.3f}")
        k3.metric("Dievaluasi di", catatan.get("split_dilaporkan", "-"))

        st.caption(
            f"Run `{catatan.get('run')}`, dari `{catatan.get('model_awal')}`, "
            f"imgsz {catatan.get('imgsz')}, berhenti di epoch "
            f"{catatan.get('epochs_berjalan')} dengan hasil terbaik epoch "
            f"{catatan.get('epoch_terbaik')}. Metrik dihitung pada "
            f"conf {catatan.get('conf_evaluasi')}, bukan pada 0,25 bawaan, "
            f"supaya ekor kurva precision recall tidak terpotong."
        )

        per_kelas = catatan.get("per_kelas", {})
        if per_kelas:
            st.dataframe(
                [
                    {
                        "kelas": k,
                        "P": round(v.get("P", 0), 3),
                        "R": round(v.get("R", 0), 3),
                        "mAP50": round(v.get("mAP50", 0), 3),
                        "mAP50-95": round(v.get("mAP50_95", 0), 3),
                    }
                    for k, v in per_kelas.items()
                ],
                width="stretch",
                hide_index=True,
            )

        st.warning(
            "Keterbatasan yang paling perlu diketahui. Kelas `no-helmet` hanya "
            "punya 94 contoh latih dan recall-nya 0,333 di test set, jadi dari "
            "tiga pelanggaran helm model rata-rata hanya menemukan satu. Itulah "
            "sebabnya tidak adanya deteksi tidak pernah dijadikan bukti bahwa "
            "pekerjanya patuh."
        )


def peringatan_pencahayaan(luma: float) -> str:
    """Kalimat peringatan kalau pencahayaan di luar rentang yang dikenal model.

    Mengembalikan string kosong kalau tidak ada masalah, supaya pemanggilnya
    tidak perlu menangani None.
    """
    if luma < LUMA_MIN:
        return (
            f"Gambar ini gelap, rata-rata luminansinya {luma:.0f} dari 255. "
            f"Pada dataset latih, 7,6 persen gambar berada di luar rentang 60 "
            f"sampai 200, jadi model jarang melihat kondisi seperti ini dan "
            f"deteksinya cenderung lebih sedikit daripada seharusnya."
        )
    if luma > LUMA_MAKS:
        return (
            f"Gambar ini sangat terang, rata-rata luminansinya {luma:.0f} dari "
            f"255. Detail helm bisa hilang tertelan cahaya. Hasil deteksi pada "
            f"gambar seperti ini perlu dipandang lebih hati-hati."
        )
    return ""


def sebagai_csv(baris: list) -> str:
    """Susun CSV tanpa pandas.

    pandas memang ikut terpasang lewat streamlit, tapi tidak dipakai langsung
    oleh kode ini dan tidak dicantumkan di requirements, jadi menulis sendiri
    membuat daftar dependensi tetap sekecil mungkin.

    Kolom `bukti` dan `alasan` mengandung koma dan titik koma, jadi pengutipan
    tidak bisa dilewati. Tanpa itu, berkasnya terbuka dengan kolom bergeser di
    Excel dan tidak ada yang menyadarinya sampai angkanya salah dibaca.
    """
    if not baris:
        return ""

    kolom = list(baris[0].keys())

    def bersih(nilai):
        teks = str(nilai)
        if any(c in teks for c in ',"\n\r'):
            return '"' + teks.replace('"', '""') + '"'
        return teks

    isi = [",".join(bersih(k) for k in kolom)]
    isi.extend(",".join(bersih(b.get(k, "")) for k in kolom) for b in baris)
    return "\n".join(isi)


def unduh_csv(baris: list, nama_berkas: str, label: str) -> None:
    if not baris:
        return
    st.download_button(
        label,
        data=sebagai_csv(baris).encode("utf-8"),
        file_name=nama_berkas,
        mime="text/csv",
        width="stretch",
    )


def sandingkan(asli: Image.Image, beranotasi: Image.Image) -> None:
    """Dua kolom, kiri asli dan kanan hasil. Elemen pertama dari materi."""
    kiri, kanan = st.columns(2)
    with kiri:
        st.image(asli, caption="Gambar asli", width="stretch")
    with kanan:
        st.image(beranotasi, caption="Vonis per pekerja", width="stretch")
