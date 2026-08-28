# Sistem Pemeriksaan Kelengkapan APD Pekerja Konstruksi

Capstone Project Module 4, Purwadhika Digital Technology School.
Object detection untuk memeriksa kelengkapan alat pelindung diri di lokasi konstruksi.

> Status pengerjaan: **hari 2 dari 15**. Bagian yang ditandai `[belum]` diisi
> sesuai urutan di `../catatan/URUTAN-KERJA.md`.

## 1. Masalah yang diselesaikan

Di lokasi konstruksi, pengawas harus memastikan setiap pekerja memakai helm dan
rompi. Memeriksanya satu per satu dari foto atau rekaman CCTV memakan waktu dan
mudah luput, terutama saat pekerjanya banyak dan sebagian saling terhalang.

Sistem ini menerima satu gambar, mengenali setiap pekerja, lalu melaporkan
**siapa** yang kelengkapan alat pelindung dirinya belum penuh, bukan sekadar
berapa helm yang terlihat.

Penggunanya pengawas lapangan atau petugas K3, yang butuh jawaban cepat dan
bukti yang bisa diperiksa, bukan angka mentah.

## 2. Dataset

`construction safety.v1i.yolov12`, sumber Roboflow Universe, lisensi CC BY 4.0,
bagian dari benchmark RF100.

| Hal | Nilai |
|---|---|
| Gambar | 1.206 |
| Split train / valid / test | 997 / 119 / 90 |
| Bounding box | 7.724 |
| Kelas | `helmet, no-helmet, no-vest, person, vest` |
| Rata-rata objek per gambar | 6,41 |

**Perhatikan urutan kelas.** `person` ada di indeks 3, bukan 0.

Distribusi kelas dan temuan lain ada di bagian 3. Pengukuran lengkap beserta
script-nya di `../catatan/98-eda-dataset.md`.

## 3. Pipeline data processing

Notebook `notebooks/01_eda_dataset.ipynb` menjalankan lima pemeriksaan sebelum
satu baris pun kode training ditulis. Angka ringkasnya tersimpan di
`laporan/eda_ringkasan.json` supaya laporan dan notebook selalu mengutip sumber
yang sama.

| # | Pemeriksaan | Hasil |
|---|---|---|
| 1 | Integritas label | **bersih**, 7.724 baris, nol bermasalah, nol berkas rusak |
| 2 | Kebersihan split | **bersih**, nol id lintas split, nol berkas md5 identik |
| 3 | Distribusi kelas | timpang berat, `no-helmet` hanya 1,7 persen |
| 4 | Ukuran objek | 68,7 persen `helmet` di bawah 2 persen luas gambar |
| 5 | Kualitas gambar | 7,6 persen menyimpang pencahayaannya, tidak ada berkas rusak |

### Distribusi kelas

| Kelas | train | valid | test | total | porsi |
|---|---|---|---|---|---|
| person | 2.362 | 241 | 214 | 2.817 | 36,5% |
| helmet | 2.116 | 232 | 195 | 2.543 | 32,9% |
| vest | 1.073 | 141 | 129 | 1.343 | 17,4% |
| no-vest | 741 | 90 | 61 | 892 | 11,5% |
| **no-helmet** | 94 | **11** | 24 | 129 | **1,7%** |

### Ukuran objek dan pemilihan resolusi

Materi menyatakan object detection paling andal untuk objek yang menutupi 2
sampai 60 persen luas gambar. Porsi objek yang berada di bawah 32 piksel setelah
letterbox, yaitu definisi objek small pada metrik COCO.

| Kelas | imgsz 640 | imgsz 960 |
|---|---|---|
| no-helmet | 47,9% | 17,0% |
| helmet | 32,6% | 14,1% |
| no-vest | 7,7% | 4,9% |
| vest | 5,8% | 1,2% |
| person | 5,1% | 1,3% |

Karena itu `imgsz` diperlakukan sebagai variabel eksperimen, bukan angka yang
ditetapkan di muka. Baseline 640, pembanding 960.

### Keputusan preprocessing

**Tidak memakai CLAHE sebagai preprocessing tetap**, meski 7,6 persen gambar
menyimpang pencahayaannya. Dua alasan. Kondisinya bervariasi, bukan seragam. Dan
preprocessing tetap harus diterapkan juga saat inference, yang menambah satu
langkah yang bisa lupa dilakukan.

Sebagai gantinya augmentasi `hsv_v` dipertahankan, supaya model terbiasa dengan
variasi pencahayaan tanpa ada yang perlu diingat saat inference.

**Tidak menormalkan piksel secara manual.** Ultralytics sudah melakukannya di
dalam, dan menormalkan dua kali merusak input.

## 4. Model

`[belum]` Hari 3 baseline, hari 8 sampai 10 eksperimen.

### Keputusan yang sudah diambil

**Augmentasi geometris dimatikan seluruhnya.** Dokumen SOAL meminta augmentasi
yang tidak mengubah geometris gambar. Slide `Object Detection Basics` halaman 50
justru mengajarkan `flipud=0.5`. SOAL yang dipakai, karena SOAL adalah dokumen
penilaian.

Yang penting disadari, Ultralytics menyalakan **empat** augmentasi geometris
secara default. `fliplr=0.5`, `mosaic=1.0`, `scale=0.5`, `translate=0.1`.
Menghapus satu baris `flipud` tidak cukup, keempatnya harus dimatikan eksplisit.

Trade-off-nya dinyatakan terbuka. Mematikan mosaic biasanya menurunkan mAP pada
dataset kecil. Penurunan itu diterima demi kepatuhan pada instruksi.

## 5. Hasil

`[belum]` Hari 11. mAP@0.5 dan mAP@0.5:0.95 di test set, tabel per kelas,
confusion matrix, kurva PR, dan speed metrics.

## 6. Lapisan analisis

`[belum]` Hari 4 dan 5.

Rancangannya asosiasi atribut ke pekerja lewat IoA, bukan aritmetika hitungan.
Alasan kuantitatifnya ada di bagian 7.

## 7. Keterbatasan yang diketahui

Dua hal sudah diketahui sejak sebelum model dilatih, keduanya hasil pengukuran
langsung terhadap file label.

**Kelas `no-helmet` sangat langka.** Hanya 129 kotak dari 7.724, atau 1,7 persen,
dan validation split hanya punya **11 instance**. Satu deteksi menggeser AP kelas
itu sekitar sembilan persen, jadi angka AP-nya di validation tidak bisa dipercaya.
Padahal itu justru kelas yang paling penting bagi keselamatan. Recall-nya
dilaporkan terpisah di test set.

**Objeknya kecil.** 68,7 persen helmet dan 88,4 persen no-helmet menutupi kurang
dari 2 persen luas gambar, di bawah rentang 2 sampai 60 persen yang disebut
materi sebagai wilayah andal object detection.

**Gambarnya padat.** Rata-rata 6,41 objek per gambar dengan maksimum 39, dan
pekerja sering berhimpitan. Ini akan menyulitkan asosiasi atribut ke pekerja
tertentu di lapisan analisis.

**Ukuran gambar sangat beragam.** Rasio aspek 0,45 sampai 3,75 dan ukuran 0,02
sampai 30,4 megapiksel. Aplikasi wajib membatasi ukuran unggahan sebelum masuk
model, karena gambar 30 megapiksel bisa mematikan server gratis.

`[belum]` Keterbatasan lain yang muncul setelah evaluasi.

## 8. Cara menjalankan ulang

`[belum]` Diisi setelah aplikasinya jadi.

```
notebooks/01_eda_dataset.ipynb     pemeriksaan data          [selesai]
notebooks/02_training.ipynb        training di Google Colab  [belum]
notebooks/03_evaluasi.ipynb        evaluasi dan eksperimen   [belum]
app.py                             aplikasi Streamlit        [belum]
```

`01_eda_dataset.ipynb` mencari zip dataset di tiga lokasi, Google Drive di
`MyDrive/capstone4/`, direktori kerja, dan folder `pilihan dataset dan aturan`.
Tidak butuh GPU, dan menyimpan hasilnya ke `laporan/eda_ringkasan.json`.

## Susunan berkas

```
capstone4-apd-konstruksi/
├── app.py                  entry point Streamlit          [belum]
├── requirements.txt        wheel CPU, versi dipin
├── packages.txt            pustaka sistem untuk OpenCV
├── models/                 bobot terpilih
├── src/
│   ├── detector.py         pemuatan model dan inference   [belum]
│   ├── analitik.py         logika analisis, tanpa impor Streamlit  [belum]
│   └── tampilan.py         komponen UI                    [belum]
├── tests/                  unit test logika analisis       [belum]
├── notebooks/
│   └── 01_eda_dataset.ipynb  lima pemeriksaan data       [selesai]
├── contoh_gambar/          tiga gambar untuk demo video    [belum]
└── laporan/
    └── eda_ringkasan.json  angka EDA, dikutip README      [selesai]
```

`src/analitik.py` sengaja tidak mengimpor Streamlit maupun Ultralytics, supaya
logikanya bisa diuji tanpa GPU dan tanpa menjalankan aplikasi.

## Catatan versi

`requirements.txt` memakai wheel CPU. Versi `ultralytics` di situ **wajib sama
persis** dengan yang dipakai saat training di Colab, karena file bobot menyimpan
referensi ke kelas Python di dalam library.

`[belum]` Versi hasil training pertama dicatat di sini setelah hari 3.

## Kredensial

Capstone ini tidak membutuhkan API key. Kalau nanti ada layanan luar yang dipakai,
kuncinya lewat `st.secrets` atau environment variable, tidak pernah di-hardcode,
tidak pernah masuk commit, dan tidak pernah terlihat di video.
