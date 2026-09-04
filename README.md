# Sistem Pemeriksaan Kelengkapan APD Pekerja Konstruksi

Capstone Project Module 4, Purwadhika Digital Technology School.
Object detection untuk memeriksa kelengkapan alat pelindung diri di lokasi konstruksi.

> Status pengerjaan: **hari 3 dari 15**. Bagian yang ditandai `[belum]` diisi
> sesuai urutan di `../catatan/URUTAN-KERJA.md`.
>
> Baseline sudah dilatih dan bobotnya sudah dipakai aplikasi. Angkanya seadanya
> dan memang begitu rencananya, pengejaran mAP dikerjakan hari 8 sampai 10.

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

Notebook `notebooks/02_training.ipynb` siap dijalankan di Google Colab. Ia
melatih baseline 30 epoch pada `imgsz` 640, memverifikasi bahwa augmentasi
geometris benar-benar mati lewat `args.yaml`, mengevaluasi di test set dengan
`conf=0.001`, lalu menyimpan bobot dan catatan versinya ke Drive.

Baseline `v1_baseline_640` dilatih 29 Agustus 2026 di Colab dengan Tesla T4.
Berhenti sendiri di epoch 27 karena early stopping, hasil terbaik di epoch 17,
total 10,6 menit. Catatan lengkapnya di `laporan/v1_baseline_640_catatan.json`.

### Rencana eksperimen hari 8 sampai 10, direvisi setelah baseline

Rencana semula menaruh "naikkan epoch" sebagai eksperimen pertama. **Itu sudah
terbantah oleh baselinenya sendiri.** Early stopping menyala di epoch 27 dengan
hasil terbaik di epoch 17, artinya model sudah berhenti membaik jauh sebelum
batas 30 epoch tercapai. Menaikkan epoch saja hampir pasti tidak menolong.

Urutannya diganti, dari yang paling ditopang bukti.

| Run | Yang diubah | Kenapa |
|---|---|---|
| v1_baseline_640 | baseline, geometris mati | sudah ada |
| v2_imgsz960 | resolusi 640 ke 960 | EDA menunjukkan 32,6 persen helmet dan 47,9 persen no-helmet di bawah 32 piksel pada 640. Ini yang paling mungkin mengangkat dua kelas terlemah |
| v3_copypaste | penanganan `no-helmet` | recall 0,333 dari 24 instance. Copy-paste augmentation menyerang persis kelas ini |
| v4_varian_s | nano ke small | terakhir, karena paling mahal dan paling jarang jadi akar masalah |
| v5_dengan_mosaic | pembanding, mosaic dinyalakan | mengukur berapa mAP yang dikorbankan demi patuh pada SOAL |

`patience` juga dinaikkan dari 10 ke 20 pada v2 dan seterusnya. Dengan resolusi
lebih tinggi, model butuh lebih banyak epoch sebelum mendatar, dan patience 10
berisiko menghentikannya terlalu cepat.

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

Baseline `v1_baseline_640`, dievaluasi di **test set** dengan `conf=0.001`.
Bukan di validation, karena validation dipakai memilih checkpoint sehingga
angkanya sudah menyesuaikan diri.

| | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| Keseluruhan | **0,722** | **0,370** |

### Per kelas

| Kelas | Instance | P | R | mAP50 | mAP50-95 | Ketemu | Terlewat |
|---|---|---|---|---|---|---|---|
| helmet | 195 | 0,867 | 0,837 | 0,876 | 0,434 | 163 | 32 |
| person | 214 | 0,865 | 0,855 | 0,848 | 0,508 | 183 | 31 |
| vest | 129 | 0,858 | 0,656 | 0,815 | 0,415 | 85 | 44 |
| no-vest | 61 | 0,731 | 0,607 | 0,652 | 0,313 | 37 | 24 |
| **no-helmet** | **24** | 0,793 | **0,333** | **0,421** | 0,182 | **8** | **16** |

### Tiga hal yang dikatakan angka ini

**Ramalan dari EDA terbukti persis.** `no-helmet` jadi kelas terburuk dengan
recall 0,333. Dari 24 pelanggaran helm di test set, model hanya menemukan 8 dan
melewatkan 16. Ini sudah diperkirakan sejak hari 2, karena kelas itu hanya punya
94 instance latih. Angkanya sendiri rapuh, 24 instance berarti satu deteksi
menggeser recall sekitar 4 poin persen.

**Model terlalu berhati-hati, dan itu arah yang salah untuk keselamatan.**
Rata-rata precision 0,823 sementara recall 0,658, selisih 16,5 poin. Artinya
model lebih sering melewatkan objek nyata daripada mengarang objek palsu. Untuk
sistem yang memeriksa keselamatan kerja, melewatkan pelanggaran jauh lebih mahal
daripada alarm palsu. Confidence threshold aplikasi perlu digeser ke bawah, dan
angkanya diambil dari kurva F1, bukan dari nilai bawaan 0,25.

**Kotaknya ketemu tapi kurang rapat.** mAP@0.5 0,722 berbanding mAP@0.5:0.95
0,370, selisihnya besar. Itu masalah lokalisasi, bukan pengenalan, dan wajar
untuk objek sekecil helm.

### Kecepatan

Dua perangkat, dua angka, karena angka tanpa konteks tidak berarti apa pun.

| Perangkat | Per gambar |
|---|---|
| Tesla T4, Colab | preprocess 1,9 ms, inference 15,8 ms, postprocess 3,9 ms |
| CPU laptop, satu thread torch | median **23 ms**, rentang 20 sampai 31 ms |

**Panggilan pertama 979 ms**, hampir empat puluh kali median. Itu lazy init
PyTorch, terjadi sekali per proses, dan bukan cacat. Tapi ia punya akibat nyata
di aplikasi. Unggahan pertama setelah aplikasi bangun dari tidur akan terasa
lambat, sedangkan unggahan berikutnya seketika.

Angka median diambil setelah tiga kali pemanasan. Mengukur tanpa membuang
panggilan pertama menghasilkan rata-rata 508 ms, dan itu menggambarkan lazy
init, bukan kecepatan modelnya.

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

Urutannya begini, dan tiap langkah berdiri sendiri.

```
notebooks/01_eda_dataset.ipynb     pemeriksaan data          [selesai]
notebooks/02_training.ipynb        training di Google Colab  [siap dijalankan]
notebooks/03_evaluasi.ipynb        evaluasi dan eksperimen   [belum]
app.py                             aplikasi Streamlit        [jalan]
```

**EDA.** `01_eda_dataset.ipynb` mencari zip dataset di tiga lokasi, Google Drive
di `MyDrive/capstone4/`, direktori kerja, dan folder `pilihan dataset dan aturan`.
Tidak butuh GPU. Hasilnya disimpan ke `laporan/eda_ringkasan.json`.

**Training.** `02_training.ipynb` dijalankan di Colab dengan GPU T4. Unggah zip
dataset ke `MyDrive/capstone4/` lebih dulu. Keluarannya bobot dan berkas catatan
versi, keduanya tersimpan ke Drive.

**Aplikasi di komputer sendiri.**

```bash
python -m venv .venv-uji
.venv-uji/bin/pip install -r requirements.txt
.venv-uji/bin/streamlit run app.py
```

Aplikasi memakai bobot di `models/`. Kalau folder itu kosong, ia jatuh ke bobot
bawaan COCO yang hanya mengenali orang, dan mengatakannya terus terang di layar.

**Uji tanpa menjalankan aplikasi.**

```bash
.venv-uji/bin/python tests/test_detector.py
```

## Susunan berkas

```
capstone4-apd-konstruksi/
├── app.py                  entry point Streamlit          [jalan]
├── requirements.txt        wheel CPU, versi dipin
├── packages.txt            pustaka sistem untuk OpenCV
├── models/
│   └── apd_v1_baseline_640.pt  5,47 MB          [selesai]
├── src/
│   ├── detector.py         pemuatan model dan inference   [selesai]
│   ├── analitik.py         logika analisis, tanpa impor Streamlit  [belum]
│   └── tampilan.py         komponen UI                    [belum]
├── tests/
│   └── test_detector.py    empat uji asap, semuanya lolos  [selesai]
├── notebooks/
│   ├── 01_eda_dataset.ipynb  lima pemeriksaan data       [selesai]
│   └── 02_training.ipynb     training baseline, berisi output  [selesai]
├── contoh_gambar/          tiga gambar untuk demo video    [belum]
└── laporan/
    └── eda_ringkasan.json  angka EDA, dikutip README      [selesai]
```

`src/analitik.py` sengaja tidak mengimpor Streamlit maupun Ultralytics, supaya
logikanya bisa diuji tanpa GPU dan tanpa menjalankan aplikasi.

## Catatan versi

`requirements.txt` diuji hari 3 di virtualenv bersih, bukan diasumsikan.

| Paket | Versi | Alasan |
|---|---|---|
| torch | `==2.11.0` | sama dengan versi di Colab |
| torchvision | `==0.26.0` | pasangan torch 2.11.0 |
| ultralytics | `==8.4.138` | **wajib sama** dengan versi saat training, berkas bobot menyimpan referensi kelas Python |
| streamlit | `==1.63.0` | ketat, verifikasi `AppTest` menjalankan `app.py` sampai selesai di Python 3.14 |
| pillow | `>=11.0` | longgar, lihat alasannya di bawah |
| pandas | tidak dicantumkan | tidak dipakai langsung, hanya ditarik streamlit |

### Kenapa sebagian dipin ketat dan sebagian longgar

Yang menentukan apakah berkas bobot bisa dimuat cuma `ultralytics`, dan lewat ia
`torch`. Ketiganya dipin ketat.

Streamlit juga dipin ketat, tapi alasannya berbeda. Ia pure Python sehingga
wheel-nya universal dan tidak pernah kena masalah yang menjatuhkan pillow. Yang
dijaga permukaan API-nya, yang berubah antar versi minor.

**Hanya `pillow` yang dilonggarkan**, karena Streamlit Community Cloud memilih
sendiri versi Python-nya dan bisa mengubahnya kapan saja. Saat ini 3.14.7.
Memin `pillow==11.0.0` membuat build gagal, sebab versi itu tidak punya wheel
untuk 3.14 sehingga pip mencoba mengompilasinya dari sumber lalu berhenti.

```
The headers or library files could not be found for zlib,
a required dependency when compiling Pillow from source.
```

Dengan batas bawah saja, resolver memilih pillow yang punya wheel `cp314`.

### Cara memutuskan mana yang dipin dan mana yang tidak

| Kelompok | Perlakuan | Alasan |
|---|---|---|
| `ultralytics`, `torch`, `torchvision` | ketat | menentukan apakah berkas bobot bisa dimuat |
| `streamlit` | ketat | permukaan API berubah antar versi minor, dan pure Python jadi tidak ada risiko wheel |
| `pillow` | longgar | punya ekstensi C, wheel-nya tergantung versi Python yang dipilih Cloud |
| `pandas` | tidak dicantumkan | tidak dipakai langsung |

Yang dilonggarkan hanya yang **terbukti** rusak kalau dipin. Melonggarkan
semuanya sekaligus pernah saya coba dan itu keliru, streamlit ikut melompat dari
1.40 ke 1.63, dua puluh tiga versi minor, menyeret pandas dan pillow ikut
berubah tanpa alasan.

### Yang diverifikasi di container, bukan diasumsikan

| Uji | Hasil |
|---|---|
| `apt-get install libgl1 libglib2.0-0t64` di `debian:trixie` | berhasil, `libGL.so.1` dan `libgthread-2.0.so.0` ada |
| `pip install -r requirements.txt` di Python 3.14.7 | berhasil |
| `import cv2, torch, ultralytics, streamlit, PIL` | berhasil |
| `streamlit run app.py` | HTTP 200 |
| `AppTest.from_file("app.py").run()` | tanpa exception, tanpa peringatan deprecation |
| `use_container_width` di `st.image` dan `st.dataframe` | masih diterima |
| streamlit | 1.40.0 | |
| pillow | 11.0.0 | |
| pandas | 2.2.3 | |

Diverifikasi pada Python 3.13.9, torch tanpa CUDA build, ukuran virtualenv 1,3 GB.

### Dua kali salah pin, dua kali ketahuan sebelum deploy

**Hari 1.** `torch==2.5.1` tidak punya wheel untuk Python 3.13, install gagal
total.

**Hari 3, setelah training.** Colab ternyata memasang `ultralytics 8.4.138` dan
`torch 2.11.0`, bukan 8.3.217 dan 2.8.0 yang saya tulis. Berkas bobot dibuat
oleh 8.4.138, jadi pin lama berisiko gagal memuatnya di aplikasi. Angka di tabel
atas diambil dari keluaran notebook, bukan dari perkiraan.

Keduanya jenis kegagalan yang tidak berbunyi sampai deploy. Itu alasan uji
pasang dijadwalkan hari 3, bukan di akhir.

### Nama paket sistem, dan kenapa `libglib2.0-0` gagal

Deploy pertama gagal di `packages.txt`, bukan di Python.

```
libglib2.0-0 : Depends: libffi7 but it is not installable
               Depends: libpcre3 but it is not installable
E: Unable to correct problems, you have held broken packages
```

Image Streamlit Community Cloud memakai **Debian 13 trixie**. Di sana
`libglib2.0-0` sudah tidak punya versi kandidat, hanya tersisa sebagai nama
virtual, sisa transisi time_t 64-bit. Sumber apt di image itu masih memuat entri
bullseye yang tertinggal, jadi apt mengambil versi bullseye 2.66.8 yang
membutuhkan `libffi7` dan `libpcre3`, dan keduanya tidak ada di trixie.

Nama yang benar `libglib2.0-0t64`. Diverifikasi di container `debian:trixie`,
bukan ditebak.

| Paket | Trixie | Keterangan |
|---|---|---|
| `libgl1` | 1.7.0-1+b2 | ada, menyediakan `libGL.so.1` |
| `libglib2.0-0` | **tidak ada kandidat** | nama virtual saja |
| `libglib2.0-0t64` | 2.84.4-3~deb13u3 | ini yang dipakai |

Glib tidak bisa dihilangkan begitu saja. Tanpa ia, `import cv2` gagal dengan
`libgthread-2.0.so.0: cannot open shared object file`, juga sudah diuji.

### Dua jebakan lingkungan yang sudah ditutup

`YOLO_CONFIG_DIR` **dibuat dulu** sebelum dipakai. Ultralytics tidak membuatnya
sendiri, ia hanya memberi peringatan lalu diam-diam menulis ke lokasi lain.

`YOLO_OFFLINE=1` sudah diuji **tidak** memblokir unduhan bobot, jadi model bawaan
tetap bisa diambil saat pertama dijalankan. Unduhannya diarahkan ke direktori
sementara yang pasti bisa ditulis, bukan ke direktori kerja yang belum tentu
punya izin tulis di server hosting.

## Kredensial

Capstone ini tidak membutuhkan API key. Kalau nanti ada layanan luar yang dipakai,
kuncinya lewat `st.secrets` atau environment variable, tidak pernah di-hardcode,
tidak pernah masuk commit, dan tidak pernah terlihat di video.
