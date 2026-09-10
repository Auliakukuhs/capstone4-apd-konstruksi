# Sistem Pemeriksaan Kelengkapan APD Pekerja Konstruksi

Capstone Project Module 4, Purwadhika Digital Technology School.
Object detection untuk memeriksa kelengkapan alat pelindung diri di lokasi konstruksi.

> Status pengerjaan: **hari 10 dari 15**. Bagian yang ditandai `[belum]` diisi
> sesuai urutan di `../catatan/URUTAN-KERJA.md`.
>
> Aplikasinya lengkap dan hidup di Streamlit Community Cloud, lapisan
> analisisnya tervalidasi di test set, empat eksperimen training sudah selesai,
> dan 61 uji lolos.
>
> Model yang dipakai `v3_oversample_nohelmet`. **mAP-nya lebih rendah daripada
> baseline, dan itu disengaja.** Alasannya di bagian 5.
>
> Sisa pekerjaan hari 11 sampai 15, yaitu evaluasi final, poles, naskah video,
> rekaman, dan pengumpulan.

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

### Rencana eksperimen hari 8 sampai 10, direvisi dua kali

Notebooknya `notebooks/03_eksperimen.ipynb`, sudah dijalankan, hasilnya di
bagian 5. Rencananya berubah dua kali, dan kedua alasannya dicatat di sini
karena keduanya temuan, bukan perubahan selera.

**Revisi pertama, setelah baseline.** Rencana semula menaruh "naikkan epoch"
sebagai eksperimen pertama. Itu terbantah oleh baselinenya sendiri. Early
stopping menyala di epoch 27 dengan hasil terbaik di epoch 17, artinya model
sudah berhenti membaik jauh sebelum batas 30 epoch tercapai. Menaikkan epoch
saja hampir pasti tidak menolong. `patience` dinaikkan dari 10 ke 20 pada v2
dan seterusnya, karena pada resolusi lebih tinggi model butuh lebih banyak
epoch sebelum mendatar.

**Revisi kedua, setelah membaca kode Ultralytics.** Rencana semula memakai
`copy_paste` untuk kelas `no-helmet`. **Itu tidak akan bekerja di dataset ini.**
Blok `CopyPaste` di `ultralytics/data/augment.py` dibuka dengan

```python
if len(labels["instances"].segments) == 0 or self.p == 0:
    return labels
```

Tanpa segmentasi ia keluar tanpa berbuat apa pun, dan seluruh 7.724 baris label
dataset ini berisi lima kolom, yaitu bbox saja. Run itu akan menghabiskan waktu
GPU lalu menghasilkan model yang identik dengan run sebelumnya, sementara
namanya menjanjikan hal lain. Ini jenis kegagalan yang tidak berbunyi.

Penggantinya oversampling tingkat data, yaitu menduplikasi gambar yang memuat
`no-helmet`. Itu bukan augmentasi sama sekali sehingga tidak menyentuh larangan
augmentasi geometris, dan efeknya sudah diukur di komputer lokal sebelum satu
menit GPU pun dipakai.

| Ukuran | Sebelum | Sesudah, duplikasi 4x |
|---|---|---|
| Gambar train | 997 | 1.144, naik 14,7 persen |
| Instance `no-helmet` | 94 | 376 |
| Porsi `no-helmet` | 1,5 persen | **4,7 persen** |
| Porsi `person` | 37,0 persen | 36,5 persen |

Hanya 49 gambar yang memuat `no-helmet`, jadi kelas lain hampir tidak bergerak.
Risikonya dinyatakan terbuka, model melihat 49 foto itu empat kali per epoch,
sehingga kalau `no-helmet` membaik tapi kelas lain memburuk, overfitting pada 49
foto itu tersangka pertamanya.

### Empat run, satu perubahan per run

| Run | Yang diubah | Dibanding | Kenapa |
|---|---|---|---|
| `v1_baseline_640` | baseline, geometris mati | | sudah ada, dan tetap kandidat |
| `v2_imgsz960` | resolusi 640 ke 960 | v1 | 32,6 persen helmet dan 47,9 persen no-helmet di bawah 32 piksel pada 640, turun ke 14,1 dan 17,0 persen pada 960 |
| `v3_oversample_nohelmet` | train split, gambar no-helmet 4x | v2 | recall no-helmet 0,333, kelas terlemah |
| `v4_varian_s` | nano ke small | v2 atau v3 | terakhir, karena paling mahal dan paling jarang jadi akar masalah |
| `v5_dengan_mosaic` | mosaic dinyalakan | v2 | **pembanding, bukan kandidat.** Mengukur berapa mAP yang dikorbankan demi patuh pada SOAL |

### Model dipilih dari kualitas vonis, bukan dari mAP

Ini keputusan yang paling perlu dipahami di bagian ini. Setiap run dinilai dua
kali, mAP di test set pada `conf=0.001`, dan **vonis per pekerja** dibandingkan
vonis dari ground truth memakai `src/analitik.py` yang sama dengan aplikasi.

Yang menentukan pemilihan adalah yang kedua. mAP mengukur kualitas kotak,
sementara yang dibaca pengawas adalah vonis, dan biaya ketiga jenis kesalahan
vonis sangat berbeda.

Aturannya ada di `skrip/eksperimen.py`, dinyatakan di muka sebelum angkanya
terlihat, dan diuji di `tests/test_eksperimen.py`.

1. Run bertanda bukan kandidat dikeluarkan, yaitu v5
2. **Penjaga cakupan.** Run yang cakupan pekerjanya jatuh lebih dari 5 poin di
   bawah yang terbaik dikeluarkan. Tanpa penjaga ini, model yang hampir tidak
   mendeteksi siapa pun akan terlihat unggul, sebab laju kesalahan dihitung
   dari pekerja yang berhasil dipasangkan dan penyebutnya menyusut
3. **Laju pembebasan keliru terendah menang.** Pelanggar yang dinyatakan
   lengkap adalah kesalahan termahal, karena ia menghentikan pemeriksaan
   terhadap orang yang justru berisiko
4. Seri diputus akurasi vonis, lalu mAP@0.5:0.95

Perhatikan bahwa mAP ada di urutan terakhir, dan itu disengaja. **Aturan ini
benar-benar diuji oleh hasilnya.** Baseline ternyata punya mAP tertinggi, jadi
kalau mAP yang dipakai memilih, seluruh eksperimen akan disimpulkan gagal.
Aturan ini memilih `v3_oversample_nohelmet` yang mAP-nya lebih rendah tapi
kedua laju kesalahannya lebih kecil. Kalau aturannya diubah sekarang setelah
angkanya terlihat, ia berhenti menjadi aturan.

Satu celah sempat ada di sini dan sudah ditutup. Catatan v1 dibuat sebelum
penilaian vonis ada, jadi ia tanpa angka itu dan otomatis tersingkir dari
pemilihan, yang berarti "pertahankan baseline" bukan hasil yang mungkin dan
eksperimen hanya bisa membaik menurut konstruksinya sendiri. Catatan v1 sudah
dilengkapi, dan ada uji yang menjaga perilakunya.

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

Model yang dipakai aplikasi **`v3_oversample_nohelmet`**, dievaluasi di **test
set** dengan `conf=0.001`. Bukan di validation, karena validation dipakai
memilih checkpoint sehingga angkanya sudah menyesuaikan diri.

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|
| `v1_baseline_640` | 640 | **0,722** | **0,370** |
| `v3_oversample_nohelmet` | 960 | 0,684 | 0,334 |

**Model terpilih justru mAP-nya lebih rendah daripada baseline.** Itu bukan
kekeliruan pemilihan, dan bagian berikut menjelaskan kenapa.

### Per kelas, model terpilih

| Kelas | Instance | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| helmet | 195 | 0,811 | 0,887 | 0,853 | 0,428 |
| person | 214 | 0,790 | 0,846 | 0,811 | 0,489 |
| vest | 129 | 0,806 | 0,721 | 0,720 | 0,371 |
| no-vest | 61 | 0,658 | 0,639 | 0,611 | 0,297 |
| **no-helmet** | **24** | 0,713 | **0,458** | 0,425 | 0,185 |

### Kenapa mAP turun tapi modelnya tetap yang dipilih

**Eksperimen ini berhasil melakukan persis yang diminta diagnosis hari 3.**
Waktu itu tertulis bahwa baseline terlalu berhati-hati, precision rata-rata
0,823 sementara recall 0,658, selisih 16,5 poin, dan untuk sistem keselamatan
arah itu salah karena melewatkan pelanggaran jauh lebih mahal daripada alarm
palsu.

| Model | P rata-rata | R rata-rata | Selisih |
|---|---|---|---|
| `v1_baseline_640` | 0,823 | 0,658 | **+16,5 poin** |
| `v2_imgsz960` | 0,712 | 0,699 | +1,4 poin |
| `v3_oversample_nohelmet` | 0,756 | 0,710 | **+4,5 poin** |

Resolusi 960 menukar precision dengan recall, dan itu memang yang diinginkan.
mAP menghukum pertukaran itu karena ia memperlakukan kedua jenis kesalahan
sebagai setara, padahal di sini keduanya tidak setara sama sekali.

Recall naik di empat dari lima kelas. `no-helmet`, kelas paling penting bagi
keselamatan dan paling lemah sejak awal, naik dari 0,333 ke 0,458.

Oversampling di v3 mengembalikan sebagian precision yang hilang di v2, dari
0,712 ke 0,756, tanpa mengorbankan recall. Kenaikannya paling besar justru di
dua kelas yang paling jarang, `no-helmet` 0,535 ke 0,713 dan `no-vest` 0,503 ke
0,658.

### Hasil seluruh eksperimen

| Run | imgsz | mAP50 | mAP50-95 | R no-helmet | Pembebasan keliru | Tuduhan palsu | Kandidat |
|---|---|---|---|---|---|---|---|
| `v1_baseline_640` | 640 | **0,722** | **0,370** | 0,333 | 0,070 | 0,067 | ya |
| `v2_imgsz960` | 960 | 0,691 | 0,336 | 0,458 | 0,054 | 0,060 | ya |
| **`v3_oversample_nohelmet`** | 960 | 0,684 | 0,334 | **0,458** | **0,036** | **0,020** | **terpilih** |
| `v4_varian_s` | 960 | 0,650 | 0,319 | 0,333 | 0,056 | 0,030 | ya |
| `v5_dengan_mosaic` | 960 | 0,673 | 0,345 | 0,351 | 0,035 | 0,078 | tidak |

Seluruhnya `batch` 16, tidak ada yang terpaksa diturunkan, jadi tiap run
benar-benar hanya membawa satu perubahan.

### Tiga hal yang dikatakan tabel ini

**Varian model lebih besar tidak menolong.** `v4_varian_s` lebih buruk pada
hampir seluruh ukuran, termasuk kembali jatuh ke recall `no-helmet` 0,333,
sementara bobotnya 19,25 MB atau 3,5 kali lipat. Ini hasil negatif dan tetap
dilaporkan. Kapasitas model bukan akar masalah di sini, jumlah contoh latih
yang jadi masalah.

**Harga kepatuhan pada SOAL ternyata mendekati nol.** `v5_dengan_mosaic`
dijalankan justru untuk mengukur berapa mAP yang dikorbankan dengan mematikan
mosaic. Dibanding `v2` yang identik kecuali mosaic, hasilnya mAP50 **turun**
dari 0,691 ke 0,673 sementara mAP50-95 naik tipis dari 0,336 ke 0,345.
Perkiraan umum bahwa mematikan mosaic merugikan mAP pada dataset kecil **tidak
terbukti di sini**. Precision rata-ratanya bahkan jatuh ke 0,640, terendah dari
seluruh run.

**Angkanya belum meyakinkan secara statistik, dan itu harus dikatakan.** Selang
kepercayaan 95 persen metode Wilson untuk ketiga perbaikan utama masih
bertumpang tindih dengan baseline.

| Ukuran | v1 | v3 | Selang v1 | Selang v3 |
|---|---|---|---|---|
| Pembebasan keliru | 4 dari 57 | 2 dari 56 | 0,028 sampai 0,167 | 0,010 sampai 0,121 |
| Tuduhan palsu | 7 dari 104 | 2 dari 101 | 0,033 sampai 0,132 | 0,005 sampai 0,069 |
| Recall `no-helmet` | 8 dari 24 | 11 dari 24 | 0,180 sampai 0,533 | 0,279 sampai 0,649 |

Test set 90 gambar terlalu kecil untuk menyatakan satu pun perbaikan itu nyata
sendiri-sendiri. Yang menopang kesimpulan adalah **arahnya konsisten di banyak
ukuran yang saling bebas**, recall naik di empat dari lima kelas, selisih
precision recall menyempit dari 16,5 ke 4,5 poin, dan kedua laju kesalahan
turun bersamaan.

### Kecepatan

| Model dan perangkat | Per gambar |
|---|---|
| `v1` di Tesla T4 Colab | preprocess 1,9 ms, inference 15,8 ms, postprocess 3,9 ms |
| `v1` di CPU laptop, imgsz 640 | median **26 ms** |
| `v3` di CPU laptop, imgsz 960 | median **51 ms** |

Resolusi 960 menggandakan waktu inference, dan itu harga yang dibayar. Pada
Streamlit Community Cloud yang hanya CPU, satu gambar masih di bawah
seperempat detik, jadi masih nyaman dipakai.

**Panggilan pertama sekitar 900 ms**, hampir empat puluh kali median. Itu lazy
init PyTorch, terjadi sekali per proses, bukan cacat. Tapi akibatnya nyata,
unggahan pertama setelah aplikasi bangun dari tidur terasa lambat sedangkan
berikutnya seketika. Angka median diambil setelah tiga kali pemanasan.

## 6. Lapisan analisis

Ada di `src/analitik.py`. Modul itu sengaja tidak mengimpor Streamlit maupun
Ultralytics, sehingga logikanya bisa diuji tanpa GPU dan tanpa menjalankan
aplikasi. Masukannya hanya daftar dict keluaran `src.detector.ke_deteksi`.

### Kenapa bukan aritmetika hitungan

Contoh di SOAL, 4 helmet, 2 vest, 2 no-vest pada 4 pekerja, kesimpulannya 2
orang tidak lengkap. Aritmetika itu benar untuk contoh itu, tapi rapuh secara
umum. Diukur pada ground truth, `vest` ditambah `no-vest` hanya 79,3 persen
dari jumlah `person`, jadi pengurangan sederhana melebih-lebihkan pelanggaran
rompi sekitar 20 persen **bahkan pada anotasi sempurna**.

Lagi pula pengurangan tidak bisa menjawab pertanyaan yang berguna di lapangan,
yaitu pekerja yang mana.

### IoA, bukan IoU

Setiap atribut diasosiasikan ke kotak `person` yang memuatnya lewat **IoA**,
yaitu luas irisan dibagi luas kotak atribut saja. Bukan IoU, yang membagi
dengan gabungan luas sehingga tertekan oleh kotak person yang jauh lebih besar.

Diukur pada kotak sungguhan, satu helm yang berada **sepenuhnya** di dalam
kotak pekerja menghasilkan IoU di bawah 0,10 tapi IoA tepat 1,00. Dengan
ambang 0,5 pada IoU, helm yang jelas-jelas milik pekerja itu justru ditolak.

Saat satu helm memenuhi ambang pada lebih dari satu pekerja, IoA seri 1,00 di
keduanya dan tidak bisa memutuskan. Yang memutuskan **posisi vertikal**. Pada
ground truth, 98,8 persen pusat helm berada di 25 persen teratas kotak person
dengan median 0,09, jadi pemenangnya pekerja yang kedalaman relatifnya paling
dekat ke 0,09.

### Tiga status, bukan dua

| Status | Artinya |
|---|---|
| `memakai` | atributnya terdeteksi dan terasosiasi |
| `tidak memakai` | kelas `no-helmet` atau `no-vest` yang terdeteksi |
| `belum dapat dipastikan` | tidak ada kotak apa pun untuk atribut itu |

Vonis per pekerja lahir dari kombinasinya. Pelanggaran yang terbukti selalu
menang atas ketidakpastian, sehingga satu atribut yang jelas tidak dipakai
sudah cukup menyatakan tidak lengkap. Sebaliknya LENGKAP hanya diberikan kalau
kedua atribut benar-benar terlihat.

Alasannya kuantitatif. Pada ground truth train dan valid, **11,8 persen kotak
person tidak punya kotak helmet sama sekali**, dan pada test set 8,9 persen.
Kalau anotator manusia saja melewatkan sebanyak itu, model pasti lebih sering.
Menyamakan "tidak terdeteksi" dengan "melanggar" memproduksi tuduhan palsu
terhadap orang yang sebenarnya patuh.

### Konflik dan urutan

Model bisa mengeluarkan `helmet` dan `no-helmet` pada pekerja yang sama saat
ragu. Yang belakangan **tidak** menimpa yang duluan, karena hasilnya lalu
bergantung pada urutan deteksi. Yang menang confidence tertinggi, dan
pertentangannya dicatat supaya bisa ditampilkan sebagai bukti.

`tests/test_analitik.py` menguji seluruh permutasi urutan deteksi pada satu
kasus berkonflik dan memastikan vonisnya tidak berubah.

### Atribut tanpa induk tidak dibuang

Helm yang tidak cocok ke satu pekerja pun dilaporkan terpisah, bukan dihilangkan
diam-diam. Jumlahnya adalah petunjuk langsung bahwa ada pekerja yang tidak
terdeteksi, dan itu informasi yang berguna bagi pengawas.

### Dua angka kepatuhan, bukan satu

| Angka | Rumus | Kapan dipakai |
|---|---|---|
| Kepatuhan | lengkap dibagi seluruh pekerja | angka pesimistis, yang belum pasti ikut penyebut |
| Kepatuhan terbaca | lengkap dibagi lengkap ditambah tidak lengkap | adil dibandingkan antar gambar |

Keduanya mengembalikan nilai kosong, bukan nol, kalau tidak ada pekerja di
gambar. "Tidak ada orang" tidak boleh terbaca sebagai "kepatuhan nol persen".

### Validasi di test set, bukan di contoh buatan

`skrip/validasi_analitik.py` menjalankan asosiasi pada **kotak ground truth**
test set. Ini menguji algoritmanya sendiri, terlepas dari kualitas deteksi.
Angkanya tersimpan di `laporan/validasi_analitik.json`.

| Ukuran | Nilai |
|---|---|
| Gambar / pekerja / atribut | 90 / 214 / 409 |
| Atribut cocok ke tepat satu pekerja | 366, **89,5 persen** |
| Atribut cocok ke lebih dari satu, tiebreak dipakai | 4 |
| Atribut tanpa induk | 39, yaitu 22 helmet, 9 vest, 8 no-vest |
| Pekerja tanpa kotak helm sama sekali | 19, 8,9 persen |
| Vonis dari ground truth | 108 lengkap, 71 tidak lengkap, 35 belum dapat dipastikan |

Baris terakhir adalah temuan terpenting di seluruh bagian ini. **35 dari 214
pekerja, 16,4 persen, tidak dapat divonis bahkan dengan anotasi manusia.** Jadi
status ketiga bukan cara menutupi kelemahan model, ia melekat pada datanya.

Porsi cocok tepat satu di test set, 89,5 persen, lebih rendah daripada 94,5
persen yang terukur di train dan valid. Sebabnya test set punya proporsi helm
yatim lebih tinggi, dan angka yang lebih rendah inilah yang dipakai, bukan yang
lebih bagus.

### Vonis prediksi model dibanding vonis ground truth

Bagian 2 skrip yang sama menjalankan model pada 90 gambar test, menjatuhkan
vonis dari prediksinya, lalu membandingkannya dengan vonis yang lahir dari
ground truth. Kotak pekerja dipasangkan lewat IoU 0,5. Ini mengukur sistem utuh
dari piksel sampai kesimpulan, bukan cuma modelnya.

| Ukuran | `v1_baseline_640` | `v3_oversample_nohelmet` |
|---|---|---|
| Pekerja ground truth / prediksi | 214 / 237 | 214 / 219 |
| Berhasil dipasangkan | 187 | 179 |
| Pekerja terlewat / palsu | 27 / 50 | 35 / 40 |
| Vonis benar | 130, 69,5 persen | 119, 66,5 persen |

Matriks vonis model terpilih, baris ground truth dan kolom prediksi.

| | pred LENGKAP | pred TIDAK LENGKAP | pred BELUM PASTI |
|---|---|---|---|
| **GT LENGKAP** (101) | 72 | **2** | 27 |
| **GT TIDAK LENGKAP** (56) | **2** | 34 | 20 |
| **GT BELUM PASTI** (22) | 4 | 5 | 13 |

Angka di atas hasil menjalankan ulang di CPU. Catatan run dari Colab mencatat
180 pasangan dan 120 vonis benar, bergeser satu pekerja karena aritmetika float
GPU dan CPU tidak identik. Selisih sebesar itu tidak mengubah kesimpulan apa
pun, tapi lebih baik disebut daripada terlihat seperti angka yang tidak cocok.

### Bukti angka bahwa status ketiga bukan sekadar kehati-hatian

Ketiga jenis kesalahan di matriks itu **biayanya sangat berbeda**, jadi akurasi
tunggal menyembunyikan yang penting.

| Jenis kesalahan | v1 | v3 |
|---|---|---|
| Pembebasan keliru, pelanggar dinyatakan lengkap | 4 dari 57, 7,0 persen | **2 dari 56, 3,6 persen** |
| Tuduhan palsu, pekerja patuh dinyatakan melanggar | 7 dari 104, 6,7 persen | **2 dari 101, 2,0 persen** |
| Tuduhan palsu **kalau status ketiga dihapus** | 29 dari 104, 27,9 persen | 29 dari 101, 28,7 persen |

Baris ketiga adalah simulasi sistem dua status, yang terpaksa membaca "tidak
terdeteksi" sebagai "melanggar". Dari 101 pekerja yang sebenarnya patuh, 29
akan dituduh melanggar. Dengan status ketiga, angkanya turun menjadi 2.

**Status ketiga memotong tuduhan palsu dari 28,7 persen menjadi 2,0 persen,
empat belas kali lipat.** Untuk sistem keselamatan yang keluarannya bisa
berujung teguran terhadap orang, pertukaran itu jelas menguntungkan.

### Harga yang dibayar, dinyatakan terbuka

Model terpilih lebih berhati-hati, dan kehati-hatian itu ada ongkosnya.

| | v1 | v3 |
|---|---|---|
| Pelanggaran yang dinyatakan tegas | 43 dari 57 | **34 dari 56** |
| Pelanggaran yang dilempar ke pemeriksaan manusia | 10 | **20** |
| Porsi seluruh pekerja yang perlu diperiksa manusia | 44 dari 187, 23,5 persen | **60 dari 179, 33,5 persen** |

Jadi v3 menyatakan lebih sedikit pelanggaran secara tegas, dan menyerahkan
lebih banyak kepada manusia. Yang penting, pelanggaran yang dilempar itu
**tidak hilang**, ia muncul sebagai kotak kuning yang menuntut pemeriksaan.
Yang benar-benar lolos hanya yang dinyatakan LENGKAP padahal melanggar, dan
jumlah itu justru turun dari 4 ke 2.

Pertukarannya bisa dinyatakan dalam satu kalimat. **Sepuluh poin persen beban
pemeriksaan manusia ditukar dengan tuduhan palsu yang turun dari 6,7 ke 2,0
persen dan pembebasan keliru yang turun dari 7,0 ke 3,6 persen.** Apakah itu
pantas dibayar adalah keputusan pengawas K3, bukan keputusan model, dan karena
itu kedua angkanya ditampilkan di aplikasi berdampingan.

### Uji

`tests/test_analitik.py`, 19 uji, semuanya lolos. Cakupannya mencakup lima
kasus yang diminta rencana kerja, yaitu pekerja lengkap, pekerja melanggar,
atribut tidak terdeteksi, atribut tanpa induk, dan nol deteksi, ditambah
pembuktian numerik IoA lawan IoU, penyelesaian konflik, ketidakpekaan terhadap
urutan, tiebreak vertikal, dan dua angka kepatuhan.

## 7. Aplikasi Streamlit

Hidup di
<https://capstone4-apd-konstruksi-kukuhsaputraaulia.streamlit.app>.

Materi memperlihatkan empat elemen yang konsisten muncul di tiga demo
aplikasi vision. Keempatnya ada, dan tempatnya begini.

| Elemen | Di aplikasi ini |
|---|---|
| Dua kolom, asli dan beranotasi | `tampilan.sandingkan`, kotak berwarna mengikuti vonis tiap pekerja |
| Panel Summary | `tampilan.panel_ringkasan`, enam metrik plus catatan atribut tanpa pekerja |
| Banner vonis berwarna dengan kalimat tindakan | `tampilan.banner`, empat tingkat |
| Sesuatu yang melampaui satu gambar | unggah banyak berkas sekaligus, dashboard total sesi di panel kiri, dan ekspor CSV per pekerja |

### Warna kotak, bukan cuma banner

Banner menyatakan keadaan keseluruhan, tapi pengawas perlu tahu **orang mana**.
Karena itu warna vonis diturunkan sampai ke tingkat objek. Hijau lengkap, merah
tidak lengkap, kuning belum dapat dipastikan, biru atribut yang berhasil
dikaitkan, dan ungu atribut yang tidak dapat dikaitkan ke siapa pun.

### Label yang menyesuaikan lebar kotak

Rata-rata gambar dataset ini berisi 6,41 objek, dan yang terpadat 39. Pada foto
sembilan pekerja berdampingan, tulisan "TIDAK LENGKAP" milik tiap orang saling
menimpa sampai tidak satu pun terbaca. Ini bukan kasus tepi, ini kondisi normal.

Karena itu label memilih sendiri bentuk terpanjang yang masih muat di lebar
kotak, dari kalimat penuh, lalu singkatan, lalu nomor pekerja saja. Vonisnya
tetap terbaca lewat warna kotak dan tabel di bawah gambar.

### Sesi dihitung ulang, bukan ditumpuk

Streamlit menjalankan ulang seluruh skrip setiap kali slider digeser. Pola yang
umum dipakai, menambahkan hasil ke `st.session_state` tiap run, punya dua cacat
sekaligus di aplikasi seperti ini.

Gambar yang sama terhitung berkali-kali setiap pengguna menyentuh apa pun. Dan
yang lebih berbahaya, riwayat yang bertahan mencampur vonis dari confidence
threshold yang berbeda ke dalam satu angka kepatuhan, sehingga angka itu tidak
berarti apa-apa.

Aplikasi ini membangun ulang `Sesi` dari seluruh berkas yang sedang terunggah,
pada ambang yang sedang aktif. Sifatnya idempoten, dan angkanya selalu
menjelaskan keadaan yang sedang terlihat di layar.

### Cache dipasang di tempat yang benar

`@st.cache_resource` untuk bobot model, sekali per proses. `@st.cache_data`
untuk hasil inference, berkunci isi berkas beserta seluruh parameter deteksi.

Efeknya, membuka expander atau menyalakan pembanding CLAHE tidak memicu
inference ulang, sementara menggeser confidence tetap memicunya, karena `conf`
ikut menjadi kunci. Yang disimpan hanya daftar dict, bukan objek `Results`
Ultralytics, karena anotasi digambar sendiri sehingga objek itu tidak
dibutuhkan lagi setelah inference selesai.

### CLAHE disediakan sebagai pembanding, bukan preprocessing

Bagian 3 menjelaskan alasan CLAHE tidak dipakai sebagai preprocessing tetap.
Di aplikasi ia tetap tersedia sebagai centang opsional yang menjalankan deteksi
kedua pada gambar yang kontrasnya disetarakan, lalu menampilkan selisih
hitungannya berdampingan.

Bedanya penting. Pengguna melihat efeknya secara sadar, dan aplikasi menyatakan
terang-terangan bahwa model dilatih tanpa CLAHE sehingga deteksi yang bertambah
belum tentu deteksi yang lebih benar. Angka resmi tetap yang dari gambar asli.

### Panel keterbatasan model ada di dalam aplikasi

`tampilan.panel_model` membaca `laporan/v1_baseline_640_catatan.json` dan
menampilkan mAP, tabel per kelas, dan satu peringatan yang menyebut angka
terburuknya, yaitu recall `no-helmet` 0,333.

Aplikasi yang menjatuhkan vonis tentang orang wajib menyatakan seberapa bisa ia
dipercaya, dan angka terburuknya justru yang paling perlu terlihat.

### Yang diverifikasi, bukan diasumsikan

`use_container_width` sudah dinyatakan usang oleh Streamlit dan tenggat
penghapusannya 31 Desember 2025, sudah lewat. Seluruh pemakaiannya diganti
menjadi `width="stretch"`.

Peringatan itu dikirim lewat logger Streamlit, bukan lewat modul `warnings`
Python, sehingga pemeriksaan yang hanya menangkap `warnings` akan melaporkan
bersih padahal tidak. Alat pemeriksanya dikalibrasi dulu pada kode yang sengaja
dibuat salah, dipastikan berbunyi di sana, baru dipakai memeriksa aplikasi ini.

| Uji | Hasil |
|---|---|
| `AppTest` jalur tanpa unggahan | tanpa exception, tanpa peringatan |
| Seluruh komponen tampilan dengan data nyata | tanpa exception, tanpa peringatan |
| Elemen yang benar-benar dirender | 10 dataframe, 27 metric, 2 banner merah, 1 banner hijau, 1 tombol unduh |
| `tests/test_tampilan.py` | 16 uji lolos |

Komponen yang memanggil `st.image` dan `st.download_button` sengaja diuji lewat
harness terpisah, karena jalur tanpa unggahan tidak pernah menyentuh keduanya
sehingga pemeriksaan `AppTest` biasa akan melewatkannya.

### Gambar contoh untuk demo

Tiga berkas di `contoh_gambar/`, seluruhnya dari test split. Yang ketiga adalah
yang paling layak ditunjukkan di video, tiga belas orang yang di mata manusia
jelas memakai APD lengkap tapi hanya empat yang bisa dipastikan model. Dua angka
kepatuhannya 23,1 dan 75,0 persen, dan jarak itu adalah ukuran kerusakan yang
dicegah status ketiga. Rinciannya di `contoh_gambar/README.md`.

## 8. Keterbatasan yang diketahui

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

### Yang baru diketahui setelah empat eksperimen

**Kelas `no-helmet` tetap yang terlemah, dan tidak ada yang benar-benar
menyembuhkannya.** Oversampling menaikkan recall dari 0,333 ke 0,458, dari 8
menjadi 11 dari 24 pelanggaran helm di test set. Jadi model masih melewatkan
lebih dari separuhnya. Varian model lebih besar justru mengembalikannya ke
0,333. Akar masalahnya jumlah contoh latih, bukan kapasitas model, dan itu
tidak bisa diselesaikan dari sisi training.

**Test setnya terlalu kecil untuk memisahkan perbaikan dari kebetulan.** Selang
kepercayaan 95 persen untuk ketiga perbaikan utama masih bertumpang tindih
dengan baseline, tabelnya di bagian 5. Kesimpulan bahwa v3 lebih baik ditopang
oleh konsistensi arah di banyak ukuran, bukan oleh satu angka yang meyakinkan
sendirian.

**Model terpilih melempar sepertiga pekerja ke pemeriksaan manusia.** Tepatnya
33,5 persen, naik dari 23,5 persen pada baseline. Sistem ini mengurangi tuduhan
palsu dengan cara mengaku tidak tahu lebih sering, dan itu berarti beban
pemeriksaan manusia bertambah. Rinciannya di bagian 6.

**Resolusi 960 menggandakan waktu inference.** Dari median 26 ms menjadi 51 ms
di CPU. Masih nyaman, tapi kalau suatu saat aplikasi ini diberi masukan video
atau batch besar, angka itu yang pertama menjadi penghalang.

**Aplikasi tidak pernah diuji pada foto di luar dataset ini.** Seluruh angka di
README berasal dari test split Roboflow yang sama sumbernya dengan train.
Perilakunya pada foto lokasi konstruksi Indonesia, dengan seragam dan warna
rompi yang berbeda, belum diketahui sama sekali.

## 9. Cara menjalankan ulang

Urutannya begini, dan tiap langkah berdiri sendiri.

```
notebooks/01_eda_dataset.ipynb     pemeriksaan data             [selesai]
notebooks/02_training.ipynb        baseline di Google Colab     [selesai]
notebooks/03_eksperimen.ipynb      empat run dan pemilihan      [selesai]
notebooks/04_evaluasi_final.ipynb  confusion matrix dan kurva   [belum]
skrip/validasi_analitik.py         vonis dibanding ground truth [selesai]
app.py                             aplikasi Streamlit           [jalan]
```

**EDA.** `01_eda_dataset.ipynb` mencari zip dataset di tiga lokasi, Google Drive
di `MyDrive/capstone4/`, direktori kerja, dan folder `pilihan dataset dan aturan`.
Tidak butuh GPU. Hasilnya disimpan ke `laporan/eda_ringkasan.json`.

**Training baseline.** `02_training.ipynb` dijalankan di Colab dengan GPU T4.
Unggah zip dataset ke `MyDrive/capstone4/` lebih dulu. Keluarannya bobot dan
berkas catatan versi, keduanya tersimpan ke Drive.

**Eksperimen.** `03_eksperimen.ipynb` menjalankan empat run, masing-masing satu
perubahan, lalu menilai keduanya dengan mAP dan dengan vonis per pekerja. Ia
mengkloning repo ini supaya memakai `src/analitik.py` yang sama dengan
aplikasi, bukan salinan yang bisa menyimpang. Tiap run menyimpan catatannya ke
Drive dan dilewati kalau catatannya sudah ada, jadi sesi Colab yang mati di
tengah jalan tidak memaksa mengulang dari awal.

**Validasi lapisan analisis.** Butuh dataset terekstrak di komputer sendiri.

```bash
.venv-uji/bin/python skrip/validasi_analitik.py /jalur/ke/dataset
```

Tanpa argumen kedua, ia memakai bobot pertama di `models/`, yang juga menjadi
pilihan bawaan aplikasi, dan resolusinya diambil dari catatan versi bobot itu.
Hasilnya ditulis ke `laporan/validasi_analitik.json`.

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
├── vendor/
│   └── opencv-python-stub/ pengalih ke headless, tanpa kode  [selesai]
├── models/
│   ├── apd_v3_oversample_nohelmet.pt  5,54 MB, dipakai  [selesai]
│   └── apd_v1_baseline_640.pt  5,47 MB, pembanding  [selesai]
├── src/
│   ├── detector.py         pemuatan model dan inference   [selesai]
│   ├── analitik.py         logika analisis, tanpa impor Streamlit  [selesai]
│   └── tampilan.py         komponen UI, menggambar dan merender  [selesai]
├── tests/
│   ├── test_detector.py    7 uji pemuatan dan pracitra    [selesai]
│   ├── test_lingkungan.py  6 uji susunan dependensi       [selesai]
│   ├── test_analitik.py    19 uji lapisan analisis        [selesai]
│   ├── test_tampilan.py    16 uji komponen tampilan       [selesai]
│   └── test_eksperimen.py  13 uji perkakas eksperimen     [selesai]
├── skrip/
│   ├── validasi_analitik.py  validasi asosiasi di test set  [selesai]
│   └── eksperimen.py       oversample dan aturan pemilihan model  [selesai]
├── notebooks/
│   ├── 01_eda_dataset.ipynb  lima pemeriksaan data       [selesai]
│   ├── 02_training.ipynb     training baseline, berisi output  [selesai]
│   ├── 03_eksperimen.ipynb   empat run hari 8 sampai 10  [selesai]
│   └── 04_evaluasi_final.ipynb  confusion matrix dan kurva PR  [belum]
├── contoh_gambar/          tiga gambar demo dari test split  [selesai]
└── laporan/
    ├── eda_ringkasan.json  angka EDA, dikutip README      [selesai]
    ├── v1..v5_catatan.json  catatan lima run training     [selesai]
    ├── perbandingan_run.json  tabel dan alasan pemilihan  [selesai]
    └── validasi_analitik.json  angka validasi asosiasi    [selesai]
```

`src/analitik.py` sengaja tidak mengimpor Streamlit maupun Ultralytics, supaya
logikanya bisa diuji tanpa GPU dan tanpa menjalankan aplikasi.

Hanya dua bobot yang masuk repo, model terpilih dan baseline sebagai
pembanding. Tiga bobot lain hasil eksperimen, yaitu `v2`, `v4`, dan `v5`,
disimpan di luar repo pada `../hasil-colab/bobot/`, karena berjumlah 30 MB dan
tidak dipakai aplikasi. Catatan versi kelimanya tetap ada di `laporan/`, sebab
angka v4 dan v5 dikutip bagian 5 sebagai hasil negatif.

## Catatan versi

`requirements.txt` diuji hari 3 di virtualenv bersih, bukan diasumsikan.

| Paket | Versi | Alasan |
|---|---|---|
| torch | `==2.11.0` | sama dengan versi di Colab |
| torchvision | `==0.26.0` | pasangan torch 2.11.0 |
| ultralytics | `==8.4.138` | **wajib sama** dengan versi saat training, berkas bobot menyimpan referensi kelas Python |
| `./vendor/opencv-python-stub` | 4.99.0 | paket lokal tanpa kode, mencegah `opencv-python` asli terunduh |
| opencv-python-headless | `>=4.10,<5` | **menggantikan `packages.txt`**, pustaka grafis datang dari wheel bukan dari apt |
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

Semuanya di `python:3.14-slim`, image yang **tidak punya satu pun** dari
`libGL.so.1`, `libxcb.so.1`, dan `libgthread-2.0.so.0`. Image itu lebih ramping
daripada image Streamlit Cloud, jadi yang lolos di sana pasti lolos di Cloud.

| Uji | pip | uv |
|---|---|---|
| `install -r requirements.txt` di Python 3.14 | berhasil | berhasil |
| `opencv-python` yang terpasang | pengalih 4.99.0 | pengalih 4.99.0 |
| `import cv2` | berhasil | berhasil |
| `import torch, ultralytics, streamlit` | berhasil | berhasil |
| `tests/test_lingkungan.py` | enam lulus | enam lulus |
| `tests/test_detector.py` | empat lulus | empat lulus |
| `AppTest.from_file("app.py").run()` | tanpa exception | tanpa exception |
| peringatan deprecation | tidak ada | tidak ada |

Sebelumnya juga diverifikasi di `debian:trixie` bahwa `use_container_width`
masih diterima `st.image` dan `st.dataframe` pada streamlit 1.63.0, dan bahwa
`streamlit run app.py` menjawab HTTP 200.

### Dua kali salah pin, dua kali ketahuan sebelum deploy

**Hari 1.** `torch==2.5.1` tidak punya wheel untuk Python 3.13, install gagal
total.

**Hari 3, setelah training.** Colab ternyata memasang `ultralytics 8.4.138` dan
`torch 2.11.0`, bukan 8.3.217 dan 2.8.0 yang saya tulis. Berkas bobot dibuat
oleh 8.4.138, jadi pin lama berisiko gagal memuatnya di aplikasi. Angka di tabel
atas diambil dari keluaran notebook, bukan dari perkiraan.

Keduanya jenis kegagalan yang tidak berbunyi sampai deploy. Itu alasan uji
pasang dijadwalkan hari 3, bukan di akhir.

### Kenapa `packages.txt` dibuang seluruhnya

Aplikasi ini pernah punya `packages.txt` berisi `libgl1` dan `libglib2.0-0t64`.
Berkas itu sekarang tidak ada, dan ketiadaannya disengaja.

Tiga deploy gagal berturut turut karena berkas itu, masing-masing dengan sebab
yang berbeda.

| Percobaan | Galat | Sebab |
|---|---|---|
| 1 | `libglib2.0-0 : Depends: libffi7 but it is not installable` | image Cloud memakai Debian 13 trixie, di sana nama itu tinggal nama virtual sisa transisi time_t 64-bit. Yang benar `libglib2.0-0t64` |
| 2 | `E: Unable to locate package NAMA` berkali-kali | `packages.txt` tidak mengenal komentar, setiap baris dikirim apa adanya ke `apt-get install` |
| 3 | `E: Release file for ... bullseye-security is expired` | sumber apt bullseye yang tertinggal di image Cloud kedaluwarsa 8 September 2026, `apt-get update` mengembalikan exit code non-nol, dan build berhenti sebelum satu paket pun sempat dicoba |

Kegagalan ketiga tidak bisa diperbaiki dari sisi repo. Ia ada di base image
Streamlit, dan menimpa semua aplikasi yang punya `packages.txt`, bukan hanya
yang ini.

Karena itu jalan keluarnya bukan menebak nama paket yang benar sekali lagi,
melainkan berhenti membutuhkan apt. `opencv-python-headless` memuat pustaka
grafisnya di dalam wheel sendiri. Begitu ia dipakai, `packages.txt` tidak punya
alasan untuk ada, dan seluruh kelas kegagalan ini hilang, termasuk yang belum
terjadi.

Menukar apt dengan headless ternyata tidak sesederhana mengganti satu baris.
Ultralytics tetap menarik `opencv-python` sebagai dependensi, dan kedua paket
memasang paket Python bernama `cv2` ke lokasi yang sama sehingga saling
menimpa. Yang dipasang belakangan yang menang.

Tiga susunan diuji di container tanpa pustaka grafis sama sekali.

| Susunan `requirements.txt` | pip | uv |
|---|---|---|
| hanya `opencv-python-headless` | gagal | gagal |
| `opencv-python` lalu `opencv-python-headless`, keduanya tingkat atas | **gagal** | berhasil |
| pengalih lokal plus `opencv-python-headless` | **berhasil** | **berhasil** |

Baris kedua yang paling penting dipahami. Ia berhasil dengan uv dan gagal
dengan pip, jadi memakainya berarti aplikasi hidup atau mati tergantung
resolver mana yang kebetulan dipakai build itu. Streamlit Cloud memakai
keduanya, uv lebih dulu lalu pip sebagai cadangan, sehingga susunan itu tidak
bisa diandalkan.

Yang dipakai susunan ketiga. `vendor/opencv-python-stub` adalah paket lokal di
dalam repo ini yang mengaku bernama `opencv-python`, tidak berisi kode apa pun,
dan hanya membawa satu dependensi ke headless. Resolver menganggap kebutuhan
ultralytics sudah terpenuhi sehingga **paket aslinya tidak pernah diunduh dari
PyPI**. Tidak ada saingan, jadi tidak ada urutan yang perlu diandalkan.

Alasan lengkapnya di `vendor/opencv-python-stub/README.md`, dan
`tests/test_lingkungan.py` menjaga susunannya tidak dirapikan tanpa sengaja.

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
