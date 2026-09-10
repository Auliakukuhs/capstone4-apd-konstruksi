# Gambar contoh untuk demo

Tiga berkas ini diambil dari **test split**, jadi model tidak pernah melihatnya
saat training maupun saat memilih checkpoint. Ketiganya dipilih dengan sengaja,
masing-masing memperlihatkan satu perilaku sistem yang berbeda.

Angka di bawah diukur pada model terpilih `apd_v3_oversample_nohelmet.pt`,
conf 0,25, IoU NMS 0,70, imgsz 960. Kalau parameternya digeser di aplikasi,
angkanya ikut berubah.

## 1. `01_pelanggaran_jelas.jpg`

600 x 399. Delapan pekerja terdeteksi, enam melanggar, dua belum dapat
dipastikan. Kepatuhan 0,0 persen, banner merah.

Yang layak ditunjuk saat demo adalah pekerja berkaus putih di kanan, yang jelas
tidak memakai rompi dan memang divonis begitu, dengan kotak `no-vest` sebagai
buktinya.

## 2. `02_seluruhnya_lengkap.jpg`

615 x 409. Enam pekerja, seluruhnya lengkap. Kepatuhan 100,0 persen, banner
hijau.

Gunanya memperlihatkan bahwa pernyataan aman memang bisa muncul, sehingga
banner merah pada gambar lain bukan sekadar keadaan bawaan sistem.

## 3. `03_banyak_belum_dapat_dipastikan.jpg`

1024 x 483. Tiga belas pekerja, enam lengkap, satu melanggar, enam belum dapat
dipastikan, dan satu kotak helm tanpa pekerja.

**Ini gambar terpenting dari ketiganya**, dan sekarang ia memperlihatkan tiga
hal sekaligus.

### Dua angka kepatuhan yang berjauhan

| Angka | Nilai | Artinya |
|---|---|---|
| Kepatuhan | 46,2 persen | 6 dari 13, yang belum pasti ikut penyebut |
| Kepatuhan yang terbaca | 85,7 persen | 6 dari 7, hanya yang statusnya terbaca |

Kalau sistem ini hanya punya dua status, enam orang yang sebenarnya patuh akan
tercatat sebagai pelanggar dan angka 46,2 persen akan dilaporkan sebagai fakta.
Jarak antara 46,2 dan 85,7 adalah ukuran kerusakan yang dicegah status ketiga.

### Satu tuduhan palsu, terlihat langsung di layar

Pekerja **#12 divonis TIDAK LENGKAP** karena mendapat kotak `no-helmet` dengan
confidence 0,85. Lihat fotonya, orang itu memakai helm putih.

Lalu perhatikan kotak ungu di sebelahnya, `helmet` dengan confidence 0,91 yang
**tidak dapat dikaitkan ke satu pekerja pun**. Kemungkinan besar helm itu
miliknya, dan `no-helmet` itu deteksi palsu.

Jangan tutupi bagian ini saat merekam video. Justru inilah yang membuat laju
tuduhan palsu 2,0 persen di README menjadi angka yang berarti, bukan klaim.
Dan perhatikan bahwa sistem melaporkan kejanggalannya sendiri, lewat kotak
ungu dan lewat kolom bukti di tabel, sehingga pengawas punya alasan untuk
meragukan vonis itu tanpa harus mempercayai atau menolaknya buta.

### Perbedaan terhadap baseline

Dengan `v1_baseline_640`, gambar yang sama menghasilkan 3 lengkap, 1 melanggar,
dan **9 belum dapat dipastikan**. Model terpilih memastikan enam orang lagi.
Ini contoh konkret kenaikan recall yang di README tampil sebagai angka.

## Catatan pencahayaan

Luminansi ketiganya 147, 119, dan 64 dari 255. Ketiganya di dalam rentang 60
sampai 200 yang dikenal model, jadi tidak ada yang memicu peringatan
pencahayaan. Gambar ketiga foto malam berlampu sorot dan nilainya 64, tepat di
atas batas bawah.

Kalau ingin memperlihatkan peringatan itu bekerja saat merekam, unggah foto
yang lebih gelap dari ini.
