# Gambar contoh untuk demo

Tiga berkas ini diambil dari **test split**, jadi model tidak pernah melihatnya
saat training maupun saat memilih checkpoint. Ketiganya dipilih dengan sengaja,
masing-masing memperlihatkan satu perilaku sistem yang berbeda.

Angka di bawah diukur pada `apd_v1_baseline_640.pt`, conf 0,25, IoU NMS 0,70,
imgsz 640. Kalau parameternya digeser di aplikasi, angkanya ikut berubah.

## 1. `01_pelanggaran_jelas.jpg`

600 x 399. Sembilan pekerja, enam melanggar, tiga belum dapat dipastikan.
Kepatuhan 0,0 persen.

Banner merah, dan kotak merah di hampir seluruh orang. Yang layak ditunjuk saat
demo adalah pekerja berkaus putih di kanan, yang jelas tidak memakai rompi dan
memang divonis begitu, dengan kotak `no-vest` sebagai buktinya.

## 2. `02_seluruhnya_lengkap.jpg`

615 x 409. Enam pekerja, seluruhnya lengkap. Kepatuhan 100,0 persen.

Banner hijau. Gunanya memperlihatkan bahwa pernyataan aman memang bisa muncul,
sehingga banner merah pada gambar lain bukan sekadar keadaan bawaan sistem.

## 3. `03_banyak_belum_dapat_dipastikan.jpg`

1024 x 483. Tiga belas pekerja, tiga lengkap, satu melanggar, sembilan belum
dapat dipastikan, dan satu kotak helm tanpa pekerja.

**Ini gambar terpenting dari ketiganya.** Mata manusia melihat tiga belas orang
yang semuanya memakai helm putih dan rompi hijau menyala. Model hanya berhasil
memastikan empat di antaranya. Sembilan sisanya tidak divonis melanggar,
melainkan ditandai perlu diperiksa.

Dua angka kepatuhannya berjauhan, dan justru itu isinya.

| Angka | Nilai | Artinya |
|---|---|---|
| Kepatuhan | 23,1 persen | 3 dari 13, yang belum pasti ikut penyebut |
| Kepatuhan yang terbaca | 75,0 persen | 3 dari 4, hanya yang statusnya terbaca |

Kalau sistem ini hanya punya dua status, sembilan orang yang sebenarnya patuh
akan tercatat sebagai pelanggar, dan angka kepatuhan 23,1 persen akan
dilaporkan sebagai fakta. Jarak antara 23,1 dan 75,0 adalah ukuran seberapa
besar kerusakan yang dicegah oleh status ketiga.

Ada juga satu kotak helm ungu tanpa pekerja di sekitar orang ke-12. Itu
petunjuk bahwa ada orang di belakang yang tidak terdeteksi, dan sistem
melaporkannya alih-alih membuangnya diam-diam.

## Catatan pencahayaan

Luminansi ketiganya 147, 119, dan 64 dari 255. Ketiganya berada di dalam
rentang 60 sampai 200 yang dikenal model, jadi tidak ada yang memicu peringatan
pencahayaan. Gambar ketiga adalah foto malam berlampu sorot dan nilainya 64,
tepat di atas batas bawah.

Kalau ingin memperlihatkan peringatan itu bekerja saat merekam video, unggah
foto yang lebih gelap dari ini.
