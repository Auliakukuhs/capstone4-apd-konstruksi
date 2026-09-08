# Pengalih `opencv-python`

Folder ini bukan library. Isinya satu berkas metadata yang mengaku bernama
`opencv-python` dan tidak membawa kode apa pun.

## Kenapa ada

`ultralytics` mewajibkan `opencv-python`. Paket itu menautkan diri ke tiga
pustaka grafis sistem, `libGL.so.1`, `libxcb.so.1`, dan
`libgthread-2.0.so.0`. Ketiganya tidak ada di image Streamlit Community Cloud.

Dulu ketiganya dipasang lewat `packages.txt`, dan itu menyebabkan tiga deploy
gagal berturut turut. Yang terakhir tidak bisa diperbaiki dari sisi repo sama
sekali, karena sumber apt bullseye yang tertinggal di image Streamlit
kedaluwarsa pada 8 September 2026 sehingga `apt-get update` mengembalikan exit
code non-nol dan seluruh build berhenti sebelum satu paket pun dicoba dipasang.

`opencv-python-headless` memuat pustaka grafisnya di dalam wheel sendiri,
sehingga apt tidak dibutuhkan. Masalahnya ia tidak bisa dipakai begitu saja.

## Kenapa tidak cukup menulis headless di requirements

Kedua paket memasang paket Python bernama `cv2` ke lokasi yang sama dan saling
menimpa. Yang dipasang belakangan yang menang. Diuji di container
`python:3.14-slim`, image tanpa satu pun pustaka grafis sistem.

| Susunan requirements | pip | uv |
|---|---|---|
| hanya `opencv-python-headless` | gagal | gagal |
| `opencv-python` lalu `opencv-python-headless`, keduanya tingkat atas | **gagal** | berhasil |
| pengalih ini plus `opencv-python-headless` | **berhasil** | **berhasil** |

Baris kedua yang menentukan keputusan. Bertumpu pada urutan berarti aplikasi
hidup atau mati tergantung resolver mana yang dipakai build itu, dan Streamlit
Cloud memakai keduanya, uv lebih dulu lalu pip sebagai cadangan.

## Cara kerjanya

`requirements.txt` menyebut `./vendor/opencv-python-stub` sebagai pemenuhan
kebutuhan `opencv-python`. Resolver melihat nama itu sudah terpenuhi oleh
paket lokal versi 4.99.0, yang memenuhi syarat ultralytics
`opencv-python!=4.13.0.90,>=4.7.0`, jadi ia tidak pernah menghubungi PyPI untuk
paket itu.

Satu satunya penyedia berkas `cv2` menjadi `opencv-python-headless`, tanpa
saingan dan tanpa bergantung pada urutan.

## Kalau suatu saat tidak dibutuhkan lagi

Begitu ultralytics melonggarkan kebutuhannya sehingga menerima headless
langsung, folder ini bisa dihapus dan barisnya dicabut dari `requirements.txt`.
`tests/test_lingkungan.py` akan memberi tahu kalau ada yang menghapusnya
sebelum waktunya.
