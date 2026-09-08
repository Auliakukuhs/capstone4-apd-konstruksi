"""Lapisan analisis. Mengubah daftar kotak menjadi vonis per pekerja.

Modul ini sengaja tidak mengimpor Streamlit maupun Ultralytics, supaya
logikanya bisa diuji tanpa GPU, tanpa bobot model, dan tanpa menjalankan
aplikasi. Masukannya cuma daftar dict hasil `src.detector.ke_deteksi`.

Tiga keputusan yang mendasari seluruh berkas ini, semuanya diambil dari
pengukuran di `catatan/98-eda-dataset.md`, bukan dari perkiraan.

**Asosiasi memakai IoA, bukan IoU.** Kotak helmet jauh lebih kecil daripada
kotak person, sehingga IoU-nya tetap kecil meski helmet berada sepenuhnya di
dalam person. IoA membagi luas irisan dengan luas kotak atribut saja, jadi
helmet yang termuat penuh bernilai 1,0. Diuji pada ground truth, ambang 0,5
memetakan 94,5 persen helmet dan 98,7 persen vest ke tepat satu person.

**Ada status ketiga.** Pada ground truth, 11,8 persen kotak person tidak punya
kotak helmet sama sekali, bahkan pada anotasi manusia. Jadi tidak adanya
deteksi helmet bukan bukti pekerja tidak memakai helm. Menyamakan keduanya
memproduksi tuduhan palsu terhadap orang yang sebenarnya patuh.

**Hitungan total tidak dipakai untuk menyimpulkan pelanggaran.** Pada ground
truth, `vest` ditambah `no-vest` hanya 79,3 persen dari jumlah `person`, jadi
pengurangan sederhana melebih-lebihkan pelanggaran vest sekitar 20 persen
bahkan pada anotasi sempurna. Lagi pula pengurangan tidak bisa menjawab
pertanyaan yang benar-benar berguna di lapangan, yaitu pekerja yang mana.
"""

from dataclasses import dataclass, field

PERSON = "person"
HELMET = "helmet"
NO_HELMET = "no-helmet"
VEST = "vest"
NO_VEST = "no-vest"

# Urutan kelas mengikuti data.yaml dataset. Perhatikan person ada di indeks 3.
KELAS = (HELMET, NO_HELMET, NO_VEST, PERSON, VEST)

MEMAKAI = "memakai"
TIDAK_MEMAKAI = "tidak memakai"
BELUM_PASTI = "belum dapat dipastikan"

LENGKAP = "LENGKAP"
TIDAK_LENGKAP = "TIDAK LENGKAP"
PERLU_DIPERIKSA = "BELUM DAPAT DIPASTIKAN"

# Ambang IoA. Divalidasi pada ground truth train dan valid.
IOA_MIN = 0.5

# Median posisi vertikal pusat helmet di dalam kotak person, 0 puncak kepala
# dan 1 telapak kaki. Dipakai hanya saat satu helmet cocok ke lebih dari satu
# person, yang terjadi pada 1,8 persen kasus.
KEDALAMAN_HELMET = 0.09


def ioa(kotak_kecil, kotak_besar) -> float:
    """Porsi luas `kotak_kecil` yang berada di dalam `kotak_besar`.

    Format xyxy. Nilainya 1,0 kalau kotak kecil termuat seluruhnya, tidak
    peduli seberapa besar selisih ukuran keduanya. Itulah bedanya dengan IoU,
    yang membagi dengan gabungan luas sehingga tertekan oleh kotak besar.
    """
    ax1, ay1, ax2, ay2 = kotak_kecil
    bx1, by1, bx2, by2 = kotak_besar

    lebar_irisan = min(ax2, bx2) - max(ax1, bx1)
    tinggi_irisan = min(ay2, by2) - max(ay1, by1)
    if lebar_irisan <= 0 or tinggi_irisan <= 0:
        return 0.0

    luas_kecil = (ax2 - ax1) * (ay2 - ay1)
    if luas_kecil <= 0:
        return 0.0

    return (lebar_irisan * tinggi_irisan) / luas_kecil


def kedalaman_relatif(kotak_kecil, kotak_besar) -> float:
    """Posisi vertikal pusat `kotak_kecil` di dalam `kotak_besar`, 0 sampai 1."""
    cy = (kotak_kecil[1] + kotak_kecil[3]) / 2
    y1, y2 = kotak_besar[1], kotak_besar[3]
    if y2 <= y1:
        return 9.9  # kotak person tidak punya tinggi, jadikan kandidat terburuk
    return (cy - y1) / (y2 - y1)


@dataclass
class Atribut:
    """Satu kotak helmet, no-helmet, vest, atau no-vest."""

    cls: str
    conf: float
    box: tuple
    ioa: float = 0.0  # terhadap pekerja yang memenanginya


@dataclass
class Pekerja:
    """Satu kotak person beserta atribut yang berhasil diasosiasikan padanya."""

    nomor: int
    box: tuple
    conf: float
    helmet: str = BELUM_PASTI
    vest: str = BELUM_PASTI
    bukti: list = field(default_factory=list)
    konflik: list = field(default_factory=list)

    @property
    def vonis(self) -> str:
        """Pelanggaran yang terbukti selalu menang atas ketidakpastian.

        Satu atribut yang jelas tidak dipakai sudah cukup untuk menyatakan
        tidak lengkap, meski atribut lain belum dapat dipastikan. Sebaliknya
        LENGKAP hanya diberikan kalau keduanya benar-benar terlihat.
        """
        if TIDAK_MEMAKAI in (self.helmet, self.vest):
            return TIDAK_LENGKAP
        if self.helmet == MEMAKAI and self.vest == MEMAKAI:
            return LENGKAP
        return PERLU_DIPERIKSA

    @property
    def alasan(self) -> str:
        """Kalimat pendek yang bisa ditempel di samping vonis."""
        kurang = []
        if self.helmet == TIDAK_MEMAKAI:
            kurang.append("helm")
        if self.vest == TIDAK_MEMAKAI:
            kurang.append("rompi")
        if kurang:
            return "tidak memakai " + " dan ".join(kurang)

        tak_terlihat = []
        if self.helmet == BELUM_PASTI:
            tak_terlihat.append("helm")
        if self.vest == BELUM_PASTI:
            tak_terlihat.append("rompi")
        if tak_terlihat:
            return " dan ".join(tak_terlihat) + " tidak terdeteksi, perlu diperiksa manusia"

        return "helm dan rompi terlihat"


@dataclass
class Ringkasan:
    pekerja: int = 0
    lengkap: int = 0
    tidak_lengkap: int = 0
    belum_pasti: int = 0
    tanpa_induk: dict = field(default_factory=dict)

    @property
    def kepatuhan(self):
        """Porsi pekerja yang terbukti lengkap, dari seluruh pekerja.

        Angka pesimistis, karena yang belum dapat dipastikan ikut menjadi
        penyebut. Mengembalikan None kalau tidak ada pekerja, bukan 0,0,
        supaya "tidak ada orang di gambar" tidak terbaca sebagai
        "kepatuhan nol persen".
        """
        if self.pekerja == 0:
            return None
        return self.lengkap / self.pekerja

    @property
    def kepatuhan_terperiksa(self):
        """Porsi lengkap di antara pekerja yang statusnya benar-benar terbaca.

        Ini angka yang adil dipakai membandingkan antar gambar, karena tidak
        menghukum gambar yang kebetulan banyak pekerjanya terhalang.
        """
        terbaca = self.lengkap + self.tidak_lengkap
        if terbaca == 0:
            return None
        return self.lengkap / terbaca


@dataclass
class HasilGambar:
    pekerja: list = field(default_factory=list)
    tanpa_induk: list = field(default_factory=list)
    hitung_kelas: dict = field(default_factory=dict)
    ringkasan: Ringkasan = field(default_factory=Ringkasan)


def hitung_kelas(deteksi) -> dict:
    """Hitungan mentah per kelas, termasuk yang nol.

    Kelas yang bernilai nol tetap ditampilkan. Kelas yang hilang dari daftar
    lebih sulit dibaca daripada kelas yang tertulis nol.
    """
    hitung = {k: 0 for k in KELAS}
    for d in deteksi:
        hitung[d["cls"]] = hitung.get(d["cls"], 0) + 1
    return hitung


def _induk_terbaik(atribut, daftar_pekerja, ioa_min):
    """Pekerja mana yang memiliki atribut ini. Mengembalikan (indeks, ioa)."""
    kandidat = []
    for i, p in enumerate(daftar_pekerja):
        skor = ioa(atribut["box"], p.box)
        if skor >= ioa_min:
            kandidat.append((skor, i))

    if not kandidat:
        return None, 0.0

    if len(kandidat) > 1 and atribut["cls"] in (HELMET, NO_HELMET):
        # Helmet yang termuat penuh di dua person bernilai IoA 1,0 pada
        # keduanya, jadi IoA tidak bisa memutuskan. Yang memutuskan posisi
        # vertikal. Pada ground truth, 98,8 persen pusat helmet berada di
        # 25 persen teratas kotak person, dengan median 0,09.
        kandidat.sort(
            key=lambda si: abs(
                kedalaman_relatif(atribut["box"], daftar_pekerja[si[1]].box)
                - KEDALAMAN_HELMET
            )
        )
    else:
        kandidat.sort(key=lambda si: si[0], reverse=True)

    skor, idx = kandidat[0]
    return idx, skor


def _putuskan(bukti, kelas_ya, kelas_tidak, nama):
    """Tentukan status satu jenis APD dari bukti yang terkumpul.

    Satu pekerja bisa menerima `helmet` dan `no-helmet` sekaligus kalau model
    ragu. Yang belakangan tidak boleh sekadar menimpa yang duluan, karena
    hasilnya lalu bergantung pada urutan deteksi. Yang menang confidence
    tertinggi, dan pertentangannya dicatat supaya bisa ditampilkan.
    """
    ya = [b for b in bukti if b.cls == kelas_ya]
    tidak = [b for b in bukti if b.cls == kelas_tidak]

    if not ya and not tidak:
        return BELUM_PASTI, []

    terbaik_ya = max((b.conf for b in ya), default=None)
    terbaik_tidak = max((b.conf for b in tidak), default=None)

    if terbaik_ya is not None and terbaik_tidak is not None:
        menang = MEMAKAI if terbaik_ya >= terbaik_tidak else TIDAK_MEMAKAI
        catatan = (
            f"{nama} berkonflik, {kelas_ya} {terbaik_ya:.2f} lawan "
            f"{kelas_tidak} {terbaik_tidak:.2f}, dimenangkan "
            f"{kelas_ya if menang == MEMAKAI else kelas_tidak}"
        )
        return menang, [catatan]

    if terbaik_ya is not None:
        return MEMAKAI, []
    return TIDAK_MEMAKAI, []


def asosiasi(deteksi, ioa_min: float = IOA_MIN) -> HasilGambar:
    """Petakan tiap atribut ke pekerja yang memakainya, lalu jatuhkan vonis.

    `deteksi` adalah daftar dict `{"cls": str, "conf": float, "box": tuple}`,
    yaitu keluaran `src.detector.ke_deteksi`.

    Atribut yang tidak cocok ke satu pekerja pun tidak dibuang diam-diam,
    melainkan dilaporkan sebagai objek tanpa induk. Jumlahnya adalah petunjuk
    langsung bahwa ada pekerja yang tidak terdeteksi, dan itu informasi yang
    berguna bagi pengawas, bukan sampah.
    """
    orang = [d for d in deteksi if d["cls"] == PERSON]
    atribut = [d for d in deteksi if d["cls"] != PERSON]

    daftar_pekerja = [
        Pekerja(nomor=i + 1, box=tuple(d["box"]), conf=d["conf"])
        for i, d in enumerate(sorted(orang, key=lambda d: d["box"][0]))
    ]
    tanpa_induk = []

    for a in atribut:
        idx, skor = _induk_terbaik(a, daftar_pekerja, ioa_min)
        satu = Atribut(cls=a["cls"], conf=a["conf"], box=tuple(a["box"]), ioa=skor)
        if idx is None:
            tanpa_induk.append(satu)
        else:
            daftar_pekerja[idx].bukti.append(satu)

    for p in daftar_pekerja:
        p.helmet, konflik_helm = _putuskan(p.bukti, HELMET, NO_HELMET, "helm")
        p.vest, konflik_rompi = _putuskan(p.bukti, VEST, NO_VEST, "rompi")
        p.konflik = konflik_helm + konflik_rompi

    yatim_per_kelas = {}
    for a in tanpa_induk:
        yatim_per_kelas[a.cls] = yatim_per_kelas.get(a.cls, 0) + 1

    ringkasan = Ringkasan(
        pekerja=len(daftar_pekerja),
        lengkap=sum(1 for p in daftar_pekerja if p.vonis == LENGKAP),
        tidak_lengkap=sum(1 for p in daftar_pekerja if p.vonis == TIDAK_LENGKAP),
        belum_pasti=sum(1 for p in daftar_pekerja if p.vonis == PERLU_DIPERIKSA),
        tanpa_induk=yatim_per_kelas,
    )

    return HasilGambar(
        pekerja=daftar_pekerja,
        tanpa_induk=tanpa_induk,
        hitung_kelas=hitung_kelas(deteksi),
        ringkasan=ringkasan,
    )


def baris_csv(hasil: HasilGambar, nama_gambar: str = "") -> list:
    """Satu baris per pekerja, siap diekspor.

    Kolom bukti sengaja ikut, supaya angka yang melahirkan vonis bisa
    diperiksa ulang tanpa membuka aplikasinya lagi.
    """
    baris = []
    for p in hasil.pekerja:
        bukti = "; ".join(f"{b.cls} conf {b.conf:.2f} ioa {b.ioa:.2f}" for b in p.bukti)
        baris.append(
            {
                "gambar": nama_gambar,
                "pekerja": p.nomor,
                "vonis": p.vonis,
                "helm": p.helmet,
                "rompi": p.vest,
                "alasan": p.alasan,
                "conf_pekerja": round(p.conf, 3),
                "box": " ".join(str(round(v, 1)) for v in p.box),
                "bukti": bukti,
                "konflik": "; ".join(p.konflik),
            }
        )
    return baris


def _persen(nilai):
    return "tidak dapat dihitung" if nilai is None else f"{nilai * 100:.1f} persen"


def laporan_teks(hasil: HasilGambar) -> str:
    """Ringkasan datar untuk notebook, CLI, dan pengecekan mata."""
    r = hasil.ringkasan
    baris = [
        "RINGKASAN KEPATUHAN APD",
        f"Pekerja terdeteksi          {r.pekerja}",
        f"  Lengkap                   {r.lengkap}",
        f"  Tidak lengkap             {r.tidak_lengkap}",
        f"  Belum dapat dipastikan    {r.belum_pasti}",
        f"Kepatuhan                   {_persen(r.kepatuhan)}"
        f"  ({r.lengkap} dari {r.pekerja})",
        f"Kepatuhan yang terbaca      {_persen(r.kepatuhan_terperiksa)}"
        f"  ({r.lengkap} dari {r.lengkap + r.tidak_lengkap})",
    ]

    if hasil.tanpa_induk:
        rincian = ", ".join(f"{n} {k}" for k, n in sorted(r.tanpa_induk.items()))
        baris.append(f"Atribut tanpa pekerja       {rincian}")

    if hasil.pekerja:
        baris.append("")
        baris.append("Rincian per pekerja")
        for p in hasil.pekerja:
            baris.append(
                f"  #{p.nomor}  helm {p.helmet:<22} rompi {p.vest:<22} {p.vonis}"
            )
            for catatan in p.konflik:
                baris.append(f"        {catatan}")

    return "\n".join(baris)


class Sesi:
    """Akumulasi lintas gambar dalam satu sesi pemakaian.

    Dipakai panel dashboard di aplikasi. Disimpan di sini, bukan di `app.py`,
    supaya perhitungannya ikut teruji dan tidak bergantung pada Streamlit.
    """

    def __init__(self):
        self.gambar = []

    def tambah(self, nama: str, hasil: HasilGambar) -> None:
        self.gambar.append((nama, hasil))

    def total(self) -> Ringkasan:
        gabungan = Ringkasan()
        for _, h in self.gambar:
            gabungan.pekerja += h.ringkasan.pekerja
            gabungan.lengkap += h.ringkasan.lengkap
            gabungan.tidak_lengkap += h.ringkasan.tidak_lengkap
            gabungan.belum_pasti += h.ringkasan.belum_pasti
            for k, n in h.ringkasan.tanpa_induk.items():
                gabungan.tanpa_induk[k] = gabungan.tanpa_induk.get(k, 0) + n
        return gabungan

    def semua_baris(self) -> list:
        baris = []
        for nama, h in self.gambar:
            baris.extend(baris_csv(h, nama))
        return baris

    def __len__(self) -> int:
        return len(self.gambar)
