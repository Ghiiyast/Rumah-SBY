import logging
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

# Folder utama proyek (tempat file .env berada)
BASE_DIR = Path(__file__).resolve().parent.parent

# Membaca file .env (tempat menyimpan API key, TIDAK ikut ke GitHub)
load_dotenv(BASE_DIR / ".env")

URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
logger = logging.getLogger(__name__)

# Nama model bisa diganti lewat .env (GROQ_MODEL) tanpa mengubah kode
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

INSTRUKSI = (
    "Kamu adalah konsultan properti di Surabaya yang ramah. Jelaskan grafik ini dalam "
    "bahasa Indonesia sehari-hari yang mengalir, jelas, dan mudah dipahami oleh orang awam "
    "dalam 1 paragraf utuh (sekitar 4 sampai 5 kalimat). Sebutkan perbandingan angka "
    "paling tinggi dan paling rendah hanya berdasarkan data yang diberikan, lalu jelaskan "
    "secara sederhana gambaran pasar yang terlihat dari grafik tersebut tanpa mengarang "
    "angka baru. Ingat bahwa 'M' artinya miliar rupiah."
)
INSTRUKSI_PREDIKSI = (
    "Kamu adalah asisten penjelas estimasi properti di Surabaya. Jelaskan hasil estimasi "
    "dalam bahasa Indonesia yang ramah dan mudah dipahami, sekitar 2 sampai 4 kalimat. "
    "Gunakan hanya fakta pada data yang diberikan. Jelaskan bahwa nilai adalah estimasi "
    "model, bukan harga transaksi atau penilaian resmi. Jangan mengarang faktor penting "
    "model, angka, atau kesimpulan yang tidak didukung data."
)
_simpan = {}


def tanya_groq(topik, fakta, instruksi=INSTRUKSI):
    """Meminta Groq menjelaskan data. Hasilnya teks, atau None kalau gagal."""
    kunci = os.environ.get("GROQ_API_KEY")
    if not kunci:
        logger.warning("GROQ_API_KEY belum dikonfigurasi; memakai ringkasan data.")
        return None

    cache_key = (topik, fakta, instruksi)
    if cache_key in _simpan:
        return _simpan[cache_key]

    try:
        respons = requests.post(
            URL_GROQ,
            headers={"Authorization": f"Bearer {kunci}"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": instruksi},
                    {"role": "user", "content": f"Topik: {topik}\nData:\n{fakta}"},
                ],
                "temperature": 0.3,
                "max_tokens": 3000,
                "reasoning_effort": "low",
            },
            timeout=10,
        )
        respons.raise_for_status()
    except requests.RequestException:
        logger.exception("Permintaan penjelasan AI gagal.")
        return None

    try:
        teks = respons.json()["choices"][0]["message"]["content"].strip()
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        logger.exception("Respons penjelasan AI tidak memiliki format yang valid.")
        return None

    if not teks:
        logger.warning("AI mengembalikan penjelasan kosong untuk grafik %s.", topik)
        return None

    _simpan[cache_key] = teks
    return teks


def _hasil_penjelasan(topik, fakta, ringkasan, ada_data):
    teks_ai = tanya_groq(topik, fakta) if ada_data else None
    return {
        "teks": teks_ai or ringkasan,
        "ai": bool(teks_ai),
    }


def penjelasan_dashboard(grafik):
    """Membuat penjelasan AI untuk kedua grafik di dashboard."""
    # Fakta untuk grafik 1 dihitung di Python, jadi angkanya pasti benar
    label = grafik["sebaran"]["label"]
    nilai = grafik["sebaran"]["nilai"]
    total = sum(nilai)
    baris = (
        [f"- {l}: {n} rumah ({n / total:.0%})" for l, n in zip(label, nilai)]
        if total else []
    )
    fakta_sebaran = (
        f"Total {total} rumah di Surabaya.\n" + "\n".join(baris)
        if total else "Tidak ada data rumah."
    )
    if total and label and nilai:
        indeks_terbanyak = max(range(min(len(label), len(nilai))), key=nilai.__getitem__)
        ringkasan_sebaran = (
            f"Sebaran terbanyak ada pada rentang {label[indeks_terbanyak]}: "
            f"{nilai[indeks_terbanyak]} rumah ({nilai[indeks_terbanyak] / total:.0%}) "
            f"dari total {total} rumah."
        )
    else:
        ringkasan_sebaran = "Belum ada data rumah untuk diringkas."

    # Fakta untuk grafik 2
    kec = grafik["kecamatan"]
    baris = [
        f"- {k}: harga tengah Rp {h:g} miliar".replace(".", ",")
        for k, h in zip(kec["label"], kec["nilai"])
    ]
    fakta_kec = "10 kecamatan dengan harga tengah tertinggi:\n" + "\n".join(baris)
    if kec["label"] and kec["nilai"]:
        indeks_tertinggi = max(
            range(min(len(kec["label"]), len(kec["nilai"]))),
            key=kec["nilai"].__getitem__,
        )
        ringkasan_kec = (
            f"Harga tengah tertinggi tercatat di {kec['label'][indeks_tertinggi]} "
            f"sebesar Rp {kec['nilai'][indeks_tertinggi]:g} miliar. "
            "Grafik menampilkan hingga 10 kecamatan dengan harga tengah tertinggi."
        )
    else:
        ringkasan_kec = "Belum ada data harga tengah kecamatan untuk diringkas."

    return {
        "sebaran": _hasil_penjelasan(
            "Sebaran harga rumah per rentang harga",
            fakta_sebaran,
            ringkasan_sebaran,
            bool(total),
        ),
        "kecamatan": _hasil_penjelasan(
            "Harga tengah rumah per kecamatan",
            fakta_kec,
            ringkasan_kec,
            bool(kec["label"] and kec["nilai"]),
        ),
    }


def penjelasan_prediksi(masukan, harga_teks, rumah_serupa):
    """Menjelaskan estimasi harga berdasarkan input dan rumah pembanding."""
    spesifikasi = (
        f"kecamatan {masukan['kecamatan']}, {masukan['jumlah_lantai']:g} lantai, "
        f"LT {masukan['luas_tanah']:g} m², LB {masukan['luas_bangunan']:g} m², "
        f"{masukan['kamar_tidur']} kamar tidur, {masukan['kamar_mandi']} kamar mandi, "
        f"sertifikat {masukan['sertifikat']}, kondisi perabotan {masukan['perabotan']}, "
        f"dan kondisi properti {masukan['kondisi']}."
    )
    fakta_pembanding = "\n".join(
        f"- {rumah['kecamatan']}: {rumah['detail']}, harga tercantum {rumah['harga']}."
        for rumah in rumah_serupa
    ) or "- Tidak ada rumah pembanding yang tersedia."
    fakta = (
        f"Estimasi model: {harga_teks}.\n"
        f"Karakteristik input: {spesifikasi}\n"
        f"Rumah pembanding terdekat dari dataset:\n{fakta_pembanding}"
    )
    ringkasan = (
        f"Model memperkirakan harga {harga_teks} untuk rumah di {masukan['kecamatan']} "
        f"dengan LT {masukan['luas_tanah']:g} m² dan LB {masukan['luas_bangunan']:g} m². "
        "Rumah pembanding di bawah memiliki karakteristik yang mendekati, tetapi "
        "harga tercantum bukan jaminan harga transaksi."
    )
    teks_ai = tanya_groq(
        "Penjelasan estimasi harga rumah Surabaya",
        fakta,
        INSTRUKSI_PREDIKSI,
    )
    return {"teks": teks_ai or ringkasan, "ai": bool(teks_ai)}