from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
LABEL_RENTANG = ["≤ 1 M", "1–2 M", "2–3 M", "3–5 M", "5–10 M", "> 10 M"]


def data_kosong():
    """Membuat bentuk data kosong agar halaman tetap bisa ditampilkan."""
    kolom = [
        "Kecamatan", "Price", "Kamar Tidur", "Kamar Mandi", "Luas Tanah",
        "Luas Bangunan", "Sertifikat", "Kondisi Perabotan", "Jumlah Lantai",
        "Kondisi Properti", "harga", "Kecamatan_Label",
    ]
    return pd.DataFrame(columns=kolom)


def muat(path=None):
    """Memuat dan membersihkan dataset rumah untuk dipakai aplikasi."""
    path = path or BASE_DIR / "data" / "rumah_surabaya.csv"
    try:
        data = pd.read_csv(path, sep=";")
    except pd.errors.EmptyDataError:
        return data_kosong()

    data.columns = data.columns.str.strip()
    data = data.drop_duplicates()
    data["harga"] = pd.to_numeric(
        data["Price"].astype(str).str.replace(".", "", regex=False).str.strip(),
        errors="coerce",
    )
    data = data.dropna(subset=["Jumlah Lantai", "Kondisi Properti", "harga"])
    data = data[data["Luas Bangunan"] <= 1500].copy()
    data["Kecamatan_Label"] = data["Kecamatan"].str.title()
    return data


def kecamatan_valid(data, nilai):
    """Mengembalikan nama kecamatan sesuai data, atau kosong jika tidak dikenal."""
    if not nilai or "Kecamatan" not in data:
        return ""
    daftar = {
        str(kecamatan).strip().casefold(): str(kecamatan).strip()
        for kecamatan in data["Kecamatan"].dropna().unique()
    }
    return daftar.get(str(nilai).strip().casefold(), "")


def _harga(data):
    if "harga" in data:
        return pd.to_numeric(data["harga"], errors="coerce")
    if "Price" not in data:
        return pd.Series(dtype="float64")
    return pd.to_numeric(
        data["Price"].astype(str).str.replace(".", "", regex=False).str.strip(),
        errors="coerce",
    )


def _angka(data, kolom):
    if kolom not in data:
        return pd.Series(np.nan, index=data.index, dtype="float64")
    return pd.to_numeric(data[kolom], errors="coerce")


def _data_filter(data, kecamatan):
    if kecamatan and "Kecamatan" in data:
        return data[data["Kecamatan"].astype(str).str.casefold() == kecamatan.casefold()]
    return data


def _grafik_distribusi(data, kecamatan):
    data = _data_filter(data, kecamatan)
    harga = _harga(data).dropna()
    nilai = pd.cut(
        harga,
        bins=[-np.inf, 1e9, 2e9, 3e9, 5e9, 10e9, np.inf],
        labels=LABEL_RENTANG,
        right=True,
    )
    jumlah = nilai.value_counts(sort=False).reindex(LABEL_RENTANG, fill_value=0)
    total = int(jumlah.sum())
    fakta = [f"Total rumah: {total}."]
    for label, count in jumlah.items():
        persen = count / total * 100 if total else 0
        fakta.append(f"{label}: {int(count)} rumah ({persen:.1f}%).")
    return {
        "labels": LABEL_RENTANG,
        "values": [int(count) for count in jumlah],
        "total": total,
        "facts": "\n".join(fakta),
    }


def _grafik_kecamatan(data, kecamatan, mode):
    if "Kecamatan" not in data:
        return {"labels": [], "values": [], "highlight": [], "facts": "Data kecamatan tidak tersedia."}

    tabel = pd.DataFrame({
        "Kecamatan": data["Kecamatan"].astype(str),
        "harga": _harga(data),
        "luas_tanah": _angka(data, "Luas Tanah"),
    }).dropna(subset=["Kecamatan", "harga"])
    if mode == "m2":
        tabel = tabel[tabel["luas_tanah"] > 0].copy()
        tabel["nilai"] = tabel["harga"] / tabel["luas_tanah"] / 1e6
        satuan = "Juta Rp/m²"
    else:
        tabel["nilai"] = tabel["harga"] / 1e9
        satuan = "Miliar Rp"

    ringkas = tabel.groupby("Kecamatan").agg(
        jumlah=("nilai", "count"),
        median=("nilai", "median"),
    )
    ringkas = ringkas[ringkas["jumlah"] >= 20].sort_values("median", ascending=False)
    labels = [str(nama).title() for nama in ringkas.index]
    values = [round(float(nilai), 3) for nilai in ringkas["median"]]
    highlight = [
        str(nama).casefold() == kecamatan.casefold() if kecamatan else False
        for nama in ringkas.index
    ]

    median_kota = float(tabel["nilai"].median()) if not tabel.empty else None
    median_pilih = None
    if kecamatan:
        dipilih = tabel[tabel["Kecamatan"].str.casefold() == kecamatan.casefold()]
        if not dipilih.empty:
            median_pilih = float(dipilih["nilai"].median())
    fakta = [f"Mode tampilan: {satuan}."]
    if median_kota is not None:
        fakta.append(f"Median seluruh kota: {median_kota:.3f} {satuan}.")
    if kecamatan:
        if median_pilih is None:
            fakta.append(f"Median {kecamatan.title()}: tidak tersedia karena tidak ada data.")
        else:
            fakta.append(
                f"Median {kecamatan.title()}: {median_pilih:.3f} {satuan} "
                f"({int((tabel['Kecamatan'].str.casefold() == kecamatan.casefold()).sum())} rumah)."
            )
    fakta.append("Peringkat kecamatan yang ditampilkan:")
    fakta.extend(
        f"{urutan}. {nama}: {nilai:.3f} {satuan}."
        for urutan, (nama, nilai) in enumerate(zip(labels, values), start=1)
    )
    return {
        "labels": labels,
        "values": values,
        "highlight": highlight,
        "has_data": not tabel.empty,
        "mode": mode,
        "unit": satuan,
        "facts": "\n".join(fakta),
    }


def _grafik_luas_harga(data, kecamatan):
    data = _data_filter(data, kecamatan)
    tabel = pd.DataFrame({
        "luas": _angka(data, "Luas Tanah"),
        "harga": _harga(data),
    }).dropna()
    tabel = tabel[(tabel["luas"] > 0) & (tabel["harga"] > 0)]
    if tabel.empty:
        return {
            "points": [], "sample_count": 0, "outside": 0,
            "x_max": 0, "y_max": 0, "facts": "Tidak ada data luas tanah dan harga.",
        }

    luas_p95 = float(tabel["luas"].quantile(0.95))
    harga_p95 = float(tabel["harga"].quantile(0.95))
    sampel = tabel.sample(n=min(1500, len(tabel)), random_state=42)
    tampak = sampel[(sampel["luas"] <= luas_p95) & (sampel["harga"] <= harga_p95)]
    di_luar = int(((tabel["luas"] > luas_p95) | (tabel["harga"] > harga_p95)).sum())

    korelasi = None
    if len(tabel) >= 2 and tabel["luas"].nunique() > 1 and tabel["harga"].nunique() > 1:
        nilai_korelasi = spearmanr(tabel["luas"], tabel["harga"]).statistic
        if np.isfinite(nilai_korelasi):
            korelasi = float(nilai_korelasi)

    kelompok = pd.cut(
        tabel["luas"],
        bins=[-np.inf, 100, 200, 300, np.inf],
        labels=["<100 m²", "100–200 m²", "200–300 m²", ">300 m²"],
        right=False,
    )
    median_kelompok = tabel.assign(kelompok=kelompok).groupby(
        "kelompok", observed=False
    )["harga"].agg(["median", "count"])
    fakta = [
        f"Jumlah rumah valid: {len(tabel)}.",
        f"Korelasi peringkat Spearman luas tanah dan harga: {korelasi:.3f}."
        if korelasi is not None
        else "Korelasi peringkat Spearman tidak tersedia.",
    ]
    fakta.extend(
        f"{nama}: median harga {baris['median'] / 1e9:.3f} miliar rupiah "
        f"({int(baris['count'])} rumah)."
        if baris["count"]
        else f"{nama}: tidak ada rumah."
        for nama, baris in median_kelompok.iterrows()
    )
    return {
        "points": [
            {"x": float(row["luas"]), "y": float(row["harga"] / 1e9)}
            for _, row in tampak.iterrows()
        ],
        "sample_count": len(sampel),
        "outside": di_luar,
        "x_max": luas_p95,
        "y_max": harga_p95 / 1e9,
        "facts": "\n".join(fakta),
    }


def buat_analisis(data, kecamatan="", mode="total"):
    """Menyiapkan data grafik dan fakta untuk halaman analisis."""
    mode = mode if mode in {"total", "m2"} else "total"
    return {
        "01": _grafik_distribusi(data, kecamatan),
        "02": _grafik_kecamatan(data, kecamatan, mode),
        "03": _grafik_luas_harga(data, kecamatan),
    }
