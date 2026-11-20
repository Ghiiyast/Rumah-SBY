import pandas as pd


def ke_angka(kolom):
    """Memastikan kolom harga berupa angka (kalau masih teks seperti ' 600.000.000 ', diubah)."""
    if pd.api.types.is_numeric_dtype(kolom):
        return kolom
    teks = kolom.astype(str).str.replace(".", "", regex=False).str.strip()
    return pd.to_numeric(teks, errors="coerce")


def siapkan_grafik(df):
    """Menyiapkan data untuk 2 grafik di dashboard (df = tabel rumah)."""
    harga = ke_angka(df["Price"])
    d = pd.DataFrame({"Kecamatan": df["Kecamatan"], "harga": harga}).dropna()

    # Grafik 1: sebaran harga -> berapa rumah di tiap rentang harga
    batas = [0, 1e9, 2e9, 3e9, 5e9, 10e9, float("inf")]
    nama = ["≤ 1 M", "1–2 M", "2–3 M", "3–5 M", "5–10 M", "> 10 M"]
    sebaran = pd.cut(d["harga"], bins=batas, labels=nama).value_counts().reindex(nama)

    # Grafik 2: 10 kecamatan dengan harga tengah (median) tertinggi, dalam miliar rupiah.
    # Hanya kecamatan yang punya minimal 20 data, supaya tidak menyesatkan.
    jumlah = d["Kecamatan"].value_counts()
    cukup = jumlah[jumlah >= 20].index
    tengah = (
        d[d["Kecamatan"].isin(cukup)]
        .groupby("Kecamatan")["harga"].median()
        .sort_values(ascending=False)
        .head(10)
    )

    # int() dan float() supaya bisa diubah ke JSON (angka dari pandas tidak bisa langsung)
    return {
        "sebaran": {
            "label": nama,
            "nilai": [int(n) for n in sebaran],
        },
        "kecamatan": {
            "label": [k.title() for k in tengah.index],
            "nilai": [round(float(h) / 1e9, 2) for h in tengah],
        },
    }