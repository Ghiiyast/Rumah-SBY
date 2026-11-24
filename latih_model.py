from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent

# Baca CSV (pemisah titik koma) dan rapikan nama kolom
df = pd.read_csv(BASE_DIR / "data" / "rumah_surabaya.csv", sep=";")
df.columns = df.columns.str.strip()

# Bersihkan kolom harga dari format teks "600.000.000" menjadi angka
df["harga"] = (
    df["Price"].astype(str).str.replace(".", "", regex=False).str.strip()
)
df["harga"] = pd.to_numeric(df["harga"], errors="coerce")

# Pilih 9 fitur utama dan buang baris yang kosong/tidak wajar
df = df.rename(columns={
    "Kecamatan": "kecamatan",
    "Sertifikat": "sertifikat",
    "Kondisi Perabotan": "perabotan",
    "Kondisi Properti": "kondisi",
    "Kamar Tidur": "kamar_tidur",
    "Kamar Mandi": "kamar_mandi",
    "Luas Tanah": "luas_tanah",
    "Luas Bangunan": "luas_bangunan",
    "Jumlah Lantai": "jumlah_lantai",
})

fitur_kategori = ["kecamatan", "sertifikat", "perabotan", "kondisi"]
fitur_angka = ["kamar_tidur", "kamar_mandi", "luas_tanah", "luas_bangunan", "jumlah_lantai"]

df = df.dropna(subset=fitur_kategori + fitur_angka + ["harga"])
df = df[(df["luas_bangunan"] <= 1500) & (df["harga"] > 0)]

X = df[fitur_kategori + fitur_angka]
y = df["harga"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ("kat", OneHotEncoder(handle_unknown="ignore"), fitur_kategori),
    ("ang", "passthrough", fitur_angka),
])

model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
])

model.fit(X_train, y_train)
prediksi = model.predict(X_test)

print(f"Jumlah data latih : {len(X_train)} baris")
print(f"Skor R2           : {r2_score(y_test, prediksi):.4f}")
print(f"Rata-rata Error   : Rp {mean_absolute_error(y_test, prediksi):,.0f}")

(BASE_DIR / "model").mkdir(exist_ok=True)
joblib.dump(model, BASE_DIR / "model" / "model_rumah.joblib")
print("Model berhasil disimpan ke model/model_rumah.joblib!")