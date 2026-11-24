import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

mae = float(mean_absolute_error(y_test, prediksi))
rmse = float(np.sqrt(mean_squared_error(y_test, prediksi)))
r2 = float(r2_score(y_test, prediksi))
galat = prediksi - y_test.to_numpy()
persen_galat = np.abs(galat) / y_test.to_numpy()

jumlah_ekor = max(1, int(np.ceil(len(y_test) * 0.1)))
data_uji = pd.DataFrame({
    "kecamatan": X_test["kecamatan"].astype(str).to_numpy(),
    "aktual": y_test.to_numpy(),
    "tebakan": prediksi,
})
termahal = data_uji.nlargest(jumlah_ekor, "aktual")
termurah = data_uji.nsmallest(jumlah_ekor, "aktual")

estimator = model
if hasattr(estimator, "regressor_"):
    estimator = estimator.regressor_
random_forest = estimator.named_steps["rf"]
nama_fitur_keluaran = estimator.named_steps["prep"].get_feature_names_out()
importance_asal = {fitur: 0.0 for fitur in fitur_kategori + fitur_angka}
for nama, nilai in zip(nama_fitur_keluaran, random_forest.feature_importances_):
    nama_asal = nama.split("__", 1)[-1]
    fitur_asal = next(
        (
            fitur for fitur in importance_asal
            if nama_asal == fitur or nama_asal.startswith(f"{fitur}_")
        ),
        None,
    )
    if fitur_asal:
        importance_asal[fitur_asal] += float(nilai)

label_fitur = {
    "kecamatan": "Kecamatan",
    "sertifikat": "Sertifikat",
    "perabotan": "Kondisi Perabotan",
    "kondisi": "Kondisi Properti",
    "kamar_tidur": "Kamar Tidur",
    "kamar_mandi": "Kamar Mandi",
    "luas_tanah": "Luas Tanah",
    "luas_bangunan": "Luas Bangunan",
    "jumlah_lantai": "Jumlah Lantai",
}
feature_importance = [
    {"fitur": label_fitur[fitur], "persen": round(nilai * 100, 2)}
    for fitur, nilai in importance_asal.items()
]
feature_importance.sort(key=lambda item: item["persen"], reverse=True)
if feature_importance:
    selisih = round(100 - sum(item["persen"] for item in feature_importance), 2)
    feature_importance[0]["persen"] = round(feature_importance[0]["persen"] + selisih, 2)

evaluasi = {
    "nama_model": "RandomForestRegressor",
    "jumlah_data_latih": int(len(X_train)),
    "jumlah_data_uji": int(len(X_test)),
    "tanggal": date.today().isoformat(),
    "metrik": {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "median_error": float(np.median(galat)),
        "dalam_10_persen": float(np.mean(persen_galat <= 0.10) * 100),
        "dalam_25_persen": float(np.mean(persen_galat <= 0.25) * 100),
        "dalam_50_persen": float(np.mean(persen_galat <= 0.50) * 100),
        "rasio_termahal": float((termahal["tebakan"] / termahal["aktual"]).mean()),
        "rasio_termurah": float((termurah["tebakan"] / termurah["aktual"]).mean()),
    },
    "feature_importance": feature_importance,
    "titik_uji": [
        [str(kecamatan), float(aktual), float(tebakan)]
        for kecamatan, aktual, tebakan in data_uji.itertuples(index=False, name=None)
    ],
}

print(f"Jumlah data latih : {len(X_train)} baris")
print(f"Skor R2           : {r2_score(y_test, prediksi):.4f}")
print(f"Rata-rata Error   : Rp {mean_absolute_error(y_test, prediksi):,.0f}")

(BASE_DIR / "model").mkdir(exist_ok=True)
joblib.dump(model, BASE_DIR / "model" / "model_rumah.joblib")
with open(BASE_DIR / "model" / "evaluasi.json", "w", encoding="utf-8") as file:
    json.dump(evaluasi, file, ensure_ascii=False, indent=2)
print("Model berhasil disimpan ke model/model_rumah.joblib!")
print("Evaluasi berhasil disimpan ke model/evaluasi.json!")