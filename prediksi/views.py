from pathlib import Path

import joblib
import pandas as pd
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, JsonResponse
from django.views.decorators.http import require_GET
from .analitik import buat_analisis, data_kosong, kecamatan_valid, muat
from django.shortcuts import render
from .grafik import siapkan_grafik
from .penjelasan_ai import (
    INSTRUKSI_ANALISIS,
    MODEL,
    penjelasan_dashboard,
    penjelasan_prediksi,
    tanya_groq,
)

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "model" / "model_rumah.joblib")

try:
    data = muat()
    DATASET_TERSEDIA = True
except FileNotFoundError:
    data = data_kosong()
    DATASET_TERSEDIA = False

def format_rupiah(nilai):
    """Ubah angka Rupiah besar menjadi format ringkas (Juta / Miliar)."""
    if nilai >= 1_000_000_000:
        return f"Rp {nilai / 1_000_000_000:,.2f} Miliar".replace(".", ",")
    return f"Rp {nilai / 1_000_000:,.0f} Juta".replace(".", ",")


def cari_rumah_serupa(masukan):
    """Mengurutkan tiga listing dataset yang paling mendekati input prediksi."""
    skor = pd.Series(0.0, index=data.index)
    kolom_kategori = {
        "kecamatan": ("Kecamatan", 1.5),
        "sertifikat": ("Sertifikat", 0.6),
        "perabotan": ("Kondisi Perabotan", 0.5),
        "kondisi": ("Kondisi Properti", 0.6),
    }
    for kunci, (kolom, bobot) in kolom_kategori.items():
        target = str(masukan[kunci]).strip().casefold()
        berbeda = data[kolom].astype(str).str.strip().str.casefold() != target
        skor += berbeda.astype(float) * bobot

    kolom_numerik = {
        "jumlah_lantai": ("Jumlah Lantai", 0.3),
        "kamar_tidur": ("Kamar Tidur", 0.5),
        "kamar_mandi": ("Kamar Mandi", 0.3),
        "luas_tanah": ("Luas Tanah", 0.9),
        "luas_bangunan": ("Luas Bangunan", 0.9),
    }
    for kunci, (kolom, bobot) in kolom_numerik.items():
        nilai_kolom = pd.to_numeric(data[kolom], errors="coerce")
        skala = nilai_kolom.quantile(0.75) - nilai_kolom.quantile(0.25)
        if not skala or pd.isna(skala):
            skala = nilai_kolom.std()
        if not skala or pd.isna(skala):
            skala = 1
        skor += (nilai_kolom.sub(masukan[kunci]).abs() / skala).clip(upper=5) * bobot

    kandidat = data.assign(_skor_serupa=skor).nsmallest(3, "_skor_serupa")
    rumah = []
    for _, row in kandidat.iterrows():
        rumah.append({
            "kecamatan": row["Kecamatan_Label"],
            "kondisi": row["Kondisi Properti"],
            "detail": (
                f"LT {int(row['Luas Tanah'])} / LB {int(row['Luas Bangunan'])} m² · "
                f"{int(row['Kamar Tidur'])} KT · {int(row['Kamar Mandi'])} KM"
            ),
            "harga": format_rupiah(row["harga"]),
        })
    return rumah


daftar_pilihan = [
    ("kecamatan", "Kecamatan (Surabaya)", sorted(data["Kecamatan"].unique())),
    ("sertifikat", "Jenis Sertifikat", sorted(data["Sertifikat"].unique())),
    ("perabotan", "Kondisi Perabotan", sorted(data["Kondisi Perabotan"].unique())),
    ("kondisi", "Kondisi Properti", sorted(data["Kondisi Properti"].unique())),
    ("jumlah_lantai", "Jumlah Lantai", [str(int(x)) for x in sorted(data["Jumlah Lantai"].unique())]),
]

if data.empty:
    batas = {}
else:
    batas = {
        "kamar_tidur": ("Kamar Tidur", int(data["Kamar Tidur"].min()), int(data["Kamar Tidur"].max())),
        "kamar_mandi": ("Kamar Mandi", int(data["Kamar Mandi"].min()), int(data["Kamar Mandi"].max())),
        "luas_tanah": ("Luas Tanah (m²)", int(data["Luas Tanah"].min()), int(data["Luas Tanah"].max())),
        "luas_bangunan": ("Luas Bangunan (m²)", int(data["Luas Bangunan"].min()), int(data["Luas Bangunan"].max())),
    }


def prediksi(request):
    nilai = {}
    masukan = None
    harga = None
    harga_teks = None
    error = None

    if request.method == "POST" and data.empty:
        nilai = request.POST.dict()
        error = "Dataset rumah belum tersedia untuk membuat prediksi."
    elif request.method == "POST":
        nilai = request.POST.dict()
        try:
            masukan = {
                "kecamatan": nilai["kecamatan"],
                "sertifikat": nilai["sertifikat"],
                "perabotan": nilai["perabotan"],
                "kondisi": nilai["kondisi"],
                "jumlah_lantai": float(nilai["jumlah_lantai"]),
                "kamar_tidur": int(nilai["kamar_tidur"]),
                "kamar_mandi": int(nilai["kamar_mandi"]),
                "luas_tanah": float(nilai["luas_tanah"]),
                "luas_bangunan": float(nilai["luas_bangunan"]),
            }
        except (KeyError, ValueError):
            error = "Semua kolom harus diisi dengan angka yang valid."
        else:
            di_luar = [
                f"{label} (antara {kecil} dan {besar})"
                for kolom, (label, kecil, besar) in batas.items()
                if not kecil <= masukan[kolom] <= besar
            ]
            if di_luar:
                error = "Nilai di luar jangkauan data: " + ", ".join(di_luar)
            else:
                data_baru = pd.DataFrame([masukan])
                hasil_rp = round(float(model.predict(data_baru)[0]))
                harga = f"{hasil_rp:,}".replace(",", ".")
                harga_teks = format_rupiah(hasil_rp)

    if request.method == "GET" and not data.empty:
        masukan_serupa = {
            nama: pilihan[0]
            for nama, _, pilihan in daftar_pilihan
        }
        masukan_serupa.update({
            "jumlah_lantai": float(masukan_serupa["jumlah_lantai"]),
            "kamar_tidur": 3,
            "kamar_mandi": 2,
            "luas_tanah": 100,
            "luas_bangunan": 120,
        })
        rumah_serupa = cari_rumah_serupa(masukan_serupa)
    elif masukan and not error:
        rumah_serupa = cari_rumah_serupa(masukan)
    else:
        rumah_serupa = []

    penjelasan = (
        penjelasan_prediksi(masukan, harga_teks, rumah_serupa)
        if harga and masukan
        else None
    )

    isian = [
        {"nama": nama, "label": label, "isi": isi, "terpilih": nilai.get(nama)}
        for nama, label, isi in daftar_pilihan
    ]

    return render(request, "prediksi/prediksi.html", {
        "halaman": "prediksi",
        "isian": isian,
        "nilai": nilai,
        "harga": harga,
        "harga_teks": harga_teks,
        "error": error,
        "rumah_serupa": rumah_serupa,
        "penjelasan": penjelasan,
    })


def saring(kode):
    """Menyaring rumah di Surabaya sesuai kategori kebutuhan."""
    d = data
    if kode == "minimalis":
        return d[d["harga"] <= 1_000_000_000]
    if kode == "keluarga":
        return d[(d["Kamar Tidur"] >= 3) & (d["harga"] <= 1_800_000_000)]
    if kode == "bertingkat":
        return d[d["Jumlah Lantai"] >= 2]
    if kode == "siap_huni":
        return d[d["Kondisi Properti"].isin(["Baru", "Bagus Sekali"])]
    if kode == "mewah":
        return d[(d["harga"] >= 5_000_000_000) & (d["Luas Tanah"] >= 250)]
    return d


daftar_kebutuhan = [
    ("minimalis", "Minimalis (≤ 1 M)", "🏡"),
    ("keluarga", "Keluarga Muda", "👨‍👩‍👧‍👦"),
    ("bertingkat", "Rumah 2+ Lantai", "🏢"),
    ("siap_huni", "Baru / Siap Huni", "✨"),
    ("mewah", "Rumah Mewah", "🏰"),
]


def dashboard(request):
    pilih = request.GET.get("kebutuhan", "")

    kebutuhan = [
        {"kode": kode, "nama": nama, "ikon": ikon, "jumlah": len(saring(kode))}
        for kode, nama, ikon in daftar_kebutuhan
    ]

    df_tabel = saring(pilih).sort_values("harga").head(10).copy()
    tabel = []
    for _, row in df_tabel.iterrows():
        tabel.append({
            "kecamatan": row["Kecamatan_Label"],
            "kamar": f"{int(row['Kamar Tidur'])} KT / {int(row['Kamar Mandi'])} KM",
            "luas": f"{int(row['Luas Tanah'])} / {int(row['Luas Bangunan'])} m²",
            "lantai": f"{int(row['Jumlah Lantai'])} Lt",
            "sertifikat": row["Sertifikat"].split(" - ")[0],
            "harga": format_rupiah(row["harga"]),
        })

    h = data["harga"]
    if h.empty:
        ringkasan = {
            "jumlah": "0",
            "rata": "Belum tersedia",
            "termurah": "Belum tersedia",
            "termahal": "Belum tersedia",
        }
    else:
        ringkasan = {
            "jumlah": f"{len(data):,}".replace(",", "."),
            "rata": format_rupiah(h.mean()),
            "termurah": format_rupiah(h.min()),
            "termahal": format_rupiah(h.max()),
        }

    grafik = siapkan_grafik(data)
    penjelasan = penjelasan_dashboard(grafik)

    return render(request, "prediksi/dashboard.html", {
        "halaman": "dashboard",
        "kebutuhan": kebutuhan,
        "pilih": pilih,
        "tabel": tabel,
        "ringkasan": ringkasan,
        "grafik": grafik,
        "penjelasan": penjelasan,
    })


def analisis(request):
    daftar_kecamatan = (
        sorted(data["Kecamatan"].dropna().astype(str).str.strip().unique())
        if "Kecamatan" in data
        else []
    )
    kecamatan = kecamatan_valid(data, request.GET.get("kecamatan", ""))
    mode = request.GET.get("mode", "total")
    if mode not in {"total", "m2"}:
        mode = "total"

    grafik = buat_analisis(data, kecamatan, mode)
    jumlah_filter = (
        int((data["Kecamatan"].astype(str).str.casefold() == kecamatan.casefold()).sum())
        if kecamatan and not data.empty
        else len(data)
    )
    if not DATASET_TERSEDIA:
        fakta_data = "File data/rumah_surabaya.csv belum tersedia."
    else:
        fakta_data = (
            f"Dihitung dari {jumlah_filter:,} rumah setelah pembersihan."
            .replace(",", ".")
        )
    jumlah_pilih = (
        int((data["Kecamatan"].astype(str).str.casefold() == kecamatan.casefold()).sum())
        if kecamatan and not data.empty
        else 0
    )
    return render(request, "prediksi/analisis.html", {
        "halaman": "analisis",
        "kecamatan": kecamatan,
        "daftar_kecamatan": daftar_kecamatan,
        "mode": mode,
        "data_json": grafik,
        "data_tersedia": DATASET_TERSEDIA and not data.empty,
        "fakta_data": fakta_data,
        "peringatan_data_sedikit": bool(kecamatan and jumlah_pilih < 20),
        "jumlah_kecamatan": jumlah_pilih,
        "grafik_distribusi": grafik["01"],
        "grafik_kecamatan": grafik["02"],
        "grafik_luas_harga": grafik["03"],
        "model_ai": MODEL,
    })


@require_GET
def analisis_penjelasan(request):
    nomor = request.GET.get("grafik", "")
    if nomor not in {"01", "02", "03"}:
        return JsonResponse({"teks": "Nomor grafik tidak valid."}, status=400)

    mode = request.GET.get("mode", "total")
    if mode not in {"total", "m2"}:
        return JsonResponse({"teks": "Mode harga tidak valid."}, status=400)

    kecamatan = kecamatan_valid(data, request.GET.get("kecamatan", ""))
    if not DATASET_TERSEDIA or data.empty:
        return JsonResponse({"teks": "Penjelasan AI sedang tidak tersedia."})

    grafik = buat_analisis(data, kecamatan, mode)[nomor]
    ada_fakta = bool(grafik.get("facts"))
    if nomor == "01":
        ada_fakta = ada_fakta and grafik["total"] > 0
    elif nomor == "02":
        ada_fakta = ada_fakta and grafik["has_data"]
    elif nomor == "03":
        ada_fakta = ada_fakta and grafik["sample_count"] > 0
    if not ada_fakta:
        return JsonResponse({"teks": "Penjelasan AI sedang tidak tersedia."})

    nama_grafik = {
        "01": "Distribusi harga rumah",
        "02": "Harga per kecamatan",
        "03": "Luas tanah dan harga",
    }[nomor]
    topik = (
        f"Grafik {nomor}: {nama_grafik}; kecamatan: {kecamatan or 'semua kecamatan'}; "
        f"mode: {mode if nomor == '02' else 'tetap'}"
    )
    teks = tanya_groq(topik, grafik["facts"], INSTRUKSI_ANALISIS)
    return JsonResponse({
        "teks": teks or "Penjelasan AI sedang tidak tersedia.",
        "tersedia": bool(teks),
    })


def dataset_view(request):
    """Menampilkan informasi, filter, tabel paginasi, dan unduhan dataset asli."""
    if not DATASET_TERSEDIA:
        raise Http404("Dataset rumah belum tersedia.")

    path_csv = BASE_DIR / "data" / "rumah_surabaya.csv"
    df_asli = pd.read_csv(path_csv, sep=";")
    df_asli.columns = df_asli.columns.str.strip()

    df = df_asli.copy()
    kolom_asli = list(df_asli.columns)

    if "Price" in df.columns:
        df["_harga_num"] = pd.to_numeric(
            df["Price"].astype(str).str.replace(".", "", regex=False).str.strip(),
            errors="coerce",
        )
    else:
        df["_harga_num"] = pd.Series(dtype="float64")

    q_kecamatan = request.GET.get("kecamatan", "").strip()
    harga_min = request.GET.get("harga_min", "").strip()
    harga_max = request.GET.get("harga_max", "").strip()
    lt_min = request.GET.get("lt_min", "").strip()
    lb_min = request.GET.get("lb_min", "").strip()

    if q_kecamatan and "Kecamatan" in df.columns:
        df = df[df["Kecamatan"].astype(str).str.contains(q_kecamatan, case=False, na=False)]

    if harga_min:
        try:
            df = df[df["_harga_num"] >= float(harga_min) * 1_000_000]
        except ValueError:
            pass

    if harga_max:
        try:
            df = df[df["_harga_num"] <= float(harga_max) * 1_000_000]
        except ValueError:
            pass

    if lt_min and "Luas Tanah" in df.columns:
        try:
            df = df[pd.to_numeric(df["Luas Tanah"], errors="coerce") >= float(lt_min)]
        except ValueError:
            pass

    if lb_min and "Luas Bangunan" in df.columns:
        try:
            df = df[pd.to_numeric(df["Luas Bangunan"], errors="coerce") >= float(lb_min)]
        except ValueError:
            pass

    df_tampil = df[kolom_asli].fillna("-")
    baris_data = df_tampil.values.tolist()

    paginator = Paginator(baris_data, 15)
    nomor_halaman = request.GET.get("page", 1)
    halaman_obj = paginator.get_page(nomor_halaman)

    query_params = request.GET.copy()
    if "page" in query_params:
        del query_params["page"]
    query_string = query_params.urlencode()

    daftar_kecamatan = (
        sorted(df_asli["Kecamatan"].dropna().astype(str).str.strip().unique())
        if "Kecamatan" in df_asli.columns
        else []
    )

    info_dataset = {
        "nama": "rumah_surabaya.csv",
        "total_baris": f"{len(df_asli):,}".replace(",", "."),
        "total_bersih": f"{len(data):,}".replace(",", "."),
        "jumlah_kolom": len(kolom_asli),
        "hasil_filter": f"{len(df_tampil):,}".replace(",", "."),
        "ukuran_kb": round(path_csv.stat().st_size / 1024, 1),
    }

    return render(request, "prediksi/dataset.html", {
        "halaman": "dataset",
        "info": info_dataset,
        "kolom": kolom_asli,
        "page_obj": halaman_obj,
        "query_string": query_string,
        "daftar_kecamatan": daftar_kecamatan,
        "ada_lt": "Luas Tanah" in kolom_asli,
        "ada_lb": "Luas Bangunan" in kolom_asli,
        "filter_aktif": {
            "kecamatan": q_kecamatan,
            "harga_min": harga_min,
            "harga_max": harga_max,
            "lt_min": lt_min,
            "lb_min": lb_min,
        },
    })


def unduh_dataset(request):
    """Mengunduh file CSV asli rumah_surabaya.csv."""
    path_csv = BASE_DIR / "data" / "rumah_surabaya.csv"
    if not path_csv.exists():
        raise Http404("File dataset tidak ditemukan.")
    return FileResponse(
        open(path_csv, "rb"),
        as_attachment=True,
        filename="rumah_surabaya.csv",
        content_type="text/csv",
    )