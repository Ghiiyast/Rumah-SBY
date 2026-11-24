import os
import tempfile
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse
import requests

from . import views
from .analitik import buat_analisis, data_kosong, kecamatan_valid, muat
from .penjelasan_ai import penjelasan_dashboard, penjelasan_prediksi


class PenjelasanAiTests(TestCase):
    def setUp(self):
        self.grafik = {
            "sebaran": {
                "label": ["≤ 1 M", "1–2 M"],
                "nilai": [3, 7],
            },
            "kecamatan": {
                "label": ["Sukolilo", "Rungkut"],
                "nilai": [2.4, 1.8],
            },
        }

    def test_ringkasan_data_tetap_muncul_tanpa_api_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with patch("prediksi.penjelasan_ai.requests.post") as post:
                hasil = penjelasan_dashboard(self.grafik)

        post.assert_not_called()
        self.assertFalse(hasil["sebaran"]["ai"])
        self.assertIn("7 rumah", hasil["sebaran"]["teks"])
        self.assertFalse(hasil["kecamatan"]["ai"])
        self.assertIn("Sukolilo", hasil["kecamatan"]["teks"])

    def test_ringkasan_data_dipakai_jika_api_gagal(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch(
                "prediksi.penjelasan_ai.requests.post",
                side_effect=requests.ConnectionError,
            ):
                hasil = penjelasan_dashboard(self.grafik)

        self.assertFalse(hasil["sebaran"]["ai"])
        self.assertIn("7 rumah", hasil["sebaran"]["teks"])
        self.assertFalse(hasil["kecamatan"]["ai"])
        self.assertIn("Sukolilo", hasil["kecamatan"]["teks"])

    def test_grafik_kosong_tidak_menghentikan_dashboard(self):
        grafik_kosong = {
            "sebaran": {"label": ["≤ 1 M"], "nilai": [0]},
            "kecamatan": {"label": [], "nilai": []},
        }

        with patch("prediksi.penjelasan_ai.requests.post") as post:
            hasil = penjelasan_dashboard(grafik_kosong)

        post.assert_not_called()
        self.assertIn("Belum ada data", hasil["sebaran"]["teks"])
        self.assertIn("Belum ada data", hasil["kecamatan"]["teks"])


class PenjelasanPrediksiTests(TestCase):
    def setUp(self):
        self.masukan = {
            "kecamatan": "sukolilo",
            "jumlah_lantai": 2.0,
            "luas_tanah": 120.0,
            "luas_bangunan": 100.0,
            "kamar_tidur": 3,
            "kamar_mandi": 2,
            "sertifikat": "SHM",
            "perabotan": "Furnished",
            "kondisi": "Bagus",
        }
        self.rumah_serupa = [{
            "kecamatan": "Sukolilo",
            "detail": "LT 120 / LB 100 m² · 3 KT · 2 KM",
            "harga": "Rp 1,20 Miliar",
        }]

    def test_ringkasan_estimasi_dipakai_jika_ai_tidak_tersedia(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with patch("prediksi.penjelasan_ai.requests.post") as post:
                hasil = penjelasan_prediksi(
                    self.masukan,
                    "Rp 1,25 Miliar",
                    self.rumah_serupa,
                )

        post.assert_not_called()
        self.assertFalse(hasil["ai"])
        self.assertIn("Rp 1,25 Miliar", hasil["teks"])
        self.assertIn("sukolilo", hasil["teks"])

    def test_penjelasan_ai_dipakai_ketika_groq_berhasil(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("prediksi.penjelasan_ai.requests.post") as post:
                post.return_value.json.return_value = {
                    "choices": [{"message": {"content": "Penjelasan hasil AI."}}],
                }
                hasil = penjelasan_prediksi(
                    self.masukan,
                    "Rp 1,25 Miliar",
                    self.rumah_serupa,
                )

        self.assertTrue(hasil["ai"])
        self.assertEqual(hasil["teks"], "Penjelasan hasil AI.")


class PredictionPageTests(TestCase):
    def test_halaman_prediksi_menampilkan_tiga_rumah_pembanding(self):
        response = self.client.get("/prediksi/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.context["rumah_serupa"]), 3)
        self.assertContains(response, "Rumah dengan karakteristik serupa")

    @patch("prediksi.views.penjelasan_prediksi", return_value={
        "teks": "Penjelasan AI untuk hasil ini.",
        "ai": True,
    })
    @patch("prediksi.views.model.predict", return_value=[1_250_000_000])
    def test_hasil_prediksi_menampilkan_penjelasan_ai(self, predict, explain):
        payload = {
            nama: pilihan[0]
            for nama, _, pilihan in views.daftar_pilihan
        }
        payload.update({
            nama: str(batas[1])
            for nama, batas in views.batas.items()
        })

        response = self.client.post("/prediksi/", payload)

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Penjelasan AI untuk hasil ini.")
        self.assertTrue(response.context["penjelasan"]["ai"])
        self.assertEqual(len(response.context["rumah_serupa"]), 3)
        explain.assert_called_once()
        predict.assert_called_once()


class AnalisisDataTests(TestCase):
    def test_hal_analisis_hanya_menampilkan_tiga_grafik(self):
        response = self.client.get(reverse("analisis"))

        self.assertEqual(response.status_code, 200)
        for nomor in ("01", "02", "03"):
            self.assertContains(response, f'id="grafik-{nomor}"')
        self.assertNotContains(response, "Feature importance")
        self.assertNotContains(response, "Prediksi vs aktual")
        self.assertNotContains(response, "Membaca performa model")
        self.assertNotContains(response, "evaluasi")
        self.assertNotContains(response, "Jalankan python latih_model.py")

    def test_filter_kecamatan_memvalidasi_nama(self):
        valid = self.client.get(reverse("analisis"), {"kecamatan": "wonokromo"})
        tidak_valid = self.client.get(reverse("analisis"), {"kecamatan": "bukan-kecamatan"})

        self.assertEqual(valid.context["kecamatan"], "wonokromo")
        jumlah = int((
            views.data["Kecamatan"].astype(str).str.casefold() == "wonokromo"
        ).sum())
        self.assertEqual(valid.context["jumlah_kecamatan"], jumlah)
        self.assertEqual(tidak_valid.context["kecamatan"], "")
        self.assertFalse(tidak_valid.context["peringatan_data_sedikit"])

    def test_endpoint_penjelasan_hanya_get_dan_memvalidasi_grafik(self):
        alamat = reverse("analisis_penjelasan")
        salah = self.client.get(alamat, {"grafik": "04"})
        post = self.client.post(alamat, {"grafik": "01"})

        self.assertEqual(salah.status_code, 400)
        self.assertEqual(post.status_code, 405)

    def test_endpoint_penjelasan_mengirim_fakta_ke_groq(self):
        with patch("prediksi.views.tanya_groq", return_value="Ringkasan dari AI.") as ai:
            response = self.client.get(
                reverse("analisis_penjelasan"),
                {"grafik": "01", "kecamatan": "wonokromo"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["tersedia"])
        self.assertEqual(response.json()["teks"], "Ringkasan dari AI.")
        self.assertIn("01", ai.call_args.args[0])
        self.assertIn("wonokromo", ai.call_args.args[0])
        self.assertNotIn("Price", ai.call_args.args[1])

    def test_endpoint_menampilkan_fallback_jika_ai_tidak_tersedia(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            with patch("prediksi.penjelasan_ai.requests.post") as post:
                response = self.client.get(
                    reverse("analisis_penjelasan"),
                    {"grafik": "01"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["tersedia"])
        self.assertEqual(response.json()["teks"], "Penjelasan AI sedang tidak tersedia.")
        post.assert_not_called()

    def test_endpoint_menggunakan_groq_dengan_key_di_server(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("prediksi.penjelasan_ai.requests.post") as post:
                post.return_value.json.return_value = {
                    "choices": [{"message": {"content": "Ringkasan grafik."}}],
                }
                response = self.client.get(
                    reverse("analisis_penjelasan"),
                    {"grafik": "01", "kecamatan": "wonokromo"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["tersedia"])
        self.assertEqual(response.json()["teks"], "Ringkasan grafik.")
        self.assertEqual(post.call_count, 1)

    def test_endpoint_tetap_mengembalikan_fallback_jika_tidak_ada_internet(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch(
                "prediksi.penjelasan_ai.requests.post",
                side_effect=requests.ConnectionError,
            ):
                response = self.client.get(
                    reverse("analisis_penjelasan"),
                    {"grafik": "01"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["tersedia"])
        self.assertEqual(response.json()["teks"], "Penjelasan AI sedang tidak tersedia.")

    def test_halaman_tidak_memerlukan_file_evaluasi(self):
        response = self.client.get(reverse("analisis"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Data dihitung langsung dari dataset asli.")
        self.assertNotContains(response, "evaluasi")

    def test_halaman_tetap_tampil_jika_dataset_hilang(self):
        with patch("prediksi.views.data", data_kosong()):
            with patch("prediksi.views.DATASET_TERSEDIA", False):
                response = self.client.get(reverse("analisis"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dataset belum tersedia")

    def test_fungsi_muat_memakai_pembersihan_data_analitik(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rumah.csv")
            with open(path, "w", encoding="utf-8") as file:
                file.write(
                    "Kecamatan; Price ;Jumlah Lantai;Kondisi Properti;Luas Bangunan\n"
                    "wonokromo; 600.000.000 ;2;Baru;70\n"
                    "wonokromo; 600.000.000 ;2;Baru;70\n"
                    "rungkut; 800.000.000 ;2;Bagus;1600\n"
                )
            data = muat(path)

        self.assertEqual(len(data), 1)
        self.assertEqual(data.iloc[0]["harga"], 600_000_000)
        self.assertEqual(kecamatan_valid(data, "WONOKROMO"), "wonokromo")
        self.assertEqual(kecamatan_valid(data, "asing"), "")

    def test_kecamatan_dengan_data_kurang_dari_dua_puluh_diberi_peringatan(self):
        data_sedikit = views.data[views.data["Kecamatan"] == "wonokromo"].head(3)
        with patch("prediksi.views.data", data_sedikit):
            response = self.client.get(reverse("analisis"), {"kecamatan": "wonokromo"})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["peringatan_data_sedikit"])
        self.assertContains(response, "n = 3")

    def test_grafik_menghitung_distribusi_dan_sampel_yang_stabil(self):
        data = views.data.head(100).copy()
        pertama = buat_analisis(data)
        kedua = buat_analisis(data)

        self.assertEqual(sum(pertama["01"]["values"]), len(data))
        self.assertEqual(pertama["03"]["points"], kedua["03"]["points"])
        self.assertLessEqual(len(pertama["03"]["points"]), 100)

    def test_analisis_hanya_menyediakan_grafik_yang_dipakai(self):
        grafik = buat_analisis(views.data.head(5))

        self.assertEqual(set(grafik), {"01", "02", "03"})
