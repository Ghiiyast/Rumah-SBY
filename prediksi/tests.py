import os
from unittest.mock import patch

from django.test import TestCase
import requests

from . import views
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
