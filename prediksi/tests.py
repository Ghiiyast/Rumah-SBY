import os
from unittest.mock import patch

from django.test import TestCase
import requests

from .penjelasan_ai import penjelasan_dashboard


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
