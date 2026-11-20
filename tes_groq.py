import os
from pathlib import Path

import requests
from dotenv import load_dotenv

env = Path(__file__).resolve().parent / ".env"
print("File .env ada?   ", env.exists(), "->", env)
load_dotenv(env)

kunci = os.environ.get("GROQ_API_KEY", "").strip()
print("Key terbaca?     ", bool(kunci), "| panjang:", len(kunci))
if not kunci:
    raise SystemExit("Key kosong. Periksa isi .env: satu baris GROQ_API_KEY=... tanpa tanda petik.")

model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
print("Model:           ", model)

try:
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {kunci}"},
        json={"model": model, "messages": [{"role": "user", "content": "Jawab satu kata: halo"}], "max_tokens": 20},
        timeout=15,
    )
except Exception as e:
    raise SystemExit(f"Gagal terhubung: {type(e).__name__}: {e}")

print("Kode status:     ", r.status_code)
print("Isi jawaban:     ", r.text[:400])