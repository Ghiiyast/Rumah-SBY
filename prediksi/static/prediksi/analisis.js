const dataAnalisis = JSON.parse(document.getElementById("data-analisis").textContent);
const warnaBatang = "rgba(59, 91, 219, 0.35)";
const warnaBatangAktif = "#3b5bdb";

if (window.Chart) {
  Chart.register({
    id: "labelNilaiBatang",
    afterDatasetsDraw(chart) {
      if (!chart.options.plugins.labelNilaiBatang) return;
      const { ctx } = chart;
      chart.data.datasets.forEach((dataset, indeks) => {
        const meta = chart.getDatasetMeta(indeks);
        meta.data.forEach((batang, nomor) => {
          const nilai = dataset.data[nomor];
          ctx.save();
          ctx.fillStyle = "#6b7280";
          ctx.font = "11px Segoe UI, sans-serif";
          ctx.textBaseline = "middle";
          if (chart.options.indexAxis === "y") {
            ctx.textAlign = "left";
            ctx.fillText(Number(nilai).toLocaleString("id-ID", { maximumFractionDigits: 2 }), batang.x + 5, batang.y);
          }
          ctx.restore();
        });
      });
    },
  });
}

function buatGrafikBatang(id, labels, values, tegak, warna) {
  const elemen = document.getElementById(id);
  if (!elemen || !window.Chart) return;
  new Chart(elemen, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: warna,
        borderRadius: 5,
      }],
    },
    options: {
      indexAxis: tegak ? "x" : "y",
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        labelNilaiBatang: id === "grafik-02",
      },
      scales: {
        x: { beginAtZero: true },
        y: { beginAtZero: true, reverse: !tegak },
      },
    },
  });
}

function buatGrafikSebaran() {
  const isi = dataAnalisis["03"];
  const elemen = document.getElementById("grafik-03");
  if (!elemen || !window.Chart) return;
  new Chart(elemen, {
    type: "scatter",
    data: {
      datasets: [{
        data: isi.points,
        backgroundColor: warnaBatangAktif,
        pointRadius: 2.5,
        pointHoverRadius: 4,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Luas tanah (m²)" },
          min: 0,
          max: isi.x_max || undefined,
        },
        y: {
          title: { display: true, text: "Harga (miliar Rp)" },
          min: 0,
          max: isi.y_max || undefined,
        },
      },
    },
  });
}

function muatPenjelasan() {
  const formulir = document.querySelector(".analisis-filter");
  const alamat = formulir ? formulir.dataset.urlPenjelasan : "";
  document.querySelectorAll(".analisis-ai[data-grafik]").forEach(async kartu => {
    const teks = kartu.querySelector(".analisis-ai-teks");
    const parameter = new URLSearchParams({
      grafik: kartu.dataset.grafik,
      kecamatan: document.getElementById("pilih-kecamatan").value,
      mode: document.getElementById("mode-harga").value,
    });
    try {
      const respons = await fetch(`${alamat}?${parameter.toString()}`);
      if (!respons.ok) throw new Error("Penjelasan tidak tersedia.");
      const hasil = await respons.json();
      teks.textContent = hasil.teks || "Penjelasan AI sedang tidak tersedia.";
    } catch (error) {
      teks.textContent = "Penjelasan AI sedang tidak tersedia.";
    }
  });
}

const distribusi = dataAnalisis["01"];
buatGrafikBatang(
  "grafik-01",
  distribusi.labels,
  distribusi.values,
  true,
  distribusi.values.map(nilai => distribusi.total && nilai === Math.max(...distribusi.values) ? warnaBatangAktif : warnaBatang),
);

const kecamatan = dataAnalisis["02"];
buatGrafikBatang(
  "grafik-02",
  kecamatan.labels,
  kecamatan.values,
  false,
  kecamatan.highlight.map(aktif => aktif ? warnaBatangAktif : warnaBatang),
);

buatGrafikSebaran();
muatPenjelasan();

const formulirAnalisis = document.querySelector(".analisis-filter");
if (formulirAnalisis) {
  formulirAnalisis.querySelectorAll("select").forEach(pilihan => {
    pilihan.addEventListener("change", () => formulirAnalisis.requestSubmit());
  });
}
