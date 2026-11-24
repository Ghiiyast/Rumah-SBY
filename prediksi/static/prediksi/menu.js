const menuSamping = document.getElementById("menu-samping");
const tombolMenu = document.querySelector(".sidebar-toggle");
const navigasiUtama = document.getElementById("navigasi-utama");

if (menuSamping && tombolMenu && navigasiUtama) {
  tombolMenu.addEventListener("click", () => {
    const terbuka = tombolMenu.getAttribute("aria-expanded") === "true";
    tombolMenu.setAttribute("aria-expanded", String(!terbuka));
    tombolMenu.setAttribute("aria-label", terbuka ? "Buka menu" : "Tutup menu");
    navigasiUtama.hidden = terbuka;
    menuSamping.classList.toggle("is-collapsed", terbuka);
  });
}
