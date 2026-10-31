// Applies the saved theme before first paint (external file: the CSP forbids inline scripts).
(function () {
  try {
    var t = localStorage.getItem("akgpt.theme");
    if (t === "light" || t === "dark" || t === "system") {
      document.documentElement.setAttribute("data-theme", t);
    }
  } catch (e) { /* storage unavailable: fall back to system theme */ }
})();
