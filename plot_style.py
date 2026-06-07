"""
plot_style.py
=============
Utilidad compartida de estilo de figuras para todas las fases.

Punto clave: la detección de LaTeX. Comprobar que el binario `latex` existe
(p. ej. `latex --version`) NO basta: una instalación PARCIAL de TeX Live
(muy común en Debian/WSL2 sin `texlive-fonts-recommended`) tiene el binario
pero le faltan paquetes como `type1ec.sty`, de modo que matplotlib falla al
medir el string trivial 'lp' en plena `savefig`.

La detección robusta hace un RENDER DE PRUEBA real con el propio TexManager de
matplotlib. Si lanza cualquier excepción, se desactiva usetex y se usa el
mathtext serif interno (sin dependencias). Así las figuras siempre se generan.
"""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def latex_available() -> bool:
    """True solo si LaTeX puede REALMENTE renderizar (no solo si el binario existe).

    Hace un render de prueba con TexManager; cachea el resultado para no pagar
    el coste (ni los warnings) más de una vez por proceso.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.texmanager import TexManager

        # Activar usetex temporalmente y forzar un render real de DVI.
        prev = plt.rcParams.get("text.usetex", False)
        plt.rcParams["text.usetex"] = True
        try:
            TexManager().get_text_width_height_descent("lp", 12)
            return True
        finally:
            plt.rcParams["text.usetex"] = prev
    except Exception:
        return False


def apply_style(extra: dict | None = None) -> bool:
    """Aplica el estilo de publicación común y devuelve si usetex está activo.

    extra: rcParams adicionales/override por figura (opcional).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usetex = latex_available()
    base = {
        "text.usetex": usetex,
        "mathtext.fontset": "dejavuserif",
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 11,
        "legend.fontsize": 9.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "ytick.minor.visible": True,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
    if usetex:
        base["text.latex.preamble"] = r"\usepackage{amsmath}"
    if extra:
        base.update(extra)
    plt.rcParams.update(base)
    return usetex
