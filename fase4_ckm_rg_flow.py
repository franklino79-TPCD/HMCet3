"""
fase4_ckm_rg_flow.py
====================
Fase IV — Métrica de Intersección y Flujo RG de la Matriz CKM.

Toma los valores "desnudos" (bare) de los elementos CKM derivados de la
geometría de intersección codim-7 a la escala M_X y los evoluciona con las
ecuaciones del Grupo de Renormalización (RG) del Modelo Estándar a 1-loop
hasta la escala electrodébil M_Z, contrastando con los valores PDG.

================================  NOTA DE HONESTIDAD CIENTÍFICA  ===============
Este script usa RGEs del SM a 1-loop GENUINAS (gauge g1,g2,g3; Yukawa top y_t;
elementos CKM dominados por y_t). NO se ajustan coeficientes para forzar el
resultado. Las conclusiones que de aquí salen son:

  * V_us (ángulo de Cabibbo): el valor desnudo geométrico θ_c = C_Σ/ln(n2/n1)
    da |V_us| ≈ 0.223 frente a PDG 0.2243  →  acierto real (~0.5 %).

  * |V_cb|, |V_ub|: el running SM a 1-loop es un efecto de ~10 %. Los valores
    desnudos del ansatz geométrico quedan a un factor ~2 (cb) y ~5 (ub) de
    PDG. El flujo RG NO cierra esa brecha. El script lo REPORTA con su
    desviación porcentual real en lugar de ocultarlo.

El término dominante del flujo CKM es correcto físicamente; lo que NO está
derivado de primeros principios son los coeficientes del ansatz desnudo
(C_Σ, ln n3, n1/n3). Cerrar la brecha requeriría o bien derivar esos
coeficientes de la métrica de intersección real, o efectos de 2-loop /
umbrales / nueva física entre M_Z y M_X.
===============================================================================

Ejecutar:  python fase4_ckm_rg_flow.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


# =========================================================================== #
#  Escalas y constantes                                                        #
# =========================================================================== #
M_Z = 91.2                 # GeV — escala electrodébil
M_X = 6.3e13               # GeV — escala de ruptura (pregeométrica)
T_X = np.log(M_X / M_Z)    # variable RG t = ln(μ/M_Z) en M_X

PI2 = 16.0 * np.pi ** 2    # 16π² (denominador 1-loop)

# Atractor topológico (números de generación)
N1, N2, N3 = 1, 207, 17
C_SIGMA = 1.2              # coeficiente de traslape de intersección

# Valores PDG en M_Z (referencia empírica)
VUS_PDG = 0.2243
VCB_PDG = 0.041
VUB_PDG = 0.0038

# Condiciones iniciales del SM en M_Z (normalización GUT para g1)
G1_MZ = 0.461              # sqrt(5/3)·g_Y
G2_MZ = 0.652
G3_MZ = 1.218              # α_s(M_Z) = g3²/4π ≈ 0.118
YT_MZ = 0.95               # y_t(M_Z) ≈ √2·m_t/v

PDF_FILE = Path("fase4_ckm_flow.pdf")


# =========================================================================== #
#  1. Valores desnudos geométricos en M_X                                     #
# =========================================================================== #
def geometric_bare_ckm() -> dict:
    """Valores CKM desnudos desde el atractor de intersección codim-7.

        θ_c⁽⁰⁾   = C_Σ / ln(n2/n1)
        |V_us|⁽⁰⁾ = sin(θ_c⁽⁰⁾)
        |V_cb|⁽⁰⁾ = sin²(θ_c⁽⁰⁾) / ln(n3)
        |V_ub|⁽⁰⁾ = sin³(θ_c⁽⁰⁾) · (n1/n3)
    """
    theta_c = C_SIGMA / np.log(N2 / N1)
    s = np.sin(theta_c)
    return {
        "theta_c": theta_c,
        "Vus": s,
        "Vcb": s ** 2 / np.log(N3),
        "Vub": s ** 3 * (N1 / N3),
    }


# =========================================================================== #
#  2. RGEs del Modelo Estándar a 1-loop                                        #
# =========================================================================== #
# Coeficientes beta de gauge a 1-loop (SM): dg_i/dt = b_i g_i³/(16π²)
B_GAUGE = np.array([41.0 / 10.0, -19.0 / 6.0, -7.0])


def rhs_gauge_yukawa(t: float, y: np.ndarray) -> list:
    """Sistema acoplado (g1, g2, g3, y_t) a 1-loop.

        dg_i/dt = b_i g_i³ / (16π²)
        dy_t/dt = y_t/(16π²) · [ 9/2 y_t² − 8 g3² − 9/4 g2² − 17/20 g1² ]
    """
    g1, g2, g3, yt = y
    dg1 = B_GAUGE[0] * g1 ** 3 / PI2
    dg2 = B_GAUGE[1] * g2 ** 3 / PI2
    dg3 = B_GAUGE[2] * g3 ** 3 / PI2
    dyt = yt / PI2 * (4.5 * yt ** 2 - 8.0 * g3 ** 2
                      - 2.25 * g2 ** 2 - 0.85 * g1 ** 2)
    return [dg1, dg2, dg3, dyt]


def ckm_top_coefficient(yt: float) -> float:
    """Coeficiente del flujo CKM dominado por el top:  (3/2) y_t² / (16π²).

    Aplica a |V_ub|, |V_cb|, |V_td|, |V_ts| (elementos que conectan con la
    tercera generación). Los elementos 1-2 (|V_us|, |V_cd|, ...) son
    esencialmente invariantes bajo el RG y se dejan fijos.
    """
    return 1.5 * yt ** 2 / PI2


# =========================================================================== #
#  3. Integración del flujo                                                    #
# =========================================================================== #
def run_rg_flow(n_pts: int = 400) -> dict:
    """Integra el flujo RG y devuelve trayectorias y valores finales.

    Estrategia de dos pasos (problema de contorno en dos puntos):
      (A) Integra (gauge, y_t) HACIA ARRIBA desde M_Z (IC conocidas) hasta
          M_X para obtener y_t(μ) en todo el rango.
      (B) Integra los elementos CKM HACIA ABAJO desde M_X (valores desnudos
          geométricos) hasta M_Z, usando y_t(μ) interpolado.
    """
    bare = geometric_bare_ckm()

    # --- (A) gauge + y_t : M_Z -> M_X ---
    sol_up = solve_ivp(
        rhs_gauge_yukawa, [0.0, T_X], [G1_MZ, G2_MZ, G3_MZ, YT_MZ],
        dense_output=True, rtol=1e-10, atol=1e-13,
    )
    yt_of_t = lambda t: sol_up.sol(t)[3]
    g1X, g2X, g3X, ytX = sol_up.y[:, -1]

    # --- (B) CKM : M_X -> M_Z ---
    def rhs_ckm(t, v):
        k = ckm_top_coefficient(yt_of_t(t))
        return [-k * v[0], -k * v[1]]   # d|V_cb|/dt, d|V_ub|/dt

    t_grid = np.linspace(T_X, 0.0, n_pts)
    sol_dn = solve_ivp(
        rhs_ckm, [T_X, 0.0], [bare["Vcb"], bare["Vub"]],
        t_eval=t_grid, dense_output=True, rtol=1e-10, atol=1e-14,
    )
    Vcb_traj, Vub_traj = sol_dn.y
    mu_grid = M_Z * np.exp(t_grid)   # escala física μ

    return {
        "bare": bare,
        "uv": {"g1": g1X, "g2": g2X, "g3": g3X, "yt": ytX,
               "alpha_s": g3X ** 2 / (4 * np.pi)},
        "mu_grid": mu_grid,
        "Vcb_traj": Vcb_traj,
        "Vub_traj": Vub_traj,
        "Vcb_MZ": Vcb_traj[-1],
        "Vub_MZ": Vub_traj[-1],
        "Vus_MZ": bare["Vus"],   # ~invariante bajo RG
    }


# =========================================================================== #
#  4. Figura de publicación                                                    #
# =========================================================================== #
def plot_flow(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from plot_style import apply_style
    apply_style({
        "xtick.minor.visible": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
    })

    mu = res["mu_grid"]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))

    # Trayectorias RG
    ax.plot(mu, res["Vcb_traj"], "-", color="#1f4e9c", lw=1.6,
            label=r"$|V_{cb}|(\mu)$")
    ax.plot(mu, res["Vub_traj"] * 10.0, "-", color="#b5431a", lw=1.6,
            label=r"$|V_{ub}|(\mu)\times 10$")

    # Líneas de referencia PDG en M_Z
    ax.axhline(VCB_PDG, ls="--", lw=1.0, color="#1f4e9c", alpha=0.7,
               label=rf"PDG $|V_{{cb}}| = {VCB_PDG}$")
    ax.axhline(VUB_PDG * 10.0, ls="--", lw=1.0, color="#b5431a", alpha=0.7,
               label=rf"PDG $|V_{{ub}}|\times 10 = {VUB_PDG*10:.3f}$")
    ax.axvline(M_Z, ls=":", lw=0.9, color="#555555")
    ax.plot([M_Z], [res["Vcb_MZ"]], "o", color="#1f4e9c", ms=6, zorder=5)
    ax.plot([M_Z], [res["Vub_MZ"] * 10.0], "o", color="#b5431a", ms=6, zorder=5)

    ax.set_xscale("log")
    ax.set_xlim(1e14, 1e2)        # descendente: M_X -> M_Z
    ax.set_xlabel(r"escala de energía  $\mu$  [GeV]")
    ax.set_ylabel(r"elemento CKM")
    ax.set_title(r"Flujo RG de la matriz CKM:  $M_X \to M_Z$  (SM 1-loop)")
    ax.grid(True, which="both")
    ax.legend(frameon=False, loc="center left", ncol=1)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(PDF_FILE)
    plt.close(fig)
    print(f"\n Figura guardada: {PDF_FILE}")


# =========================================================================== #
#  Programa principal                                                          #
# =========================================================================== #
def _pct(calc: float, ref: float) -> float:
    return abs(calc - ref) / ref * 100.0


def main() -> None:
    print("=" * 72)
    print(" FASE IV — Flujo RG de la Matriz CKM  (geometría codim-7 → M_Z)")
    print("=" * 72)
    print(f" M_X = {M_X:.3e} GeV   M_Z = {M_Z} GeV   t_X = ln(M_X/M_Z) = {T_X:.3f}")
    print(f" Atractor topológico: n1={N1}, n2={N2}, n3={N3},  C_Σ={C_SIGMA}")
    print("-" * 72)

    res = run_rg_flow()
    bare = res["bare"]
    uv = res["uv"]

    print(f" θ_c⁽⁰⁾ = {bare['theta_c']:.5f} rad")
    print(f" UV (M_X):  y_t = {uv['yt']:.4f},  α_s = {uv['alpha_s']:.4f}")
    print("-" * 72)

    # Tabla de contraste M_X (desnudo) → M_Z (corrido) vs PDG
    print(f" {'elemento':<10} {'desnudo M_X':>13} {'corrido M_Z':>13} "
          f"{'PDG':>9} {'desv %':>9}")
    print(" " + "-" * 70)
    rows = [
        ("|V_us|", bare["Vus"], res["Vus_MZ"], VUS_PDG),
        ("|V_cb|", bare["Vcb"], res["Vcb_MZ"], VCB_PDG),
        ("|V_ub|", bare["Vub"], res["Vub_MZ"], VUB_PDG),
    ]
    for name, b, mz, pdg in rows:
        print(f" {name:<10} {b:>13.6f} {mz:>13.6f} {pdg:>9.4f} {_pct(mz, pdg):>8.2f}%")

    print("=" * 72)
    print(" LECTURA HONESTA DEL RESULTADO:")
    print(f"   • |V_us| (Cabibbo): {_pct(res['Vus_MZ'], VUS_PDG):.1f}% de PDG → acierto geométrico.")
    print(f"   • |V_cb|: factor {VCB_PDG/res['Vcb_MZ']:.2f} corto de PDG tras el RG.")
    print(f"   • |V_ub|: factor {VUB_PDG/res['Vub_MZ']:.2f} corto de PDG tras el RG.")
    print("   El running SM 1-loop (~10%) NO cierra la brecha cb/ub: los")
    print("   coeficientes del ansatz desnudo no están derivados de la métrica.")
    print("=" * 72)

    plot_flow(res)
    print(" Fase IV completada.")


if __name__ == "__main__":
    main()
