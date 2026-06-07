"""
fase2b_tcs_index.py
===================
Fase II-b — Inyección de Topología TCS (Twisted Connected Sum).

Fuerza el vacío desde el sector trivial Q=0 (toro plano T^4) al sector
topológico Q = N_gen = 3, emulando el cuello cilíndrico de una geometría de
Suma Conexa Torcida con invariante eta no nulo.

MÉTODO — Flujo abeliano constante ('t Hooft flux) en T^4
--------------------------------------------------------
Para U(1) en un toro 4D con condiciones de contorno periódicas, una
configuración clásica de field-strength CONSTANTE en dos 2-toros ortogonales
tiene carga topológica EXACTAMENTE cuantizada:

        Q = m_{01} · m_{23}

donde m_{μν} ∈ Z es el flujo magnético (número de Chern) a través del plano
(μ,ν). Cada flujo se realiza con una plaqueta uniforme  b = 2π m / L²  más un
"twist" de fase coherente en los enlaces de frontera (construcción de
't Hooft) que cierra la periodicidad sin romper la cuantización.

  -> Para Q_target = 3 elegimos  m_{01} = 3,  m_{23} = 1  =>  Q = 3.

Observaciones de implementación:
  * El twist de frontera produce una plaqueta de esquina grande en la
    representación NO-COMPACTA (R), pero módulo 2π equivale a +b. Como el
    estimador geométrico usa sin(P) (periódico), lee el flujo correctamente.
  * NO se aplica jnp.mod en pasos intermedios: el twist se inyecta una sola
    vez en la inicialización (cold start), y la dinámica leapfrog evoluciona
    libremente en R como exige el motor O(1).
  * Los defectos de 2-forma se acoplan CONSTRUCTIVAMENTE: añadimos un fondo
    entero constante n_{01}, n_{23} (co-cerrado por ser espacialmente
    constante) alineado con los planos de flujo, anclando el sector Q=3 vía
    el término S_BF = κ Σ n P.

Salida:
  fase2b_carga_confinada.pdf  — histograma anclado en el atractor Q=3.

Ejecutar:  python fase2b_tcs_index.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from config import CFG
from hmc import hmc_step, ising_heatbath, init_ising
from lattice_math import (
    all_plaquettes,
    codiff_2form,
    plaquette,
    make_coclosed_defects,
)
# Reutilizamos el operador de carga topológica y el análisis de la Fase II
from fase2_index import compute_topological_charge, summarize


# =========================================================================== #
#  Parámetros de la Fase II-b                                                  #
# =========================================================================== #
BETA = 1.10
J = 0.50
KAPPA = 0.20
Q_TARGET = 3            # N_gen objetivo
N_THERM = 1000          # relajación UV preservando el flujo de fondo
N_MEAS = 2000           # medición de <Q>

PDF_FILE = Path("fase2b_carga_confinada.pdf")


# =========================================================================== #
#  Inicialización fría (cold start)                                            #
# =========================================================================== #
def init_gauge_cold(link_shape) -> jnp.ndarray:
    """Campo de gauge frío θ = 0.

    A diferencia del hot-start aleatorio, el cold-start NO contamina el sector
    topológico con ruido UV de amplitud O(1); así el flujo clásico inyectado
    (amplitud b ~ 2π·3/256 ≈ 0.07) NO queda sepultado por las fluctuaciones y
    el sector Q=3 sobrevive a la termalización.
    """
    return jnp.zeros(link_shape, dtype=jnp.float64)


# =========================================================================== #
#  Inyección de flujo topológico ('t Hooft constant flux)                     #
# =========================================================================== #
def _add_constant_flux(theta: jnp.ndarray, a: int, b: int, m: int, L: int) -> jnp.ndarray:
    """Añade un flujo magnético entero m a través del 2-toro (a,b).

    Construcción de 't Hooft que produce plaqueta uniforme  bf = 2π m / L²:

        θ_b(x)               += bf · x_a
        θ_a(x)|_{x_a = L-1}  += -bf · L · x_b      (twist de cierre)

    Resultado: P_{ab}(x) ≡ bf (mod 2π) en TODA la red, con flujo total 2π m.
    Vectorizado con broadcasting de las mallas de coordenadas (sin grafos).
    """
    bf = 2.0 * np.pi * m / (L ** 2)
    coords = jnp.meshgrid(*[jnp.arange(L) for _ in range(4)], indexing="ij")
    xa = coords[a].astype(theta.dtype)
    xb = coords[b].astype(theta.dtype)

    # Rampa lineal en θ_b: genera la plaqueta uniforme bf.
    theta = theta.at[..., b].add(bf * xa)
    # Twist de frontera en θ_a sólo en la capa x_a = L-1 (cierra la periodicidad).
    boundary = coords[a] == (L - 1)
    theta = theta.at[..., a].add(jnp.where(boundary, -bf * L * xb, 0.0))
    return theta


def inject_topological_flux(theta: jnp.ndarray, Q_target: int = 3, L: int = 16) -> jnp.ndarray:
    """Inyecta un fondo topológico clásico con carga Q = Q_target.

    Factoriza Q_target = m_{01} · m_{23} eligiendo m_{01} = Q_target, m_{23} = 1.
    Superpone los dos flujos abelianos constantes ortogonales sobre `theta`.
    """
    theta = _add_constant_flux(theta, 0, 1, Q_target, L)   # m_{01} = Q_target
    theta = _add_constant_flux(theta, 2, 3, 1, L)          # m_{23} = 1
    return theta


# =========================================================================== #
#  Defectos co-cerrados ALINEADOS con el flujo (acoplamiento constructivo)    #
# =========================================================================== #
def make_aligned_defects(key, cfg, Q_target: int = 3) -> jnp.ndarray:
    """Defectos co-cerrados + fondo entero constante alineado a los flujos.

    Un 2-forma espacialmente CONSTANTE es trivialmente co-cerrado (δn = 0,
    pues n(x) - n(x-ν) = 0). Lo superponemos a los defectos aleatorios
    co-cerrados de `make_coclosed_defects` para acoplar S_BF = κ Σ n P
    constructivamente a los planos de flujo (0,1) y (2,3), pinchando (pinning)
    el sector Q = Q_target durante la dinámica.
    """
    n = make_coclosed_defects(key, cfg, amplitude=1)
    # Fondo constante alineado: una unidad en cada plano de flujo.
    c01 = 1.0      # plano (0,1) — donde vive m_{01}
    c23 = 1.0      # plano (2,3) — donde vive m_{23}
    n = n.at[..., 0, 1].add(c01)
    n = n.at[..., 1, 0].add(-c01)
    n = n.at[..., 2, 3].add(c23)
    n = n.at[..., 3, 2].add(-c23)
    return jnp.round(n)


# =========================================================================== #
#  Cadena HMC Fase II-b                                                        #
# =========================================================================== #
def run_phase2b(key, theta, s, n) -> tuple:
    """Termaliza preservando el flujo y mide Q. Devuelve historial de Q."""
    D = CFG.lattice.D
    eps = CFG.hmc.epsilon
    n_md = CFG.hmc.n_md

    sep = "─" * 66
    Q0 = float(compute_topological_charge(theta))
    print(f"\n Carga inyectada (cold, pre-HMC):  Q = {Q0:+.5f}  (target {Q_TARGET})")

    # ---- Termalización: relaja UV, preserva el flujo de fondo ----
    print(f"\n{sep}")
    print(f" Termalización  β={BETA}  {N_THERM} trayectorias (preservando flujo)")
    print(sep)
    t0 = time.time()
    for i in range(N_THERM):
        key, k_hmc, k_ising = jax.random.split(key, 3)
        theta, _, _ = hmc_step(k_hmc, theta, s, n, BETA, J, KAPPA, eps, n_md, D)
        s = ising_heatbath(k_ising, theta, s, J, D)
        if (i + 1) % 100 == 0:
            Q = float(compute_topological_charge(theta))
            print(f"  therm {i+1:4d}/{N_THERM}   Q = {Q:+.4f}")
    print(f" Termalización completada en {time.time()-t0:.1f} s")

    # ---- Medición ----
    print(f"\n{sep}")
    print(f" Medición  {N_MEAS} trayectorias  —  Q en tiempo real")
    print(sep)
    print(f"  {'traj':>5}  {'Q':>10}  acc   dH")
    print(f"  {'─'*5}  {'─'*10}  ───   ──────")

    Q_vals: list[float] = []
    t1 = time.time()
    for i in range(N_MEAS):
        key, k_hmc, k_ising = jax.random.split(key, 3)
        theta, accepted, dH = hmc_step(k_hmc, theta, s, n, BETA, J, KAPPA, eps, n_md, D)
        s = ising_heatbath(k_ising, theta, s, J, D)

        Q = float(compute_topological_charge(theta))
        Q_vals.append(Q)
        if (i + 1) % 50 == 0 or i < 10:
            acc_sym = "✓" if int(accepted) else "✗"
            print(f"  {i+1:5d}  {Q:+10.5f}  {acc_sym}    {float(dH):+.4f}")

    print(f"\n Medición completada en {time.time()-t1:.1f} s")
    return key, theta, s, Q_vals


# =========================================================================== #
#  Resumen terminal                                                            #
# =========================================================================== #
def print_summary(stats: dict) -> None:
    sep = "=" * 66
    print(f"\n{sep}")
    print("  RESULTADOS  Fase II-b — Sector Topológico Inyectado (TCS)")
    print(sep)
    print(f"  Q_target           = {Q_TARGET}")
    print(f"  N_meas             = {stats['n']}")
    print(f"  <Q>                = {stats['mean_Q']:+.6f}  ±  {stats['err_Q']:.6f}")
    print(f"  redondeo entero    = {round(stats['mean_Q'])}")
    print(f"  std(Q)             = {stats['std_Q']:.6f}")
    print(f"  χ_t = <Q²>/V       = {stats['chi_t']:.4e}")
    print(f"  τ_int (Sokal)      = {stats['tau_int']:.2f}")
    print(f"  Q ∈ [{stats['Q_min']:+.3f}, {stats['Q_max']:+.3f}]")
    print(f"\n  N_gen = <Q> ≈ {stats['mean_Q']:.3f}  →  {round(stats['mean_Q'])} "
          f"generaciones quirales ancladas.")
    print(sep)


# =========================================================================== #
#  Histograma de publicación                                                   #
# =========================================================================== #
def plot_histogram(Q_vals: list[float], stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    from plot_style import apply_style
    apply_style({"xtick.minor.visible": True})

    arr = np.array(Q_vals)
    mean_Q = stats["mean_Q"]
    err_Q = stats["err_Q"]

    q_lo = int(np.floor(arr.min())) - 1
    q_hi = int(np.ceil(arr.max())) + 1
    bins = np.arange(q_lo - 0.5, q_hi + 1.0, 0.25)   # binning fino

    fig, ax = plt.subplots(figsize=(6.6, 4.8))

    ax.hist(
        arr, bins=bins, color="#2a6f4f", alpha=0.80,
        edgecolor="white", linewidth=0.4,
        label=f"$Q$ muestras ($N={stats['n']}$)",
    )
    # Atractor objetivo Q = 3
    ax.axvline(
        Q_TARGET, color="#b5431a", lw=1.8, ls="-",
        label=f"Atractor TCS  $Q = N_{{\\rm gen}} = {Q_TARGET}$",
    )
    # <Q> medido
    ax.axvline(
        mean_Q, color="#1f4e9c", lw=1.4, ls="--",
        label=f"$\\langle Q \\rangle = {mean_Q:+.3f} \\pm {err_Q:.3f}$",
    )

    ax.set_xlabel("$Q$")
    ax.set_ylabel("Frecuencia")
    ax.set_title(
        f"Confinamiento Quiral por Inyección TCS — Red $16^4$\n"
        f"$\\beta={BETA}$, $\\kappa={KAPPA}$,  "
        f"$\\langle Q \\rangle = {mean_Q:+.4f}$,  "
        f"$\\chi_t = {stats['chi_t']:.3e}$"
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(PDF_FILE)
    plt.close(fig)
    print(f"\n Histograma guardado: {PDF_FILE}")


# =========================================================================== #
#  Programa principal                                                          #
# =========================================================================== #
def main() -> None:
    print("=" * 66)
    print(" FASE II-b — Inyección de Topología TCS  (Q -> N_gen = 3)")
    print(" Twisted Connected Sum  |  flujo 't Hooft abeliano constante")
    print(f" U(1)-Ising-Defectos | Red {CFG.lattice.L}^4 | β={BETA} | κ={KAPPA}")
    print("=" * 66)
    print(f" Dispositivos JAX  : {jax.devices()}")
    print(f" x64 habilitado    : {jax.config.jax_enable_x64}")

    lat = CFG.lattice
    key = jax.random.PRNGKey(CFG.hmc.seed + 314)
    key, k_s, k_n = jax.random.split(key, 3)

    # 1. Cold start + inyección de flujo topológico Q=3
    theta = init_gauge_cold(lat.link_shape)
    theta = inject_topological_flux(theta, Q_target=Q_TARGET, L=lat.L)

    # 2. Materia de Ising (hot) y defectos co-cerrados ALINEADOS al flujo
    s = init_ising(k_s, lat.shape)
    n = make_aligned_defects(k_n, lat, Q_target=Q_TARGET)

    coclosure = float(jnp.max(jnp.abs(codiff_2form(n, lat.D))))
    print(f" Defectos co-cerrados (alineados)  max|δn| = {coclosure:.2e}")
    assert coclosure < 1e-9, "δn ≠ 0: los defectos alineados no son co-cerrados."

    # Warm-up JIT
    print(" Compilando kernels JIT (warm-up) ...")
    _ = float(compute_topological_charge(theta))
    print(" Kernels compilados.")

    key, theta, s, Q_vals = run_phase2b(key, theta, s, n)

    stats = summarize(Q_vals, lat.volume)
    print_summary(stats)
    plot_histogram(Q_vals, stats)
    print(" Fase II-b completada.")


if __name__ == "__main__":
    main()
