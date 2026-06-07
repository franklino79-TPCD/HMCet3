"""
fase2_index.py
==============
Fase II — Resolución del Confinamiento Quiral (N_gen = 3).

Mide la carga topológica geométrica Q de la red en la fase deconfinada
(β = 1.10) usando el operador de Atiyah-Singer discreto:

    Q = 1/(32π²) · Σ_x Σ_{μνρσ} ε_{μνρσ} sin(P_{μν}(x)) sin(P_{ρσ}(x))

La suma se evalúa con un único `jnp.einsum` vectorizado sobre todos los
16^4 sitios. Las plaquetas se construyen con jnp.roll (sin grafos), igual
que en lattice_math.py.

Física:
  * En U(1) compacto con defectos de 2-forma (BF), los monopolos magnéticos
    actúan como instantones y pueden cargar Q.
  * El invariante η de frontera co-dimensión 6 del modelo pre-geométrico
    predice |<Q>| → N_gen = 3.
  * La susceptibilidad topológica χ_t = <Q²>/V mide las fluctuaciones
    instantónicas.

Salidas
-------
  fase2_carga_topologica.pdf  — histograma de publicación (300 DPI)
  (valores de Q se imprimen en tiempo real en la terminal)

Ejecutar:  python fase2_index.py
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from config import CFG
from hmc import hmc_step, ising_heatbath, init_gauge, init_ising
from lattice_math import all_plaquettes, codiff_2form, make_coclosed_defects


# =========================================================================== #
#  Parámetros de la Fase II                                                    #
# =========================================================================== #
BETA = 1.10       # fase deconfinada (por encima de β_c ≈ 1.00)
J = 0.50
KAPPA = 0.20
N_THERM = 500     # trayectorias de termalización
N_MEAS = 2000     # trayectorias de medición

PDF_FILE = Path("fase2_carga_topologica.pdf")


# =========================================================================== #
#  Tensor de Levi-Civita  ε_{μνρσ}  (constante de compilación para XLA)       #
# =========================================================================== #
def _build_levi_civita_4d() -> jnp.ndarray:
    """Tensor completamente antisimétrico ε_{μνρσ} en 4D.

    Calculado una sola vez en tiempo de importación del módulo.
    Al ser una constante de JAX capturada por cierre en la función @jit,
    XLA la trata como inmediata (literal baked-in) y no genera tráfico
    de memoria durante el kernel de carga topológica.
    """
    def perm_sign(p: tuple) -> int:
        s, lst = 1, list(p)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i] > lst[j]:
                    s = -s
        return s

    eps = np.zeros((4, 4, 4, 4), dtype=np.float64)
    for perm in itertools.permutations(range(4)):
        eps[perm] = perm_sign(perm)
    return jnp.array(eps)


_EPS4: jnp.ndarray = _build_levi_civita_4d()  # shape (4,4,4,4)


# =========================================================================== #
#  Operador de carga topológica  (JIT-compilado, vectorizado)                  #
# =========================================================================== #
@jax.jit
def compute_topological_charge(theta: jnp.ndarray) -> jnp.ndarray:
    """Carga topológica discreta Q por el operador geométrico sin-sin.

    Implementación:
      1. all_plaquettes → tensor P_{μν}(x), forma (L,L,L,L,D,D).
         Construido con jnp.roll sin grafos (contigüidad de memoria garantizada).
      2. sinP = sin(P)  — preserva rango topológico y es diferenciable en R.
      3. Contracción con ε vía einsum:
             q(x) = ε_{mnrs} sinP_{mn}(x) sinP_{rs}(x)
         XLA fusiona el einsum en un único kernel GPU; el tensor ε está
         baked-in como constante escalar (no genera carga de VRAM extra).
      4. Q = Σ_x q(x) / (32π²).

    La antisimetría de P garantiza que los 24 términos no nulos de ε
    contribuyen de forma consistente:
       ε_{mnrs} sin(P_{mn}) sin(P_{rs})  con P_{nm} = -P_{mn}.
    """
    D = theta.shape[-1]
    P = all_plaquettes(theta, D)       # (L,L,L,L,4,4)
    sinP = jnp.sin(P)
    # Contracción completa sobre índices de Lorentz; '...' = índices espaciales
    q_density = jnp.einsum("mnrs,...mn,...rs->...", _EPS4, sinP, sinP)
    return jnp.sum(q_density) / (32.0 * jnp.pi ** 2)


@jax.jit
def topological_susceptibility(Q_arr: jnp.ndarray, volume: int) -> jnp.ndarray:
    """χ_t = <Q²> / V  (susceptibilidad topológica por unidad de volumen)."""
    return jnp.mean(Q_arr ** 2) / volume


# =========================================================================== #
#  Análisis estadístico (numpy — post-JAX)                                     #
# =========================================================================== #
def _sokal_autocorr_time(x: np.ndarray) -> float:
    """τ_int por el criterio de ventana automática de Madras-Sokal."""
    n = len(x)
    xc = x - x.mean()
    c0 = float(np.dot(xc, xc)) / n
    if c0 < 1e-30:
        return 0.5
    tau = 0.5
    for t in range(1, n // 2):
        ct = float(np.dot(xc[: n - t], xc[t:])) / (n - t)
        tau += ct / c0
        if t >= 5 * tau:
            break
    return max(tau, 0.5)


def summarize(Q_vals: list[float], volume: int) -> dict:
    """Estadísticos de Q sobre el historial de medición."""
    arr = np.array(Q_vals, dtype=np.float64)
    n = len(arr)
    mean_Q = arr.mean()
    tau = _sokal_autocorr_time(arr)
    err_Q = np.sqrt(2.0 * tau * arr.var(ddof=0) / n)
    chi_t = float(np.mean(arr ** 2)) / volume

    return {
        "n": n,
        "mean_Q": mean_Q,
        "err_Q": err_Q,
        "std_Q": arr.std(),
        "mean_Q2": float(np.mean(arr ** 2)),
        "chi_t": chi_t,
        "tau_int": tau,
        "Q_min": arr.min(),
        "Q_max": arr.max(),
    }


# =========================================================================== #
#  Cadena HMC Fase II                                                          #
# =========================================================================== #
def run_phase2(
    key: jnp.ndarray,
    theta: jnp.ndarray,
    s: jnp.ndarray,
    n: jnp.ndarray,
) -> tuple:
    """Ejecuta termalización y barrido de medición para Fase II.

    Returns
    -------
    key, theta, s  : estado actualizado
    Q_vals         : lista de Q por trayectoria medida
    """
    D = CFG.lattice.D
    eps = CFG.hmc.epsilon
    n_md = CFG.hmc.n_md

    # ---- Termalización ----
    sep = "─" * 64
    print(f"\n{sep}")
    print(f" Termalización  β={BETA}  {N_THERM} trayectorias HMC")
    print(sep)
    t0 = time.time()
    for i in range(N_THERM):
        key, k_hmc, k_ising = jax.random.split(key, 3)
        theta, _, _ = hmc_step(k_hmc, theta, s, n, BETA, J, KAPPA, eps, n_md, D)
        s = ising_heatbath(k_ising, theta, s, J, D)
        if (i + 1) % 100 == 0:
            Q = float(compute_topological_charge(theta))
            print(f"  therm {i+1:4d}/{N_THERM}   Q = {Q:+.4f}")
    print(f" Termalización completada en {time.time()-t0:.1f} s\n")

    # ---- Medición ----
    print(sep)
    print(f" Medición  {N_MEAS} trayectorias  —  Q en tiempo real")
    print(sep)
    print(f"  {'traj':>5}  {'Q':>10}  acc  dH")
    print(f"  {'─'*5}  {'─'*10}  ───  ──────")

    Q_vals: list[float] = []
    t1 = time.time()
    for i in range(N_MEAS):
        key, k_hmc, k_ising = jax.random.split(key, 3)
        theta, accepted, dH = hmc_step(k_hmc, theta, s, n, BETA, J, KAPPA, eps, n_md, D)
        s = ising_heatbath(k_ising, theta, s, J, D)

        Q = float(compute_topological_charge(theta))
        Q_vals.append(Q)

        acc_sym = "✓" if int(accepted) else "✗"
        print(f"  {i+1:5d}  {Q:+10.5f}  {acc_sym}    {float(dH):+.4f}")

    print(f"\n Medición completada en {time.time()-t1:.1f} s")
    return key, theta, s, Q_vals


# =========================================================================== #
#  Resumen terminal                                                             #
# =========================================================================== #
def print_summary(stats: dict) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  RESULTADOS  Fase II — Carga Topológica  Q")
    print(sep)
    print(f"  N_meas             = {stats['n']}")
    print(f"  <Q>                = {stats['mean_Q']:+.6f}  ±  {stats['err_Q']:.6f}")
    print(f"  |<Q>|              = {abs(stats['mean_Q']):.6f}")
    print(f"  std(Q)             = {stats['std_Q']:.6f}")
    print(f"  <Q²>               = {stats['mean_Q2']:.6f}")
    print(f"  χ_t = <Q²>/V       = {stats['chi_t']:.4e}")
    print(f"  τ_int (Sokal)      = {stats['tau_int']:.2f}")
    print(f"  Q ∈ [{stats['Q_min']:+.3f}, {stats['Q_max']:+.3f}]")
    print(f"\n  N_gen ~ |<Q>|  =  {abs(stats['mean_Q']):.3f}   (predicción: 3)")
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
    apply_style({
        "legend.fontsize": 10,
        "xtick.minor.visible": True,
        "lines.linewidth": 1.4,
    })

    arr = np.array(Q_vals)
    mean_Q = stats["mean_Q"]
    err_Q = stats["err_Q"]

    # Bins centrados en enteros: [n-0.5, n+0.5]
    q_lo = int(np.floor(arr.min())) - 1
    q_hi = int(np.ceil(arr.max())) + 1
    bins = np.arange(q_lo - 0.5, q_hi + 1.0, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.hist(
        arr, bins=bins, color="#1f4e9c", alpha=0.78,
        edgecolor="white", linewidth=0.5,
        label=f"$Q$ muestras  ($N = {stats['n']}$)",
    )
    # <Q>
    ax.axvline(
        mean_Q, color="#b5431a", lw=1.6, ls="-",
        label=f"$\\langle Q \\rangle = {mean_Q:+.3f} \\pm {err_Q:.3f}$",
    )
    # |<Q>| — predicción N_gen
    ax.axvline(
        abs(mean_Q), color="#2ca05a", lw=1.2, ls="--",
        label=f"$|\\langle Q \\rangle| \\approx N_{{\\rm gen}} = {abs(mean_Q):.2f}$",
    )
    # Marca entera más cercana a |<Q>|
    N_gen_int = round(abs(mean_Q))
    ax.axvline(
        N_gen_int, color="#888888", lw=0.9, ls=":",
        label=f"$N_{{\\rm gen}} = {N_gen_int}$ (predicción)",
    )

    ax.set_xlabel("$Q$")
    ax.set_ylabel("Frecuencia")
    ax.set_title(
        f"Carga Topológica — Red $16^4$,  "
        f"$\\beta = {BETA}$,  $\\kappa = {KAPPA}$\n"
        f"$\\langle Q \\rangle = {mean_Q:+.4f} \\pm {err_Q:.4f}$,"
        f"  $\\chi_t = {stats['chi_t']:.3e}$"
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
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
    print("=" * 64)
    print(" FASE II — Resolución del Confinamiento Quiral")
    print(" Índice Topológico Q (Atiyah-Singer lattice)")
    print(f" U(1)-Ising-Defectos | Red {CFG.lattice.L}^4 | β={BETA} | κ={KAPPA}")
    print("=" * 64)
    print(f" Dispositivos JAX  : {jax.devices()}")
    print(f" x64 habilitado    : {jax.config.jax_enable_x64}")

    lat = CFG.lattice

    # Semilla diferente a Fase I para decorrelacionar los ensembles
    key = jax.random.PRNGKey(CFG.hmc.seed + 42)
    key, k_g, k_s, k_n = jax.random.split(key, 4)

    theta = init_gauge(k_g, lat.link_shape)
    s = init_ising(k_s, lat.shape)
    n = make_coclosed_defects(k_n, lat, amplitude=1)

    coclosure = float(jnp.max(jnp.abs(codiff_2form(n, lat.D))))
    print(f" Defectos co-cerrados  max|δn| = {coclosure:.2e}")
    assert coclosure < 1e-9, "δn ≠ 0: defectos no co-cerrados."

    # Warm-up JIT: compilar compute_topological_charge antes del loop principal
    print(" Compilando kernels JIT (warm-up) ...")
    _ = float(compute_topological_charge(theta))
    print(" Kernels compilados.\n")

    key, theta, s, Q_vals = run_phase2(key, theta, s, n)

    stats = summarize(Q_vals, lat.volume)
    print_summary(stats)
    plot_histogram(Q_vals, stats)

    print(" Fase II completada.")


if __name__ == "__main__":
    main()
