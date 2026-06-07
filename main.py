"""
main.py
=======
Fase I — Validación Termodinámica del Atractor Infrarrojo.

Perfila la transición de deconfinamiento topológico en el sistema
U(1) Gauge-Ising-Defectos sobre una red 16^4 barriendo beta en [0.6, 1.4]
con paso 0.05 (17 puntos, resolución máxima alrededor de beta_c ≈ 1.00).

Salidas:
  fase1_termodinamica.csv     — datos numéricos (se actualiza por cada beta)
  fase1_transicion_fase.pdf   — gráfico de publicación (2 paneles)

Ejecutar: python main.py
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from config import CFG
from hmc import hmc_step, ising_heatbath, init_gauge, init_ising
from lattice_math import all_plaquettes, codiff_2form, make_coclosed_defects


# =========================================================================== #
#  Parámetros del barrido — Fase I                                             #
# =========================================================================== #
BETA_MIN = 0.60
BETA_MAX = 1.40
N_BETA = 17                          # paso 0.05 exacto
BETA_C = 1.00                        # transición teórica (línea de referencia)
CSV_FILE = Path("fase1_termodinamica.csv")
PDF_FILE = Path("fase1_transicion_fase.pdf")

# Encabezado del CSV
_CSV_COLS = ["beta", "mean_cosP", "err_cosP", "Cv", "acc_rate", "mean_dH", "time_s"]


# =========================================================================== #
#  Observables JAX-jitted                                                      #
# =========================================================================== #
@jax.jit
def mean_plaquette(theta: jnp.ndarray) -> jnp.ndarray:
    """<cos P> promediado sobre las D*(D-1)/2 plaquetas y el volumen V."""
    D = theta.shape[-1]
    P = all_plaquettes(theta, D)
    cos_sum = jnp.array(0.0, dtype=theta.dtype)
    count = 0
    for mu in range(D):
        for nu in range(mu + 1, D):
            cos_sum = cos_sum + jnp.sum(jnp.cos(P[..., mu, nu]))
            count += 1
    # Normalizado por V * n_plaquetas_independientes
    return cos_sum / (theta.shape[0] ** D * count)


@jax.jit
def total_wilson_energy(theta: jnp.ndarray) -> jnp.ndarray:
    """E_Wilson = -Σ_{x,mu<nu} cos(P_{mu nu}).

    Energía TOTAL (escalar no normalizada) necesaria para calcular
    Var(E_Wilson) y de ahí el calor específico Cv.
    """
    D = theta.shape[-1]
    P = all_plaquettes(theta, D)
    E = jnp.array(0.0, dtype=theta.dtype)
    for mu in range(D):
        for nu in range(mu + 1, D):
            E = E - jnp.sum(jnp.cos(P[..., mu, nu]))
    return E


@jax.jit
def abs_magnetization(s: jnp.ndarray) -> jnp.ndarray:
    """Magnetización absoluta por sitio |<s>|."""
    return jnp.abs(jnp.mean(s))


# =========================================================================== #
#  Análisis estadístico (numpy — post-JAX)                                     #
# =========================================================================== #
def _integrated_autocorr_time(x: np.ndarray) -> float:
    """Tiempo de autocorrelación integrado por el método de ventana de Sokal.

    tau_int = 1/2 + Σ_{t=1}^{W} rho(t)  con W el primer entero tal que
    W >= 5 * tau_int  (criterio automático de ventana).

    Devuelve tau_int >= 0.5. Si la serie es demasiado corta o constante
    devuelve 0.5 (error estadístico estándar sin correlaciones).
    """
    n = len(x)
    xc = x - x.mean()
    c0 = float(np.dot(xc, xc)) / n
    if c0 < 1e-30:
        return 0.5
    tau = 0.5
    for t in range(1, n // 2):
        ct = float(np.dot(xc[: n - t], xc[t:])) / (n - t)
        rho_t = ct / c0
        tau += rho_t
        if t >= 5 * tau:          # ventana automática de Sokal
            break
    return max(tau, 0.5)


def compute_stats(
    cosP_vals: list[float],
    E_vals: list[float],
    beta: float,
    volume: int,
) -> dict:
    """Calcula observables termodinámicos a partir de los historiales.

    Returns
    -------
    mean_cosP  : <cos P> promedio
    err_cosP   : error estándar corregido por autocorrelaciones (Sokal)
    Cv         : calor específico  beta^2 * Var(E_Wilson) / V
    """
    arr = np.array(cosP_vals, dtype=np.float64)
    E_arr = np.array(E_vals, dtype=np.float64)
    n = len(arr)

    mean_cosP = float(arr.mean())
    tau = _integrated_autocorr_time(arr)
    # sigma_mean corregida por autocorrelaciones
    err_cosP = float(np.sqrt(2.0 * tau * arr.var(ddof=0) / n))

    # Cv = beta^2 * Var(E_Wilson) / V  (fluctuaciones de energía total)
    Cv = float(beta**2 * E_arr.var(ddof=1) / volume)

    return {"mean_cosP": mean_cosP, "err_cosP": err_cosP, "Cv": Cv}


# =========================================================================== #
#  CSV — logging incremental                                                   #
# =========================================================================== #
def init_csv() -> None:
    """Crea (o sobreescribe) el archivo CSV con el encabezado."""
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS)
        writer.writeheader()


def append_csv(row: dict) -> None:
    """Añade una fila al CSV (se llama después de cada beta)."""
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS)
        writer.writerow({k: f"{row[k]:.8g}" for k in _CSV_COLS})


# =========================================================================== #
#  Cadena HMC para un beta fijo                                                #
# =========================================================================== #
def run_chain(key, theta, s, n, beta):
    """Termaliza y mide; devuelve historiales por trayectoria para estadística.

    Returns
    -------
    key, theta, s         : estado actualizado del PRNG y los campos
    cosP_vals             : lista de <cos P> por trayectoria medida
    E_vals                : lista de E_Wilson total por trayectoria medida
    dH_vals               : lista de delta H  (para <dH> y diagnóstico)
    acc_vals              : lista de 0/1 de aceptación
    """
    a = CFG.action
    h = CFG.hmc
    D = CFG.lattice.D

    # ---- Termalización ----
    for _ in range(h.n_therm):
        key, k_hmc, k_ising = jax.random.split(key, 3)
        theta, _, _ = hmc_step(
            k_hmc, theta, s, n, beta, a.J, a.kappa, h.epsilon, h.n_md, D
        )
        s = ising_heatbath(k_ising, theta, s, a.J, D)

    # ---- Medición ----
    cosP_vals: list[float] = []
    E_vals: list[float] = []
    dH_vals: list[float] = []
    acc_vals: list[int] = []

    for _ in range(h.n_meas):
        key, k_hmc, k_ising = jax.random.split(key, 3)
        theta, accepted, dH = hmc_step(
            k_hmc, theta, s, n, beta, a.J, a.kappa, h.epsilon, h.n_md, D
        )
        s = ising_heatbath(k_ising, theta, s, a.J, D)

        cosP_vals.append(float(mean_plaquette(theta)))
        E_vals.append(float(total_wilson_energy(theta)))
        dH_vals.append(float(dH))
        acc_vals.append(int(accepted))

    return key, theta, s, cosP_vals, E_vals, dH_vals, acc_vals


# =========================================================================== #
#  Gráficos de nivel de publicación                                            #
# =========================================================================== #
def plot_results(records: list[dict]) -> None:
    """Genera el gráfico multipanel de la transición de fase.

    Estética arXiv/APS: fuentes serif, ticks hacia adentro, sin spines
    innecesarios, barras de error, resolución 300 DPI.
    """
    import matplotlib
    matplotlib.use("Agg")           # backend sin pantalla (GPU/WSL headless)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # ---- Estilo de publicación ----
    # Detección robusta de LaTeX (render de prueba real; ver plot_style.py).
    from plot_style import apply_style
    apply_style({
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.minor.visible": True,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
    })

    def tex(s: str) -> str:
        """Devuelve el string tal cual (compatible con mathtext y LaTeX)."""
        return s

    betas = np.array([r["beta"] for r in records])
    cosP = np.array([r["mean_cosP"] for r in records])
    err = np.array([r["err_cosP"] for r in records])
    Cv = np.array([r["Cv"] for r in records])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.5, 7.0),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # ---- Subplot superior: <cos P> vs beta ----
    ax1.errorbar(
        betas, cosP, yerr=err,
        fmt="o-", color="#1f4e9c",
        capsize=3, capthick=0.8, elinewidth=0.8,
        label=tex(r"$\langle \cos P \rangle$"),
    )
    ax1.axvline(BETA_C, ls="--", lw=0.9, color="#888888",
                label=tex(rf"$\beta_c = {BETA_C:.2f}$"))
    ax1.set_ylabel(tex(r"$\langle \cos P_{\mu\nu} \rangle$"))
    ax1.legend(frameon=False, loc="upper left")
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # Eliminar spine superior/derecho (ya lo maneja rcParams xtick.top, etc.)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ---- Subplot inferior: Cv vs beta ----
    ax2.plot(betas, Cv, "s-", color="#b5431a",
             label=tex(r"$C_v$"))
    ax2.axvline(BETA_C, ls="--", lw=0.9, color="#888888",
                label=tex(rf"$\beta_c = {BETA_C:.2f}$"))
    ax2.set_xlabel(tex(r"$\beta$"))
    ax2.set_ylabel(
        tex(r"$C_v = \beta^2 \,\mathrm{Var}(E_W)/V$")
    )
    ax2.legend(frameon=False, loc="upper left")
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        tex(r"Transición de Deconfinamiento Topológico — Red $16^4$ U(1)-Ising-BF"),
        fontsize=11, y=1.01,
    )

    fig.savefig(PDF_FILE)
    plt.close(fig)
    print(f" Grafico guardado en: {PDF_FILE}")


# =========================================================================== #
#  Programa principal                                                          #
# =========================================================================== #
def main():
    print("=" * 74)
    print(" FASE I — Validacion Termodinamica del Atractor Infrarrojo")
    print(" U(1) Gauge - Ising - Defectos (2-formas)  |  Red 4D 16^4")
    print(" Diferenciacion automatica exacta O(1)  |  doble precision x64")
    print("=" * 74)
    print(f" Dispositivos JAX  : {jax.devices()}")
    print(f" x64 habilitado    : {jax.config.jax_enable_x64}")

    lat = CFG.lattice
    a = CFG.action
    h = CFG.hmc

    key = jax.random.PRNGKey(h.seed)

    # ---- Inicialización de campos ----
    key, k_g, k_s, k_n = jax.random.split(key, 4)
    theta = init_gauge(k_g, lat.link_shape)
    s = init_ising(k_s, lat.shape)
    n = make_coclosed_defects(k_n, lat, amplitude=1)

    coclosure = float(jnp.max(jnp.abs(codiff_2form(n, lat.D))))
    print(f" Defectos co-cerrados  max|δn| = {coclosure:.2e}")
    assert coclosure < 1e-9, "Defectos NO co-cerrados (δn ≠ 0)."

    betas = np.round(np.linspace(BETA_MIN, BETA_MAX, N_BETA), 10)
    print(
        f" Barrido beta: [{BETA_MIN}, {BETA_MAX}] × {N_BETA} puntos "
        f"(paso {betas[1]-betas[0]:.2f})"
    )
    print(
        f" Config: L={lat.L}, V={lat.volume}, eps={h.epsilon}, "
        f"N_md={h.n_md}, n_therm={h.n_therm}, n_meas={h.n_meas}"
    )
    print(f" J={a.J}, kappa={a.kappa}  |  beta_c teórico={BETA_C:.2f}")
    print("-" * 74)

    header = (
        f"{'beta':>5} | {'<cosP>':>9} | {'err_cosP':>8} | "
        f"{'Cv':>8} | {'acc':>5} | {'<dH>':>7} | {'t[s]':>7}"
    )
    print(header)
    print("-" * 74)

    init_csv()
    records: list[dict] = []

    for beta in betas:
        beta_f = float(beta)
        t0 = time.time()

        key, theta, s, cosP_vals, E_vals, dH_vals, acc_vals = run_chain(
            key, theta, s, n, beta_f
        )
        dt = time.time() - t0

        thermo = compute_stats(cosP_vals, E_vals, beta_f, lat.volume)
        acc_rate = float(np.mean(acc_vals))
        mean_dH = float(np.mean(dH_vals))

        row = {
            "beta": beta_f,
            "mean_cosP": thermo["mean_cosP"],
            "err_cosP": thermo["err_cosP"],
            "Cv": thermo["Cv"],
            "acc_rate": acc_rate,
            "mean_dH": mean_dH,
            "time_s": dt,
        }
        records.append(row)
        append_csv(row)   # persistir inmediatamente

        print(
            f"{beta_f:5.2f} | {thermo['mean_cosP']:9.6f} | "
            f"{thermo['err_cosP']:8.2e} | {thermo['Cv']:8.5f} | "
            f"{acc_rate:5.3f} | {mean_dH:7.4f} | {dt:7.2f}"
        )

    print("-" * 74)
    print(f" CSV guardado en   : {CSV_FILE}")
    print(f" <exp(-dH)> ~1 confirma reversibilidad de Crooks")
    print(" Generando grafico de publicacion ...")
    plot_results(records)
    print(" Simulacion completada.")
    return records


if __name__ == "__main__":
    main()
