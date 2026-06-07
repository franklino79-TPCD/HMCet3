"""
smoke_test.py
=============
Validación rápida (red pequeña 4^4) de la integridad del motor HMC:

  1. Los defectos generados son co-cerrados:  delta n = 0.
  2. El leapfrog es reversible:  (theta,p) -> integrar -> invertir p ->
     integrar -> recupera el estado inicial a precision de maquina (x64).
  3. <exp(-dH)> ~ 1 sobre un lote de trayectorias (identidad de Jarzynski).
  4. El gradiente AD coincide con diferencias finitas.

Ejecutar:  python smoke_test.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from config import LatticeConfig
from action import total_action, action_grad
from hmc import leapfrog, hmc_step, init_gauge, init_ising
from lattice_math import codiff_2form, make_coclosed_defects

D = 4
lat = LatticeConfig(L=4, D=4)
beta, J, kappa = 1.0, 0.5, 0.2
eps, n_md = 0.01, 30

key = jax.random.PRNGKey(0)
k_g, k_s, k_n = jax.random.split(key, 3)
theta = init_gauge(k_g, lat.link_shape)
s = init_ising(k_s, lat.shape)
n = make_coclosed_defects(k_n, lat, amplitude=1)


def test_coclosed():
    err = float(jnp.max(jnp.abs(codiff_2form(n, D))))
    print(f"[1] co-cerradura  max|delta n| = {err:.3e}")
    assert err < 1e-9
    # Asegura que los defectos no son trivialmente cero.
    assert float(jnp.max(jnp.abs(n))) > 0.0
    print("    -> defectos co-cerrados y no triviales  OK")


def test_reversibility():
    p0 = jax.random.normal(jax.random.PRNGKey(7), theta.shape, dtype=theta.dtype)
    th1, p1 = leapfrog(theta, p0, s, n, beta, J, kappa, eps, n_md, D)
    # Invertir momento e integrar de vuelta.
    th2, p2 = leapfrog(th1, -p1, s, n, beta, J, kappa, eps, n_md, D)
    err_th = float(jnp.max(jnp.abs(th2 - theta)))
    err_p = float(jnp.max(jnp.abs(-p2 - p0)))
    print(f"[2] reversibilidad  d_theta={err_th:.2e}  d_p={err_p:.2e}")
    assert err_th < 1e-9 and err_p < 1e-9
    print("    -> leapfrog reversible a precision x64  OK")


def test_grad_vs_fd():
    g = action_grad(theta, s, n, beta, J, kappa, D)
    # Diferencia finita en un enlace.
    idx = (0, 0, 0, 0, 0)
    h = 1e-6
    tp = theta.at[idx].add(h)
    tm = theta.at[idx].add(-h)
    fd = (
        float(total_action(tp, s, n, beta, J, kappa, D))
        - float(total_action(tm, s, n, beta, J, kappa, D))
    ) / (2 * h)
    print(f"[3] AD={float(g[idx]):.6f}  vs  FD={fd:.6f}")
    assert abs(float(g[idx]) - fd) < 1e-5
    print("    -> jax.grad coincide con diferencias finitas  OK")


def test_crooks():
    keys = jax.random.split(jax.random.PRNGKey(123), 64)
    th = theta
    dHs = []
    for k in keys:
        th, _, dH = hmc_step(k, th, s, n, beta, J, kappa, eps, n_md, D)
        dHs.append(float(jnp.exp(-dH)))
    mean = sum(dHs) / len(dHs)
    print(f"[4] <exp(-dH)> = {mean:.4f}  (debe ~1)")
    assert 0.5 < mean < 1.8
    print("    -> identidad de fluctuacion satisfecha  OK")


if __name__ == "__main__":
    print("Smoke test  (red 4^4, x64) ...")
    test_coclosed()
    test_reversibility()
    test_grad_vs_fd()
    test_crooks()
    print("\nTODOS LOS TESTS PASARON.")
