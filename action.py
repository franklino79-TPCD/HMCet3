"""
action.py
=========
Acción euclídea total del sistema acoplado U(1)-Ising-Defectos en 4D.

    S = S_Wilson + S_Ising + S_BF

  S_Wilson = -beta  * sum_{x, mu<nu} cos(P_{mu nu}(x))
  S_Ising  = -J     * sum_{x, mu}    s_x * s_{x+mu} * cos(theta_mu(x))
  S_BF     =  kappa * sum_{x, mu<nu} n_{mu nu}(x) * P_{mu nu}(x)

El término B^F (S_BF) es LINEAL en la plaqueta. Es el responsable de las
posibles divergencias de energía y de la lentitud crítica; por eso el
integrador debe evolucionar el gauge libremente en R (ver hmc.py).

`total_action` es una función pura y diferenciable de `theta`. Los campos
`s` (Ising) y `n` (defectos) entran como argumentos estáticos durante la
trayectoria molecular: el leapfrog solo propaga el campo de gauge.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from lattice_math import plaquette, shift_fwd


@partial(jax.jit, static_argnames=("D",))
def wilson_action(theta: jnp.ndarray, beta: float, D: int) -> jnp.ndarray:
    """S_Wilson = -beta * sum_{x, mu<nu} cos(P_{mu nu})."""
    total = jnp.array(0.0, dtype=theta.dtype)
    for mu in range(D):
        for nu in range(mu + 1, D):
            P = plaquette(theta, mu, nu)
            total = total + jnp.sum(jnp.cos(P))
    return -beta * total


@partial(jax.jit, static_argnames=("D",))
def ising_action(theta: jnp.ndarray, s: jnp.ndarray, J: float, D: int) -> jnp.ndarray:
    """S_Ising = -J * sum_{x, mu} s_x s_{x+mu} cos(theta_mu(x)).

    El acoplamiento materia-gauge usa cos(theta_mu) como factor de transporte
    paralelo escalar a lo largo del enlace mu.
    """
    total = jnp.array(0.0, dtype=theta.dtype)
    for mu in range(D):
        s_fwd = shift_fwd(s, mu)
        total = total + jnp.sum(s * s_fwd * jnp.cos(theta[..., mu]))
    return -J * total


@partial(jax.jit, static_argnames=("D",))
def bf_action(theta: jnp.ndarray, n: jnp.ndarray, kappa: float, D: int) -> jnp.ndarray:
    """S_BF = kappa * sum_{x, mu<nu} n_{mu nu} * P_{mu nu}  (término lineal)."""
    total = jnp.array(0.0, dtype=theta.dtype)
    for mu in range(D):
        for nu in range(mu + 1, D):
            P = plaquette(theta, mu, nu)
            total = total + jnp.sum(n[..., mu, nu] * P)
    return kappa * total


@partial(jax.jit, static_argnames=("D",))
def total_action(
    theta: jnp.ndarray,
    s: jnp.ndarray,
    n: jnp.ndarray,
    beta: float,
    J: float,
    kappa: float,
    D: int,
) -> jnp.ndarray:
    """Acción total S(theta; s, n).

    Diferenciable en `theta`. XLA fusiona las tres contribuciones en un
    grafo único, maximizando el throughput en la RTX 5090.
    """
    return (
        wilson_action(theta, beta, D)
        + ising_action(theta, s, J, D)
        + bf_action(theta, n, kappa, D)
    )


# --------------------------------------------------------------------------- #
#  Fuerza global  dS/dtheta  vía diferenciación automática en modo inverso
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("D",))
def action_grad(
    theta: jnp.ndarray,
    s: jnp.ndarray,
    n: jnp.ndarray,
    beta: float,
    J: float,
    kappa: float,
    D: int,
) -> jnp.ndarray:
    """Gradiente exacto dS/dtheta (Restricción algorítmica #2).

    Se usa `jax.grad` en modo reverse-mode AD. El coste es O(1) en el sentido
    de que una sola pasada de retropropagación entrega la fuerza completa
    sobre TODOS los enlaces simultáneamente (no hay barrido sitio-a-sitio).
    """
    grad_fn = jax.grad(total_action, argnums=0)
    return grad_fn(theta, s, n, beta, J, kappa, D)
