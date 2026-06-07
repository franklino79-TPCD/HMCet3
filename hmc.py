"""
hmc.py
======
Dinámica molecular Hamiltoniana, integrador leapfrog reversible y criterio
de aceptación Metropolis-Hastings.

Restricciones algorítmicas obligatorias implementadas aquí:

 (#3) EVOLUCIÓN LIBRE (Non-compact Leapfrog):
      Durante los N_md pasos, theta evoluciona libremente en R. NO se aplica
      jnp.mod ni ningún truncamiento durante la trayectoria. Esto preserva la
      diferenciabilidad C^1 que jax.grad necesita y evita las discontinuidades
      de fuerza que el término lineal S_BF amplificaría.

 (#4) COMPACTIFICACIÓN DIFERIDA:
      theta = jnp.mod(theta, 2*pi) se aplica UNA sola vez, GLOBALMENTE, y solo
      DESPUÉS del paso de aceptación/rechazo de Metropolis. La acción es
      invariante bajo theta -> theta + 2*pi, de modo que esta proyección no
      altera el balance detallado.

El integrador leapfrog es simpléctico y exactamente reversible (al nivel de
la precisión de máquina x64), lo que garantiza la relación de Crooks y por
tanto <exp(-dH)> = 1.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from action import action_grad, total_action

TWO_PI = 2.0 * jnp.pi


# --------------------------------------------------------------------------- #
#  Hamiltoniano de la dinámica ficticia
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("D",))
def hamiltonian(theta, p, s, n, beta, J, kappa, D):
    """H = T(p) + S(theta) = 1/2 * sum p^2 + S(theta; s, n)."""
    kinetic = 0.5 * jnp.sum(p * p)
    return kinetic + total_action(theta, s, n, beta, J, kappa, D)


# --------------------------------------------------------------------------- #
#  Integrador leapfrog (le-frog / velocity Verlet)  -- NON-COMPACT
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("D", "n_md"))
def leapfrog(theta, p, s, n, beta, J, kappa, eps, n_md, D):
    """Integra (theta, p) durante n_md pasos con paso eps.

    Esquema kick-drift-kick:
        p     <- p - (eps/2) * dS/dtheta
        repetir (n_md-1):  theta <- theta + eps*p ;  p <- p - eps*dS/dtheta
        theta <- theta + eps*p
        p     <- p - (eps/2) * dS/dtheta

    theta NO se reduce módulo 2*pi en ningún momento (Restricción #3).
    El bucle interno usa jax.lax.scan para que XLA lo despliegue como un
    único kernel persistente en la RTX 5090.
    """
    grad = lambda th: action_grad(th, s, n, beta, J, kappa, D)

    # --- half-kick inicial ---
    p = p - 0.5 * eps * grad(theta)

    def body(carry, _):
        th, mom = carry
        th = th + eps * mom                 # drift (evolución libre en R)
        mom = mom - eps * grad(th)          # full kick
        return (th, mom), None

    # n_md - 1 pasos completos drift+kick.
    (theta, p), _ = jax.lax.scan(body, (theta, p), None, length=n_md - 1)

    # --- último drift + half-kick final ---
    theta = theta + eps * p
    p = p - 0.5 * eps * grad(theta)

    return theta, p


# --------------------------------------------------------------------------- #
#  Una trayectoria HMC completa con Metropolis-Hastings
# --------------------------------------------------------------------------- #
@partial(jax.jit, static_argnames=("D", "n_md"))
def hmc_step(key, theta, s, n, beta, J, kappa, eps, n_md, D):
    """Ejecuta una trayectoria HMC y devuelve el nuevo estado.

    Returns
    -------
    theta_new : campo de gauge COMPACTIFICADO en [0, 2*pi) (post-Metropolis).
    accepted  : bool (0/1) de aceptación.
    dH        : delta del Hamiltoniano H_new - H_old (diagnóstico de Crooks).
    """
    key_p, key_acc = jax.random.split(key)

    # 1. Refresco de momentos: p ~ N(0, 1) (heat-bath gaussiano).
    p0 = jax.random.normal(key_p, shape=theta.shape, dtype=theta.dtype)

    # 2. Hamiltoniano inicial.
    H0 = hamiltonian(theta, p0, s, n, beta, J, kappa, D)

    # 3. Integración molecular NON-COMPACT (theta libre en R).
    theta_prop, p_prop = leapfrog(
        theta, p0, s, n, beta, J, kappa, eps, n_md, D
    )

    # 4. Hamiltoniano propuesto (inversión de momento implícita: H par en p).
    H1 = hamiltonian(theta_prop, p_prop, s, n, beta, J, kappa, D)

    dH = H1 - H0

    # 5. Criterio de Metropolis-Hastings: accept con prob min(1, exp(-dH)).
    u = jax.random.uniform(key_acc, dtype=theta.dtype)
    accepted = (u < jnp.exp(-dH)) | (dH < 0.0)

    theta_accepted = jnp.where(accepted, theta_prop, theta)

    # 6. COMPACTIFICACIÓN DIFERIDA (Restricción #4): solo aquí, post-Metropolis.
    theta_new = jnp.mod(theta_accepted, TWO_PI)

    return theta_new, accepted.astype(jnp.int32), dH


# --------------------------------------------------------------------------- #
#  Inicialización de campos
# --------------------------------------------------------------------------- #
def init_gauge(key, link_shape) -> jnp.ndarray:
    """Campo de gauge caliente: theta_mu(x) ~ Uniforme[0, 2*pi)."""
    return jax.random.uniform(
        key, shape=link_shape, minval=0.0, maxval=float(TWO_PI),
        dtype=jnp.float64,
    )


def init_ising(key, shape) -> jnp.ndarray:
    """Campo de Ising caliente: s_x ~ {-1, +1} equiprobable."""
    bits = jax.random.bernoulli(key, p=0.5, shape=shape)
    return jnp.where(bits, 1.0, -1.0).astype(jnp.float64)


# --------------------------------------------------------------------------- #
#  Actualización del campo de Ising: heat-bath checkerboard (even/odd)
# --------------------------------------------------------------------------- #
from lattice_math import shift_fwd, shift_bwd  # noqa: E402


def _parity_mask(shape, D, parity: int) -> jnp.ndarray:
    """Máscara booleana de los sitios cuya suma de coordenadas == parity (mod 2)."""
    grids = jnp.meshgrid(*[jnp.arange(s) for s in shape], indexing="ij")
    coord_sum = sum(grids)
    return (coord_sum % 2) == parity


@partial(jax.jit, static_argnames=("D",))
def _local_field(theta, s, J, D):
    """Campo local h_x = J * sum_mu [ cos(theta_mu(x)) s_{x+mu}
                                    + cos(theta_mu(x-mu)) s_{x-mu} ].

    La energía local de Ising es E(s_x) = -s_x * h_x.
    """
    h = jnp.zeros_like(s)
    for mu in range(D):
        c_fwd = jnp.cos(theta[..., mu])              # enlace x -> x+mu
        c_bwd = shift_bwd(jnp.cos(theta[..., mu]), mu)  # enlace x-mu -> x
        h = h + J * (c_fwd * shift_fwd(s, mu) + c_bwd * shift_bwd(s, mu))
    return h


@partial(jax.jit, static_argnames=("D",))
def ising_heatbath(key, theta, s, J, D):
    """Un barrido heat-bath del campo de Ising con descomposición checkerboard.

    Actualiza primero los sitios pares y luego los impares; dentro de cada
    sublattice los espines son condicionalmente independientes, de modo que el
    update vectorizado respeta el balance detallado.

        P(s_x = +1) = 1 / (1 + exp(-2 * h_x))
    """
    shape = s.shape
    for parity in (0, 1):
        key, sub = jax.random.split(key)
        h = _local_field(theta, s, J, D)
        p_up = jax.nn.sigmoid(2.0 * h)
        u = jax.random.uniform(sub, shape=shape, dtype=s.dtype)
        s_new = jnp.where(u < p_up, 1.0, -1.0)
        mask = _parity_mask(shape, D, parity)
        s = jnp.where(mask, s_new, s)
    return s
