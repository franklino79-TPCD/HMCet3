"""
lattice_math.py
===============
Operadores topológicos sobre tensores densos contiguos.

REGLA DE ORO (Restricción algorítmica #1):
La adyacencia en la red se resuelve EXCLUSIVAMENTE mediante `jnp.roll`.
No se usan grafos, listas de adyacencia ni gather/scatter con índices.
Esto mantiene los accesos a memoria perfectamente coalesced y permite a
XLA fusionar los kernels para saturar el bus GDDR7 de la RTX 5090.

Convención de índices:
  * Los ejes 0..D-1 son las coordenadas espacio-temporales (L cada uno).
  * El último eje (o los dos últimos) indexa(n) la(s) componente(s) de Lorentz.

Convención de derivada exterior (forward difference):
  (d f)(x) usa f en x y en x + mu_hat. El vecino "forward" x + mu_hat se
  obtiene con `jnp.roll(f, shift=-1, axis=mu)`, de modo que el valor en la
  posición lógica x corresponde al campo evaluado en x + mu_hat.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from config import LatticeConfig


# --------------------------------------------------------------------------- #
#  Operadores de shift (desplazamiento de red)
# --------------------------------------------------------------------------- #
def shift_fwd(f: jnp.ndarray, mu: int) -> jnp.ndarray:
    """Devuelve f evaluado en x + mu_hat (vecino forward) vía jnp.roll.

    result[x] = f[x + mu_hat]   (con condiciones de contorno periódicas).
    """
    return jnp.roll(f, shift=-1, axis=mu)


def shift_bwd(f: jnp.ndarray, mu: int) -> jnp.ndarray:
    """Devuelve f evaluado en x - mu_hat (vecino backward) vía jnp.roll.

    result[x] = f[x - mu_hat]   (con condiciones de contorno periódicas).
    """
    return jnp.roll(f, shift=+1, axis=mu)


# --------------------------------------------------------------------------- #
#  Plaquetas del campo de gauge U(1)
# --------------------------------------------------------------------------- #
def plaquette(theta: jnp.ndarray, mu: int, nu: int) -> jnp.ndarray:
    """Fase de la plaqueta orientada en el plano (mu, nu).

        P_{mu nu}(x) = theta_mu(x) + theta_nu(x + mu)
                       - theta_mu(x + nu) - theta_nu(x)

    `theta` tiene forma (L,L,L,L,D); `theta[..., mu]` es la fase del enlace
    en dirección mu. El resultado tiene forma (L,L,L,L).

    NOTA: NO se aplica ninguna reducción modular aquí. P se mantiene en R
    para preservar la diferenciabilidad C^1 durante la trayectoria molecular
    (Restricción algorítmica #3).
    """
    th_mu = theta[..., mu]
    th_nu = theta[..., nu]
    return (
        th_mu
        + shift_fwd(th_nu, mu)
        - shift_fwd(th_mu, nu)
        - th_nu
    )


def all_plaquettes(theta: jnp.ndarray, D: int) -> jnp.ndarray:
    """Tensor antisimétrico de plaquetas P_{mu nu}(x), forma (L,L,L,L,D,D).

    Vectorizado: se construye el tensor completo para que XLA lo procese en
    un único kernel fusionado. Solo los pares mu<nu son independientes; el
    resto se obtiene por antisimetría P_{nu mu} = -P_{mu nu}.
    """
    spatial = theta.shape[:-1]
    P = jnp.zeros((*spatial, D, D), dtype=theta.dtype)
    for mu in range(D):
        for nu in range(mu + 1, D):
            p = plaquette(theta, mu, nu)
            P = P.at[..., mu, nu].set(p)
            P = P.at[..., nu, mu].set(-p)
    return P


# --------------------------------------------------------------------------- #
#  Co-diferencial (codifferential) y construcción de defectos co-cerrados
# --------------------------------------------------------------------------- #
def codiff_2form(n: jnp.ndarray, D: int) -> jnp.ndarray:
    """Codiferencial discreto delta de una 2-forma -> 1-forma.

        (delta n)_mu(x) = sum_nu [ n_{mu nu}(x) - n_{mu nu}(x - nu) ]

    Una 2-forma es CO-CERRADA cuando delta n = 0 para todo (mu, x).
    Devuelve un tensor de forma (L,L,L,L,D).
    """
    spatial = n.shape[:-2]
    out = jnp.zeros((*spatial, D), dtype=n.dtype)
    for mu in range(D):
        acc = jnp.zeros(spatial, dtype=n.dtype)
        for nu in range(D):
            if nu == mu:
                continue
            n_mn = n[..., mu, nu]
            acc = acc + (n_mn - shift_bwd(n_mn, nu))
        out = out.at[..., mu].set(acc)
    return out


def _perm_sign(perm) -> int:
    """Signo de una permutación (paridad por conteo de inversiones)."""
    s = 1
    p = list(perm)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def random_antisym_3form(key, cfg: LatticeConfig, amplitude: int = 1) -> jnp.ndarray:
    """3-forma entera totalmente antisimétrica A_{rho mu nu}(x).

    En 4D hay C(4,3)=4 componentes independientes (tripletes ordenados). Se
    muestrea un campo entero por triplete y se rellena el tensor completo con
    los signos de permutación. Forma: (L,L,L,L,D,D,D).
    """
    import itertools

    D = cfg.D
    A = jnp.zeros((*cfg.shape, D, D, D), dtype=jnp.float64)
    triples = [t for t in itertools.combinations(range(D), 3)]
    keys = jax.random.split(key, len(triples))
    for k, (a, b, c) in zip(keys, triples):
        g = jax.random.randint(
            k, shape=cfg.shape, minval=-amplitude, maxval=amplitude + 1
        ).astype(jnp.float64)
        for perm in itertools.permutations((a, b, c)):
            A = A.at[..., perm[0], perm[1], perm[2]].set(_perm_sign(perm) * g)
    return A


def codiff_3form(A: jnp.ndarray, D: int) -> jnp.ndarray:
    """Codiferencial discreto delta de una 3-forma -> 2-forma.

        (delta A)_{mu nu}(x) = sum_rho [ A_{rho mu nu}(x)
                                       - A_{rho mu nu}(x - rho) ]

    Usa la MISMA convención (índice antepuesto, diferencia backward) que
    `codiff_2form`, de modo que se cumple la nilpotencia delta^2 = 0. Por
    tanto delta(delta A) = 0 EXACTAMENTE: una 2-forma así es co-cerrada por
    construcción. Forma de salida: (L,L,L,L,D,D).
    """
    spatial = A.shape[:-3]
    n = jnp.zeros((*spatial, D, D), dtype=A.dtype)
    for mu in range(D):
        for nu in range(D):
            acc = jnp.zeros(spatial, dtype=A.dtype)
            for rho in range(D):
                a = A[..., rho, mu, nu]
                acc = acc + (a - shift_bwd(a, rho))
            n = n.at[..., mu, nu].set(acc)
    return n


def make_coclosed_defects(key, cfg: LatticeConfig, amplitude: int = 1) -> jnp.ndarray:
    """Genera una 2-forma entera ESTÁTICA y co-cerrada (delta n = 0).

    Receta topológicamente rigurosa basada en la nilpotencia del codiferencial:
      1. Muestrear una 3-forma entera antisimétrica aleatoria A.
      2. n = delta A   ->   delta n = delta(delta A) = 0 EXACTAMENTE.

    A diferencia del dual de Hodge punto-a-punto (que en la red mezcla la red
    directa y la dual y NO conserva la co-cerradura), esta construcción produce
    enteros exactos que satisfacen delta n = 0 a precision de maquina.
    """
    A = random_antisym_3form(key, cfg, amplitude=amplitude)
    n = codiff_3form(A, cfg.D)
    return jnp.round(n)
