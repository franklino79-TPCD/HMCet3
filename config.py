"""
config.py
=========
Hiperparámetros globales y configuración de JAX/XLA para el motor HMC
U(1) Gauge - Ising - Defectos (2-formas) en 4D.

IMPORTANTE: La doble precisión (x64) se habilita ANTES de cualquier
operación con arrays de JAX. Esto es un requisito ESTRICTO para garantizar:
  * Reversibilidad de Crooks en el integrador leapfrog.
  * Balance detallado en el paso de Metropolis-Hastings.
  * Conservación aproximada del Hamiltoniano <exp(-dH)> = 1.

Hardware objetivo: NVIDIA GeForce RTX 5090 (32 GB VRAM), WSL2 / Miniconda3.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Configuración de JAX / XLA  (debe ejecutarse antes de importar jax.numpy)
# --------------------------------------------------------------------------- #
# Permite que XLA pre-asigne el 90% de la VRAM de la 5090 para evitar
# fragmentación y saturar el ancho de banda de memoria (~1.7 TB/s GDDR7).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
# Flag de XLA orientado a latency-hiding en arquitecturas Blackwell.
# (Solo se aplica en backend GPU; es inocuo en CPU.)
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_enable_latency_hiding_scheduler=true",
)

import jax  # noqa: E402

# >>> REQUISITO CRÍTICO: doble precisión global <<<
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class LatticeConfig:
    """Geometría de la red hipercúbica 4D."""

    L: int = 16          # extensión por dimensión
    D: int = 4           # dimensionalidad espacio-temporal (4D)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return (self.L, self.L, self.L, self.L)

    @property
    def volume(self) -> int:
        return self.L ** self.D

    @property
    def link_shape(self) -> Tuple[int, ...]:
        """Tensor 5D denso de enlaces: (L,L,L,L,D) -> theta_mu(x)."""
        return (*self.shape, self.D)

    @property
    def plaquette_shape(self) -> Tuple[int, ...]:
        """Tensor 6D de 2-formas: (L,L,L,L,D,D) -> n_{mu nu}(x)."""
        return (*self.shape, self.D, self.D)


@dataclass(frozen=True)
class ActionConfig:
    """Acoplamientos de la acción total S = S_Wilson + S_Ising + S_BF."""

    beta: float = 1.0    # acoplamiento de gauge (se barre en main.py)
    J: float = 0.5       # acoplamiento Ising-gauge
    kappa: float = 0.2   # acoplamiento topológico B^F (término lineal)


@dataclass(frozen=True)
class HMCConfig:
    """Parámetros de la dinámica molecular y del muestreo."""

    epsilon: float = 0.01    # step size del leapfrog
    n_md: int = 30           # pasos de molecular dynamics por trayectoria
    n_therm: int = 200       # trayectorias de termalización por beta
    n_meas: int = 500        # trayectorias de medición por beta
    seed: int = 20260607     # semilla PRNG reproducible


@dataclass(frozen=True)
class SweepConfig:
    """Barrido termodinámico en beta."""

    beta_min: float = 0.5
    beta_max: float = 1.5
    n_beta: int = 11         # 0.5, 0.6, ..., 1.5


@dataclass(frozen=True)
class Config:
    """Contenedor global de configuración."""

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    hmc: HMCConfig = field(default_factory=HMCConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


# Instancia por defecto importable: `from config import CFG`
CFG = Config()
