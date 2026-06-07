# Motor HMC — U(1) Gauge · Ising · Defectos (2-formas) en 4D

Motor **Hybrid Monte Carlo** con **diferenciación automática exacta** (JAX/XLA)
para un sistema acoplado U(1)-gauge / materia de Ising / defectos topológicos
de 2-forma sobre una red hipercúbica 4D (`16⁴`), en doble precisión.

> Diseñado para una sola **NVIDIA RTX 5090 (32 GB)** bajo WSL2 / Miniconda3.

> 📋 **Programa experimental completo (Fases I–III):** ver
> [`PIPELINE.md`](PIPELINE.md) — desde la transición de fase topológica
> (`β_c=1.0`) hasta la razón de masas leptónicas `m_μ/m_e ≈ 207`.

---

## Acción total

```
S = S_Wilson + S_Ising + S_BF

S_Wilson = -β  · Σ_{x, μ<ν}  cos(P_{μν}(x))
S_Ising  = -J  · Σ_{x, μ}    s_x · s_{x+μ} · cos(θ_μ(x))
S_BF     =  κ  · Σ_{x, μ<ν}  n_{μν}(x) · P_{μν}(x)
```

con la plaqueta `P_{μν}(x) = θ_μ(x) + θ_ν(x+μ) − θ_μ(x+ν) − θ_ν(x)`.

---

## Restricciones algorítmicas implementadas

| # | Restricción | Dónde |
|---|-------------|-------|
| 1 | **Tensores contiguos**: adyacencia sólo con `jnp.roll` (sin grafos) | `lattice_math.py` |
| 2 | **AD en modo inverso**: fuerza `∂S/∂θ` vía `jax.grad` | `action.py::action_grad` |
| 3 | **Leapfrog non-compact**: θ evoluciona libre en ℝ, **sin** `jnp.mod` durante la trayectoria (preserva diferenciabilidad C¹) | `hmc.py::leapfrog` |
| 4 | **Compactificación diferida**: `θ = mod(θ, 2π)` sólo **después** de Metropolis | `hmc.py::hmc_step` |

Los **defectos** `n_{μν}` son enteros **estáticos y co-cerrados** (`δn = 0`),
construidos como `n = δA` de una 3-forma entera aleatoria `A`; la nilpotencia
del codiferencial (`δ² = 0`) garantiza `δn = 0` a precisión de máquina.

---

## Estructura modular

```
config.py        Hiperparámetros + configuración JAX/XLA (x64, flags GPU).
lattice_math.py  Shifts topológicos (jnp.roll), plaquetas, codiferencial, defectos.
action.py        S_Wilson, S_Ising, S_BF, acción total y gradiente AD.
hmc.py           Leapfrog reversible, Metropolis-Hastings, heat-bath de Ising.
main.py          Bucle termodinámico de barrido en β ∈ [0.5, 1.5].
smoke_test.py    Validación: co-cerradura, reversibilidad, AD vs FD, Crooks.
```

---

## Uso

```bash
# Instalar JAX (elige backend)
pip install -U "jax[cuda12]"   # GPU RTX 5090
pip install -U "jax[cpu]"      # CPU

# Validar la integridad numérica (red 4⁴, rápido)
python smoke_test.py

# Barrido completo de producción (red 16⁴)
python main.py
```

### Parámetros por defecto (`config.py`)

`ε = 0.01`, `N_md = 30`, `J = 0.5`, `κ = 0.2`, `β` barrido `0.5 → 1.5`.

---

## Diagnósticos de corrección

`smoke_test.py` verifica las propiedades que garantizan el balance detallado:

```
[1] co-cerradura  max|δn|     = 0.000e+00   (defectos exactamente co-cerrados)
[2] reversibilidad  dθ ~ 1e-16            (leapfrog reversible en x64)
[3] AD == diferencias finitas             (gradiente exacto)
[4] <exp(-dH)> ≈ 1                         (identidad de fluctuación / Crooks)
```
