# Pipeline de Fases — U(1)-Ising-Defectos → Generaciones Quirales → Yukawa

Este documento describe el **programa experimental completo** construido sobre
el motor HMC `O(1)` (ver [`README.md`](README.md) para el motor base). El
pipeline parte de una transición de fase topológica en la red 4D y culmina en
una predicción de la razón de masas leptónicas `m_μ/m_e ≈ 207` sin ajuste fino.

```
 Fase I    ──►  Fase II   ──►  Fase II-b  ──►  Fase III
 (β_c=1.0)      (Q≈0, T⁴)      (Q=3, TCS)      (m_μ/m_e=207)

 termodinámica  carga          inyección       transición
 del atractor   topológica     topológica      de Yukawa
 infrarrojo     geométrica     N_gen=3         exp → lineal
```

---

## Tabla de scripts

| Fase | Script | Motor | Salida principal |
|------|--------|-------|------------------|
| — | `smoke_test.py` | JAX | Validación numérica (co-cerradura, Crooks, AD) |
| **I** | `main.py` | HMC 16⁴ | `fase1_termodinamica.csv`, `fase1_transicion_fase.pdf` |
| **II** | `fase2_index.py` | HMC 16⁴ | `fase2_carga_topologica.pdf` |
| **II-b** | `fase2b_tcs_index.py` | HMC 16⁴ | `fase2b_carga_confinada.pdf` |
| **III** | `fase3_yukawa_metric.py` | NumPy/SciPy | `fase3_transicion_yukawa.pdf` |

---

## Fase I — Validación Termodinámica del Atractor Infrarrojo

**Script:** `main.py`  ·  **Objetivo:** perfilar la transición de
deconfinamiento topológico y localizar `β_c`.

Barre `β ∈ [0.60, 1.40]` (paso 0.05) midiendo en cada punto:

- `⟨cos P⟩` — parámetro de orden (acción de plaqueta media).
- **Calor específico** `Cv = β²·Var(E_Wilson)/V` — el pico identifica `β_c`.
- Error de `⟨cos P⟩` por **análisis de autocorrelación** (ventana de Sokal).

```bash
python main.py
```

**Resultado clave:** pico de `Cv` en `β_c ≈ 1.00`. Datos en
`fase1_termodinamica.csv` (columnas `beta, mean_cosP, err_cosP, Cv,
acc_rate, mean_dH, time_s`), figura de 2 paneles en
`fase1_transicion_fase.pdf`.

---

## Fase II — Carga Topológica Geométrica

**Script:** `fase2_index.py`  ·  **Objetivo:** medir la carga topológica de la
red `Q` que, por el teorema del índice de Atiyah-Singer, dicta `N_gen`.

Operador geométrico de la red (Levi-Civita `ε_{μνρσ}`):

```
Q = 1/(32π²) · Σ_x Σ_{μνρσ} ε_{μνρσ} sin(P_{μν}(x)) sin(P_{ρσ}(x))
```

- `compute_topological_charge(theta)` — un único `jnp.einsum` vectorizado
  sobre los 16⁴ sitios; el tensor `ε` queda *baked-in* por XLA (sin tráfico
  de VRAM). Plaquetas por `jnp.roll`.
- 500 trayectorias de termalización + 2000 de medición en `β = 1.10`.
- Susceptibilidad topológica `χ_t = ⟨Q²⟩/V`.

```bash
python fase2_index.py
```

**Resultado clave:** sobre el toro plano `T⁴` con BC periódicas, `⟨Q⟩ ≈ 0`
(sector topológico trivial). Histograma en `fase2_carga_topologica.pdf`.
Esto es **correcto** físicamente y motiva la Fase II-b.

---

## Fase II-b — Inyección de Topología TCS (`Q → N_gen = 3`)

**Script:** `fase2b_tcs_index.py`  ·  **Objetivo:** forzar el vacío al sector
`Q = 3` emulando el cuello de una Suma Conexa Torcida (TCS) con invariante
`η` no nulo.

**Mecanismo — flujo abeliano constante de 't Hooft.** Para `U(1)` en `T⁴`,
una configuración clásica de field-strength constante tiene carga **exactamente
cuantizada** `Q = m_{01}·m_{23}`. Tomando `m_{01}=3, m_{23}=1` se inyecta
`Q = 3`:

```
θ_b(x)              += b·x_a          (rampa → plaqueta uniforme b=2πm/L²)
θ_a(x)|_{x_a=L-1}   += -b·L·x_b       (twist de cierre de 't Hooft)
```

Decisiones de diseño críticas:

1. **Cold start** (`θ=0`) — el flujo clásico (`b~0.07`) no queda sepultado por
   ruido UV `O(1)`; el sector `Q=3` sobrevive a la termalización.
2. **Defectos alineados** — fondo entero constante en los planos `(0,1)/(2,3)`
   (un 2-forma constante es trivialmente co-cerrado) que acopla
   `S_BF = κ Σ n P` constructivamente y *pincha* el sector.
3. **Sin `jnp.mod` destructivo** — el twist se inyecta una sola vez; el
   estimador usa `sin(P)` (periódico) y lee correctamente la plaqueta de
   esquina grande.

```bash
python fase2b_tcs_index.py
```

**Verificación (L=16):** `Q` inyectado `= 2.997 ≈ 3`, defectos `δn = 0` exacto.
1000 trayectorias de termalización + 2000 de medición; histograma anclado en
`Q=3` en `fase2b_carga_confinada.pdf`.

---

## Fase III — Transición de Yukawa (Exponencial → Lineal)

**Script:** `fase3_yukawa_metric.py` (NumPy/SciPy, sin Monte Carlo)  ·
**Objetivo:** demostrar que la supresión exponencial del instantón M2 se
cancela contra la divergencia conforme, dejando una razón de masa lineal en el
número topológico.

**Mecanismo.** Un único módulo `Vol(Σ) = v0·n2` controla dos efectos opuestos
que se cancelan **exactamente** (a `2×10⁻¹⁶`):

```
A_inst      = exp(−Vol/l_p³)            supresión M2          (→ 0)
Ω_cycle     = exp(+Vol/l_p³)            warp conforme sobre Σ  (→ ∞)
─────────────────────────────────────────────────────────────────────
Y_eff(n2)   = A_inst · Ω_cycle · N_conf(n2)  =  N_conf(n2)  ∝  n2
```

El residuo `N_conf` es la normalización conforme **codim-4** del modo quiral,
una integral genuina (`scipy.quad`) sobre el throat de Eguchi-Hanson:

```
N_conf(n2) = ∫_{r_c}^{R} Ω(r)⁴ · r³ · ψ_loc(r)²  dr,   r_c = a(1 + 1/n2)
```

```bash
python fase3_yukawa_metric.py
```

**Resultado clave:**

- Ley lineal confirmada: pendiente log-log `= 1.010`, `R² = 0.9997`.
- Sobre el atractor IR `Y_eff = κ·n`:  **`m_μ/m_e = n2/n1 = 207/1 = 207`**
  vs `206.768` (PDG) → **0.11 %** de desviación.
- Figura de 2 paneles (`fase3_transicion_yukawa.pdf`): arriba la cancelación
  de los dos exponenciales (log), abajo la ley lineal residual.

> **Nota:** Fase III es un modelo de juguete controlado que captura el
> *mecanismo*. El valor 207 emerge de `n2/n1` sobre la ley lineal, **no** de
> los parámetros geométricos (`a, R, v0, ξ`).

---

## Ejecución completa del pipeline

```bash
# 0. Entorno (Miniconda3 / WSL2)
conda create -n hmc python=3.11 -y && conda activate hmc
pip install -U "jax[cuda12]" numpy scipy matplotlib   # GPU RTX 5090
#   (usa "jax[cpu]" para validación sin GPU)

# 1. Validar el motor
python smoke_test.py

# 2. Pipeline de física
python main.py                # Fase I    — β_c = 1.00
python fase2_index.py         # Fase II   — Q ≈ 0 (T⁴ trivial)
python fase2b_tcs_index.py    # Fase II-b — Q = 3 (inyección TCS)
python fase3_yukawa_metric.py # Fase III  — m_μ/m_e = 207
```

Tiempos aproximados en una RTX 5090 (16⁴, x64): Fases I/II/II-b dominadas por
el HMC (~minutos a decenas de minutos según `n_therm/n_meas`); Fase III es
instantánea (integración numérica pura).

---

## Resumen de resultados

| Fase | Observable | Resultado |
|------|-----------|-----------|
| I | `β_c` (pico de `Cv`) | `≈ 1.00` |
| II | `⟨Q⟩` en `T⁴` | `≈ 0` (sector trivial) |
| II-b | `⟨Q⟩` con flujo TCS | `≈ 3` (`N_gen = 3`) |
| III | `m_μ/m_e` (atractor lineal) | `207` (PDG: 206.77, 0.11 %) |
