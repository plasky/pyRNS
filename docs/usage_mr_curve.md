# mr_curve.py — mass-radius sequences

`mr_curve.py` computes a sequence of non-rotating equilibrium models over
a range of central densities and produces a four-panel publication figure.

---

## Command-line interface

```
python3 mr_curve.py [options]
```

### EOS selection

| Flag | Description |
|------|-------------|
| `--poly` | Polytropic EOS (default if `-f` not given) |
| `-N N` | Polytropic index (default 1.0) |
| `-f EOS_FILE` | Tabulated EOS in RNS format |

### Sequence parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--emin` | 0.05 (poly) / 10¹⁴ g cm⁻³ (tab) | Minimum central density |
| `--emax` | 1.5 (poly) / 3×10¹⁵ g cm⁻³ (tab) | Maximum central density |
| `--npts` | 30 | Number of equally-spaced models |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--out FILE` | `mr_data.txt` | Text table of results |
| `--plot` / `--no-plot` | `--plot` | Produce figure |
| `--savefig FILE` | `mr_data.pdf` | Override figure filename |

---

## Output files

### Text table (`mr_data.txt`)

Space-separated columns, one row per model:

| Column (tabulated) | Column (polytrope) | Description |
|--------------------|--------------------|-------------|
| ε_c [g cm⁻³] | ε_c [code] | Central energy density |
| M [M☉] | M [code] | Gravitational mass |
| M₀ [M☉] | M₀ [code] | Baryon mass |
| R_e [km] | R_e [code] | Circumferential equatorial radius |
| Ω_K [rad s⁻¹] | Ω_K [code] | Keplerian angular velocity |

### Figure (`mr_data.pdf`)

Four panels, publication quality:

1. **M–R diagram** — gravitational mass vs equatorial radius, with M_max marked
2. **M vs ε_c** — mass vs central density, with secondary Ω_K axis
3. **Compactness** — GM/(c²R) vs R, with Buchdahl (4/9) and BH (1/2) limits
   (tabulated EOS only)
4. **Binding energy** — (M₀ − M)/M × 100% vs M

---

## Examples

### Polytrope N=1, 30 models

```bash
python3 mr_curve.py --poly -N 1.0 --npts 30
```

### Polytrope N=1.5, custom density range

```bash
python3 mr_curve.py --poly -N 1.5 --emin 0.01 --emax 2.0 --npts 40
```

### Tabulated EOS eosA

```bash
python3 mr_curve.py -f eos/eosA --npts 20
# → mr_data.txt  mr_data.pdf
```

### Custom output paths

```bash
python3 mr_curve.py -f eos/eosL \
  --out results/eosL_seq.txt \
  --savefig results/eosL_mr.pdf
```

### Suppress plot (script use)

```bash
python3 mr_curve.py --poly --npts 50 --no-plot
```

---

## Python API

```python
import numpy as np
from mr_curve import (
    nonrotating_sequence,
    make_plots,
    _model,
    _units,
)
from pyRNS import (
    make_grid, precompute_legendre, compute_kernels,
    load_eos, C, KSCALE,
)

# ── Grid (done once) ──────────────────────────────────────────────────────
s_gp, mu = make_grid()
P_2n, P1_2n1, sn = precompute_legendre(mu)
fk, gk = compute_kernels(s_gp)

# ── EOS ───────────────────────────────────────────────────────────────────
log_e, log_p, log_h, log_n0, n_tab = load_eos("eos/eosA")
eos = {"type": "tab", "Gamma_P": 2.0,
       "log_e": log_e, "log_p": log_p,
       "log_h": log_h, "log_n0": log_n0, "n_tab": n_tab}

common = dict(
    e_surface    = 7.8 * C**2 * KSCALE,
    p_surface    = 1.01e8 * KSCALE,
    enthalpy_min = 1.0 / C**2,
    s_gp=s_gp, mu_arr=mu,
    P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk,
)

# ── Compute the sequence ──────────────────────────────────────────────────
e_min = 1e14 * C**2 * KSCALE    # code units
e_max = 3e15 * C**2 * KSCALE
seq = nonrotating_sequence(
    eos, is_tab=True,
    e_min=e_min, e_max=e_max, n_pts=25,
    **common,
)
# seq shape: (5, n_valid)  — e_c, M, M0, Re, OK

# ── Plot ──────────────────────────────────────────────────────────────────
make_plots(seq, is_tab=True, eos_label="eosA", savepdf="eosA_mr.pdf")

# ── Access individual columns ─────────────────────────────────────────────
u = _units(is_tab=True)   # conversion factors + labels
ec, M, M0, R, OK = seq
print(f"M_max = {max(M)*u['cM']:.3f}  {u['lM']}")
print(f"R     = {R[M.argmax()]*u['cR']:.2f}  {u['lR']}")
```

### Single model via `_model`

```python
M, M0, Re, OK, J, Omega = _model(
    e_c=1e15 * C**2 * KSCALE,
    eos=eos,
    r_ratio=1.0,      # non-rotating
    accuracy=1e-4,
    cf=0.5,
    **common,
)
```

---

## Unit conventions

### Tabulated EOS

All quantities in the output file and figure are in SI-adjacent units:

| Quantity | Unit |
|----------|------|
| ε_c | g cm⁻³ |
| M, M₀ | M☉ |
| R_e | km |
| Ω_K | rad s⁻¹ |

### Polytrope

Polytropes have no absolute physical scale without specifying the
polytropic constant K.  All quantities are dimensionless:

| Quantity | Unit |
|----------|------|
| ε_c | code units |
| M, M₀ | code units |
| R_e | code units |
| Ω_K | code units |

The compactness M/R_e and dimensionless Keplerian frequency
Ω_K √(R_e³/M) are displayed in the figure title and are meaningful
in geometric units G = c = 1.

---

## Performance

On a modern laptop with `--npts 30`:

- Polytrope: ~10 s (grid setup dominates)
- Tabulated EOS: ~15 s

Grid pre-computation (`make_grid`, `precompute_legendre`, `compute_kernels`)
takes ~3–5 s and is done once per script run.
