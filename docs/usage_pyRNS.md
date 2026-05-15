# pyRNS.py — single-star models

`pyRNS.py` computes equilibrium properties of a single neutron star and
saves a diagnostic PDF automatically.

---

## Command-line interface

```
python3 pyRNS.py [options]
```

### Required arguments

| Flag | Description |
|------|-------------|
| `-e E_CENTER` | Central energy density. For tabulated EOS: g cm⁻³. For polytrope: code units. |

### EOS selection (one required)

| Flag | Description |
|------|-------------|
| `-f EOS_FILE` | Path to a tabulated EOS file (RNS format, see [EOS format](eos_format.md)) |
| `-q poly -N N` | Polytropic EOS $p = \rho_0^\Gamma$ with $\Gamma = 1 + 1/N$ |

### Optional arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--accuracy` | `1e-5` | Convergence threshold $|Δr_e|/r_e$ |
| `--cf` | `1.0` | SOR relaxation factor (reduce to 0.3–0.5 if solver oscillates) |
| `--plot` / `--no-plot` | `--plot` | Produce diagnostic PDF |
| `--savefig FILE` | auto | Override auto-generated PDF filename |
| `--verbose` | off | Print per-iteration convergence details |

---

## Output

### Terminal (tabulated EOS)

```
eosA,  MDIVxSDIV=65x129
ratio   e_15    M       M_0     r_star  spin    Omega_K I       J/M^2
        g/cm^3  sun     sun     km      s-1     s-1     g cm^2

1.000   1.0     0.835   0.861   9.948   0.0     10603.4 0.00    0.000
```

| Column | Units | Description |
|--------|-------|-------------|
| `ratio` | — | Polar-to-equatorial axis ratio |
| `e_15` | 10¹⁵ g cm⁻³ | Central energy density |
| `M` | M☉ | Gravitational mass |
| `M_0` | M☉ | Baryon mass |
| `r_star` | km | Circumferential equatorial radius |
| `spin` | rad s⁻¹ | Angular velocity Ω |
| `Omega_K` | rad s⁻¹ | Keplerian angular velocity |
| `I` | g cm² | Moment of inertia (= J/Ω) |
| `J/M^2` | — | Dimensionless spin parameter cJ/(GM²) |

### Terminal (polytrope)

Same columns but M, M₀, R, Ω in code units; e column is the input code density.

### PDF plot

Saved as `star_<eos>_<density>.pdf` (or `--savefig` path).
Four panels in SI / publication units:

1. **Energy density** ε(r) in kg m⁻³ vs radius in km
2. **Pressure** p(r) in Pa vs radius in km
3. **Enclosed mass** M(<r) in M☉ vs radius in km
4. **Spacetime potentials** ν(r) and ζ(r) (dimensionless, G=c=1)

For rotating stars a fifth panel shows a colour map of ε in the
meridional (x, z) plane.

---

## Examples

### Polytrope N=1

```bash
python3 pyRNS.py -q poly -N 1.0 -e 0.5
# → star_poly_N1.0_e0.50.pdf
```

### Polytrope N=1.5 at two densities

```bash
python3 pyRNS.py -q poly -N 1.5 -e 0.1
python3 pyRNS.py -q poly -N 1.5 -e 1.0
```

### Tabulated EOS — eosA at 10¹⁵ g cm⁻³

```bash
python3 pyRNS.py -f eos/eosA -e 1e15
# → star_eosA_1.0e+15.pdf
```

### Save figure with custom name

```bash
python3 pyRNS.py -f eos/eosL -e 5e14 --savefig results/eosL_star.pdf
```

### Suppress plot

```bash
python3 pyRNS.py -f eos/eosA -e 1e15 --no-plot
```

---

## Python API

```python
import numpy as np
from pyRNS import (
    make_grid, precompute_legendre, compute_kernels,
    load_eos, make_center,
    sphere, spin, mass_radius,
    plot_star,
    C, G, MSUN, KAPPA, KSCALE,
    SDIV, MDIV,
)

# ── Set up the grid (done once — expensive) ──────────────────────────────
s_gp, mu = make_grid()
P_2n, P1_2n1, sn = precompute_legendre(mu)
fk, gk = compute_kernels(s_gp)

# ── Load a tabulated EOS ──────────────────────────────────────────────────
log_e, log_p, log_h, log_n0, n_tab = load_eos("eos/eosA")
eos = {
    "type": "tab",
    "Gamma_P": 2.0,         # dummy — not used for tabulated
    "log_e": log_e, "log_p": log_p,
    "log_h": log_h, "log_n0": log_n0,
    "n_tab": n_tab,
}

# ── Central conditions ────────────────────────────────────────────────────
e_center     = 1e15 * C**2 * KSCALE      # 10¹⁵ g/cm³ in code units
e_surface    = 7.8  * C**2 * KSCALE
p_surface    = 1.01e8 * KSCALE
enthalpy_min = 1.0 / C**2

p_center, h_center = make_center(e_center, eos)

# ── Allocate field arrays ─────────────────────────────────────────────────
rho   = np.zeros((SDIV, MDIV))
gama  = np.zeros((SDIV, MDIV))
alpha = np.zeros((SDIV, MDIV))
omega = np.zeros((SDIV, MDIV))
energy    = np.zeros((SDIV, MDIV))
pressure  = np.zeros((SDIV, MDIV))
enthalpy  = np.zeros((SDIV, MDIV))
velocity_sq = np.zeros((SDIV, MDIV))

# ── Non-rotating seed from TOV ───────────────────────────────────────────
rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
    s_gp, mu, e_center, p_center, h_center,
    p_surface, e_surface, eos
)

# ── Self-consistent field (non-rotating: returns immediately) ─────────────
r_e, Omega = spin(
    s_gp, mu, eos, h_center, enthalpy_min,
    rho, gama, alpha, omega,
    energy, pressure, enthalpy, velocity_sq,
    r_ratio=1.0, r_e_in=r_e,
    accuracy=1e-5, cf=1.0,
    f_rho_kern=fk, f_gama_kern=gk,
    P_2n=P_2n, P1_2n1=P1_2n1, sin_2n1_theta=sn,
)

# ── Global quantities ─────────────────────────────────────────────────────
Mass, Mass_0, J, R_e, Omega_K, v_plus, v_minus = mass_radius(
    s_gp, mu, eos,
    rho, gama, alpha, omega,
    energy, pressure, enthalpy, velocity_sq,
    r_ratio=1.0, e_surface=e_surface, r_e=r_e, Omega=Omega,
)

print(f"M  = {Mass / MSUN:.4f} M_sun")
print(f"R  = {R_e / 1e5:.4f} km")
print(f"ΩK = {Omega_K * C / KAPPA**0.5:.1f} rad/s")

# ── Diagnostic plot ───────────────────────────────────────────────────────
plot_star(
    s_gp, mu, eos,
    rho, gama, alpha, omega,
    energy, pressure, enthalpy, velocity_sq,
    r_e, Omega, r_ratio=1.0, e_center=e_center,
    Mass=Mass, Mass_0=Mass_0, R_e=R_e, Omega_K=Omega_K, J=J,
    savefig="my_star.pdf",
)
```

### Polytropic EOS

```python
eos = {"type": "poly", "Gamma_P": 2.0}   # N=1 polytrope (Gamma = 2)
e_center = 0.5   # code units
p_center, h_center = make_center(e_center, eos)
e_surface = p_surface = enthalpy_min = 0.0
```

---

## Accuracy notes

| Quantity | Accuracy | Notes |
|----------|---------|-------|
| R_e | < 1% | Tested against C-code reference |
| M (tabulated) | ~15% high | GR integrand includes pressure terms |
| M₀ | < 1% | |
| Ω_K | ~1–2% | Schwarzschild approximation |
