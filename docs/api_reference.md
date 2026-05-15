# API reference

All public functions across the three modules.

---

## pyRNS.py

### Grid and setup

---

#### `make_grid() → (s_gp, mu)`

Create the compactified radial and angular grids.

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `s_gp` | `(SDIV,)` | Radial grid: s ∈ [0, SMAX) |
| `mu` | `(MDIV,)` | Angular grid: μ = cos θ ∈ [0, 1] |

**Constants used**: `SDIV=129`, `MDIV=65`, `SMAX=0.9999`.

---

#### `precompute_legendre(mu) → (P_2n, P1_2n1, sin_2n1_theta)`

Pre-compute Legendre-polynomial arrays on the angular grid.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mu` | `(MDIV,)` | Angular grid from `make_grid` |

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `P_2n` | `(MDIV, LMAX+1)` | P_{2n}(μ), n=0…LMAX |
| `P1_2n1` | `(MDIV, LMAX+1)` | Associated Legendre P¹_{2n-1}(μ) |
| `sin_2n1_theta` | `(MDIV, LMAX+1)` | sin((2n-1)θ) / sin θ |

---

#### `compute_kernels(s_gp) → (f_rho, f_gama)`

Pre-compute the Green's-function radial kernel arrays for the
self-consistent field iteration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `s_gp` | `(SDIV,)` | Radial grid from `make_grid` |

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `f_rho` | `(SDIV, LMAX+1, SDIV)` | Kernel for ρ potential |
| `f_gama` | `(SDIV, LMAX+1, SDIV)` | Kernel for γ potential |

---

### EOS functions

---

#### `load_eos(eos_file) → (log_e, log_p, log_h, log_n0, n_tab)`

Load a tabulated EOS in RNS format.

| Parameter | Type | Description |
|-----------|------|-------------|
| `eos_file` | `str` | Path to EOS file |

**Returns**: five arrays for use as `eos` dict entries, all log₁₀ of code-unit quantities.

---

#### `make_center(e_center, eos) → (p_center, h_center)`

Compute central pressure and enthalpy from central energy density.

| Parameter | Type | Description |
|-----------|------|-------------|
| `e_center` | `float` | Central energy density (code units) |
| `eos` | `dict` | EOS dictionary |

**`eos` dict keys**

| Key | Required | Description |
|-----|----------|-------------|
| `type` | yes | `"tab"` or `"poly"` |
| `Gamma_P` | yes | Adiabatic index Γ (poly only) |
| `log_e`, `log_p`, `log_h`, `log_n0` | tab only | Log₁₀ tables |
| `n_tab` | tab only | Number of table points |

---

### TOV and initialisation

---

#### `solve_tov(e_center, p_center, p_surface, eos) → tuple`

Integrate the TOV equations in isotropic coordinates.

**Returns** `(r_is_arr, r_arr, m_arr, p_arr, nu_arr, lambda_arr, r_is_final, m_final)`

| Name | Description |
|------|-------------|
| `r_is_arr` | Isotropic radial coordinate grid [RDIV+1] |
| `r_arr` | Schwarzschild r coordinate [RDIV+1] |
| `m_arr` | Enclosed mass [RDIV+1] |
| `p_arr` | Pressure [RDIV+1] |
| `nu_arr` | Lapse ν [RDIV+1] |
| `lambda_arr` | Conformal factor log(r/r_is) [RDIV+1] |
| `r_is_final` | Isotropic surface radius |
| `m_final` | Total gravitational mass (code units) |

---

#### `sphere(s_gp, mu, e_center, p_center, h_center, p_surface, e_surface, eos) → (rho, gama, alpha, omega, r_e)`

Initialise the metric potentials from the TOV solution (non-rotating seed).

All five outputs are `(SDIV, MDIV)` arrays in code units (divided by r_e²),
except `r_e` which is a scalar.

---

#### `spin(s_gp, mu_arr, eos, h_center, enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq, r_ratio, r_e_in, *, accuracy, cf, f_rho_kern, f_gama_kern, P_2n, P1_2n1, sin_2n1_theta, verbose) → (r_e, Omega)`

Self-consistent field iteration.

For `r_ratio = 1.0` (non-rotating) returns immediately with the sphere
solution and fills the matter arrays.  For `r_ratio < 1.0` attempts the
rotating-star iteration (currently diverges — see [physics](physics.md)).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r_ratio` | — | Polar-to-equatorial axis ratio (1.0 = sphere) |
| `r_e_in` | — | Initial equatorial isotropic radius |
| `accuracy` | `1e-5` | Convergence threshold on r_e |
| `cf` | `1.0` | SOR relaxation factor |
| `verbose` | `False` | Print iteration details |

All six field arrays (`rho`, `gama`, `alpha`, `omega`, `energy`, `pressure`)
are updated **in-place**.

**Returns** `(r_e, Omega)` — converged equatorial radius and angular velocity.

---

### Observables

---

#### `mass_radius(s_gp, mu, eos, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq, r_ratio, e_surface, r_e, Omega) → (Mass, Mass_0, J, R_e, Omega_K, v_plus, v_minus)`

Compute global equilibrium quantities by numerical integration.

| Output | Units (tabulated) | Units (poly) | Description |
|--------|------------------|--------------|-------------|
| `Mass` | g | code | Gravitational mass |
| `Mass_0` | g | code | Baryon mass |
| `J` | g cm² s⁻¹ | code | Angular momentum |
| `R_e` | cm | code | Circumferential equatorial radius |
| `Omega_K` | code | code | Keplerian angular velocity |
| `v_plus` | — | — | Co-rotating ZAMO velocity profile |
| `v_minus` | — | — | Counter-rotating ZAMO velocity profile |

Convert to display units:

```python
from pyRNS import MSUN, KAPPA, C
M_sun   = Mass / MSUN
R_km    = R_e / 1e5
OmK_SI  = Omega_K * C / KAPPA**0.5
```

---

### Plotting

---

#### `plot_star(s_gp, mu_arr, eos, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq, r_e, Omega, r_ratio, e_center, Mass, Mass_0, R_e, Omega_K, J, *, savefig)`

Save a four-panel diagnostic figure to `savefig` (PDF).
No window is opened (`Agg` backend).  The `publication` style is applied
automatically.

| Parameter | Description |
|-----------|-------------|
| `savefig` | Path to output PDF (required; auto-generated by CLI) |

---

#### `_apply_style()`

Apply the bundled `publication.mplstyle` and set the `Agg` backend.
Called automatically by all plot functions.  Can be called manually
before any custom `matplotlib` code to match the package style.

```python
from pyRNS import _apply_style
_apply_style()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# ... your plot code ...
fig.savefig("my_plot.pdf")
```

---

### Derivative helpers

#### `grad_s(f) → df_ds`
#### `grad_m(f) → df_dm`

Finite-difference first derivatives on the (SDIV × MDIV) grid using
central differences (one-sided at boundaries).

---

### Constants

```python
from pyRNS import C, G, MSUN, KAPPA, KSCALE, PI, SDIV, MDIV, LMAX, DS, DM, SMAX
```

| Name | Value | Description |
|------|-------|-------------|
| `C` | 2.9979 × 10¹⁰ cm/s | Speed of light |
| `G` | 6.6732 × 10⁻⁸ cm³/(g s²) | Gravitational constant |
| `MSUN` | 1.9870 × 10³³ g | Solar mass |
| `KAPPA` | 1.3468 × 10¹³ cm | RNS length scale |
| `KSCALE` | 1.1127 × 10⁻³⁵ | Pressure scale factor |
| `PI` | π | — |
| `SDIV` | 129 | Radial grid points |
| `MDIV` | 65 | Angular grid points |
| `LMAX` | 10 | Max Legendre mode |
| `DS` | SMAX/(SDIV-1) | Radial step |
| `DM` | 1/(MDIV-1) | Angular step |
| `SMAX` | 0.9999 | Outer radial boundary |

---

## mr_curve.py

---

#### `nonrotating_sequence(eos, *, is_tab, e_min, e_max, n_pts, e_surface, p_surface, enthalpy_min, s_gp, mu_arr, P_2n, P1_2n1, sn, fk, gk) → seq`

Compute a sequence of `n_pts` non-rotating models over [e_min, e_max].

**Returns** `seq` — `numpy.ndarray` of shape `(5, n_valid)`:

```
seq[0]  e_c    central energy density (code units)
seq[1]  M      gravitational mass
seq[2]  M0     baryon mass
seq[3]  Re     equatorial circumferential radius
seq[4]  OK     Keplerian angular velocity
```

---

#### `make_plots(seq, is_tab, eos_label="", savepdf=None)`

Save a four-panel M–R sequence figure to `savepdf` (PDF).

---

#### `_model(e_c, eos, *, e_surface, p_surface, enthalpy_min, s_gp, mu_arr, P_2n, P1_2n1, sn, fk, gk, r_ratio=1.0, accuracy=1e-4, cf=0.5, verbose=False) → (M, M0, Re, OK, J, Omega)`

Compute a single equilibrium model.

---

#### `_units(is_tab) → dict`

Return a dict of conversion factors and axis labels:

```python
u = _units(is_tab=True)
u["cM"]    # Mass conversion factor (1/MSUN for tab, 1.0 for poly)
u["cR"]    # Radius conversion (1/1e5 cm→km for tab)
u["cec"]   # Density conversion (1/(C²·KSCALE) for tab)
u["cOm"]   # Angular velocity (C/√κ for tab)
u["lM"]    # Axis label string
u["lR"]    # Axis label string
```

---

## muses_eos.py

---

#### `list_eos(family="cold_ns", verbose=True, refresh=False) → catalog`

Fetch (or load from cache) the CompOSE EOS catalog.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `verbose` | `True` | Print formatted table |
| `refresh` | `False` | Re-fetch from CompOSE, ignoring cache |

**Returns** list of dicts, each with keys:
`id`, `url_path`, `name`, `downloads`, `thermo_path`, `base_path`.

---

#### `find_eos(name_or_id, catalog=None) → entry`

Find a single EOS entry by name (substring) or numeric ID.

Raises `ValueError` if zero or more than one match is found.

---

#### `download_eos(eos_entry, outdir="eos/", force=False) → local_dir`

Download `eos.t`, `eos.nb`, `eos.yq`, `eos.thermo` for one EOS.
Existing files are reused (prints `Using cached`) unless `force=True`.

---

#### `parse_compose(local_dir, n_min_fm3=1e-4, n_max_fm3=None) → dict`

Parse CompOSE raw files in `local_dir` and return arrays in CGS/RNS units.

**Returns** dict with keys: `e`, `p`, `h`, `n0` (all `numpy.ndarray`),
`m_ref_MeV`, `n_pts`.

---

#### `write_rns_eos(data, outfile)`

Write the parsed EOS to RNS tabulated format at `outfile`.

---

#### `fetch_and_convert(name_or_id, outdir="eos/", n_min=1e-4, n_max=None, force=False) → (rns_file, eos_name)`

Full pipeline: catalog lookup → download → parse → write.
All steps are cached; use `force=True` to redo.

---

#### `_load_mr_data(name_or_id, outdir="eos/", force=False) → (R_arr, M_arr, label, txt_out)`

Fetch (or load) the `eos.mr` file for one EOS.
Returns arrays in physical units (km, M☉).

---

#### `plot_mr_multi(datasets, savefig, title=None)`

Plot M–R curves for one or more EOS on one figure.

| Parameter | Type | Description |
|-----------|------|-------------|
| `datasets` | `list[(R, M, label)]` | One tuple per EOS |
| `savefig` | `str` | Output PDF path |
| `title` | `str` or `None` | Optional suptitle |

---

#### `run_mrseq(names_or_ids, outdir="eos/", force=False, plot=True, savefig=None)`

High-level function: fetch M–R data for one or more EOS and plot.

```python
# Single
run_mrseq("RG(SK255)")

# Multiple — one figure
run_mrseq(["RG(SK255)", "HS(DD2)"], savefig="comparison.pdf")

# By ID
run_mrseq([94, 184])
```

---

#### `run_star(name_or_id, e_center_g_cm3=1e15, outdir="eos/", n_min=1e-4, force=False, plot=True, savefig=None, verbose=False)`

Fetch and convert an EOS, then call `pyRNS.py` for a single model.
