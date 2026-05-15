# EOS file format

pyRNS uses the tabulated EOS format from the original RNS C code.
All 15 original RNS EOS files are included in the `eos/` directory and
can be used directly.  Custom EOS tables must follow the same format.

---

## RNS tabulated EOS format

### File structure

```
n_tab
e₁   p₁   h₁   n0₁
e₂   p₂   h₂   n0₂
...
eₙ   pₙ   hₙ   n0ₙ
```

- **Line 1**: integer `n_tab` — number of tabulated density points
- **Lines 2 … n_tab+1**: four space-separated floating-point values per row

### Column definitions

| Column | Symbol | CGS units | Description |
|--------|--------|-----------|-------------|
| 1 | ε/c² | g cm⁻³ | Total energy density divided by c² |
| 2 | p | dyne cm⁻² | Pressure |
| 3 | h | cm² s⁻² | Specific enthalpy (see below) |
| 4 | n₀ | cm⁻³ | Baryon number density |

> **ε/c² vs ε**: The first column stores the *mass-density equivalent*
> ε/c² in g cm⁻³.  In the non-relativistic limit this equals the baryon
> mass density ρ₀.  At nuclear densities it includes internal and
> interaction energies.

### Specific enthalpy

The enthalpy column stores the **thermodynamic integral**:

$$
h = c^2 \int_0^p \frac{dp'}{\varepsilon(p') + p'}
\quad [\text{cm}^2\,\text{s}^{-2}]
$$

This quantity is zero at zero pressure (stellar surface) and increases
monotonically into the core.  In the non-relativistic limit:

$$
h \approx \frac{p}{\varepsilon}\,c^2 \approx \frac{p}{\rho_0}
\quad [\text{cm}^2\,\text{s}^{-2}]
$$

> **Do not use** μ_B (baryon chemical potential) directly as h — it
> includes the rest-mass contribution c² and does not go to zero at the
> surface.

### Ordering

Rows must be sorted by **strictly increasing** pressure (and energy
density).  The `load_eos` function enforces monotonicity and raises an
error if the table is inconsistent.

### Example — first rows of eosA

```
102
3.95008e+01  1.27820e+14  1.000000e+00  2.379569e+25
5.01830e+01  2.19340e+14  2.035474e+12  3.023074e+25
6.30713e+01  3.76390e+14  4.803682e+12  3.799473e+25
...
```

Row 1: ε/c² = 39.5 g/cm³ (outer crust), p = 1.28×10¹⁴ dyne/cm²,
h ≈ 0 (low density), n₀ = 2.38×10²⁵ cm⁻³.

---

## Unit conversions

### Loading

`load_eos` converts the file columns to internal code units using:

```python
log_e  = log10(e_cgs  * C**2 * KSCALE)   # code energy density
log_p  = log10(p_cgs  * KSCALE)           # code pressure
log_h  = log10(h_cgs  / C**2)             # code enthalpy (= h/c²)
log_n0 = log10(n0_cgs)                    # number density (unchanged)
```

where `KSCALE = KAPPA * G / C**4` ≈ 1.11 × 10⁻³⁵ and
`KAPPA = 10⁻¹⁵ × c²/G` ≈ 1.35 × 10¹³ cm.

All interpolations inside the solver are done in log space for numerical
stability across the wide dynamic range of neutron-star matter.

### Recovering physical values from code units

| Code quantity | Physical conversion |
|---------------|---------------------|
| `e_code` | ε [g cm⁻³] = `e_code / (C**2 * KSCALE)` |
| `p_code` | p [dyne cm⁻²] = `p_code / KSCALE` |
| `h_code` | h [cm² s⁻²] = `h_code * C**2` (h_code = h/c²) |
| `n0` (unchanged) | n₀ [cm⁻³] |

### SI equivalents

| CGS | SI |
|-----|----|
| 1 g cm⁻³ | 1000 kg m⁻³ |
| 1 dyne cm⁻² | 0.1 Pa |
| 1 M☉ = 1.989 × 10³³ g | — |
| 1 km | 10⁵ cm |

---

## Surface and atmosphere conditions

pyRNS uses two cut-off conditions to define the stellar surface:

| Parameter | Typical tabulated value | Role |
|-----------|------------------------|------|
| `e_surface` | 7.8 × c² × KSCALE ≈ 7.8 g cm⁻³ | Minimum energy density treated as matter |
| `p_surface` | 1.01 × 10⁸ × KSCALE | Surface pressure |
| `enthalpy_min` | 1/c² | Minimum enthalpy; points below = vacuum |

For polytropes all three are zero (the EOS goes all the way to zero density).

---

## Creating a custom EOS table

1. Compute (ε, p, n_B) as a function of baryon density using your
   many-body code of choice.

2. Compute the enthalpy by numerical integration:

   ```python
   import numpy as np

   # eps [g/cm³], p [dyne/cm²], n0 [cm⁻³]  — all sorted by increasing p
   c = 2.9979e10   # cm/s
   h = np.zeros(len(p))
   for i in range(1, len(p)):
       dp    = p[i]   - p[i-1]
       emid  = 0.5*(eps[i]+eps[i-1]) * c**2   # erg/cm³
       pmid  = 0.5*(p[i]+p[i-1])
       h[i]  = h[i-1] + dp / (emid + pmid)
   h *= c**2   # → cm²/s²
   ```

3. Write the file:

   ```python
   n_tab = len(eps)
   with open("my_eos.dat", "w") as f:
       f.write(f"{n_tab}\n")
       for i in range(n_tab):
           f.write(f"{eps[i]:.8e} {p[i]:.8e} {h[i]:.8e} {n0[i]:.8e}\n")
   ```

4. Test with pyRNS:

   ```bash
   python3 pyRNS.py -f my_eos.dat -e 1e15 --no-plot
   ```

---

## Bundled EOS files

The `eos/` directory contains 15 tabulated EOS from the original RNS
code.  See the `eos/EOS.INDEX` file for a full description of each model.

| File | Reference | Status |
|------|-----------|--------|
| eosA | Pandharipande neutron matter | ✓ works |
| eosAU | | ✓ works |
| eosB | | ✓ works |
| eosC | | ✓ works |
| eosF | | ✓ works |
| eosFP | Friedman-Pandharipande | limited density range |
| eosFPS | | ✓ works |
| eosG | | ✓ works |
| eosL | | ✓ works (stiff) |
| eosN | | ✓ works (stiff) |
| eosNV | | limited density range (max 1.6×10¹⁴ g/cm³) |
| eosO | | ✓ works |
| eosUU | | ✓ works |
| eosWNV | | non-standard column order |
| eosWS | | ✓ works |

---

## CompOSE format (reference)

CompOSE `eos.thermo` uses a different (normalised) format.
The `muses_eos.py` converter handles the translation automatically.
See [usage_muses.md](usage_muses.md) for details.

The interpreted column mapping used by `muses_eos.py`:

| CompOSE col | Physical quantity | Unit |
|-------------|-------------------|------|
| 0 | p | MeV fm⁻³ |
| 1 | s/n_B = 0 | (T=0) |
| 2 | (ε/n_B − m_ref)/m_ref | dimensionless |
| 3 | μ_Q/m_ref | dimensionless |
| 4 | μ_L/m_ref | dimensionless |
| 5/6 | (μ_B − m_ref)/m_ref | dimensionless |

Unit conversions from CompOSE to RNS:

| CompOSE | × factor | RNS |
|---------|----------|-----|
| p [MeV fm⁻³] | 1.6022 × 10³³ | p [dyne cm⁻²] |
| ε [MeV fm⁻³] | 1.7827 × 10¹² | ε [g cm⁻³] |
| n_B [fm⁻³] | 10³⁹ | n₀ [cm⁻³] |
| h = ∫dp/(ε+p)·c² | — | h [cm² s⁻²] |
