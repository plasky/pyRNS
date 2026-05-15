# gw_posterior.py — GW posterior predictive distributions

`gw_posterior.py` reads posterior samples from a
[Bilby](https://lscsoft.docs.ligo.org/bilby/) parameter-estimation run,
converts component masses and tidal deformabilities to (mass, radius) pairs
using universal neutron-star relations, and plots the posterior predictive
distribution (PPD) for the M–R plane.

---

## Physical pipeline

For a binary neutron-star (BNS) gravitational-wave event, Bilby produces
posterior samples of (among other things):

- **m₁, m₂** — component masses [M☉]
- **Λ₁, Λ₂** — dimensionless tidal deformabilities

The tidal deformability is related to the star's internal structure by

$$
\Lambda = \frac{2}{3}\,k_2 \left(\frac{c^2 R}{G M}\right)^5 = \frac{2}{3}\,\frac{k_2}{C^5}
$$

where C = GM/(c²R) is the **compactness** and k₂ is the tidal Love number.
Because k₂ and C are approximately universal functions of Λ, we can invert
to get R from any (M, Λ) pair without assuming a specific EOS.

### Universal Λ–C relations

Two fits are implemented:

#### De et al. (2018) — `--relation de2018` (default)

> S. De et al., *PRL* **121**, 091102 (2018), Eq. (3)

$$
C = 0.371 - 0.0391\ln\Lambda + 0.001056\,(\ln\Lambda)^2
$$

Valid for C ∈ [0.10, 0.35], fitted to a large set of nuclear-matter EOS models.

#### Yagi–Yunes (2013) — `--relation yy2013`

> K. Yagi & N. Yunes, *PRD* **88**, 023009 (2013), Table I

Polynomial fit of ln Λ as a function of η = ln C, inverted numerically:

$$
\ln\Lambda = a_1 + a_2\eta + a_3\eta^2 + a_4\eta^3 + a_5\eta^4,
\quad \eta = \ln C
$$

Once C is known, the circumferential radius follows directly:

$$
R\,[\text{km}] = \frac{GM}{c^2 C} \times 10^{-5}
$$

---

## Command-line interface

```
python3 gw_posterior.py <posterior_file> [options]
```

### Required argument

| Argument | Description |
|----------|-------------|
| `posterior_file` | Bilby result file — `.json`, `.hdf5`, `.h5`, `.csv`, or `.txt` |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--comp {1,2,both}` | `both` | Include component 1, 2, or both NS in the PPD |
| `--relation {de2018,yy2013}` | `de2018` | Universal Λ–C relation to use |
| `--n-samples N` | all | Thin posterior to N samples |
| `--mr-eos FILE` | none | Overlay a pyRNS tabulated EOS M-R curve |
| `--savefig FILE` | auto | Output PDF path |
| `--title TEXT` | event label | Figure suptitle |
| `--credible L1,L2` | `50,90` | Credible levels to draw (%) |
| `--no-plot` | off | Suppress figure; still writes text table |

---

## Examples

### Basic — both components

```bash
python3 gw_posterior.py GW170817_posterior.json
# → GW170817_posterior_mr_ppd.pdf
# → GW170817_posterior_mr_ppd.txt
```

### Single component only

```bash
python3 gw_posterior.py posterior.json --comp 1
```

### Overlay an EOS M-R curve

```bash
python3 gw_posterior.py posterior.json --mr-eos eos/eosA
```

The EOS curve is computed on-the-fly using `pyRNS.py`'s TOV solver and
overlaid in black on the credible-region contours.

### Use the Yagi–Yunes relation

```bash
python3 gw_posterior.py posterior.json --relation yy2013
```

### Custom credible levels and output

```bash
python3 gw_posterior.py posterior.json \
    --credible 68,95 \
    --title "GW170817 M–R posterior predictive" \
    --savefig results/GW170817_mr.pdf
```

### No plot — text table only

```bash
python3 gw_posterior.py posterior.json --no-plot
```

### Thinned samples (faster for testing)

```bash
python3 gw_posterior.py posterior.json --n-samples 500
```

---

## Supported Bilby file formats

`gw_posterior.py` reads Bilby output files without requiring the `bilby`
package to be installed.  If `bilby` is available it is used natively;
otherwise the file is parsed manually.

| Format | Extension | Notes |
|--------|-----------|-------|
| Bilby JSON (v1/v2) | `.json` | Standard Bilby result file |
| Bilby HDF5 (v2) | `.hdf5`, `.h5` | Requires `h5py` if bilby not installed |
| Plain CSV | `.csv` | Column headers must include parameter names |
| Plain text | `.txt`, `.dat` | Whitespace-separated, first row = header |

### Parameter name recognition

The following parameter names are recognised automatically (case-insensitive):

| Physical quantity | Recognised names |
|------------------|-----------------|
| Mass 1 | `mass_1`, `m1`, `mass1`, `M1` |
| Mass 2 | `mass_2`, `m2`, `mass2`, `M2` |
| Chirp mass | `chirp_mass`, `Mc`, `mchirp` |
| Mass ratio | `mass_ratio`, `q` |
| Λ₁ | `lambda_1`, `Lambda_1`, `lambda1`, `tidal_1`, `l1` |
| Λ₂ | `lambda_2`, `Lambda_2`, `lambda2`, `tidal_2`, `l2` |

If component masses are absent but chirp mass and mass ratio are present,
the individual masses are reconstructed:

$$
M_{\rm tot} = \mathcal{M}_c\,(1+q)^{1/5}\,q^{-3/5}, \qquad
m_1 = M_{\rm tot}/(1+q), \quad m_2 = q\,m_1
$$

---

## Output files

### PDF figure (`<basename>_mr_ppd.pdf`)

Two-panel publication figure:

| Panel | Description |
|-------|-------------|
| M–R plane | Filled 50% and 90% KDE credible-region contours; optional EOS curve |
| *(single panel)* | If only one panel is produced, it fills the full figure |

The figure uses the bundled `publication` matplotlib style automatically
(Times New Roman, LaTeX labels, no interactive window).

### Text table (`<basename>_mr_ppd.txt`)

Space-separated, three columns:

```
# M [M_sun]   R [km]   component
1.441234e+00  1.218765e+01   1
1.291876e+00  1.158234e+01   2
...
```

The `component` column is `1` or `2`, useful for downstream analysis that
needs to distinguish the heavier from the lighter star.

---

## Terminal output

```
Reading GW170817_posterior.json …
  3000 posterior samples loaded
  label: GW170817
Computing M-R PPD  (component=both, relation=de2018) …
  6000 (M, R) pairs generated

M-R posterior predictive summary:
  Component       M [M☉]                  R [km]
  ────────        ──────────────────────  ──────────────────────
  comp 1          1.44 [1.37, 1.52]       12.2 [10.7, 14.0]
  comp 2          1.29 [1.23, 1.36]       11.6 [10.1, 13.5]
  combined        1.37 [1.24, 1.51]       11.9 [10.3, 13.8]

  Saved text table → GW170817_posterior_mr_ppd.txt
  Saved M-R PPD → GW170817_posterior_mr_ppd.pdf
```

The 90% credible interval is quoted as `[5th percentile, 95th percentile]`.

---

## Python API

### Read a Bilby file

```python
from gw_posterior import read_bilby

posterior, meta = read_bilby("GW170817_posterior.json")
# posterior : dict of {parameter: np.ndarray}
# meta      : dict with label, n_samples, log_evidence, …

print(meta["n_samples"])          # 3000
print(posterior["mass_1"][:5])    # first 5 mass-1 samples
```

### Convert (M, Λ) → R

```python
import numpy as np
from gw_posterior import lambda_to_radius, lambda_to_compactness

mass = np.array([1.35, 1.4, 1.45])   # M☉
lam  = np.array([400,  350,  300])    # Λ

R = lambda_to_radius(mass, lam)           # km,  De et al. (2018)
R_yy = lambda_to_radius(mass, lam, relation="yy2013")

C = lambda_to_compactness(lam)            # dimensionless compactness
```

### Compute the full PPD

```python
from gw_posterior import read_bilby, compute_mr_ppd, print_summary

posterior, meta = read_bilby("posterior.json")

M_arr, R_arr, labels = compute_mr_ppd(
    posterior,
    component="both",     # '1', '2', or 'both'
    relation="de2018",    # 'de2018' or 'yy2013'
    n_samples=2000,       # thin if needed
)

print_summary(M_arr, R_arr, labels)
```

### Plot the PPD

```python
from gw_posterior import plot_mr_ppd

plot_mr_ppd(
    M_arr, R_arr, labels,
    savefig="GW170817_mr.pdf",
    title="GW170817",
    credible_levels=(0.50, 0.90),
    mr_eos_file="eos/eosA",        # optional EOS overlay
    event_label="GW170817",
)
```

### Full pipeline in one call

```python
from gw_posterior import read_bilby, compute_mr_ppd, print_summary, plot_mr_ppd

posterior, meta = read_bilby("posterior.json")
M, R, labels   = compute_mr_ppd(posterior)
print_summary(M, R, labels)
plot_mr_ppd(M, R, labels, savefig="mr_ppd.pdf", mr_eos_file="eos/eosA")
```

---

## Accuracy and caveats

| Source of uncertainty | Impact |
|-----------------------|--------|
| Universal Λ–C relation (De 2018) | ±5–10% in R across the NS EOS family |
| Yagi–Yunes relation | Similar scatter; slightly different at low C |
| GW posterior width | Dominates for real events (broad Λ posteriors) |
| Assumption of cold, non-rotating NS | Valid for NS inspiral; neglects spin and thermal effects |

The dominant uncertainty is always the width of the GW posterior itself.
The universal-relation conversion adds a systematic of ~5–10% in radius,
which is subdominant to the current GW measurement precision for most events.

---

## References

1. **De et al. (2018)**, *PRL* **121**, 091102 — universal Λ–compactness polynomial
2. **Yagi & Yunes (2013)**, *PRD* **88**, 023009 — I-Love-Q universal relations
3. **Hinderer et al. (2010)**, *ApJ* **677**, 1216 — tidal deformability definition
4. **Abbott et al. (2018)**, *PRL* **121**, 161101 — GW170817 tidal measurements
5. **Ashton et al. (2019)**, *ApJS* **241**, 27 — Bilby parameter estimation code
