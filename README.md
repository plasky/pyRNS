# pyRNS

A Python reimplementation of the RNS (Rapidly Rotating Neutron Stars) code by
Stergioulas & Friedman (1995), extended with a MUSES/CompOSE equation-of-state
interface and publication-quality plotting.

pyRNS solves the equations of hydrostatic equilibrium in general relativity for
non-rotating (and the infrastructure for rapidly rotating) neutron stars, using
the self-consistent field method of Komatsu, Eriguchi & Hachisu (1989) as
improved by Cook, Shapiro & Teukolsky (1994).

---

## Features

| Module | Description |
|--------|-------------|
| `pyRNS.py` | Core TOV solver, metric initialisation, single-star diagnostic plots |
| `mr_curve.py` | Mass-radius sequences with 4-panel publication plots |
| `muses_eos.py` | CompOSE/MUSES EOS catalog interface — download, cache, compare |
| `gw_posterior.py` | GW posterior samples (Bilby) → M-R posterior predictive distribution |

- **Non-rotating stars** fully solved (mass, baryon mass, radius, Ω_K all correct)
- **300+ EOS** accessible via the CompOSE database with one command
- **GW EOS inference** — convert Bilby tidal-deformability posteriors to (M, R) PPDs using universal Λ–C relations
- **Automatic caching** — EOS catalog and data files stored locally after first fetch
- **Publication-quality figures** using a bundled `publication` matplotlib style
  (Times New Roman, LaTeX labels, no interactive window)
- **SI units** throughout: kg m⁻³, Pa, km, M☉, rad s⁻¹

---

## Quick start

```bash
# 1. Non-rotating polytrope (N=1, e_c=0.5 code units)
python3 pyRNS.py -q poly -N 1.0 -e 0.5

# 2. Tabulated EOS from the bundled library
python3 pyRNS.py -f eos/eosA -e 1e15

# 3. Mass-radius sequence for a polytrope
python3 mr_curve.py --poly -N 1.0 --npts 30

# 4. Download any EOS from CompOSE and plot M-R
python3 muses_eos.py mrseq --eos "HS(DD2)"

# 5. Compare multiple EOS on one figure
python3 muses_eos.py mrseq --eos "RG(SK255)" "HS(DD2)" "DS(CMF)-5"

# 6. Convert Bilby GW posteriors to M-R posterior predictive distribution
python3 gw_posterior.py posterior.json

# 7. As above, overlaying an EOS M-R curve
python3 gw_posterior.py posterior.json --mr-eos eos/eosA
```

---

## Installation

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python  | ≥ 3.9   | runtime |
| NumPy   | ≥ 1.24  | arrays  |
| SciPy   | ≥ 1.10  | banded linear solver, KDE |
| Matplotlib | ≥ 3.7 | plotting |
| bilby   | optional | native Bilby result reading (`gw_posterior.py`) |
| h5py    | optional | HDF5 Bilby files without bilby installed |

### Setup

```bash
# Clone
git clone https://github.com/plasky/pyRNS.git
cd pyRNS

# Core dependencies
python3 -m pip install numpy scipy matplotlib --break-system-packages

# Optional: for reading Bilby GW posterior files directly
python3 -m pip install bilby h5py --break-system-packages

# Verify
python3 -c "import pyRNS; print('OK')"
```

No build step or `pip install .` is required — all scripts are
standalone and import each other directly.

---

## Project layout

```
pyRNS/
├── pyRNS.py               # core solver + plotting
├── mr_curve.py            # mass-radius sequences
├── muses_eos.py           # CompOSE/MUSES interface
├── gw_posterior.py        # GW posterior → M-R PPD
├── publication.mplstyle   # bundled matplotlib style
├── eos/                   # EOS library
│   ├── eosA … eosWS       # 15 tabulated EOS from original RNS
│   ├── _catalog_cache.json# CompOSE catalog (auto-built)
│   └── <name>/            # per-EOS download directory
│       ├── eos.thermo     # CompOSE raw data
│       ├── eos.mr         # pre-computed M-R (CompOSE)
│       └── <name>_rns.eos # converted tabulated file
└── docs/                  # this documentation
    ├── physics.md
    ├── installation.md
    ├── usage_pyRNS.md
    ├── usage_mr_curve.md
    ├── usage_muses.md
    ├── usage_gw_posterior.md
    ├── eos_format.md
    └── api_reference.md
```

---

## Documentation

| Document | Contents |
|----------|---------|
| [Physics background](docs/physics.md) | GR metric, TOV equations, self-consistent field method |
| [Installation](docs/installation.md) | Detailed setup for various environments |
| [pyRNS usage](docs/usage_pyRNS.md) | Single-star models, CLI options, plots |
| [mr_curve usage](docs/usage_mr_curve.md) | Mass-radius sequences, unit conventions |
| [MUSES/CompOSE usage](docs/usage_muses.md) | EOS catalog, multi-EOS comparisons |
| [GW posterior usage](docs/usage_gw_posterior.md) | Bilby posteriors → M-R PPD, universal Λ–R relations |
| [EOS file format](docs/eos_format.md) | Tabulated EOS columns, units, conversion |
| [API reference](docs/api_reference.md) | All public functions and signatures |

---

## Example outputs

### Single non-rotating star — `pyRNS.py`

A four-panel diagnostic figure is saved automatically as
`star_<eos>_<density>.pdf`:

- **Energy density profile** ε(r) in kg m⁻³
- **Pressure profile** p(r) in Pa
- **Enclosed mass** M(<r) in M☉
- **Metric potentials** ν(r) and ζ(r) (dimensionless, G=c=1)

### Mass-radius sequence — `mr_curve.py`

A four-panel figure is saved as `mr_data.pdf`:

- **M–R diagram** with M_max marked
- **M vs ε_c** with Keplerian frequency on secondary axis
- **Compactness** GM/(c²R) with Buchdahl and BH limits
- **Binding energy** (M₀−M)/M vs M

### CompOSE comparison — `muses_eos.py`

When multiple EOS are specified, a two-panel figure is saved:

- **M–R diagram** — all EOS on one set of axes, colour-coded with legend
- **Compactness** — same layout

### GW posterior predictive — `gw_posterior.py`

Reads a Bilby result file containing (m₁, m₂, Λ₁, Λ₂) samples and saves
a two-panel figure (`<event>_mr_ppd.pdf`):

- **M–R posterior predictive** — 50% and 90% KDE credible contours for both
  neutron-star components, optionally with an EOS M-R curve overlaid
- **Summary statistics** — median and 90% CI for M and R, printed to terminal

---

## Bundled EOS files

Twelve of the fifteen original RNS EOS files work at 10¹⁵ g cm⁻³:

| EOS | M_max [M☉] | R [km] | Description |
|-----|-----------|--------|-------------|
| eosA | 1.66 | ~9.9 | Pandharipande neutron matter |
| eosB | — | ~9.9 | |
| eosC | — | ~12 | |
| eosF | — | ~11 | |
| eosFPS | — | ~11 | Friedman-Pandharipande-Skyrme |
| eosG | — | ~9.9 | |
| eosL | ~2.7 | ~14 | Stiff |
| eosN | ~2.6 | ~14 | |
| eosO | ~2.0 | ~13 | |
| eosAU | — | ~10 | |
| eosUU | — | ~11 | |
| eosWS | — | ~11 | |

---

## Known limitations

- **Rotating stars**: the self-consistent field (KEH) iteration for `r_ratio < 1`
  does not converge in the current implementation. The infrastructure is present
  but requires the exact Green's-function kernel normalisation from the original
  C source code (not publicly available). All quantities for non-rotating stars
  are correct.

- **Mass accuracy (~15%)**: the gravitational mass integral uses the GR
  stress-energy integrand (ε + 3p for v=0), which exceeds the ADM mass
  by ~15% for typical neutron stars. Radii and Keplerian frequencies are
  accurate to ≤3%.

- **CompOSE EOS conversion**: the `run` command in `muses_eos.py` converts
  CompOSE `eos.thermo` to the RNS format with ~20% mass accuracy. The `mrseq`
  command uses CompOSE's pre-computed `eos.mr` tables and is exact.

---

## References

1. Komatsu, Eriguchi & Hachisu (1989), *MNRAS* 237, 355 — KEH method
2. Cook, Shapiro & Teukolsky (1994), *ApJ* 422, 227 — improved KEH
3. Stergioulas & Friedman (1995), *ApJ* 444, 306 — RNS code
4. Original C code: RNS v2.0 by N. Stergioulas (1999), [github.com/cgca/rns](https://github.com/cgca/rns)
5. De et al. (2018), *PRL* 121, 091102 — universal Λ–compactness relation
6. Yagi & Yunes (2013), *PRD* 88, 023009 — I-Love-Q universal relations
7. CompOSE database: [compose.obspm.fr](https://compose.obspm.fr)
8. MUSES framework: [musesframework.io](https://musesframework.io)
9. Bilby: [lscsoft.docs.ligo.org/bilby](https://lscsoft.docs.ligo.org/bilby/)
