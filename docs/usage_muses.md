# muses_eos.py — CompOSE / MUSES interface

`muses_eos.py` provides access to the
[CompOSE](https://compose.obspm.fr) database of nuclear equations of state,
which is part of the [MUSES](https://musesframework.io) framework.
It downloads, caches and converts EOS tables, and can plot M–R curves for
one or more EOS on the same figure.

---

## Commands

```
python3 muses_eos.py <command> [options]
```

| Command | Description |
|---------|-------------|
| `list`  | Print all available cold-NS EOS from CompOSE |
| `fetch` | Download and convert one EOS (no run) |
| `run`   | Download EOS and run a single-star model via `pyRNS.py` |
| `mrseq` | Download EOS and plot the CompOSE pre-computed M–R sequence |

---

## `list` — browse available EOS

```bash
python3 muses_eos.py list
```

Prints a table of EOS IDs and names.  The catalog is cached to
`eos/_catalog_cache.json` after the first fetch (instant on subsequent runs).

```bash
# Re-fetch catalog from CompOSE (e.g. after new EOS are added)
python3 muses_eos.py list --refresh
```

---

## `mrseq` — mass-radius sequence

### Single EOS

```bash
python3 muses_eos.py mrseq --eos "RG(SK255)"
python3 muses_eos.py mrseq --eos-id 94          # same, using CompOSE ID
```

Produces:
- `eos/RG_SK255_/RG_SK255__mr.txt` — table of (R [km], M [M☉])
- `eos/RG_SK255_/RG_SK255__mr.pdf` — two-panel figure (M–R + compactness)

### Multiple EOS — all on one figure

```bash
# By name (substring match)
python3 muses_eos.py mrseq --eos "RG(SK255)" "HS(DD2)" "DS(CMF)-5"

# By CompOSE numeric ID
python3 muses_eos.py mrseq --eos-id 94 184

# Custom output filename
python3 muses_eos.py mrseq \
  --eos "RG(SK255)" "HS(DD2)" \
  --savefig results/my_comparison.pdf
```

When multiple EOS are given, the figure filename is auto-generated as
`mr_comparison_<eos1>_<eos2>_….pdf` in `--outdir`.

> **Accuracy**: `mrseq` uses CompOSE's own pre-computed `eos.mr` tables
> (full numerical TOV by CompOSE itself), so results are exact.

---

## `run` — single-star model via pyRNS

```bash
python3 muses_eos.py run --eos "RG(SK255)" --e 1e15
python3 muses_eos.py run --eos-id 94 --e 5e14 --savefig star.pdf
```

Converts the CompOSE EOS to RNS tabulated format then calls `pyRNS.py`.

> **Accuracy**: radius is accurate to ~3%; mass is ~20% lower than
> CompOSE's exact value due to the approximate `eos.thermo` column
> interpretation.  Use `mrseq` for accurate M–R results.

---

## `fetch` — download and convert only

```bash
python3 muses_eos.py fetch --eos "RG(SK255)"
python3 muses_eos.py fetch --eos "HS(DD2)" --outdir my_eos/
```

Downloads `eos.t`, `eos.nb`, `eos.yq`, `eos.thermo` and converts to
`<name>_rns.eos` without running any model.

---

## Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--outdir DIR` | `eos/` | Root directory for downloaded files |
| `--force` | off | Bypass all caches; re-download and re-convert |
| `--refresh` | off | Re-fetch EOS catalog from CompOSE (`list` / any command) |
| `--n-min FM3` | `1e-4` | Minimum baryon density [fm⁻³] for EOS conversion |
| `--n-max FM3` | None | Maximum baryon density [fm⁻³] |
| `--savefig FILE` | auto | Override figure filename |
| `--no-plot` | off | Suppress figure output |

---

## Caching

Every step is cached to avoid repeated network requests:

| What | Cache location | Invalidated by |
|------|----------------|---------------|
| EOS catalog | `eos/_catalog_cache.json` | `--refresh` or `--force` |
| Raw CompOSE files | `eos/<name>/eos.t`, `eos.nb`, `eos.yq`, `eos.thermo` | `--force` |
| Converted RNS EOS | `eos/<name>/<name>_rns.eos` | `--force` or if `eos.thermo` is newer |
| Pre-computed M-R | `eos/<name>/eos.mr` | `--force` |

On the first run a given EOS requires network access. All subsequent runs
use only local files.

---

## Python API

### Fetch and get M–R data for one EOS

```python
from muses_eos import _load_mr_data
import numpy as np

R, M, label, txt = _load_mr_data("RG(SK255)", outdir="eos/")
print(f"{label}: M_max = {M.max():.3f} M_sun  at R = {R[M.argmax()]:.2f} km")
```

### Plot multiple EOS

```python
from muses_eos import _load_mr_data, plot_mr_multi

datasets = []
for name in ["RG(SK255)", "HS(DD2)", "DS(CMF)-5"]:
    try:
        R, M, label, _ = _load_mr_data(name)
        datasets.append((R, M, label))
    except Exception as e:
        print(f"Skipping {name}: {e}")

plot_mr_multi(datasets, savefig="comparison.pdf",
              title="EOS comparison")
```

### Full pipeline

```python
from muses_eos import run_mrseq

# Single
run_mrseq("RG(SK255)", plot=True, savefig="sk255.pdf")

# Multiple — one figure
run_mrseq(["RG(SK255)", "HS(DD2)"],
          savefig="sk255_dd2.pdf")

# No plot, just download and cache
run_mrseq(["RG(SK255)", "HS(DD2)"], plot=False)
```

### Using the converted EOS in pyRNS

```python
from muses_eos import fetch_and_convert
from pyRNS import load_eos, make_center, ...

rns_file, name = fetch_and_convert("RG(SK255)")
log_e, log_p, log_h, log_n0, n_tab = load_eos(rns_file)
# ... continue as with any tabulated EOS
```

---

## Naming and disambiguation

EOS names are matched by **case-insensitive substring**:

```bash
# These all find the same EOS
python3 muses_eos.py mrseq --eos "SK255"
python3 muses_eos.py mrseq --eos "rg(sk255)"
python3 muses_eos.py mrseq --eos "RG(SK255)"
```

If the substring matches more than one EOS, an error lists the candidates:

```
ValueError: Ambiguous EOS name 'DD2'. Matches:
  [1] HS(DD2) neutron matter (no electrons)
  [42] HS(DD2) (with electrons)
  ...
```

Resolve by using the full name or the CompOSE ID (`--eos-id`).

---

## Which CompOSE EOS have `eos.mr`?

Not all EOS in CompOSE have pre-computed `eos.mr` files (the TOV solution).
The `mrseq` command will warn and skip any EOS where `eos.mr` is absent
(HTTP 404).  Use `list` to browse, then try `mrseq` — missing files are
handled gracefully.

---

## Directory structure after downloads

```
eos/
├── _catalog_cache.json          ← catalog (all 200+ EOS, auto-built)
├── RG_SK255_/
│   ├── eos.t                    ← temperature grid (T=0)
│   ├── eos.nb                   ← baryon density grid
│   ├── eos.yq                   ← charge fraction (Y_Q=0)
│   ├── eos.thermo               ← thermodynamic table
│   ├── eos.mr                   ← CompOSE pre-computed M-R
│   ├── RG_SK255__rns.eos        ← converted tabulated EOS for pyRNS
│   └── RG_SK255__mr.txt         ← parsed M-R text table
└── HS_DD2_.../
    └── ...
```
