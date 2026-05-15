# Installation

## Requirements

| Package | Minimum version | Notes |
|---------|----------------|-------|
| Python | 3.9 | f-strings, `math.isfinite` |
| NumPy | 1.24 | `np.trapezoid` (NumPy ≥ 2.0) |
| SciPy | 1.10 | `scipy.linalg.solve_banded` |
| Matplotlib | 3.7 | Agg backend, `constrained_layout` |

All packages are available via `pip`.  No compiled extensions are required.

---

## Standard install (macOS / Linux)

```bash
# Clone the repository
git clone <repo-url>
cd pyRNS

# Install dependencies
python3 -m pip install numpy scipy matplotlib

# Confirm
python3 -c "import pyRNS; print('pyRNS import OK')"
```

If you are on a managed system (e.g. a university HPC) that uses PEP 668
you may need:

```bash
python3 -m pip install numpy scipy matplotlib --break-system-packages
# or
python3 -m pip install numpy scipy matplotlib --user
```

---

## Conda environment (recommended for isolation)

```bash
conda create -n pyrns python=3.11 numpy scipy matplotlib
conda activate pyrns
cd pyRNS
python3 -c "import pyRNS; print('OK')"
```

---

## HPC modules

On systems using `module`:

```bash
module load python/3.11 numpy scipy matplotlib
cd pyRNS
python3 -c "import pyRNS; print('OK')"
```

---

## Verifying the installation

Run a quick non-rotating polytrope model:

```bash
python3 pyRNS.py -q poly -N 1.0 -e 0.5 --no-plot
```

Expected output (last data line):

```
1.000   0.500   0.188   0.179   0.733   0.000   0.184   0.00   0.000
```

Columns: r_ratio, e_c, M [M☉], M₀ [M☉], R [km], Ω [rad/s], Ω_K [rad/s], I, J/M².

---

## LaTeX / font requirements for plots

The bundled `publication.mplstyle` enables `text.usetex = True` and uses
Times New Roman.  This requires:

- A working LaTeX installation (`pdflatex`, or `xelatex` / `lualatex`)
- The `type1cm` and `dvipng` packages (usually part of TeX Live or MiKTeX)

On macOS with Homebrew:

```bash
brew install --cask mactex-no-gui
```

On Ubuntu/Debian:

```bash
sudo apt install texlive-latex-base texlive-fonts-recommended dvipng
```

If LaTeX is not available, edit `publication.mplstyle` and set:

```
text.usetex : False
font.family : serif
```

Plots will still be saved as PDFs but labels will use the default
matplotlib font rather than Times New Roman.

---

## No-install usage (portable copy)

Because pyRNS has no compiled extensions, the entire directory can be
copied to any machine with Python ≥ 3.9 and the three pip packages
installed.  The `publication.mplstyle` file and the `eos/` directory
travel with the code and reproduce identical results on any host.
