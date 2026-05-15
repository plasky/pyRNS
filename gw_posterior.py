"""
gw_posterior.py  —  GW posterior samples → mass-radius posterior predictive
============================================================================

Reads Bilby parameter-estimation output files containing posterior samples
of component masses and tidal deformabilities (Λ₁, Λ₂), converts them to
(mass, radius) pairs using the universal Λ–compactness relation, and plots
the posterior predictive distribution (PPD) for the neutron-star M–R plane.

Supported Bilby output formats
-------------------------------
  .json   — Bilby v1/v2 result JSON  (native or via the bilby package)
  .hdf5   — Bilby v2 result HDF5
  .h5     — same
  .csv    — plain comma-separated posterior table
  .txt    — space/tab-separated posterior table

Usage
-----
  # Plot M-R PPD from a BNS posterior file
  python3 gw_posterior.py posterior.json

  # Include only component 1
  python3 gw_posterior.py posterior.json --comp 1

  # Overlay an EOS M-R curve
  python3 gw_posterior.py posterior.json --mr-eos eos/eosA

  # Custom output
  python3 gw_posterior.py posterior.json --savefig GW170817_mr.pdf

Universal Λ–R relation
-----------------------
The conversion Λ → compactness C = GM/(c²R) → R uses the polynomial fit of
De et al. (2018, PRL 121 091102), valid for C ∈ [0.10, 0.35]:

    C = 0.371 − 0.0391 ln Λ + 0.001056 (ln Λ)²

An alternative Yagi–Yunes (2013) fit is also available via --relation yy.

References
----------
  De et al. (2018), PRL 121, 091102  — universal Λ–C relation
  Yagi & Yunes (2013), PRD 88, 023009 — I-Love-Q universality
  Bilby: https://lscsoft.docs.ligo.org/bilby/
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_G    = 6.6732e-8    # cm³ g⁻¹ s⁻²
_C    = 2.9979e10    # cm s⁻¹
_MSUN = 1.9870e33   # g


# ---------------------------------------------------------------------------
# Universal Λ → compactness → radius relations
# ---------------------------------------------------------------------------

def lambda_to_compactness(lam, relation="de2018"):
    """
    Convert dimensionless tidal deformability Λ to compactness C = GM/(c²R).

    Parameters
    ----------
    lam      : array-like  Tidal deformability Λ (dimensionless, Λ > 0)
    relation : str         'de2018' (default) or 'yy2013'

    Returns
    -------
    C : ndarray  Compactness (dimensionless, ~ 0.10–0.35 for NS)

    Notes
    -----
    de2018 : De et al. (2018) PRL 121 091102, Eq. (3):
        C = 0.371 − 0.0391 ln(Λ) + 0.001056 [ln(Λ)]²

    yy2013 : Yagi & Yunes (2013) PRD 88 023009, inverted from their
        polynomial fit of ln(Λ) as a function of C.  Solved numerically.
    """
    lam = np.asarray(lam, dtype=float)
    ln_lam = np.log(np.maximum(lam, 1e-10))

    if relation.lower() in ("de2018", "de"):
        # De et al. 2018 — direct polynomial C(ln Λ)
        C = 0.371 - 0.0391 * ln_lam + 0.001056 * ln_lam**2
    elif relation.lower() in ("yy2013", "yy", "yagi-yunes"):
        # Yagi & Yunes 2013 (PRD 88 023009), Table I, normal NS fit:
        #   ln Λ = a₁ η⁰ + a₂ η¹ + a₃ η² + a₄ η³ + a₅ η⁴, η ≡ ln C
        # We invert numerically with Newton–Raphson.
        a = np.array([1.5004e+01, -1.4393e+00, 4.0743e-01,
                      -7.5118e-03, 1.0706e-03])
        def f(eta):
            return (a[0] + a[1]*eta + a[2]*eta**2
                    + a[3]*eta**3 + a[4]*eta**4) - ln_lam
        def df(eta):
            return a[1] + 2*a[2]*eta + 3*a[3]*eta**2 + 4*a[4]*eta**3

        eta = np.full_like(ln_lam, -2.0)   # initial guess C ~ exp(-2) ≈ 0.14
        for _ in range(50):
            h = f(eta) / np.maximum(np.abs(df(eta)), 1e-15) * np.sign(df(eta))
            eta -= h
            if np.all(np.abs(h) < 1e-12):
                break
        C = np.exp(eta)
    else:
        raise ValueError(f"Unknown relation '{relation}'. Use 'de2018' or 'yy2013'.")

    return np.clip(C, 0.05, 0.5)   # physical sanity bounds


def lambda_to_radius(mass_msun, lam, relation="de2018"):
    """
    Convert (mass [M☉], Λ) → circumferential radius [km].

    Parameters
    ----------
    mass_msun : array-like  Gravitational mass in solar masses
    lam       : array-like  Dimensionless tidal deformability Λ
    relation  : str         Universal relation to use (see lambda_to_compactness)

    Returns
    -------
    R_km : ndarray  Equatorial radius in km
    """
    mass_msun = np.asarray(mass_msun, dtype=float)
    lam       = np.asarray(lam,       dtype=float)
    C         = lambda_to_compactness(lam, relation=relation)
    # R = G M / (c² C)   [cm]
    R_cm = _G * mass_msun * _MSUN / (_C**2 * C)
    return R_cm / 1e5    # → km


# ---------------------------------------------------------------------------
# Bilby posterior file I/O
# ---------------------------------------------------------------------------

def read_bilby(filename):
    """
    Read a Bilby result file and return the posterior as a dict of arrays.

    Supported formats: .json, .hdf5 / .h5, .csv, .txt

    Returns
    -------
    posterior : dict  {parameter_name: np.ndarray}
    meta      : dict  {label, outdir, n_samples, sampler, ...}
    """
    ext = os.path.splitext(filename)[1].lower()

    # ── Try the bilby package first (handles all formats natively) ─────────
    try:
        import bilby                                     # noqa: F401
        result = bilby.result.read_in_result(filename)
        df = result.posterior
        posterior = {col: df[col].values for col in df.columns}
        meta = {
            "label":     getattr(result, "label",     "unknown"),
            "outdir":    getattr(result, "outdir",    ""),
            "n_samples": len(df),
            "sampler":   getattr(result, "sampler",   "unknown"),
            "log_evidence": getattr(result, "log_evidence", float("nan")),
        }
        return posterior, meta
    except ImportError:
        pass   # bilby not installed — fall back to manual parsing
    except Exception as exc:
        print(f"  bilby.read_in_result failed ({exc}); trying manual parse …",
              flush=True)

    # ── Manual JSON parsing ─────────────────────────────────────────────────
    if ext == ".json":
        import json
        with open(filename) as fh:
            data = json.load(fh)

        # Bilby v1/v2 JSON layout: data['posterior']['content'] is a dict
        # of {param: [values]} or data['posterior'] is a DataFrame-like dict
        post_node = data.get("posterior", {})
        if isinstance(post_node, dict) and "content" in post_node:
            raw = post_node["content"]
        elif isinstance(post_node, dict):
            raw = post_node
        elif isinstance(data, dict) and all(
                isinstance(v, list) for v in data.values()):
            raw = data
        else:
            raise ValueError(
                f"Cannot find posterior samples in JSON file '{filename}'. "
                "Expected keys: data['posterior']['content'] or similar."
            )

        posterior = {k: np.array(v) for k, v in raw.items()
                     if isinstance(v, (list, tuple))}
        meta = {
            "label":     data.get("label", ""),
            "outdir":    data.get("outdir", ""),
            "n_samples": len(next(iter(posterior.values()), [])),
            "sampler":   data.get("sampler", ""),
            "log_evidence": data.get("log_evidence", float("nan")),
        }
        return posterior, meta

    # ── Manual HDF5 parsing ─────────────────────────────────────────────────
    if ext in (".hdf5", ".h5"):
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required to read HDF5 files.  "
                "Install with: pip install h5py"
            )
        posterior = {}
        meta = {}
        with h5py.File(filename, "r") as fh:
            # Standard Bilby HDF5 layout: /posterior/{param}
            post_grp = fh.get("posterior", fh)
            for key in post_grp.keys():
                posterior[key] = post_grp[key][:]
            meta["n_samples"] = len(next(iter(posterior.values()), []))
            meta["label"] = fh.attrs.get("label", "")
            meta["log_evidence"] = float(
                fh["log_evidence"][()] if "log_evidence" in fh else "nan"
            )
        return posterior, meta

    # ── CSV / plain text ───────────────────────────────────────────────────
    if ext in (".csv", ".txt", ".dat"):
        import pandas as pd
        sep = "," if ext == ".csv" else r"\s+"
        df  = pd.read_csv(filename, sep=sep)
        posterior = {col: df[col].values for col in df.columns}
        meta = {"n_samples": len(df), "label": os.path.basename(filename)}
        return posterior, meta

    raise ValueError(
        f"Unrecognised file extension '{ext}'.  "
        "Expected .json, .hdf5, .h5, .csv, or .txt"
    )


# ---------------------------------------------------------------------------
# Parameter name resolution
# ---------------------------------------------------------------------------

_MASS_ALIASES = {
    "1": ["mass_1", "m1", "mass1", "M1"],
    "2": ["mass_2", "m2", "mass2", "M2"],
}
_LAMBDA_ALIASES = {
    "1": ["lambda_1", "Lambda_1", "lambda1", "tidal_1", "l1", "L1"],
    "2": ["lambda_2", "Lambda_2", "lambda2", "tidal_2", "l2", "L2"],
}
_CHIRP_ALIASES  = ["chirp_mass", "Mc", "mchirp", "chirp_mass_source"]
_Q_ALIASES      = ["mass_ratio", "q", "mass_ratio_and_q"]
_LTILDE_ALIASES = ["lambda_tilde", "Lambda_tilde", "lambdat", "lam_tilde"]


def _find_key(posterior, aliases):
    for a in aliases:
        if a in posterior:
            return a
    return None


def _extract_masses_lambdas(posterior, component):
    """
    Return (masses [M☉], lambdas) arrays for the requested component(s).

    component : '1', '2', or 'both'
    """
    results = []

    for comp in (["1", "2"] if component == "both" else [str(component)]):
        m_key = _find_key(posterior, _MASS_ALIASES[comp])
        l_key = _find_key(posterior, _LAMBDA_ALIASES[comp])

        # Fall back: reconstruct masses from chirp mass + mass ratio
        if m_key is None:
            mc_key = _find_key(posterior, _CHIRP_ALIASES)
            q_key  = _find_key(posterior, _Q_ALIASES)
            if mc_key and q_key:
                Mc = posterior[mc_key]
                q  = posterior[q_key]       # q = m2/m1 ≤ 1
                m_total = Mc * (1 + q)**0.2 / q**0.6
                if comp == "1":
                    mass = m_total / (1 + q)
                else:
                    mass = m_total * q / (1 + q)
            else:
                print(f"  WARNING: no mass parameter found for component {comp}; "
                      "skipping.", flush=True)
                continue
        else:
            mass = posterior[m_key]

        if l_key is None:
            print(f"  WARNING: no Λ parameter found for component {comp}; "
                  "skipping.", flush=True)
            continue

        lam = posterior[l_key]

        # Filter unphysical samples
        ok  = (lam > 0) & np.isfinite(lam) & np.isfinite(mass) & (mass > 0)
        results.append((mass[ok], lam[ok], comp))

    if not results:
        raise ValueError(
            "Could not find mass/Λ parameters in the posterior.\n"
            f"  Available keys: {sorted(posterior.keys())}"
        )
    return results


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_mr_ppd(posterior, component="both", relation="de2018",
                   n_samples=None, rng_seed=42):
    """
    Compute the mass-radius posterior predictive distribution.

    Parameters
    ----------
    posterior  : dict   Posterior sample dict from read_bilby()
    component  : str    '1', '2', or 'both' (default)
    relation   : str    Universal relation: 'de2018' (default) or 'yy2013'
    n_samples  : int    Thin to this many samples (None = use all)
    rng_seed   : int    RNG seed for thinning

    Returns
    -------
    M_arr : ndarray   Masses [M☉] for all (mass, radius) pairs
    R_arr : ndarray   Radii  [km]
    labels: list[str] Component label for each pair ('1' or '2')
    """
    rng = np.random.default_rng(rng_seed)
    all_M, all_R, all_labels = [], [], []

    for mass, lam, comp in _extract_masses_lambdas(posterior, component):
        if n_samples is not None and len(mass) > n_samples:
            idx   = rng.choice(len(mass), n_samples, replace=False)
            mass  = mass[idx]; lam = lam[idx]
        R = lambda_to_radius(mass, lam, relation=relation)
        all_M.append(mass); all_R.append(R); all_labels.extend([comp]*len(mass))

    return np.concatenate(all_M), np.concatenate(all_R), all_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mr_ppd(M_arr, R_arr, labels=None, *,
                savefig,
                title=None,
                credible_levels=(0.50, 0.90),
                mr_eos_file=None,
                mr_curve_data=None,
                event_label="GW event"):
    """
    Plot the M-R posterior predictive distribution as KDE credible contours.

    Parameters
    ----------
    M_arr, R_arr : ndarray   Mass [M☉] and radius [km] samples
    labels       : list[str] Component labels ('1' or '2'); None = ignore
    savefig      : str       Output PDF path (required)
    title        : str       Figure title
    credible_levels : tuple  Credible-interval levels to draw (default 50%, 90%)
    mr_eos_file  : str       Path to a tabulated RNS EOS file to overlay M-R curve
    mr_curve_data: tuple     (R_arr, M_arr) from a pre-computed M-R sequence
    event_label  : str       Legend label for the posterior
    """
    from pyRNS import _apply_style
    _apply_style()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    # ── Overlay EOS M-R curve if requested ───────────────────────────────
    if mr_curve_data is not None:
        R_eos, M_eos = mr_curve_data
        ax.plot(R_eos, M_eos, "k-", lw=1.5, label="EOS M–R", zorder=3)

    elif mr_eos_file is not None and os.path.isfile(mr_eos_file):
        try:
            _compute_and_overlay(mr_eos_file, ax)
        except Exception as exc:
            print(f"  Could not overlay EOS curve: {exc}", flush=True)

    # ── 2-D KDE ──────────────────────────────────────────────────────────
    # Use a fine grid for the KDE
    R_min, R_max = max(5,  np.percentile(R_arr, 0.5)), \
                   min(25, np.percentile(R_arr, 99.5))
    M_min, M_max = max(0.5, np.percentile(M_arr, 0.5)), \
                   min(3.5, np.percentile(M_arr, 99.5))

    n_grid = 200
    R_grid = np.linspace(R_min, R_max, n_grid)
    M_grid = np.linspace(M_min, M_max, n_grid)
    RR, MM = np.meshgrid(R_grid, M_grid)

    # Fit KDE on (R, M) — note: column order for meshgrid
    stack = np.vstack([R_arr, M_arr])
    kde   = gaussian_kde(stack, bw_method="scott")
    ZZ    = kde(np.vstack([RR.ravel(), MM.ravel()])).reshape(RR.shape)

    # Convert probability levels to density thresholds
    # Sort ZZ descending, find cumulative fractions
    ZZ_sorted = np.sort(ZZ.ravel())[::-1]
    ZZ_cumsum = np.cumsum(ZZ_sorted) / ZZ_sorted.sum()
    levels = []
    for cl in sorted(credible_levels, reverse=True):
        idx = np.searchsorted(ZZ_cumsum, cl)
        levels.append(ZZ_sorted[min(idx, len(ZZ_sorted)-1)])
    levels = sorted(levels)

    # Draw filled contours + lines
    colours = plt.cm.Blues(np.linspace(0.35, 0.75, len(credible_levels)))
    cf = ax.contourf(RR, MM, ZZ, levels=[levels[0], ZZ.max()],
                     colors=[colours[-1]], alpha=0.55, zorder=1)
    if len(levels) > 1:
        ax.contourf(RR, MM, ZZ, levels=[levels[-1], ZZ.max()],
                    colors=[colours[0]], alpha=0.35, zorder=1)
    # Draw labelled proxy patches for the legend
    import matplotlib.patches as mpatches
    legend_handles = []
    for i, (lv, cl) in enumerate(zip(levels, sorted(credible_levels, reverse=True))):
        ax.contour(RR, MM, ZZ, levels=[lv], colors=[colours[-(i+1)]],
                   linewidths=1.5, zorder=2)
        legend_handles.append(
            mpatches.Patch(facecolor=colours[-(i+1)], alpha=0.7,
                           label=f"${int(cl*100):d}\\%$ credible region")
        )

    # Scatter points if few samples, or small marker for density intuition
    if len(M_arr) < 500:
        ax.scatter(R_arr, M_arr, s=2, color="steelblue", alpha=0.4,
                   zorder=0, label=event_label)

    ax.set_xlabel(r"$R_e$  [km]")
    ax.set_ylabel(r"$M$  [$M_\odot$]")
    ax.set_xlim(R_min, R_max)
    ax.set_ylim(M_min, M_max)
    ax.grid(True, alpha=0.25)

    if title:
        ax.set_title(title)
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    fig.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved M-R PPD → {savefig}")


def _compute_and_overlay(eos_file, ax):
    """Compute and overlay a pyRNS M-R curve on ax."""
    import warnings
    warnings.filterwarnings("ignore")
    from pyRNS import (
        make_grid, precompute_legendre, compute_kernels,
        load_eos, make_center, sphere, spin, mass_radius,
        C, G, MSUN, KAPPA, KSCALE, SDIV, MDIV,
    )
    import numpy as np

    s_gp, mu = make_grid()
    P_2n, P1_2n1, sn = precompute_legendre(mu)
    fk, gk = compute_kernels(s_gp)

    log_e, log_p, log_h, log_n0, n_tab = load_eos(eos_file)
    eos = {"type": "tab", "Gamma_P": 2.0,
           "log_e": log_e, "log_p": log_p,
           "log_h": log_h, "log_n0": log_n0, "n_tab": n_tab}

    e_surf = 7.8 * C**2 * KSCALE; p_surf = 1.01e8 * KSCALE
    e_min  = 10**log_e[1]; e_max = 10**log_e[-2]

    M_seq, R_seq = [], []
    for ec in np.linspace(e_min, e_max, 20):
        try:
            p_c, h_c = make_center(ec, eos)
            rho = np.zeros((SDIV, MDIV)); gama = np.zeros((SDIV, MDIV))
            alpha = np.zeros((SDIV, MDIV)); omega = np.zeros((SDIV, MDIV))
            en = np.zeros((SDIV, MDIV)); pr = np.zeros((SDIV, MDIV))
            enth = np.zeros((SDIV, MDIV)); vsq = np.zeros((SDIV, MDIV))
            rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
                s_gp, mu, ec, p_c, h_c, p_surf, e_surf, eos)
            r_e, Om = spin(s_gp, mu, eos, h_c, 1/C**2,
                           rho, gama, alpha, omega, en, pr, enth, vsq,
                           1.0, r_e,
                           f_rho_kern=fk, f_gama_kern=gk,
                           P_2n=P_2n, P1_2n1=P1_2n1, sin_2n1_theta=sn)
            Mv, M0v, Jv, Rev, OKv, _, _ = mass_radius(
                s_gp, mu, eos, rho, gama, alpha, omega,
                en, pr, enth, vsq, 1.0, e_surf, r_e, 0.0)
            if np.isfinite(Mv) and Mv > 0:
                M_seq.append(Mv / MSUN); R_seq.append(Rev / 1e5)
        except Exception:
            pass

    if M_seq:
        ax.plot(R_seq, M_seq, "k-", lw=1.5, label="EOS M–R", zorder=3)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(M_arr, R_arr, labels=None):
    """Print median and 90% credible intervals for M and R."""
    print("\nM-R posterior predictive summary:")
    print(f"  {'Component':<12} {'M [M☉]':>22}  {'R [km]':>22}")
    print(f"  {'─'*12:<12} {'median [5%, 95%]':>22}  {'median [5%, 95%]':>22}")

    def _fmt(arr):
        med  = np.median(arr)
        lo   = np.percentile(arr, 5)
        hi   = np.percentile(arr, 95)
        return f"{med:.2f} [{lo:.2f}, {hi:.2f}]"

    if labels and len(set(labels)) > 1:
        for comp in sorted(set(labels)):
            mask = np.array(labels) == comp
            print(f"  {'comp '+comp:<12} {_fmt(M_arr[mask]):>22}  "
                  f"{_fmt(R_arr[mask]):>22}")
    print(f"  {'combined':<12} {_fmt(M_arr):>22}  {_fmt(R_arr):>22}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert Bilby GW posteriors to M-R posterior predictive distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 gw_posterior.py posterior.json
  python3 gw_posterior.py posterior.json --comp 1
  python3 gw_posterior.py posterior.json --mr-eos eos/eosA
  python3 gw_posterior.py posterior.json --relation yy2013
  python3 gw_posterior.py posterior.json --savefig GW170817_mr.pdf
  python3 gw_posterior.py posterior.json --n-samples 2000 --no-plot
""",
    )
    parser.add_argument("posterior_file",
                        help="Bilby result file (.json, .hdf5, .h5, .csv, .txt)")
    parser.add_argument("--comp",      default="both",
                        choices=["1", "2", "both"],
                        help="Which component(s) to include [default: both]")
    parser.add_argument("--relation",  default="de2018",
                        choices=["de2018", "yy2013"],
                        help="Universal Λ–C relation [default: de2018]")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Thin posterior to this many samples")
    parser.add_argument("--mr-eos",    default=None, metavar="EOS_FILE",
                        help="Tabulated RNS EOS file to overlay M-R curve")
    parser.add_argument("--savefig",   default=None, metavar="FILE",
                        help="Output PDF [default: <posteriorfile>_mr_ppd.pdf]")
    parser.add_argument("--title",     default=None,
                        help="Figure title")
    parser.add_argument("--no-plot",   dest="plot", action="store_false",
                        default=True, help="Skip figure output")
    parser.add_argument("--credible",  default="50,90",
                        help="Comma-separated credible levels in %% [default: 50,90]")
    args = parser.parse_args()

    # ── Read posterior ────────────────────────────────────────────────────
    print(f"Reading {args.posterior_file} …", flush=True)
    posterior, meta = read_bilby(args.posterior_file)
    print(f"  {meta.get('n_samples','?')} posterior samples loaded")
    if meta.get("label"):
        print(f"  label: {meta['label']}")

    # ── Compute M-R PPD ───────────────────────────────────────────────────
    print(f"Computing M-R PPD  (component={args.comp}, "
          f"relation={args.relation}) …", flush=True)
    M_arr, R_arr, labels = compute_mr_ppd(
        posterior,
        component=args.comp,
        relation=args.relation,
        n_samples=args.n_samples,
    )
    print(f"  {len(M_arr)} (M, R) pairs generated")

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(M_arr, R_arr, labels)

    # ── Save samples ──────────────────────────────────────────────────────
    base    = os.path.splitext(args.posterior_file)[0]
    txt_out = base + "_mr_ppd.txt"
    header  = "M [M_sun]   R [km]   component"
    np.savetxt(txt_out,
               np.column_stack([M_arr, R_arr,
                                 [int(l) for l in labels]]),
               header=header, fmt=["%.6e", "%.6e", "%d"])
    print(f"  Saved text table → {txt_out}")

    # ── Plot ──────────────────────────────────────────────────────────────
    if args.plot:
        savefig = args.savefig or (base + "_mr_ppd.pdf")
        levels  = tuple(float(x) / 100 for x in args.credible.split(","))
        plot_mr_ppd(
            M_arr, R_arr, labels,
            savefig=savefig,
            title=args.title or meta.get("label", ""),
            credible_levels=levels,
            mr_eos_file=args.mr_eos,
            event_label=meta.get("label", "GW event"),
        )


if __name__ == "__main__":
    main()
