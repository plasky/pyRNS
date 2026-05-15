"""
mr_curve.py — Build mass-radius sequences and publication-quality plots with pyRNS.

Usage
-----
  python3 mr_curve.py --poly -N 1.0        # polytrope N=1, plots on by default
  python3 mr_curve.py --poly --no-plot      # suppress plots
  python3 mr_curve.py -f eosA               # tabulated EOS

Unit conventions (matching the NS literature)
----------------------------------------------
  Mass       : M_☉  (solar masses)
  Radius     : km
  Density    : g cm⁻³
  Pressure   : dyne cm⁻²
  Ω (angular velocity) : rad s⁻¹

For polytropes without a physical scale, all quantities remain in code units
and are labelled accordingly.
"""

import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from pyRNS import (
    make_grid, precompute_legendre, compute_kernels,
    make_center, sphere, spin, mass_radius,
    load_eos,
    C, G, MSUN, KAPPA, KSCALE, SDIV, MDIV,
)


# ── unit helpers ──────────────────────────────────────────────────────────────

def _units(is_tab):
    """Return (label_M, label_R, label_ec, conv_M, conv_R, conv_ec, conv_Omega)."""
    if is_tab:
        return dict(
            lM   = r"$M$ [$M_\odot$]",
            lR   = r"$R_e$ [km]",
            lec  = r"$\varepsilon_c$ [$\rm g\,cm^{-3}$]",
            lOmK = r"$\Omega_K$ [rad s$^{-1}$]",
            lJ   = r"$J$ [$\rm g\,cm^2\,s^{-1}$]",
            cM   = 1.0 / MSUN,
            cR   = 1.0 / 1e5,          # cm → km
            cec  = 1.0 / (C**2 * KSCALE),
            cOm  = C / KAPPA**0.5,     # code → rad/s
        )
    else:
        return dict(
            lM   = r"$M$ [code units]",
            lR   = r"$R_e$ [code units]",
            lec  = r"$\varepsilon_c$ [code units]",
            lOmK = r"$\Omega_K$ [code units]",
            lJ   = r"$J$ [code units]",
            cM   = 1.0,
            cR   = 1.0,
            cec  = 1.0,
            cOm  = 1.0,
        )


# ── single-model helper ───────────────────────────────────────────────────────

def _model(e_c, eos, *, e_surface, p_surface, enthalpy_min,
           s_gp, mu_arr, P_2n, P1_2n1, sn, fk, gk,
           r_ratio=1.0, accuracy=1e-4, cf=0.5, verbose=False):
    """Compute one equilibrium model. Returns (M,M0,Re,OK,J,Omega)."""
    rho   = np.zeros((SDIV, MDIV));  gama  = np.zeros((SDIV, MDIV))
    alpha = np.zeros((SDIV, MDIV));  omega = np.zeros((SDIV, MDIV))
    en    = np.zeros((SDIV, MDIV));  pr    = np.zeros((SDIV, MDIV))
    enth  = np.zeros((SDIV, MDIV));  vsq   = np.zeros((SDIV, MDIV))

    p_c, h_c = make_center(e_c, eos)
    rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
        s_gp, mu_arr, e_c, p_c, h_c, p_surface, e_surface, eos)

    r_e, Omega = spin(
        s_gp, mu_arr, eos, h_c, enthalpy_min,
        rho, gama, alpha, omega, en, pr, enth, vsq,
        r_ratio, r_e,
        accuracy=accuracy, cf=cf,
        f_rho_kern=fk, f_gama_kern=gk,
        P_2n=P_2n, P1_2n1=P1_2n1, sin_2n1_theta=sn,
        verbose=verbose)

    M, M0, J, Re, OK, _, _ = mass_radius(
        s_gp, mu_arr, eos,
        rho, gama, alpha, omega, en, pr, enth, vsq,
        r_ratio, e_surface, r_e, Omega)

    return M, M0, Re, OK, J, Omega


# ── non-rotating sequence ─────────────────────────────────────────────────────

def nonrotating_sequence(eos, *, is_tab, e_min, e_max, n_pts,
                         e_surface, p_surface, enthalpy_min,
                         s_gp, mu_arr, P_2n, P1_2n1, sn, fk, gk):
    """Return arrays (e_c, M, M0, Re, Omega_K) for non-rotating models."""
    e_centers = np.linspace(e_min, e_max, n_pts)
    rows = []
    u = _units(is_tab)

    for i, ec in enumerate(e_centers):
        try:
            M, M0, Re, OK, J, Omega = _model(
                ec, eos,
                e_surface=e_surface, p_surface=p_surface,
                enthalpy_min=enthalpy_min,
                s_gp=s_gp, mu_arr=mu_arr,
                P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk)

            if not np.isfinite(M) or M <= 0:
                print(f"  e_c={ec*u['cec']:.3e}: skipped", flush=True)
                continue

            rows.append((ec, M, M0, Re, OK))
            print(f"  [{i+1:2d}/{n_pts}]  ε_c={ec*u['cec']:.3e}  "
                  f"M={M*u['cM']:.3f}  Re={Re*u['cR']:.3f}  "
                  f"Ω_K={OK*u['cOm']:.3f}", flush=True)

        except Exception as exc:
            print(f"  e_c={ec*u['cec']:.3e}: error ({exc})", flush=True)

    if not rows:
        return None
    return np.array(rows).T   # shape (5, n_valid)  — e_c, M, M0, Re, OK


# ── plots ─────────────────────────────────────────────────────────────────────

def make_plots(seq, is_tab, eos_label="", savepdf=None):
    """
    Four-panel figure for a non-rotating mass-radius sequence.

    Panels
    ------
    1. Mass-radius diagram  M(R_e)
    2. Mass vs central density  M(ε_c)
    3. Compactness  GM/(c²R) vs R
    4. Binding energy fraction  (M_0 − M)/M vs M
    """
    try:
        from pyRNS import _apply_style
        _apply_style()
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[mr_curve] matplotlib not available – skipping plots.")
        return

    u = _units(is_tab)
    ec_arr, M_arr, M0_arr, Re_arr, OK_arr = seq

    # Convert to display units
    M  = M_arr  * u["cM"]
    M0 = M0_arr * u["cM"]
    R  = Re_arr * u["cR"]
    ec = ec_arr * u["cec"]
    OK = OK_arr * u["cOm"]

    # Compactness  C = GM/(c² R)   [dimensionless]
    # For tabulated: M in g, R in km → need G M / (c² R)
    if is_tab:
        compactness = G * (M * MSUN) / (C**2 * (R * 1e5))
    else:
        # code units: use M and R directly (no physical meaning without K)
        compactness = M / R

    # Binding energy fraction
    binding = (M0 - M) / np.maximum(M, 1e-30)   # (M0_bary − M_grav) / M_grav

    # Index of maximum mass
    imax = np.argmax(M)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    col_main  = "#1a73e8"
    col_max   = "red"
    ms        = 5
    lw        = 1.8
    star_kw   = dict(marker="*", ms=13, color=col_max, zorder=5, ls="none")

    # ── 1. M-R diagram ────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(R, M, "-o", color=col_main, lw=lw, ms=ms, label="Non-rotating")
    ax.plot(R[imax], M[imax], **star_kw, label=f"$M_{{\\rm max}}={M[imax]:.3f}$")
    ax.set_xlabel(u["lR"])
    ax.set_ylabel(u["lM"])
    ax.set_title("Mass–Radius diagram")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.annotate(" $M_{\\rm max}$", xy=(R[imax], M[imax]),
                xytext=(R[imax] + 0.04 * (R.max() - R.min()), M[imax]),
                fontsize=9, color=col_max,
                arrowprops=dict(arrowstyle="->", color=col_max, lw=0.8))

    # ── 2. M vs ε_c  +  Ω_K secondary axis ──────────────────────────────
    ax = axes[1]
    ax.plot(ec, M, "-o", color="tab:orange", lw=lw, ms=ms)
    ax.plot(ec[imax], M[imax], **star_kw)
    ax.set_xlabel(u["lec"])
    ax.set_ylabel(u["lM"])
    ax.set_title(r"Mass vs central density")
    if is_tab:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.25)
    ax.axvline(ec[imax], ls="--", color=col_max, alpha=0.4)
    # Ω_K on a secondary axis
    ax2 = ax.twinx()
    ax2.plot(ec, OK, "--s", color="tab:red", lw=1.2, ms=3, alpha=0.7)
    ax2.set_ylabel(u["lOmK"], color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")

    # ── 3. Compactness ────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(R, compactness, "-o", color="tab:green", lw=lw, ms=ms)
    ax.plot(R[imax], compactness[imax], **star_kw)
    ax.set_xlabel(u["lR"])
    clabel = (r"Compactness  $GM/(c^2 R)$" if is_tab
              else r"Compactness  $M/R$  [code]")
    ax.set_ylabel(clabel)
    ax.set_title("Compactness")
    if is_tab:
        # Buchdahl limit (9/8 × Schwarzschild: GM/c²R = 4/9)
        ax.axhline(4.0 / 9.0, ls="--", color="gray", alpha=0.6,
                   label="Buchdahl (4/9)")
        ax.axhline(0.5, ls=":", color="gray", alpha=0.4,
                   label="Black-hole limit (1/2)")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # ── 4. Binding energy ────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(M, binding * 100.0, "-o", color="tab:purple", lw=lw, ms=ms)
    ax.plot(M[imax], binding[imax] * 100.0, **star_kw)
    ax.set_xlabel(u["lM"])
    ax.set_ylabel(r"Binding energy  $(M_0 - M)/M$ [%]")
    ax.set_title("Gravitational binding energy")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, ls="-", color="gray", lw=0.8, alpha=0.5)

    title = f"Non-rotating sequence — {eos_label}" if eos_label else "Non-rotating sequence"
    fig.suptitle(title, fontsize=13, y=1.01)

    fig.savefig(savepdf, dpi=150, bbox_inches="tight")
    print(f"  Saved sequence plot → {savepdf}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="pyRNS: mass-radius sequence builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 mr_curve.py --poly -N 1.0           # N=1 polytrope (30 models, plots on)
  python3 mr_curve.py --poly --no-plot         # suppress plots
  python3 mr_curve.py -f eosA                  # tabulated EOS
  python3 mr_curve.py --poly --savefig mr.pdf  # save plot to file
""")
    # EOS
    ap.add_argument("--poly",  action="store_true", help="polytrope (default)")
    ap.add_argument("-N",      type=float, default=1.0,  help="polytropic index N")
    ap.add_argument("-f",      metavar="EOS", default="", help="tabulated EOS file")
    # Sequence parameters
    ap.add_argument("--emin",  type=float, default=None, help="min central density")
    ap.add_argument("--emax",  type=float, default=None, help="max central density")
    ap.add_argument("--npts",  type=int,   default=30,   help="number of models [30]")
    # Output / plots
    ap.add_argument("--out",   default="mr_data.txt", help="output text file")
    ap.add_argument("--plot",  dest="plot", action="store_true",  default=True,
                    help="produce plots (default: on)")
    ap.add_argument("--no-plot", dest="plot", action="store_false",
                    help="suppress plots")
    ap.add_argument("--savefig", metavar="FILE", default=None,
                    help="save figure to FILE (e.g. mr.pdf)")
    args = ap.parse_args()

    is_poly = args.poly or (not args.f)
    is_tab  = not is_poly
    Gamma_P = 1.0 + 1.0 / args.N if is_poly else 2.0

    # ── EOS ──────────────────────────────────────────────────────────────
    eos = {"type": "poly" if is_poly else "tab", "Gamma_P": Gamma_P}
    if is_tab:
        if not os.path.isfile(args.f):
            ap.error(f"EOS file not found: {args.f}")
        log_e, log_p, log_h, log_n0, n_tab = load_eos(args.f)
        eos.update({"log_e": log_e, "log_p": log_p,
                    "log_h": log_h, "log_n0": log_n0, "n_tab": n_tab})

    if is_poly:
        e_min        = args.emin if args.emin is not None else 0.05
        e_max        = args.emax if args.emax is not None else 1.5
        e_surface    = 0.0;  p_surface = 0.0;  enthalpy_min = 0.0
    else:
        e_min        = args.emin if args.emin is not None else (5e14 * C**2 * KSCALE)
        e_max        = args.emax if args.emax is not None else (3e15 * C**2 * KSCALE)
        e_surface    = 7.8   * C**2 * KSCALE
        p_surface    = 1.01e8 * KSCALE
        enthalpy_min = 1.0 / C**2

    # ── grid (precomputed once) ───────────────────────────────────────────
    print("Setting up grid and precomputing kernels...", flush=True)
    s_gp, mu_arr      = make_grid()
    P_2n, P1_2n1, sn  = precompute_legendre(mu_arr)
    fk, gk            = compute_kernels(s_gp)
    common = dict(e_surface=e_surface, p_surface=p_surface,
                  enthalpy_min=enthalpy_min,
                  s_gp=s_gp, mu_arr=mu_arr,
                  P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk)

    # ── non-rotating sequence ─────────────────────────────────────────────
    label = (f"Polytrope $N={args.N}$" if is_poly
             else os.path.basename(args.f))
    print(f"\nNon-rotating sequence  ({args.npts} models)  —  {label}",
          flush=True)
    seq = nonrotating_sequence(eos, is_tab=is_tab,
                               e_min=e_min, e_max=e_max, n_pts=args.npts,
                               **common)
    if seq is None:
        print("No valid models computed."); return

    ec_arr, M_arr, M0_arr, Re_arr, OK_arr = seq
    u = _units(is_tab)
    np.savetxt(args.out,
               np.column_stack([ec_arr * u["cec"],
                                 M_arr  * u["cM"],
                                 M0_arr * u["cM"],
                                 Re_arr * u["cR"],
                                 OK_arr * u["cOm"]]),
               header=(f"e_c [{u['lec']}]   M [{u['lM']}]   "
                       f"M0 [{u['lM']}]   Re [{u['lR']}]   "
                       f"Omega_K [{u['lOmK']}]"),
               fmt="%.6e")

    imax = np.argmax(M_arr)
    print(f"\nSaved {seq.shape[1]} models to '{args.out}'")
    print(f"Maximum mass: M = {M_arr[imax]*u['cM']:.4f} {u['lM']}  "
          f"at ε_c = {ec_arr[imax]*u['cec']:.3e} {u['lec']}")
    print(f"  Radius at M_max: R_e = {Re_arr[imax]*u['cR']:.3f} {u['lR']}")

    # ── plots ─────────────────────────────────────────────────────────────
    if args.plot:
        savepdf = args.savefig or args.out.replace(".txt", ".pdf")
        make_plots(seq, is_tab=is_tab, eos_label=label, savepdf=savepdf)


if __name__ == "__main__":
    main()
