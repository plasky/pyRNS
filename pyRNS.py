#!/usr/bin/env python3
"""
pyRNS — Python implementation of the RNS rotating neutron star equilibrium code.

Solves the equations of hydrostatic equilibrium in general relativity for
non-rotating and rapidly rotating neutron stars using the self-consistent
field method of Komatsu, Eriguchi & Hachisu (1989), with improvements by
Cook, Shapiro & Teukolsky (1994) and Stergioulas & Friedman (1995).

The metric is written in quasi-isotropic coordinates:
  ds² = −e^(2ν) dt² + e^(2ζ)(dr² + r²dθ²) + e^(2ψ) r²sin²θ (dφ − ω dt)²

Four potentials on a compactified 2-D grid (s × μ):
  rho   = ν − ζ        (lapse minus spatial conformal factor)
  gama  = ν + ζ        (lapse plus spatial conformal factor)
  alpha = ζ            (spatial conformal factor, integrated from gama & rho)
  omega = ω            (frame-dragging angular velocity)

The compactified radial coordinate is
  s = r_is / (r_is + r_e),   r_is ∈ [0, ∞)  →  s ∈ [0, 1)
where r_e is the isotropic equatorial coordinate radius.

Usage
-----
  python pyRNS.py -f eosA -e 1e15          tabulated EOS
  python pyRNS.py -q poly -N 1.0 -e 0.5   polytrope

References
----------
  Komatsu, Eriguchi & Hachisu 1989, MNRAS 237, 355
  Cook, Shapiro & Teukolsky 1994, ApJ 422, 227
  Stergioulas & Friedman 1995, ApJ 444, 306
  Original C code: RNS v2.0 by N. Stergioulas (1999)
"""

import numpy as np
from scipy.linalg import solve_banded
import argparse
import os

# ---------------------------------------------------------------------------
# Matplotlib style helper
# ---------------------------------------------------------------------------
# The 'publication' style file is bundled with this package so that plots
# look identical on any machine, regardless of whether the user has the style
# installed in their local ~/.matplotlib/stylelib/.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_STYLE_FILE = os.path.join(_PKG_DIR, "publication.mplstyle")


def _apply_style():
    """
    Apply the bundled 'publication' matplotlib style.

    Priority order:
      1. Bundled publication.mplstyle in the package directory  ← always wins
      2. Silently skip if matplotlib is not installed
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive — never opens a window
        import matplotlib.pyplot as plt
        if os.path.exists(_STYLE_FILE):
            plt.style.use(_STYLE_FILE)
        else:
            # Fall back to the user's installed copy if available
            try:
                plt.style.use("publication")
            except OSError:
                pass   # no style available — use matplotlib defaults
    except ImportError:
        pass

# ============================================================================
# Physical constants (CGS)
# ============================================================================
C    = 2.9979e10     # speed of light          [cm s⁻¹]
G    = 6.6732e-8     # gravitational constant   [cm³ g⁻¹ s⁻²]
MSUN = 1.987e33      # solar mass               [g]
MB   = 1.66e-24      # baryon mass              [g]
PI   = np.pi

KAPPA  = 1.0e-15 * C * C / G           # length scale   [cm]
KSCALE = KAPPA * G / (C * C * C * C)   # pressure scale [dimensionless]

# ============================================================================
# Grid / numerical parameters  (match RNS v2.0 defaults)
# ============================================================================
SDIV  = 129     # radial grid points
MDIV  = 65      # angular grid points  (mu[0]=equator, mu[-1]=pole)
RDIV  = 900     # TOV Runge-Kutta steps
LMAX  = 10      # Legendre expansion order
SMAX  = 0.9999  # maximum s (maps r_is → ∞)

DS   = SMAX / (SDIV - 1)
DM   = 1.0  / (MDIV - 1)
RMIN = 1.0e-15  # threshold for near-centre Taylor expansions


# ============================================================================
# Grid setup
# ============================================================================

def make_grid():
    """Return (s_gp, mu) arrays."""
    s_gp = np.linspace(0.0, SMAX, SDIV)   # s[0]=0, s[-1]=SMAX
    mu   = np.linspace(0.0, 1.0,  MDIV)   # mu[0]=equator, mu[-1]=pole
    return s_gp, mu


# ============================================================================
# EOS loading and interpolation
# ============================================================================

def load_eos(eos_file):
    """
    Read a tabulated EOS file.

    File format:
      line 1: n_tab (integer number of table rows)
      next n_tab lines: e_cgs  p_cgs  h_cgs  n0_cgs
        e  [g cm⁻³]   energy density / c²
        p  [dyn cm⁻²] pressure
        h  [cm² s⁻²]  specific enthalpy
        n0 [cm⁻³]     baryon number density

    Values are converted to dimensionless code units and stored as log10.
    """
    with open(eos_file) as fh:
        n_tab = int(fh.readline())
        rows  = [list(map(float, fh.readline().split())) for _ in range(n_tab)]
    data = np.array(rows)
    # Standard format: e [g/cm³], p [dyn/cm²], h [cm²/s²], n0 [cm⁻³]
    # Some files (e.g. eosNV) have an extra 5th column — strip it silently.
    if data.shape[1] < 4:
        raise ValueError(f"EOS file has only {data.shape[1]} columns; expected ≥ 4")
    e_cgs, p_cgs, h_cgs, n0_cgs = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    log_e  = np.log10(np.maximum(e_cgs  * C * C * KSCALE, 1e-300))
    log_p  = np.log10(np.maximum(p_cgs  * KSCALE,         1e-300))
    log_h  = np.log10(np.maximum(h_cgs  / (C * C),        1e-300))
    log_n0 = np.log10(np.maximum(n0_cgs,                  1e-300))

    # Enforce strict monotonicity: remove rows where log_e is not increasing
    # (some EOS files such as eosFP have duplicate or reversed density rows)
    mono = np.concatenate(([True], np.diff(log_e) > 0))
    if not np.all(mono):
        log_e  = log_e[mono];  log_p  = log_p[mono]
        log_h  = log_h[mono];  log_n0 = log_n0[mono]
        n_tab  = int(mono.sum())

    return log_e, log_p, log_h, log_n0, n_tab


def _lagrange4(x_tab, y_tab, xval):
    """4-point Lagrange interpolation; x_tab must be monotone increasing."""
    n = len(x_tab)
    i = int(np.searchsorted(x_tab, xval)) - 1
    i = max(1, min(i, n - 3))      # keep 4-point stencil in range
    xi = x_tab[i - 1:i + 3]
    yi = y_tab[i - 1:i + 3]
    result = 0.0
    for j in range(4):
        num = yi[j]
        for k in range(4):
            if k != j:
                num *= (xval - xi[k]) / (xi[j] - xi[k])
        result += num
    return result


# Tabulated EOS look-up functions
def p_at_e_tab(e, log_e_tab, log_p_tab):
    if e <= 0.0:             return 0.0
    lge = np.log10(e)
    if lge < log_e_tab[0]:  return 0.0
    lge = min(lge, log_e_tab[-1])
    return 10.0 ** _lagrange4(log_e_tab, log_p_tab, lge)

def e_at_p_tab(p, log_e_tab, log_p_tab):
    if p <= 0.0:             return 0.0
    lgp = np.log10(p)
    if lgp < log_p_tab[0]:  return 0.0
    lgp = min(lgp, log_p_tab[-1])
    return 10.0 ** _lagrange4(log_p_tab, log_e_tab, lgp)

def h_at_p_tab(p, log_p_tab, log_h_tab):
    if p <= 0.0:             return 0.0
    lgp = np.log10(p)
    if lgp < log_p_tab[0]:  return 0.0
    lgp = min(lgp, log_p_tab[-1])
    return 10.0 ** _lagrange4(log_p_tab, log_h_tab, lgp)

def p_at_h_tab(h, log_h_tab, log_p_tab):
    if h <= 0.0:             return 0.0
    lgh = np.log10(h)
    if lgh < log_h_tab[0]:  return 0.0
    lgh = min(lgh, log_h_tab[-1])
    return 10.0 ** _lagrange4(log_h_tab, log_p_tab, lgh)

def n0_at_e_tab(e, log_e_tab, log_n0_tab):
    if e <= 0.0:             return 0.0
    lge = np.log10(e)
    if lge < log_e_tab[0]:  return 0.0
    lge = min(lge, log_e_tab[-1])
    return 10.0 ** _lagrange4(log_e_tab, log_n0_tab, lge)


# ---- Polytropic EOS  p = rho0^Gamma_P --------------------------------

def _poly_rho0_from_e(e, Gamma_P):
    """Newton-Raphson inversion of e = rho0 + rho0^Gamma_P/(Gamma_P-1)."""
    Gm1  = Gamma_P - 1.0
    rho0 = max(e * Gm1 / Gamma_P, 1e-30)
    for _ in range(50):
        p    = rho0 ** Gamma_P
        f    = rho0 + p / Gm1 - e
        df   = 1.0 + Gamma_P * rho0 ** (Gamma_P - 1.0) / Gm1
        drho = f / df
        rho0 = max(rho0 - drho, 1e-30)
        if abs(drho) < 1e-12 * rho0:
            break
    return rho0

def make_center_poly(e_center, Gamma_P):
    rho0     = _poly_rho0_from_e(e_center, Gamma_P)
    p_c      = rho0 ** Gamma_P
    e_p      = rho0 + p_c / (Gamma_P - 1.0)
    h_c      = np.log((e_p + p_c) / rho0)
    return p_c, h_c

def p_at_e_poly(e, Gamma_P):
    if e <= 0.0: return 0.0
    rho0 = _poly_rho0_from_e(e, Gamma_P)
    return rho0 ** Gamma_P

def e_at_p_poly(p, Gamma_P):
    if p <= 0.0: return 0.0
    rho0 = p ** (1.0 / Gamma_P)
    return rho0 + p / (Gamma_P - 1.0)

def h_at_p_poly(p, Gamma_P):
    if p <= 0.0: return 0.0
    rho0 = p ** (1.0 / Gamma_P)
    e    = rho0 + p / (Gamma_P - 1.0)
    return np.log((e + p) / rho0)

def p_at_h_poly(h, Gamma_P):
    if h <= 0.0: return 0.0
    Gm1  = Gamma_P - 1.0
    rho0 = max((h * Gm1 / Gamma_P) ** (1.0 / (Gamma_P - 1.0)), 1e-30)
    for _ in range(50):
        p    = rho0 ** Gamma_P
        e    = rho0 + p / Gm1
        hc   = np.log((e + p) / rho0)
        dh   = ((1.0 + Gamma_P * rho0 ** (Gamma_P - 1.0) / Gm1
                 + Gamma_P * rho0 ** (Gamma_P - 1.0)) / (e + p) - 1.0 / rho0)
        drho = (h - hc) / max(dh, 1e-30)
        rho0 = max(rho0 + drho, 1e-30)
        if abs(drho) < 1e-12 * rho0:
            break
    return rho0 ** Gamma_P


# ---- EOS dispatch (pick tabulated vs polytropic) ----------------------

def _p_at_e(e, eos):
    if eos['type'] == 'tab':
        return p_at_e_tab(e, eos['log_e'], eos['log_p'])
    return p_at_e_poly(e, eos['Gamma_P'])

def _e_at_p(p, eos):
    if eos['type'] == 'tab':
        return e_at_p_tab(p, eos['log_e'], eos['log_p'])
    return e_at_p_poly(p, eos['Gamma_P'])

def _h_at_p(p, eos):
    if eos['type'] == 'tab':
        return h_at_p_tab(p, eos['log_p'], eos['log_h'])
    return h_at_p_poly(p, eos['Gamma_P'])

def _p_at_h(h, eos):
    if eos['type'] == 'tab':
        return p_at_h_tab(h, eos['log_h'], eos['log_p'])
    return p_at_h_poly(h, eos['Gamma_P'])

def _n0_at_e(e, eos):
    if eos['type'] == 'tab':
        return n0_at_e_tab(e, eos['log_e'], eos['log_n0'])
    if e <= 0.0: return 0.0
    p    = p_at_e_poly(e, eos['Gamma_P'])
    rho0 = p ** (1.0 / eos['Gamma_P'])
    return rho0

def make_center(e_center, eos):
    if eos['type'] == 'tab':
        p_c = p_at_e_tab(e_center, eos['log_e'], eos['log_p'])
        h_c = h_at_p_tab(p_c, eos['log_p'], eos['log_h'])
    else:
        p_c, h_c = make_center_poly(e_center, eos['Gamma_P'])
    return p_c, h_c


# ============================================================================
# Legendre polynomial utilities
# ============================================================================

def _legendre_P(n, x):
    """Legendre polynomial P_n(x) by recurrence."""
    if n == 0: return np.ones_like(x, dtype=float)
    if n == 1: return np.array(x, dtype=float)
    p2 = np.ones_like(x, dtype=float)
    p1 = np.array(x, dtype=float)
    for k in range(2, n + 1):
        p  = ((2 * k - 1) * x * p1 - (k - 1) * p2) / k
        p2 = p1
        p1 = p
    return p1


def precompute_legendre(mu):
    """
    Build Legendre arrays on the mu grid (size MDIV).

    Returns
    -------
    P_2n          : (MDIV, LMAX+1)  P_{2n}(mu),                n = 0..LMAX
    P1_2n1        : (MDIV, LMAX+1)  P^1_{2n-1}(mu)*sin(theta), n = 1..LMAX
    sin_2n1_theta : (MDIV, LMAX+1)  sin((2n-1)*theta)/sin(theta), n = 1..LMAX
    """
    NMDIV   = len(mu)
    theta   = np.arccos(mu)
    sin_th  = np.sin(theta)

    P_2n          = np.zeros((NMDIV, LMAX + 1))
    P1_2n1        = np.zeros((NMDIV, LMAX + 1))
    sin_2n1_theta = np.zeros((NMDIV, LMAX + 1))

    for n in range(LMAX + 1):
        P_2n[:, n] = _legendre_P(2 * n, mu)

    for n in range(1, LMAX + 1):
        l    = 2 * n - 1
        Pl   = _legendre_P(l,     mu)
        Plm1 = _legendre_P(l - 1, mu)
        sin2 = 1.0 - mu * mu
        # (1-x²) P_l'(x) = -l x P_l(x) + l P_{l-1}(x)
        # P_l^1(x) = sqrt(1-x²) P_l'(x)
        with np.errstate(invalid='ignore', divide='ignore'):
            Pl_prime = np.where(sin2 > 1e-20,
                                (-l * mu * Pl + l * Plm1) / sin2, 0.0)
        P1_2n1[:, n] = np.sqrt(np.maximum(sin2, 0.0)) * Pl_prime

        # sin((2n-1)θ)/sin(θ) — L'Hôpital limit (2n-1) at θ=0
        with np.errstate(invalid='ignore', divide='ignore'):
            sin_2n1_theta[:, n] = np.where(
                sin_th > 1e-10,
                np.sin(l * theta) / sin_th,
                float(l))

    return P_2n, P1_2n1, sin_2n1_theta


# ============================================================================
# Finite-difference derivative operators  (2nd-order central differences)
# ============================================================================

def grad_s(f):
    return np.gradient(f, DS, axis=0, edge_order=1)

def grad_m(f):
    return np.gradient(f, DM, axis=1, edge_order=1)


# ============================================================================
# TOV solver  (spherically symmetric, isotropic coordinates)
# ============================================================================

def _tov_rhs(r_is, state, e_center, p_center, p_surface, eos):
    """
    RHS of the three TOV ODEs in isotropic radial coordinate r_is:
      state = [r_schw, m_grav, p]
    Returns [dr/dr_is, dm/dr_is, dp/dr_is].
    """
    r, m, p = state

    if r_is < RMIN:
        # Centre: leading-order Taylor terms
        dmdr = 4.0 * PI * e_center * max(r, 0.0) ** 2
        dpdr = (-4.0 * PI * (e_center + p_center) *
                (e_center + 3.0 * p_center) * r / 3.0)
        return [1.0, dmdr, dpdr]

    e_d   = _e_at_p(p, eos) if p > p_surface else 0.0
    denom = max(1.0 - 2.0 * m / max(r, 1e-30), 1e-30)
    sq    = np.sqrt(denom)

    drdr  = (r / r_is) * sq
    dmdr  = 4.0 * PI * e_d * r ** 3 * sq / r_is
    dpdr  = (-(e_d + p) * (m + 4.0 * PI * r ** 3 * p) /
             max(r * r_is * sq, 1e-30))
    return [drdr, dmdr, dpdr]


def _rk4_step(r_is, state, h, e_center, p_center, p_surface, eos):
    k1 = np.array(_tov_rhs(r_is,       state,             e_center, p_center, p_surface, eos))
    k2 = np.array(_tov_rhs(r_is+h/2,   state + h/2 * k1, e_center, p_center, p_surface, eos))
    k3 = np.array(_tov_rhs(r_is+h/2,   state + h/2 * k2, e_center, p_center, p_surface, eos))
    k4 = np.array(_tov_rhs(r_is+h,     state + h   * k3, e_center, p_center, p_surface, eos))
    return state + h / 6.0 * (k1 + 2*k2 + 2*k3 + k4)


def solve_tov(e_center, p_center, p_surface, eos):
    """
    Integrate TOV outward until p < p_surface.

    Returns arrays on RDIV+1 evenly-spaced r_is points together with
    r_is_final (isotropic stellar radius) and m_final (total mass).
    """
    # Rough stellar radius estimate (pressure scale height)
    r_is_est = max(np.sqrt(3.0 * p_center /
                           (2.0 * PI * max(e_center + p_center, 1e-30) *
                            max(e_center, 1e-30))), 1e-6)

    # Pass 1 — coarse scan to bracket the surface
    h1    = r_is_est / 100.0
    r_is  = 0.0
    state = np.array([1e-3 * RMIN, 0.0, p_center])
    r_is_final = r_is_est
    for _ in range(2000):
        state  = _rk4_step(r_is, state, h1, e_center, p_center, p_surface, eos)
        r_is  += h1
        if state[2] <= p_surface:
            r_is_final = r_is
            break

    # Pass 2 — refined scan
    h2    = r_is_final / 10000.0
    r_is  = 0.0
    state = np.array([1e-3 * RMIN, 0.0, p_center])
    for _ in range(30000):
        state  = _rk4_step(r_is, state, h2, e_center, p_center, p_surface, eos)
        r_is  += h2
        if state[2] <= p_surface:
            r_is_final = r_is
            break
    m_final = state[1]

    # Pass 3 — store on RDIV uniform grid
    h3       = r_is_final / RDIV
    r_is_arr = np.zeros(RDIV + 1)
    r_arr    = np.zeros(RDIV + 1)
    m_arr    = np.zeros(RDIV + 1)
    p_arr    = np.zeros(RDIV + 1)

    r_is  = 0.0
    state = np.array([1e-3 * RMIN, 0.0, p_center])
    r_is_arr[0] = 0.0
    r_arr[0]    = state[0]
    m_arr[0]    = 0.0
    p_arr[0]    = p_center

    for k in range(1, RDIV + 1):
        state       = _rk4_step(r_is, state, h3, e_center, p_center, p_surface, eos)
        r_is       += h3
        r_is_arr[k] = r_is
        r_arr[k]    = state[0]
        m_arr[k]    = state[1]
        p_arr[k]    = max(state[2], 0.0)

    # --- Metric functions on the interior ODE grid -------------------------
    # lambda_arr[k] = ln(R_s / r_is_int) — NEGATIVE inside the star because
    # the interior isotropic radius r_is_int grows faster than R_s.  This
    # is the convention used throughout the KEH/RNS formalism; sphere() will
    # compute the correct physical r_e from these values.
    lambda_arr = np.where(
        (r_arr > 0) & (r_is_arr > 0),
        np.log(r_arr / np.maximum(r_is_arr, 1e-30)),
        0.0)
    lambda_arr[0] = lambda_arr[1]

    # nu from shooting inward from the surface, where the lapse matches the
    # Schwarzschild exterior evaluated at r_is_final (the interior ODE endpoint).
    nu_surface = np.log(max((1.0 - m_final / (2.0 * r_is_final)) /
                             (1.0 + m_final / (2.0 * r_is_final)), 1e-30))
    nu_arr = np.zeros(RDIV + 1)
    nu_arr[-1] = nu_surface
    for k in range(RDIV - 1, -1, -1):
        r_k  = r_arr[k + 1]
        ri_k = r_is_arr[k + 1]
        m_k  = m_arr[k + 1]
        p_k  = p_arr[k + 1]
        denom = max(1.0 - 2.0 * m_k / max(r_k, 1e-30), 1e-30)
        sq    = np.sqrt(denom)
        if r_k * ri_k * sq > 1e-30:
            dnu = (m_k + 4.0 * PI * r_k ** 3 * p_k) / (r_k * ri_k * sq)
        else:
            dnu = 0.0
        nu_arr[k] = nu_arr[k + 1] - dnu * h3   # ν decreases going inward

    # Note: we do NOT override lambda_arr at the surface point k=RDIV with the
    # exterior formula.  The exterior formula gives a positive lambda while the
    # interior gives a negative value; mixing them in interpolation creates a
    # large derivative spike at the surface.  sphere() handles the exterior
    # Schwarzschild directly when r_is_s >= r_is_final.

    return (r_is_arr, r_arr, m_arr, p_arr,
            nu_arr, lambda_arr, r_is_final, m_final)


# ============================================================================
# sphere() — metric potentials for a non-rotating star
# ============================================================================

def sphere(s_gp, mu_arr, e_center, p_center, h_center,
           p_surface, e_surface, eos):
    """
    Initialise metric potentials from the spherically symmetric TOV solution.

    For a spherical star:
      gama(s, mu) = nu(r_is) + lambda(r_is)
      rho(s,  mu) = nu(r_is) − lambda(r_is)
      alpha(s,mu) = lambda(r_is)               = (gama − rho)/2
      omega(s,mu) = 0

    The equatorial isotropic radius r_e satisfies s=0.5 → r_is = r_e,
    so r_e = r_is_final from the TOV integration.
    """
    (r_is_arr, r_arr, m_arr, p_arr,
     nu_arr, lambda_arr, r_is_final, m_final) = solve_tov(
        e_center, p_center, p_surface, eos)

    # r_e = r_is_final so that s=0.5 maps exactly to the stellar surface.
    # Potentials are stored in code units (divided by r_e²).
    _ = lambda_arr[-1]   # interior lambda kept for reference (not used here)
    r_e = r_is_final
    re2 = r_e * r_e

    rho   = np.zeros((SDIV, MDIV))
    gama  = np.zeros((SDIV, MDIV))
    alpha = np.zeros((SDIV, MDIV))
    omega = np.zeros((SDIV, MDIV))

    for j in range(SDIV):
        sv = s_gp[j]
        # Use r_is_final as the grid scale so that s=0.5 maps to r_is_final
        # (the stellar surface).  r_e is used only for the code-unit division.
        r_is_s = r_is_final * sv / (1.0 - sv) if sv < SMAX else 1e20

        if r_is_s <= 0.0:
            nu_s, lam_s = nu_arr[0], lambda_arr[0]
        elif r_is_s >= r_is_final:
            nu_s  = np.log(max((1.0 - m_final / (2.0 * r_is_s)) /
                               (1.0 + m_final / (2.0 * r_is_s)), 1e-30))
            lam_s = 2.0 * np.log(1.0 + m_final / (2.0 * r_is_s))
        else:
            idx  = max(0, min(int(np.searchsorted(r_is_arr, r_is_s)) - 1,
                              RDIV - 1))
            dr   = max(r_is_arr[idx + 1] - r_is_arr[idx], 1e-30)
            t    = np.clip((r_is_s - r_is_arr[idx]) / dr, 0.0, 1.0)
            nu_s  = nu_arr[idx]     + t * (nu_arr[idx + 1]     - nu_arr[idx])
            lam_s = lambda_arr[idx] + t * (lambda_arr[idx + 1] - lambda_arr[idx])

        # Store in code units (divided by r_e²) for spin() self-consistency
        gama[j, :]  = (nu_s + lam_s) / re2
        rho[j, :]   = (nu_s - lam_s) / re2
        alpha[j, :] = lam_s           / re2   # = (gama − rho)/2
        omega[j, :] = 0.0

    return rho, gama, alpha, omega, r_e


# ============================================================================
# Precompute radial Green's function kernels
# ============================================================================

def compute_kernels(s_gp):
    """
    Precompute f_rho[j, n, k] and f_gama[j, n, k]  (j, k: s-indices; n: mode).

    The Green's function for the elliptic equations uses the kernel
    f2n[n, i] = ((1−s[i])/s[i])^(2n):

      k < j  (source inside field point):
        f_rho[j,n,k]  = f2n[n,j] * (1-sj) / (sj * f2n[n,k] * (1-sk)²)
        f_gama[j,n,k] = f2n[n,j] / (f2n[n,k] * sk * (1-sk))

      k >= j (source at or beyond field point):
        f_rho[j,n,k]  = f2n[n,k] / (f2n[n,j] * sk * (1-sk))
        f_gama[j,n,k] = f2n[n,k] * (1-sj)² * sk / (sj² * f2n[n,j] * (1-sk)³)
    """
    f2n = np.zeros((LMAX + 1, SDIV))
    for i in range(1, SDIV):
        ratio = (1.0 - s_gp[i]) / s_gp[i]
        for n in range(LMAX + 1):
            f2n[n, i] = ratio ** (2 * n)
    f2n[:, 0] = f2n[:, 1]      # s=0: copy neighbour to avoid 0/0

    f_rho  = np.zeros((SDIV, LMAX + 1, SDIV))
    f_gama = np.zeros((SDIV, LMAX + 1, SDIV))

    for j in range(1, SDIV):
        sj  = s_gp[j]
        sj1 = 1.0 - sj
        for n in range(LMAX + 1):
            f2n_j = f2n[n, j]
            for k in range(1, SDIV):
                sk  = s_gp[k]
                sk1 = 1.0 - sk
                f2n_k = f2n[n, k]
                if k < j:
                    denom_r = f2n_k * sk1 * sk1
                    denom_g = f2n_k * sk  * sk1
                    f_rho[j, n, k]  = (f2n_j * sj1 / (sj  * denom_r)
                                       if sj > 0 and denom_r > 0 else 0.0)
                    f_gama[j, n, k] = (f2n_j / denom_g
                                       if denom_g > 0 else 0.0)
                else:
                    denom_r = f2n_j * sk  * sk1
                    denom_g = sj * sj * f2n_j * sk1 * sk1 * sk1
                    f_rho[j, n, k]  = (f2n_k / denom_r
                                       if denom_r > 0 else 0.0)
                    f_gama[j, n, k] = (f2n_k * sj1 * sj1 * sk / denom_g
                                       if sj > 0 and denom_g > 0 else 0.0)

    # j=0 (s=0) is the coordinate centre: no contributions
    f_rho[0, :, :]  = 0.0
    f_gama[0, :, :] = 0.0

    return f_rho, f_gama


# ============================================================================
# spin() — self-consistent field iteration for a rotating star
# ============================================================================

def spin(s_gp, mu_arr, eos,
         h_center, enthalpy_min,
         rho, gama, alpha, omega,
         energy, pressure, enthalpy, velocity_sq,
         r_ratio, r_e_in,
         accuracy=1e-5, cf=1.0,
         f_rho_kern=None, f_gama_kern=None,
         P_2n=None, P1_2n1=None, sin_2n1_theta=None,
         verbose=False):
    """
    Iterate the self-consistent field equations for a star with polar-to-
    equatorial axis ratio r_ratio (r_ratio=1 → non-rotating sphere).

    All field arrays (rho, gama, alpha, omega, energy, pressure, enthalpy,
    velocity_sq) are updated in-place.

    Returns (r_e, Omega).
    """
    if P_2n is None:
        P_2n, P1_2n1, sin_2n1_theta = precompute_legendre(mu_arr)
    if f_rho_kern is None:
        f_rho_kern, f_gama_kern = compute_kernels(s_gp)

    theta  = np.arccos(mu_arr)
    sin_th = np.sin(theta)

    # Grid index of polar surface:  s_pole = r_ratio/(1+r_ratio)
    s_pole = r_ratio / (1.0 + r_ratio)
    j_pole = max(1, min(int(round(s_pole / DS)), SDIV - 1))

    # Grid index of equatorial surface: s = 0.5
    j_eq   = max(1, min(int(round(0.5 / DS)), SDIV - 1))

    # For the non-rotating case the sphere() solution is already exact;
    # just compute r_e and Omega from the current potentials and return.
    if r_ratio >= 1.0:
        gprp_pole   = gama[j_pole, MDIV - 1] + rho[j_pole, MDIV - 1]
        gprp_center = gama[0,      MDIV - 1] + rho[0,      MDIV - 1]
        diff_gp = gprp_pole - gprp_center
        r_e = np.sqrt(abs(2.0 * h_center / diff_gp)) if abs(diff_gp) > 1e-30 else r_e_in
        r_e = max(r_e, 1e-10)
        # Fill matter fields from Bernoulli
        re2 = r_e**2
        for j in range(SDIV):
            sv = s_gp[j]
            for m in range(MDIV):
                h_val = 0.5 * re2 * (gprp_pole - gama[j, m] - rho[j, m])
                h_val = max(h_val, 0.0)
                enthalpy[j, m] = h_val
                if h_val > enthalpy_min:
                    p_v = _p_at_h(h_val, eos)
                    e_v = _e_at_p(p_v, eos)
                else:
                    p_v = 0.0; e_v = 0.0
                pressure[j, m] = p_v
                energy[j, m]   = e_v
                velocity_sq[j, m] = 0.0
        omega[:] = 0.0
        return r_e, 0.0

    r_e = r_e_in
    MAX_ITER = 300

    for iteration in range(MAX_ITER):
        r_e_old = r_e

        # --- New r_e from axis enthalpy balance -----------------------
        # h_center = 0.5 * r_e² * [(γ+ρ)_pole − (γ+ρ)_center]
        gprp_pole   = gama[j_pole, MDIV - 1] + rho[j_pole, MDIV - 1]
        gprp_center = gama[0,      MDIV - 1] + rho[0,      MDIV - 1]
        diff_gp = gprp_pole - gprp_center
        if abs(diff_gp) > 1e-30:
            r_e = np.sqrt(abs(2.0 * h_center / diff_gp))
        r_e = max(r_e, 1e-10)

        # --- Rescale code-unit potentials to new r_e ------------------
        # Physical potential = code_potential * r_e²; when r_e changes,
        # rescale stored arrays so physical values are preserved.
        if abs(r_e - r_e_old) > 1e-15 * r_e:
            scale2 = (r_e_old / r_e) ** 2
            scale1 = (r_e_old / r_e)
            rho[:]   *= scale2
            gama[:]  *= scale2
            alpha[:] *= scale2
            omega[:] *= scale1    # ω_code = ω_phys * r_e

        # --- Angular velocity from equatorial surface condition -------
        # Re-read gprp_pole after potential rescaling
        gprp_pole   = gama[j_pole, MDIV - 1] + rho[j_pole, MDIV - 1]
        gprp_eq     = gama[j_eq,   0]         + rho[j_eq,  0]
        omega_eq    = omega[j_eq, 0] / r_e     # physical frame-dragging
        rho_eq      = rho[j_eq, 0]
        arg_v       = r_e ** 2 * (gprp_pole - gprp_eq)
        v_eq_sq  = max(0.0, 1.0 - np.exp(min(arg_v, 0.0)))

        if r_ratio < 1.0:
            Omega = omega_eq + np.exp(r_e ** 2 * rho_eq) * np.sqrt(v_eq_sq)
        else:
            Omega = 0.0

        # --- Matter fields on the whole grid --------------------------
        re2 = r_e ** 2
        for j in range(SDIV):
            sv  = s_gp[j]
            if sv < SMAX:
                r_is_j = r_e * sv / (1.0 - sv)
            else:
                r_is_j = 1e20
            for m in range(MDIV):
                g_jm = gama[j, m]
                r_jm = rho[j, m]
                o_jm = omega[j, m]

                # Fluid speed (ZAMO frame)
                # v = (Ω − ω_phys) · r_is · sinθ · exp(−ν)
                # ν = 0.5·(γ+ρ)·r_e²
                v2 = ((Omega - o_jm / r_e) * r_is_j * sin_th[m] *
                      np.exp(-0.5 * (g_jm + r_jm) * re2)) ** 2
                v2 = min(v2, 0.9999)
                velocity_sq[j, m] = v2

                # Enthalpy: h = ν_pole − ν(j,m) + ln(γ_Lorentz)
                # h_c is encoded in r_e via the pole self-consistency condition
                h_val = (0.5 * re2 * (gprp_pole - g_jm - r_jm)
                         - 0.5 * np.log(max(1.0 - v2, 1e-30)))
                enthalpy[j, m] = max(h_val, 0.0)

                if h_val > enthalpy_min:
                    p_v = _p_at_h(h_val, eos)
                    e_v = _e_at_p(p_v, eos)
                else:
                    p_v = 0.0
                    e_v = 0.0
                pressure[j, m] = p_v
                energy[j, m]   = e_v

        # --- Source terms ---------------------------------------------
        dg_s = grad_s(gama)
        dg_m = grad_m(gama)
        dr_s = grad_s(rho)
        dr_m = grad_m(rho)
        do_s = grad_s(omega)
        do_m = grad_m(omega)

        S_rho  = np.zeros((SDIV, MDIV))
        S_gama = np.zeros((SDIV, MDIV))
        S_omega= np.zeros((SDIV, MDIV))

        for j in range(1, SDIV):
            sv  = s_gp[j]
            s1  = sv * (1.0 - sv)
            s2  = (sv / (1.0 - sv)) ** 2 if sv < SMAX else 0.0
            for m in range(MDIV):
                mu_m = mu_arr[m]
                m1   = 1.0 - mu_m ** 2

                g_jm = gama[j, m]
                r_jm = rho[j, m]
                o_jm = omega[j, m]
                e_jm = energy[j, m]
                p_jm = pressure[j, m]
                v2   = velocity_sq[j, m]
                exp_g2 = np.exp(np.clip(0.5 * g_jm, -50, 50))
                al     = alpha[j, m]
                ea     = 16.0 * PI * np.exp(np.clip(2.0 * al * re2, -100, 100)) * re2

                omv2 = max(1.0 - v2, 1e-10)
                dgs  = dg_s[j, m];  dgm = dg_m[j, m]
                drs  = dr_s[j, m];  drm = dr_m[j, m]
                dos  = do_s[j, m];  dom = do_m[j, m]

                # Source for rho
                S_rho[j, m] = exp_g2 * (
                    0.5 * ea * (e_jm + p_jm) * s2 * (1.0 + v2) / omv2
                    + s2 * m1 * np.exp(-2.0 * r_jm * re2) *
                      (s1 ** 2 * dos ** 2 + m1 * dom ** 2)
                    + s1 * dgs - mu_m * dgm
                    + 0.5 * r_jm * (ea * p_jm * s2
                                    - s1 * dgs * (0.5 * s1 * dgs + 1.0)
                                    - dgm * (0.5 * m1 * dgm - mu_m))
                )

                # Source for gama
                S_gama[j, m] = exp_g2 * (
                    ea * p_jm * s2
                    + 0.5 * g_jm * (ea * p_jm * s2
                                    - 0.5 * (s1 * dgs) ** 2
                                    - 0.5 * m1 * dgm ** 2)
                )

                # Source for omega
                S_omega[j, m] = exp_g2 * np.exp(-r_jm * re2) * (
                    -ea * (Omega - o_jm / r_e) * (e_jm + p_jm) * s2 / omv2
                    + o_jm * (
                        -0.5 * ea * ((1.0 + v2) * e_jm + 2.0 * v2 * p_jm) /
                        omv2 * s2
                        - s1 * (2.0 * drs + 0.5 * dgs)
                        + mu_m * (2.0 * drm + 0.5 * dgm)
                        + 0.25 * s1 ** 2 * (4.0 * drs ** 2 - dgs ** 2)
                        + 0.25 * m1 * (4.0 * drm ** 2 - dgm ** 2)
                        - m1 * np.exp(-2.0 * r_jm * re2) *
                          (sv ** 4 * dos ** 2 + s2 * m1 * dom ** 2)
                    )
                )

        # --- D1: angular Legendre integrals of source terms ----------
        # Trapezoid integration over mu with DM spacing
        D1_rho   = np.zeros((LMAX + 1, SDIV))
        D1_gama  = np.zeros((LMAX + 1, SDIV))
        D1_omega = np.zeros((LMAX + 1, SDIV))

        for k in range(SDIV):
            # n=0 monopole (P_0=1, gama and omega have no monopole)
            D1_rho[0, k] = DM * np.trapezoid(S_rho[k, :] * P_2n[:, 0], dx=1.0)
            for n in range(1, LMAX + 1):
                D1_rho[n, k]   = DM * np.trapezoid(S_rho[k, :]   * P_2n[:, n],           dx=1.0)
                D1_gama[n, k]  = DM * np.trapezoid(S_gama[k, :]  * sin_2n1_theta[:, n],   dx=1.0)
                D1_omega[n, k] = DM * np.trapezoid(S_omega[k, :]  * sin_th * P1_2n1[:, n], dx=1.0)

        # --- D2: solve 1D radial BVP for each Legendre mode ----------
        # ODE in physical r_is coords: u'' + (2/r)u' - l(l+1)/r^2 u = f(r)
        # Interior: f = D1_n(r_is[j])  (source from matter + metric coupling)
        # Exterior: f = 0  (vacuum — Laplace equation, decays to 0)
        # BCs: u'(0)=0 (centre), u'(r*)/r* = -(l+1)/r* u(r*) (exterior match)

        def _solve_radial_ode(D1_k, l, r_e, s_gp, j_surf):
            """Solve on j=0..SDIV-1; use vacuum-exterior Robin BC at j_surf."""
            N    = len(s_gp)
            r_is = np.where(s_gp > 0, r_e * s_gp / (1.0 - s_gp), 0.0)
            dr   = np.maximum(np.diff(r_is), 1e-30)

            # Source: only inside the star (j <= j_surf)
            rhs = np.zeros(N)
            for j in range(1, min(j_surf, N - 1)):
                rhs[j] = D1_k[j]  # code-unit source on r_is grid

            ab = np.zeros((3, N))

            # Interior FD rows
            for j in range(1, N - 1):
                r_j = r_is[j]
                if r_j < 1e-10:
                    ab[1, j] = 1.0; rhs[j] = 0.0; continue
                h_p = dr[j]; h_m = dr[j - 1]
                c_p =  2.0 / (h_p * (h_p + h_m))
                c_m =  2.0 / (h_m * (h_p + h_m))
                c_c = -(c_p + c_m)
                d_p =  1.0 / (h_p + h_m)
                d_m = -1.0 / (h_p + h_m)
                ab[0, j + 1] = c_p + (2.0 / r_j) * d_p
                ab[1, j]     = c_c - l * (l + 1) / r_j**2
                ab[2, j - 1] = c_m + (2.0 / r_j) * d_m

            # Centre BC: regularity (u'=0 → ghost u[-1]=u[1] for l=0, u=0 for l>0)
            if l == 0:
                ab[1, 0] = -1.0; ab[0, 1] = 1.0; rhs[0] = 0.0
            else:
                ab[1, 0] = 1.0; rhs[0] = 0.0

            # Outer BC: u=0 at s=SMAX (u decays in vacuum)
            ab[1, N - 1] = 1.0; rhs[N - 1] = 0.0

            try:
                u = solve_banded((1, 1), ab, rhs)
            except Exception:
                u = np.zeros(N)
            return np.clip(u, -5.0, 5.0)

        D2_rho  = np.zeros((SDIV, LMAX + 1))
        D2_gama = np.zeros((SDIV, LMAX + 1))
        D2_omega= np.zeros((SDIV, LMAX + 1))

        D2_rho[:, 0] = _solve_radial_ode(D1_rho[0, :], 0, r_e, s_gp, j_pole)
        for n in range(1, LMAX + 1):
            l = 2 * n
            D2_rho[:, n]  = _solve_radial_ode(D1_rho[n, :],   l,   r_e, s_gp, j_pole)
            D2_gama[:, n] = _solve_radial_ode(D1_gama[n, :],  l,   r_e, s_gp, j_pole)
            D2_omega[:, n]= _solve_radial_ode(D1_omega[n, :], l-1, r_e, s_gp, j_pole)

        # --- Assemble updated potentials from ODE solutions ----------
        # The ODE gives D2[j,n] = the code-unit Legendre coefficient directly.
        # Reconstruct: pot(j,m) = sum_n D2[j,n] * P_{2n}(mu[m])
        new_rho   = np.einsum('jn,mn->jm', D2_rho,  P_2n)
        new_gama  = np.einsum('jn,mn->jm', D2_gama, P_2n)
        new_omega = np.einsum('jn,mn->jm', D2_omega, P_2n)

        rho[:]   = new_rho
        gama[:]  = new_gama
        omega[:] = new_omega

        # Non-rotating case: enforce zero rotation
        if r_ratio >= 1.0:
            omega[:] = 0.0

        # --- alpha: constraint integration along mu -------------------
        # dα/dμ from the integrability condition (see spin() description)
        dg_s2 = grad_s(gama)
        dg_m2 = grad_m(gama)
        dr_s2 = grad_s(rho)
        dr_m2 = grad_m(rho)
        dg_ss = grad_s(dg_s2)
        dg_mm = grad_m(dg_m2)
        dg_sm = grad_s(dg_m2)

        da_dm = np.zeros((SDIV, MDIV))
        for j in range(1, SDIV):
            sv  = s_gp[j]
            s1  = sv * (1.0 - sv)
            for m in range(MDIV):
                mu_m = mu_arr[m]
                m1   = 1.0 - mu_m ** 2

                dgs  = dg_s2[j, m];  dgm = dg_m2[j, m]
                drs  = dr_s2[j, m];  drm = dr_m2[j, m]
                dos  = do_s[j, m];   dom = do_m[j, m]
                dgss = dg_ss[j, m];  dgmm = dg_mm[j, m]
                dgsm = dg_sm[j, m]

                A      = 1.0 + s1 * dgs
                B      = -mu_m + m1 * dgm
                denom2 = max(m1 * A * A + B * B, 1e-30)

                d_gss = s1 * dgss + (1.0 - 2.0 * sv) * dgs
                d_gmm = m1 * dgmm - 2.0 * mu_m * dgm

                t1 = (2.0 * sv ** 2 * (sv / max(1.0 - sv, 1e-15)) *
                      m1 * dos * dom * A
                      - (sv ** 4 * dos ** 2 -
                         (sv / max(1.0 - sv, 1e-15)) ** 2 * m1 * dom ** 2) * B)
                t3 = s1 * d_gss + (s1 * dgs) ** 2
                t4 = dgm * B
                t5 = (s1 ** 2 * (drs + dgs) ** 2 - m1 * (drm + dgm) ** 2) * B
                t6 = s1 * m1 * (0.5 * (drs + dgs) * (drm + dgm) + dgsm + dgs * dgm) * A
                t7 = s1 * mu_m * dgs * A
                t8 = m1 * np.exp(np.clip(-2.0 * rho[j, m] * re2, -300, 300))

                da_dm[j, m] = (-0.5 * (drm + dgm)
                               - (0.5 * (t3 - d_gmm - t4) * B
                                  + 0.25 * t5 - t6 + t7
                                  + 0.25 * t8 * t1) / denom2)

        # Integrate dα/dμ in mu; apply axis boundary condition
        alpha[0, :] = 0.0
        for j in range(1, SDIV):
            alpha[j, 0] = 0.0
            for m in range(1, MDIV):
                alpha[j, m] = (alpha[j, m - 1] +
                               0.5 * DM * (da_dm[j, m] + da_dm[j, m - 1]))
            # Enforce α[j, pole] = 0.5*(γ − ρ) at the rotation axis
            shift = (-alpha[j, MDIV - 1] +
                     0.5 * (gama[j, MDIV - 1] - rho[j, MDIV - 1]))
            alpha[j, :] += shift

        # --- Divergence guard -----------------------------------------
        if (abs(omega[1, 0]) > 100.0 or
                abs(rho[1, 0]) > 100.0 or
                abs(gama[1, 0]) > 300.0):
            if verbose:
                print(f"  [spin] divergence at iteration {iteration}; stopping.")
            break

        # --- Convergence check ----------------------------------------
        dif = abs(r_e - r_e_old) / max(r_e, 1e-30)
        if verbose:
            print(f"  iter {iteration:3d}: r_e={r_e:.6e}  Ω={Omega:.4e}"
                  f"  dif={dif:.2e}")
        if dif < accuracy and iteration >= 1:
            break

    return r_e, Omega


# ============================================================================
# mass_radius() — global integral quantities
# ============================================================================

def mass_radius(s_gp, mu_arr, eos,
                rho, gama, alpha, omega,
                energy, pressure, enthalpy, velocity_sq,
                r_ratio, e_surface, r_e, Omega):
    """
    Compute gravitational mass M, baryon mass M_0, angular momentum J,
    circumferential equatorial radius R_e, Keplerian frequency Omega_K,
    and co/counter-rotating particle velocities v_plus, v_minus.
    """
    is_tab = eos['type'] == 'tab'
    re2    = r_e ** 2
    v2     = velocity_sq
    sin_th = np.sqrt(np.maximum(1.0 - mu_arr ** 2, 0.0))

    # Baryon mass density on grid
    rho0 = np.zeros((SDIV, MDIV))
    for j in range(SDIV):
        for m in range(MDIV):
            e_jm = energy[j, m]
            if e_jm > e_surface:
                if is_tab:
                    n0 = _n0_at_e(e_jm, eos)
                    rho0[j, m] = n0 * MB * KSCALE * C * C
                else:
                    p_jm = pressure[j, m]
                    h_jm = enthalpy[j, m]
                    rho0[j, m] = (e_jm + p_jm) * np.exp(-h_jm)

    # Volume-element factor (√s/(1-s))^4 = s²/(1-s)^4 in s-coordinates.
    # This arises from r_is² dr_is = r_e³ s²/(1-s)^4 ds (the coordinate Jacobian).
    with np.errstate(invalid='ignore', divide='ignore'):
        jac = np.where(s_gp < SMAX,
                       s_gp**2 / (1.0 - s_gp)**4,
                       0.0)

    exp2ag  = np.exp(np.clip(2.0 * alpha * re2 + gama  * re2,             -100, 100))
    exp2agr = np.exp(np.clip(2.0 * alpha * re2 + 0.5 * (gama - rho) * re2, -100, 100))
    exp2agp = np.exp(np.clip(2.0 * alpha * re2 + (gama + rho) * re2,        -100, 100))

    omv2 = np.maximum(1.0 - v2, 1e-10)

    # ---- Gravitational mass ----------------------------------------
    D_M = np.zeros((SDIV, MDIV))
    for j in range(1, SDIV):
        D_M[j, :] = (exp2ag[j, :] * jac[j] *
                     ((energy[j, :] + pressure[j, :]) * (1.0 + v2[j, :]) / omv2[j, :]
                      + 2.0 * pressure[j, :]))
    Mass_int = np.trapezoid(np.trapezoid(D_M, dx=DM, axis=1), dx=DS)
    if is_tab:
        Mass = 4.0 * PI * np.sqrt(KAPPA) * C * C * r_e ** 3 / G * Mass_int
    else:
        Mass = 4.0 * PI * r_e ** 3 * Mass_int

    # ---- Baryon mass -----------------------------------------------
    D_M0 = np.zeros((SDIV, MDIV))
    for j in range(1, SDIV):
        D_M0[j, :] = (exp2agr[j, :] * jac[j] *
                      rho0[j, :] / np.sqrt(omv2[j, :]))
    Mass_0_int = np.trapezoid(np.trapezoid(D_M0, dx=DM, axis=1), dx=DS)
    if is_tab:
        Mass_0 = 4.0 * PI * np.sqrt(KAPPA) * C * C * r_e ** 3 / G * Mass_0_int
    else:
        Mass_0 = 4.0 * PI * r_e ** 3 * Mass_0_int

    # ---- Angular momentum ------------------------------------------
    D_J = np.zeros((SDIV, MDIV))
    for j in range(1, SDIV):
        D_J[j, :] = (exp2agp[j, :] * jac[j] * sin_th *
                     (energy[j, :] + pressure[j, :]) *
                     np.sqrt(v2[j, :]) / omv2[j, :])
    J_int = np.trapezoid(np.trapezoid(D_J, dx=DM, axis=1), dx=DS)
    if is_tab:
        J = 4.0 * PI * KAPPA * C * C * C * r_e ** 4 / G * J_int
    else:
        J = 4.0 * PI * r_e ** 4 * J_int

    # ---- Circumferential equatorial radius -------------------------
    j_eq    = max(1, min(int(round(0.5 / DS)), SDIV - 1))
    gama_eq = gama[j_eq, 0]
    rho_eq  = rho[j_eq,  0]
    exp_arg = np.clip(0.5 * (gama_eq - rho_eq) * re2, -50, 50)
    if is_tab:
        R_e = np.sqrt(KAPPA) * r_e * np.exp(exp_arg)
    else:
        R_e = r_e * np.exp(exp_arg)

    # ---- Keplerian angular velocity at equatorial surface ----------
    # The metric-gradient formula is corrupted by the coordinate
    # discontinuity at the interior/exterior boundary.  Instead derive
    # Omega_K from the Schwarzschild formula Omega_K = sqrt(M/R_S^3),
    # valid for both non-rotating and slowly-rotating stars.
    #   m_code = M_phys * G / (c^2 * sqrt(KAPPA))   [code mass]
    #   R_S_code = R_e / sqrt(KAPPA)                 [code Schwarzschild radius]
    #   Omega_K_code = sqrt(m_code / R_S_code^3)
    #   Omega_K_SI   = Omega_K_code * c / sqrt(KAPPA)
    if is_tab:
        m_code   = Mass * G / (C * C * np.sqrt(KAPPA))
        R_S_code = max(R_e / np.sqrt(KAPPA), 1e-30)
    else:
        m_code   = Mass          # code-unit mass for polytrope
        R_S_code = max(R_e, 1e-30)

    om_eq   = omega[j_eq, 0]
    Omega_K = np.sqrt(m_code / R_S_code**3) + om_eq / r_e

    # ---- Co/counter-rotating particle velocities -------------------
    v_plus  = np.zeros(SDIV)
    v_minus = np.zeros(SDIV)
    for j in range(1, SDIV):
        j1i = min(j + 1, SDIV - 1)
        j0i = max(j - 1, 1)
        dg = (gama[j1i, 0] - gama[j0i, 0]) / (2.0 * DS)
        dr = (rho[j1i,  0] - rho[j0i,  0]) / (2.0 * DS)
        do = (omega[j1i, 0] - omega[j0i, 0]) / (2.0 * DS)
        sv = s_gp[j]
        s1 = sv * (1.0 - sv)
        A  = 2.0 + re2 * s1 * (dg - dr)
        num_sq = re2 * s1 * (dg + dr)
        disc   = (num_sq / max(A, 1e-20)) ** 2 - 1.0
        if disc >= 0.0:
            rot_shift    = re2 * s1 * do / max(A, 1e-20)
            v_plus[j]   = max(0.0,  np.sqrt(disc) + rot_shift)
            v_minus[j]  = max(0.0,  np.sqrt(disc) - rot_shift)

    return Mass, Mass_0, J, R_e, Omega_K, v_plus, v_minus


# ============================================================================
# Main driver
# ============================================================================


# ============================================================================
# Plotting  (requires matplotlib; silently skipped if unavailable)
# ============================================================================

def plot_star(s_gp, mu_arr, eos,
              rho, gama, alpha, omega,
              energy, pressure, enthalpy, velocity_sq,
              r_e, Omega, r_ratio, e_center,
              Mass, Mass_0, R_e, Omega_K, J,
              savefig="star.pdf"):
    """
    Save a multi-panel diagnostic figure for a single equilibrium model.

    The figure is always written to *savefig* (default ``star.pdf``).
    No interactive window is opened.

    Panels
    ------
    1. Energy-density profile ε(r)
    2. Pressure profile p(r)
    3. Enclosed-mass profile M(<r)
    4. Metric potentials ν(r) and ζ(r) = (γ−ρ)/2
    5. (Rotating only) Colour map of ε in the meridional plane

    Units  (SI throughout)
    ----------------------
    Tabulated EOS : radius in km, ε in kg m⁻³, p in Pa, M in M☉
    Polytrope     : radius r/R_e, ε/ε_c, p/p_c, M/M (dimensionless — no
                    physical scale is defined without specifying the
                    polytropic constant K)
    """
    _apply_style()
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("[plot_star] matplotlib not available – skipping plots.")
        return

    is_tab    = eos['type'] == 'tab'
    re2       = r_e ** 2
    j_pole    = max(1, min(int(round(0.5 / DS)), SDIV - 1))
    gprp_pole = gama[j_pole, MDIV - 1] + rho[j_pole, MDIV - 1]

    # ── radial grid (equatorial, j = 0 … j_pole) ──────────────────────────
    idx    = np.arange(j_pole + 1)
    sv     = s_gp[idx]
    r_is   = np.where(sv > 0, r_e * sv / (1.0 - sv), 0.0)
    zeta   = alpha[idx, 0] * re2
    R_circ = r_is * np.exp(np.clip(zeta, -20, 20))

    # Circumferential equatorial radius R_e (used for normalisation)
    R_circ_surface = R_circ[j_pole]   # circumferential radius at stellar surface

    if is_tab:
        # SI unit conversions
        # radius:  1 code-length unit → sqrt(KAPPA) cm;  /1e5 → km
        # density: 1 code unit = 1/(C²·KSCALE) g/cm³ = 10¹⁵ g/cm³ at e_code=1;
        #          × 1000 → kg/m³
        # pressure: 1 code unit = 1/KSCALE dyne/cm²; × 0.1 → Pa
        km_fac   = np.sqrt(KAPPA) / 1e5           # code-length → km
        rho_SI   = 1e3 / (C**2 * KSCALE)          # code e      → kg m⁻³
        pres_SI  = 0.1 / KSCALE                   # code p      → Pa
        R_plot   = R_circ * km_fac                 # [km]
        R_e_plot = R_circ_surface * km_fac         # stellar surface [km]
        xlabel   = r"Circumferential radius  $r$  [km]"
        elabel   = r"Energy density  $\varepsilon$  [kg m$^{-3}$]"
        plabel   = r"Pressure  $p$  [Pa]"
        Mlabel   = r"Enclosed mass  $M(<r)$  [$M_\odot$]"
    else:
        # Normalised (dimensionless) — polytrope has no absolute physical scale
        rho_SI   = 1.0
        pres_SI  = 1.0
        R_plot   = R_circ / max(R_circ_surface, 1e-30)   # r / R_e
        R_e_plot = 1.0
        xlabel   = r"Circumferential radius  $r / R_e$"
        elabel   = r"Energy density  $\varepsilon / \varepsilon_c$"
        plabel   = r"Pressure  $p / p_c$"
        Mlabel   = r"Enclosed mass  $M(<r) / M$"

    # ── radial profiles ───────────────────────────────────────────────────
    e_prof = np.zeros(len(idx))
    p_prof = np.zeros(len(idx))
    for k, j in enumerate(idx):
        h_val = 0.5 * re2 * (gprp_pole - gama[j, 0] - rho[j, 0])
        if h_val > 0:
            pv           = _p_at_h(h_val, eos)
            e_prof[k]    = _e_at_p(pv, eos) * rho_SI
            p_prof[k]    = pv * pres_SI

    if not is_tab:
        ec0 = e_prof[0] if e_prof[0] > 0 else 1.0
        pc0 = p_prof[0] if p_prof[0] > 0 else 1.0
        e_prof /= ec0
        p_prof /= pc0

    # Enclosed mass (Newtonian proxy, scaled to total)
    dR    = np.diff(R_plot)
    Rmid  = 0.5 * (R_plot[:-1] + R_plot[1:])
    emid  = 0.5 * (e_prof[:-1] + e_prof[1:])
    M_enc = np.zeros(len(R_plot))
    M_enc[1:] = np.cumsum(4.0 * PI * emid * Rmid**2 * dR)
    if is_tab:
        M_enc = M_enc / max(M_enc[-1], 1e-30) * max(Mass / MSUN, 1e-30)
    else:
        M_enc = M_enc / max(M_enc[-1], 1e-30)

    # Metric potentials
    nu_prof   = 0.5 * (gama[idx, 0] + rho[idx, 0]) * re2
    zeta_prof = 0.5 * (gama[idx, 0] - rho[idx, 0]) * re2

    # ── layout ────────────────────────────────────────────────────────────
    is_rotating = r_ratio < 0.999
    n_rows      = 3 if is_rotating else 2
    fig, axes   = plt.subplots(n_rows, 2,
                               figsize=(10, 4 * n_rows),
                               constrained_layout=True)
    colour = "#1f77b4"
    surf_kw = dict(ls="--", color="gray", alpha=0.6)

    # (0,0) energy density
    a = axes[0, 0]
    a.plot(R_plot, e_prof, color=colour, lw=1.8)
    if is_tab: a.set_yscale("log")
    a.axvline(R_e_plot, label=r"$R_e$", **surf_kw)
    a.set_xlabel(xlabel); a.set_ylabel(elabel)
    a.set_title("Energy-density profile")
    a.legend(fontsize=9); a.grid(True, alpha=0.25)

    # (0,1) pressure
    a = axes[0, 1]
    a.plot(R_plot, p_prof, color="tab:orange", lw=1.8)
    if is_tab: a.set_yscale("log")
    a.axvline(R_e_plot, **surf_kw)
    a.set_xlabel(xlabel); a.set_ylabel(plabel)
    a.set_title("Pressure profile")
    a.grid(True, alpha=0.25)

    # (1,0) enclosed mass
    a = axes[1, 0]
    a.plot(R_plot, M_enc, color="tab:green", lw=1.8)
    a.axvline(R_e_plot, **surf_kw)
    a.set_xlabel(xlabel); a.set_ylabel(Mlabel)
    a.set_title("Enclosed-mass profile")
    a.grid(True, alpha=0.25)

    # (1,1) metric potentials
    a = axes[1, 1]
    a.plot(R_plot, nu_prof,   color="tab:red",    lw=1.8, label=r"$\nu$ (lapse)")
    a.plot(R_plot, zeta_prof, color="tab:purple", lw=1.8, ls="--",
           label=r"$\zeta$ (conformal)")
    a.axvline(R_e_plot, **surf_kw)
    a.set_xlabel(xlabel); a.set_ylabel("Metric potential  (dimensionless)")
    a.set_title("Spacetime potentials  ($G=c=1$)")
    a.legend(fontsize=9); a.grid(True, alpha=0.25)

    # (2, full-width) meridional density contour — rotating only
    if is_rotating:
        ax_mer = fig.add_subplot(n_rows, 1, n_rows)
        axes[2, 0].remove(); axes[2, 1].remove()

        theta_arr = np.arccos(mu_arr)
        r_is_2d   = np.where(s_gp[:j_pole+1] > 0,
                              r_e * s_gp[:j_pole+1] / (1.0 - s_gp[:j_pole+1]), 0.0)
        R_2d = (np.outer(r_is_2d, np.ones(MDIV))
                * np.exp(np.clip(alpha[:j_pole+1, :] * re2, -20, 20)))
        e_2d = np.zeros((j_pole + 1, MDIV))
        for j in range(j_pole + 1):
            for m in range(MDIV):
                h_val = 0.5 * re2 * (gprp_pole - gama[j, m] - rho[j, m])
                if h_val > 0:
                    pv = _p_at_h(h_val, eos)
                    e_2d[j, m] = _e_at_p(pv, eos) * rho_SI

        fac  = km_fac if is_tab else 1.0 / max(R_circ_surface, 1e-30)
        x_2d = R_2d * np.sin(theta_arr)[np.newaxis, :] * fac
        z_2d = R_2d * np.cos(theta_arr)[np.newaxis, :] * fac

        x_f = np.hstack([-x_2d[:, ::-1], x_2d])
        z_f = np.hstack([ z_2d[:, ::-1], z_2d])
        e_f = np.hstack([ e_2d[:, ::-1], e_2d])
        x_f = np.hstack([x_f, x_f[:, ::-1]])
        z_f = np.hstack([z_f, -z_f[:, ::-1]])
        e_f = np.hstack([e_f, e_f[:, ::-1]])

        vmax = e_f.max()
        vmin = max(vmax * 1e-3, 1e-30)
        cf = ax_mer.contourf(x_f, z_f, e_f, levels=20,
                             norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
                             cmap="inferno")
        plt.colorbar(cf, ax=ax_mer, label=elabel)
        ax_mer.set_xlabel(xlabel.replace("Circumferential radius", "$x$"))
        ax_mer.set_ylabel(xlabel.replace("Circumferential radius", "$z$"))
        ax_mer.set_title("Meridional density map")
        ax_mer.set_aspect("equal")
        ax_mer.grid(True, alpha=0.2, color="white", ls=":")

    # ── super-title ───────────────────────────────────────────────────────
    if is_tab:
        ec_SI    = e_center / (C**2 * KSCALE) * 1e3   # g/cm³ → kg/m³
        Omega_SI = Omega    * C / np.sqrt(KAPPA)       # code  → rad/s
        OK_SI    = Omega_K  * C / np.sqrt(KAPPA)
        state    = "Rotating" if is_rotating else "Non-rotating"
        suptitle = (f"{state} neutron star   "
                    f"$M = {Mass/MSUN:.3f}\\,M_\\odot$,  "
                    f"$R_e = {R_e/1e5:.2f}$ km,  "
                    f"$\\varepsilon_c = {ec_SI:.2e}$ kg m$^{{-3}}$,  "
                    f"$\\Omega_K = {OK_SI:.0f}$ rad s$^{{-1}}$")
    else:
        # Polytrope in geometric units (G = c = 1).
        # Absolute mass and radius have no physical scale without specifying K;
        # the dimensionless compactness M/R_e is physically meaningful.
        N_poly    = 1.0 / max(eos.get('Gamma_P', 2.0) - 1.0, 1e-9)
        compact   = Mass / max(R_e, 1e-30)           # M/R_e in geometric units
        state     = "Rotating" if is_rotating else "Non-rotating"
        spin_frac = Omega / max(Omega_K, 1e-30) if is_rotating else 0.0
        suptitle  = (f"{state} polytrope  $N={N_poly:.2g}$   "
                     f"Compactness $M/R_e = {compact:.3f}$   "
                     f"$\\Omega_K\\sqrt{{R_e^3/M}} = "
                     f"{Omega_K * np.sqrt(R_e**3 / max(Mass, 1e-30)):.3f}$")
        if is_rotating:
            suptitle += f"   $\\Omega/\\Omega_K = {spin_frac:.3f}$"
    fig.suptitle(suptitle, fontsize=11)

    fig.savefig(savefig, dpi=150, bbox_inches="tight")
    print(f"  Saved star plot → {savefig}")
    plt.close(fig)


def _print_header(eos_file, is_poly, n_P):
    if not is_poly:
        print(f"{eos_file},  MDIVxSDIV={MDIV}x{SDIV}")
    else:
        print(f"polytrope N={n_P:.3f},  MDIVxSDIV={MDIV}x{SDIV}")
    print("ratio\te_15\tM\tM_0\tr_star\tspin\tOmega_K\tI\tJ/M^2")
    if not is_poly:
        print("\tg/cm^3\tsun\tsun\tkm\ts-1\ts-1\tg cm^2\t")
    print()


def _print_result(r_ratio, e_center, Mass, Mass_0, R_e, Omega, Omega_K, J, is_poly):
    if abs(Omega) > 0.0:
        I_val = J / Omega
    else:
        I_val = 0.0

    if not is_poly:
        e_15       = e_center / (C * C * KSCALE * 1e15)
        I_45       = I_val / 1e45
        M2         = max(Mass, 1e-30) ** 2
        Jm2        = C * J / (G * M2)
        # Convert angular velocities from code units → rad s⁻¹
        conv_omega = C / np.sqrt(KAPPA)
        print(f"{r_ratio:.3f}\t{e_15:.1f}\t"
              f"{Mass/MSUN:.3f}\t{Mass_0/MSUN:.3f}\t"
              f"{R_e/1e5:.3f}\t{Omega*conv_omega:.1f}\t"
              f"{Omega_K*conv_omega:.1f}\t{I_45:.2f}\t{Jm2:.3f}")
    else:
        Jm2 = J / max(Mass_0 ** 2, 1e-60)
        print(f"{r_ratio:.3f}\t{e_center:.3f}\t"
              f"{Mass:.3f}\t{Mass_0:.3f}\t"
              f"{R_e:.3f}\t{Omega:.3f}\t"
              f"{Omega_K:.3f}\t{I_val:.2f}\t{Jm2:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="pyRNS: rotating neutron star equilibrium solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pyRNS.py -f eosA -e 1e15
  python pyRNS.py -q poly -N 1.0 -e 0.5
""")
    parser.add_argument("-f",  metavar="EOS_FILE",  default="",
                        help="tabulated EOS filename")
    parser.add_argument("-e",  metavar="E_CENTER",  type=float, required=True,
                        help="central energy density [g/cm³] (tab) or code units (poly)")
    parser.add_argument("-q",  metavar="EOS_TYPE",  default="tab",
                        choices=["tab", "poly"],
                        help="EOS type: tab or poly  [default: tab]")
    parser.add_argument("-N",  metavar="POLY_N",    type=float, default=1.0,
                        help="polytropic index N  [default: 1.0]")
    parser.add_argument("--accuracy", type=float, default=1e-5,
                        help="convergence threshold  [default: 1e-5]")
    parser.add_argument("--cf",       type=float, default=1.0,
                        help="SOR relaxation factor  [default: 1.0]")
    parser.add_argument("--verbose",  action="store_true",
                        help="print per-iteration details")
    parser.add_argument("--plot",  dest="plot",  action="store_true",  default=True,
                        help="produce diagnostic plots (default: on)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="suppress all plots")
    parser.add_argument("--savefig", metavar="FILE", default=None,
                        help="save figure to FILE (e.g. star.pdf)")
    args = parser.parse_args()

    is_poly = (args.q == "poly")
    Gamma_P = 1.0 + 1.0 / args.N if is_poly else 2.0

    # Build EOS dict
    eos = {"type": args.q, "Gamma_P": Gamma_P}
    if not is_poly:
        if not args.f:
            parser.error("specify an EOS file with -f for tabulated EOS.")
        if not os.path.isfile(args.f):
            parser.error(f"EOS file not found: {args.f}")
        log_e, log_p, log_h, log_n0, n_tab = load_eos(args.f)
        eos.update({"log_e": log_e, "log_p": log_p,
                    "log_h": log_h, "log_n0": log_n0, "n_tab": n_tab})

    # Central thermodynamic conditions
    if not is_poly:
        e_center     = args.e * C * C * KSCALE
        e_surface    = 7.8   * C * C * KSCALE
        p_surface    = 1.01e8 * KSCALE
        enthalpy_min = 1.0 / (C * C)
    else:
        e_center     = args.e
        e_surface    = 0.0
        p_surface    = 0.0
        enthalpy_min = 0.0

    p_center, h_center = make_center(e_center, eos)

    # Grid and precomputed arrays (done once)
    s_gp, mu_arr = make_grid()
    print("Precomputing Legendre arrays...", flush=True)
    P_2n, P1_2n1, sin_2n1_theta = precompute_legendre(mu_arr)
    print("Precomputing Green's function kernels...", flush=True)
    f_rho_kern, f_gama_kern = compute_kernels(s_gp)

    # Allocate field arrays
    rho          = np.zeros((SDIV, MDIV))
    gama         = np.zeros((SDIV, MDIV))
    alpha        = np.zeros((SDIV, MDIV))
    omega        = np.zeros((SDIV, MDIV))
    energy       = np.zeros((SDIV, MDIV))
    pressure     = np.zeros((SDIV, MDIV))
    enthalpy_arr = np.zeros((SDIV, MDIV))
    velocity_sq  = np.zeros((SDIV, MDIV))

    _print_header(args.f, is_poly, args.N)

    # Non-rotating seed solution from TOV
    print("Computing spherical seed star...", flush=True)
    rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
        s_gp, mu_arr, e_center, p_center, h_center,
        p_surface, e_surface, eos)

    r_ratio = 1.0

    # Non-rotating model (r_ratio=1 → Omega=0, handled exactly in spin())
    r_e, Omega = spin(s_gp, mu_arr, eos,
                      h_center, enthalpy_min,
                      rho, gama, alpha, omega,
                      energy, pressure, enthalpy_arr, velocity_sq,
                      r_ratio, r_e,
                      accuracy=args.accuracy, cf=args.cf,
                      f_rho_kern=f_rho_kern, f_gama_kern=f_gama_kern,
                      P_2n=P_2n, P1_2n1=P1_2n1,
                      sin_2n1_theta=sin_2n1_theta,
                      verbose=args.verbose)

    Mass, Mass_0, J, R_e, Omega_K, _, _ = mass_radius(
        s_gp, mu_arr, eos,
        rho, gama, alpha, omega,
        energy, pressure, enthalpy_arr, velocity_sq,
        r_ratio, e_surface, r_e, Omega)

    _print_result(r_ratio, e_center, Mass, Mass_0, R_e, Omega, Omega_K, J, is_poly)

    if args.plot:
        # Auto-generate a sensible PDF name if --savefig was not supplied
        if args.savefig:
            pdf_path = args.savefig
        elif not is_poly:
            # e.g. star_eosA_1.0e+15.pdf
            base = os.path.splitext(os.path.basename(args.f))[0]
            pdf_path = f"star_{base}_{args.e:.1e}.pdf"
        else:
            # e.g. star_poly_N1.0_e0.50.pdf
            pdf_path = f"star_poly_N{args.N}_e{args.e:.2f}.pdf"

        plot_star(s_gp, mu_arr, eos,
                  rho, gama, alpha, omega,
                  energy, pressure, enthalpy_arr, velocity_sq,
                  r_e, Omega, r_ratio, e_center,
                  Mass, Mass_0, R_e, Omega_K, J,
                  savefig=pdf_path)

    # Spin up toward the Keplerian limit
    # NOTE: the full self-consistent field iteration for rotating stars (the
    # Komatsu-Eriguchi-Hachisu / Stergioulas algorithm) requires an exact
    # reproduction of the original RNS Green's-function kernel normalisation
    # that is only available in the original C source.  The implementation
    # below attempts the iteration but may not converge to the correct answer.
    dr       = 0.05
    diff     = Omega_K - Omega
    old_diff = diff
    rho_saved   = rho.copy()
    gama_saved  = gama.copy()
    alpha_saved = alpha.copy()
    omega_saved = omega.copy()
    r_e_saved   = r_e

    while diff > 0.0:
        r_ratio -= dr
        rho[:]   = rho_saved
        gama[:]  = gama_saved
        alpha[:] = alpha_saved
        omega[:] = omega_saved
        r_e      = r_e_saved

        try:
            r_e, Omega = spin(s_gp, mu_arr, eos,
                              h_center, enthalpy_min,
                              rho, gama, alpha, omega,
                              energy, pressure, enthalpy_arr, velocity_sq,
                              r_ratio, r_e,
                              accuracy=args.accuracy, cf=args.cf,
                              f_rho_kern=f_rho_kern, f_gama_kern=f_gama_kern,
                              P_2n=P_2n, P1_2n1=P1_2n1,
                              sin_2n1_theta=sin_2n1_theta,
                              verbose=args.verbose)

            Mass, Mass_0, J, R_e, Omega_K, _, _ = mass_radius(
                s_gp, mu_arr, eos,
                rho, gama, alpha, omega,
                energy, pressure, enthalpy_arr, velocity_sq,
                r_ratio, e_surface, r_e, Omega)

            # Sanity check: if results are clearly unphysical, stop
            if not np.isfinite(Mass) or Mass <= 0 or Mass > 1e6:
                print(f"  [Note] Rotating-star iteration did not converge for "
                      f"r_ratio={r_ratio:.3f}; stopping spin-up.", flush=True)
                break

            _print_result(r_ratio, e_center, Mass, Mass_0, R_e,
                          Omega, Omega_K, J, is_poly)

        except Exception as exc:
            print(f"  [Warning] spin() failed for r_ratio={r_ratio:.3f}: {exc}",
                  flush=True)
            break

        old_diff = diff
        diff     = Omega_K - Omega


if __name__ == "__main__":
    main()
