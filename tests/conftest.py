"""
Shared pytest fixtures for pyRNS tests.

The grid, Legendre arrays, and kernels are precomputed once per session
(they are deterministic and expensive) and reused across all tests.
"""
import json
import os
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Session-scoped fixtures — computed once, shared across all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def grid():
    from pyRNS import make_grid
    return make_grid()


@pytest.fixture(scope="session")
def legendre(grid):
    from pyRNS import precompute_legendre
    s_gp, mu = grid
    return precompute_legendre(mu)


@pytest.fixture(scope="session")
def kernels(grid):
    from pyRNS import compute_kernels
    s_gp, mu = grid
    return compute_kernels(s_gp)


@pytest.fixture(scope="session")
def poly_eos():
    """N=1 polytrope EOS dict."""
    return {"type": "poly", "Gamma_P": 2.0}


@pytest.fixture(scope="session")
def poly_center(poly_eos):
    """Central conditions for N=1 polytrope at e_c=0.5."""
    from pyRNS import make_center
    e_c = 0.5
    p_c, h_c = make_center(e_c, poly_eos)
    return e_c, p_c, h_c


@pytest.fixture(scope="session")
def sphere_solution(grid, poly_eos, poly_center):
    """Pre-computed sphere() result for N=1 polytrope."""
    from pyRNS import sphere, SDIV, MDIV
    s_gp, mu = grid
    e_c, p_c, h_c = poly_center
    rho   = np.zeros((SDIV, MDIV))
    gama  = np.zeros((SDIV, MDIV))
    alpha = np.zeros((SDIV, MDIV))
    omega = np.zeros((SDIV, MDIV))
    rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
        s_gp, mu, e_c, p_c, h_c, 0.0, 0.0, poly_eos)
    return rho, gama, alpha, omega, r_e


@pytest.fixture(scope="session")
def full_star(grid, legendre, kernels, poly_eos, poly_center, sphere_solution):
    """
    Full non-rotating N=1 star: spin() + mass_radius() result.
    Returns (rho, gama, alpha, omega, energy, pressure, enthalpy,
             velocity_sq, r_e, Omega, M, M0, J, Re, Omega_K).
    """
    from pyRNS import spin, mass_radius, SDIV, MDIV
    s_gp, mu = grid
    P_2n, P1_2n1, sn = legendre
    fk, gk = kernels
    e_c, p_c, h_c = poly_center
    rho0, gama0, alpha0, omega0, r_e0 = sphere_solution

    # Work on copies so sphere_solution fixture is not mutated
    rho   = rho0.copy()
    gama  = gama0.copy()
    alpha = alpha0.copy()
    omega = omega0.copy()
    en    = np.zeros((SDIV, MDIV))
    pr    = np.zeros((SDIV, MDIV))
    enth  = np.zeros((SDIV, MDIV))
    vsq   = np.zeros((SDIV, MDIV))

    r_e, Om = spin(
        s_gp, mu, poly_eos, h_c, 0.0,
        rho, gama, alpha, omega, en, pr, enth, vsq,
        1.0, r_e0,
        f_rho_kern=fk, f_gama_kern=gk,
        P_2n=P_2n, P1_2n1=P1_2n1, sin_2n1_theta=sn,
    )
    M, M0, J, Re, OK, vp, vm = mass_radius(
        s_gp, mu, poly_eos,
        rho, gama, alpha, omega, en, pr, enth, vsq,
        1.0, 0.0, r_e, 0.0,
    )
    return rho, gama, alpha, omega, en, pr, enth, vsq, r_e, Om, M, M0, J, Re, OK


@pytest.fixture(scope="session")
def tab_eos():
    """Load eosA tabulated EOS (returns None if file absent)."""
    eos_path = os.path.join(os.path.dirname(__file__), "..", "eos", "eosA")
    if not os.path.exists(eos_path):
        return None
    from pyRNS import load_eos, C, KSCALE
    log_e, log_p, log_h, log_n0, n_tab = load_eos(eos_path)
    return {
        "type": "tab", "Gamma_P": 2.0,
        "log_e": log_e, "log_p": log_p,
        "log_h": log_h, "log_n0": log_n0,
        "n_tab": n_tab,
        "_path": eos_path,
    }


@pytest.fixture(scope="session")
def synthetic_posterior(tmp_path_factory):
    """
    Write a small synthetic Bilby JSON posterior to a temp file
    and return its path.
    """
    rng = np.random.default_rng(42)
    n   = 200
    Mc  = 1.188 + rng.normal(0, 0.004, n)
    q   = np.clip(0.88 + rng.normal(0, 0.06, n), 0.5, 1.0)
    m1  = Mc * (1 + q)**0.2 / q**0.6
    m2  = q * m1
    lam1 = np.exp(rng.normal(np.log(350), 0.5, n)).clip(10, 5000)
    lam2 = np.exp(rng.normal(np.log(550), 0.5, n)).clip(10, 5000)

    doc = {
        "label": "test_event",
        "posterior": {"content": {
            "mass_1": m1.tolist(), "mass_2": m2.tolist(),
            "lambda_1": lam1.tolist(), "lambda_2": lam2.tolist(),
            "chirp_mass": Mc.tolist(), "mass_ratio": q.tolist(),
        }},
    }
    tmp = tmp_path_factory.mktemp("posterior")
    path = str(tmp / "test_posterior.json")
    with open(path, "w") as f:
        json.dump(doc, f)
    return path
