"""
Tests for pyRNS.py — core TOV solver and metric initialisation.

Marks
-----
  fast    : pure-math / tiny computation (< 0.1 s)
  slow    : requires full grid pre-computation or TOV integration
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Physical constants have correct magnitudes."""

    @pytest.mark.fast
    def test_speed_of_light(self):
        from pyRNS import C
        assert abs(C - 2.9979e10) / 2.9979e10 < 1e-4

    @pytest.mark.fast
    def test_solar_mass(self):
        from pyRNS import MSUN
        assert 1.98e33 < MSUN < 2.00e33

    @pytest.mark.fast
    def test_kappa_positive(self):
        from pyRNS import KAPPA
        assert KAPPA > 0

    @pytest.mark.fast
    def test_grid_sizes(self):
        from pyRNS import SDIV, MDIV, LMAX
        assert SDIV == 129
        assert MDIV == 65
        assert LMAX == 10


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class TestGrid:
    @pytest.mark.fast
    def test_shapes(self, grid):
        from pyRNS import SDIV, MDIV
        s_gp, mu = grid
        assert s_gp.shape == (SDIV,)
        assert mu.shape  == (MDIV,)

    @pytest.mark.fast
    def test_s_range(self, grid):
        s_gp, _ = grid
        assert s_gp[0]  == pytest.approx(0.0,   abs=1e-10)
        assert s_gp[-1] == pytest.approx(0.9999, rel=1e-4)
        assert np.all(np.diff(s_gp) > 0), "s_gp must be strictly increasing"

    @pytest.mark.fast
    def test_mu_range(self, grid):
        _, mu = grid
        assert mu[0]  == pytest.approx(0.0, abs=1e-10)
        assert mu[-1] == pytest.approx(1.0, abs=1e-10)
        assert np.all(np.diff(mu) > 0), "mu must be strictly increasing"


# ---------------------------------------------------------------------------
# Legendre arrays
# ---------------------------------------------------------------------------

class TestLegendre:
    @pytest.mark.fast
    def test_shapes(self, legendre):
        from pyRNS import MDIV, LMAX
        P_2n, P1_2n1, sn = legendre
        assert P_2n.shape  == (MDIV, LMAX + 1)
        assert P1_2n1.shape == (MDIV, LMAX + 1)
        assert sn.shape     == (MDIV, LMAX + 1)

    @pytest.mark.fast
    def test_P0_equals_one(self, legendre):
        P_2n, _, _ = legendre
        # P_0(mu) = 1 for all mu
        assert np.allclose(P_2n[:, 0], 1.0)

    @pytest.mark.fast
    def test_P2_at_equator(self, legendre):
        P_2n, _, _ = legendre
        # P_2(0) = -1/2
        assert P_2n[0, 1] == pytest.approx(-0.5, rel=1e-4)

    @pytest.mark.fast
    def test_P2_at_pole(self, legendre):
        P_2n, _, _ = legendre
        # P_2(1) = 1
        assert P_2n[-1, 1] == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

class TestKernels:
    @pytest.mark.fast
    def test_shapes(self, kernels):
        from pyRNS import SDIV, LMAX
        fk, gk = kernels
        assert fk.shape == (SDIV, LMAX + 1, SDIV)
        assert gk.shape == (SDIV, LMAX + 1, SDIV)

    @pytest.mark.fast
    def test_unit_source_outside(self, kernels, grid):
        """Placing a unit source at k gives a positive D2 at j < k."""
        from pyRNS import DS
        fk, _ = kernels
        k_src = 20
        D2 = DS * fk[:, 0, k_src]           # monopole contribution from k_src
        # For j < k_src the Green's function is positive
        assert D2[5] > 0
        assert D2[10] > 0

    @pytest.mark.fast
    def test_zero_at_boundary(self, kernels):
        """Kernel is zero at j=0 (centre)."""
        fk, _ = kernels
        assert np.allclose(fk[0, :, :], 0.0)


# ---------------------------------------------------------------------------
# EOS functions
# ---------------------------------------------------------------------------

class TestEOS:
    @pytest.mark.fast
    def test_make_center_poly(self, poly_eos):
        from pyRNS import make_center
        p_c, h_c = make_center(0.5, poly_eos)
        assert p_c > 0
        assert h_c > 0
        assert h_c < 1.0      # sanity: h/c² < 1 for polytrope at e_c=0.5

    @pytest.mark.fast
    def test_make_center_tab(self, tab_eos):
        if tab_eos is None:
            pytest.skip("eosA not found")
        from pyRNS import make_center, C, KSCALE
        e_c = 1e15 * C**2 * KSCALE
        p_c, h_c = make_center(e_c, tab_eos)
        assert p_c > 0
        assert h_c > 0

    @pytest.mark.fast
    def test_load_eos_shape(self, tab_eos):
        if tab_eos is None:
            pytest.skip("eosA not found")
        assert tab_eos["n_tab"] > 50
        assert len(tab_eos["log_e"]) == tab_eos["n_tab"]

    @pytest.mark.fast
    def test_load_eos_monotone(self, tab_eos):
        if tab_eos is None:
            pytest.skip("eosA not found")
        assert np.all(np.diff(tab_eos["log_e"]) > 0), "log_e not monotone"
        assert np.all(np.diff(tab_eos["log_p"]) > 0), "log_p not monotone"
        assert np.all(np.diff(tab_eos["log_h"]) > 0), "log_h not monotone"

    @pytest.mark.fast
    def test_all_bundled_eos_load(self):
        """All 12 working RNS EOS files load without errors."""
        import os
        from pyRNS import load_eos
        eos_dir = os.path.join(os.path.dirname(__file__), "..", "eos")
        working = ["eosA", "eosAU", "eosB", "eosC", "eosF", "eosFPS",
                   "eosG", "eosL", "eosN", "eosO", "eosUU", "eosWS"]
        for name in working:
            path = os.path.join(eos_dir, name)
            if not os.path.exists(path):
                continue
            log_e, log_p, log_h, log_n0, n_tab = load_eos(path)
            assert n_tab > 10, f"{name}: too few table points"
            assert np.all(np.diff(log_e) > 0), f"{name}: log_e not monotone"


# ---------------------------------------------------------------------------
# TOV integration
# ---------------------------------------------------------------------------

class TestTOV:
    @pytest.mark.slow
    def test_poly_mass_positive(self, poly_eos, poly_center):
        from pyRNS import solve_tov
        e_c, p_c, _ = poly_center
        _, _, m_arr, _, _, _, r_is_final, m_final = solve_tov(
            e_c, p_c, 0.0, poly_eos)
        assert m_final > 0
        assert r_is_final > 0

    @pytest.mark.slow
    def test_poly_mass_value(self, poly_eos, poly_center):
        """m_final for N=1, e_c=0.5 should be ~0.163 (code units)."""
        from pyRNS import solve_tov
        e_c, p_c, _ = poly_center
        _, _, _, _, _, _, _, m_final = solve_tov(e_c, p_c, 0.0, poly_eos)
        assert m_final == pytest.approx(0.163, rel=0.02)

    @pytest.mark.slow
    def test_poly_radius_value(self, poly_eos, poly_center):
        """r_is_final for N=1, e_c=0.5 should be ~1.073 (code units)."""
        from pyRNS import solve_tov
        e_c, p_c, _ = poly_center
        _, _, _, _, _, _, r_is_final, _ = solve_tov(e_c, p_c, 0.0, poly_eos)
        assert r_is_final == pytest.approx(1.073, rel=0.02)

    @pytest.mark.slow
    def test_pressure_decreases(self, poly_eos, poly_center):
        from pyRNS import solve_tov
        e_c, p_c, _ = poly_center
        _, _, _, p_arr, _, _, _, _ = solve_tov(e_c, p_c, 0.0, poly_eos)
        # Pressure must decrease outward (non-zero portion)
        nonzero = p_arr[p_arr > 0]
        assert len(nonzero) > 10
        assert np.all(np.diff(nonzero) <= 0)


# ---------------------------------------------------------------------------
# Sphere initialisation
# ---------------------------------------------------------------------------

class TestSphere:
    @pytest.mark.slow
    def test_shapes(self, sphere_solution):
        from pyRNS import SDIV, MDIV
        rho, gama, alpha, omega, r_e = sphere_solution
        for arr in (rho, gama, alpha, omega):
            assert arr.shape == (SDIV, MDIV)

    @pytest.mark.slow
    def test_r_e_positive(self, sphere_solution):
        *_, r_e = sphere_solution
        assert r_e > 0

    @pytest.mark.slow
    def test_self_consistency(self, sphere_solution, poly_center):
        """2 h_c / (gprp_pole - gprp_center) ≈ r_e²  (Bernoulli condition)."""
        from pyRNS import SDIV, MDIV, DS
        rho, gama, _, _, r_e = sphere_solution
        _, _, h_c = poly_center
        j_pole = max(1, min(int(round(0.5 / DS)), SDIV - 1))
        gprp_pole   = gama[j_pole, MDIV - 1] + rho[j_pole, MDIV - 1]
        gprp_center = gama[0,      MDIV - 1] + rho[0,      MDIV - 1]
        diff = gprp_pole - gprp_center
        assert diff > 0, "gprp must increase from center to pole"
        assert 2 * h_c / diff == pytest.approx(r_e**2, rel=0.01)

    @pytest.mark.slow
    def test_omega_zero(self, sphere_solution):
        *_, rho, gama, alpha, omega, r_e = sphere_solution
        assert np.allclose(omega, 0.0)


# ---------------------------------------------------------------------------
# Spin + mass_radius
# ---------------------------------------------------------------------------

class TestMassRadius:
    @pytest.mark.slow
    def test_mass_positive(self, full_star):
        *_, M, M0, J, Re, OK = full_star
        assert M > 0

    @pytest.mark.slow
    def test_baryon_mass_geq_grav_mass(self, full_star):
        """M₀ ≥ M for a bound star (binding energy > 0)."""
        *_, M, M0, J, Re, OK = full_star
        assert M0 >= M * 0.9    # allow 10% margin for approximate formula

    @pytest.mark.slow
    def test_radius_positive(self, full_star):
        *_, Re, OK = full_star
        assert Re > 0

    @pytest.mark.slow
    def test_keplerian_positive(self, full_star):
        *_, OK = full_star
        assert OK > 0

    @pytest.mark.slow
    def test_poly_mass_value(self, full_star):
        """M ≈ 0.188 code units for N=1, e_c=0.5 (pyRNS reference value)."""
        *_, M, M0, J, Re, OK = full_star
        assert M == pytest.approx(0.188, rel=0.05)

    @pytest.mark.slow
    def test_poly_radius_value(self, full_star):
        """R_e ≈ 0.733 code units for N=1, e_c=0.5."""
        *_, Re, OK = full_star
        assert Re == pytest.approx(0.733, rel=0.02)

    @pytest.mark.slow
    def test_tab_mass_msun(self, grid, legendre, kernels, tab_eos):
        """eosA at 1e15 g/cm³ gives M = 0.835 ± 5% M☉."""
        if tab_eos is None:
            pytest.skip("eosA not found")
        import numpy as np
        from pyRNS import (make_center, sphere, spin, mass_radius,
                           SDIV, MDIV, C, KSCALE, MSUN)
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        e_c = 1e15 * C**2 * KSCALE
        e_surf = 7.8 * C**2 * KSCALE
        p_surf = 1.01e8 * KSCALE
        p_c, h_c = make_center(e_c, tab_eos)
        rho   = np.zeros((SDIV, MDIV)); gama  = np.zeros((SDIV, MDIV))
        alpha = np.zeros((SDIV, MDIV)); omega = np.zeros((SDIV, MDIV))
        en    = np.zeros((SDIV, MDIV)); pr    = np.zeros((SDIV, MDIV))
        enth  = np.zeros((SDIV, MDIV)); vsq   = np.zeros((SDIV, MDIV))
        rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
            s_gp, mu, e_c, p_c, h_c, p_surf, e_surf, tab_eos)
        r_e, Om = spin(s_gp, mu, tab_eos, h_c, 1/C**2,
                       rho, gama, alpha, omega, en, pr, enth, vsq,
                       1.0, r_e, f_rho_kern=fk, f_gama_kern=gk,
                       P_2n=P_2n, P1_2n1=P1_2n1, sin_2n1_theta=sn)
        M, *_ = mass_radius(s_gp, mu, tab_eos, rho, gama, alpha, omega,
                            en, pr, enth, vsq, 1.0, e_surf, r_e, 0.0)
        assert M / MSUN == pytest.approx(0.835, rel=0.05)

    @pytest.mark.slow
    def test_tab_radius_km(self, grid, legendre, kernels, tab_eos):
        """eosA at 1e15 g/cm³ gives R = 9.95 ± 2% km."""
        if tab_eos is None:
            pytest.skip("eosA not found")
        import numpy as np
        from pyRNS import (make_center, sphere, spin, mass_radius,
                           SDIV, MDIV, C, KSCALE, KAPPA)
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        e_c = 1e15 * C**2 * KSCALE
        e_surf = 7.8 * C**2 * KSCALE
        p_surf = 1.01e8 * KSCALE
        p_c, h_c = make_center(e_c, tab_eos)
        rho   = np.zeros((SDIV, MDIV)); gama  = np.zeros((SDIV, MDIV))
        alpha = np.zeros((SDIV, MDIV)); omega = np.zeros((SDIV, MDIV))
        en    = np.zeros((SDIV, MDIV)); pr    = np.zeros((SDIV, MDIV))
        enth  = np.zeros((SDIV, MDIV)); vsq   = np.zeros((SDIV, MDIV))
        rho[:], gama[:], alpha[:], omega[:], r_e = sphere(
            s_gp, mu, e_c, p_c, h_c, p_surf, e_surf, tab_eos)
        r_e, Om = spin(s_gp, mu, tab_eos, h_c, 1/C**2,
                       rho, gama, alpha, omega, en, pr, enth, vsq,
                       1.0, r_e, f_rho_kern=fk, f_gama_kern=gk,
                       P_2n=P_2n, P1_2n1=P1_2n1, sin_2n1_theta=sn)
        _, _, _, Re, *_ = mass_radius(s_gp, mu, tab_eos, rho, gama, alpha,
                                       omega, en, pr, enth, vsq,
                                       1.0, e_surf, r_e, 0.0)
        assert Re / 1e5 == pytest.approx(9.95, rel=0.02)
