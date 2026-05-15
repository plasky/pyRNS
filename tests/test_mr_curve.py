"""Tests for mr_curve.py — mass-radius sequences."""
import numpy as np
import pytest


class TestUnits:
    @pytest.mark.fast
    def test_poly_units_keys(self):
        from mr_curve import _units
        u = _units(is_tab=False)
        for key in ("lM", "lR", "lec", "lOmK", "cM", "cR", "cec", "cOm"):
            assert key in u

    @pytest.mark.fast
    def test_poly_unit_conversions_unity(self):
        from mr_curve import _units
        u = _units(is_tab=False)
        assert u["cM"]  == 1.0
        assert u["cR"]  == 1.0
        assert u["cec"] == 1.0

    @pytest.mark.fast
    def test_tab_unit_conversions(self):
        from mr_curve import _units
        from pyRNS import MSUN, KAPPA, C
        u = _units(is_tab=True)
        assert u["cM"]  == pytest.approx(1.0 / MSUN, rel=1e-4)
        assert u["cR"]  == pytest.approx(1.0 / 1e5,  rel=1e-4)
        assert u["cOm"] == pytest.approx(C / KAPPA**0.5, rel=1e-4)


class TestSingleModel:
    @pytest.mark.slow
    def test_model_returns_tuple(self, grid, legendre, kernels, poly_eos):
        from mr_curve import _model
        from pyRNS import C, KSCALE
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        result = _model(
            0.5, poly_eos,
            e_surface=0.0, p_surface=0.0, enthalpy_min=0.0,
            s_gp=s_gp, mu_arr=mu,
            P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk,
        )
        assert len(result) == 6

    @pytest.mark.slow
    def test_model_physical_values(self, grid, legendre, kernels, poly_eos):
        from mr_curve import _model
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        M, M0, Re, OK, J, Omega = _model(
            0.5, poly_eos,
            e_surface=0.0, p_surface=0.0, enthalpy_min=0.0,
            s_gp=s_gp, mu_arr=mu,
            P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk,
        )
        assert M  > 0
        assert M0 > 0
        assert Re > 0
        assert OK > 0
        assert np.isfinite(M) and np.isfinite(Re)


class TestSequence:
    @pytest.mark.slow
    def test_sequence_shape(self, grid, legendre, kernels, poly_eos):
        from mr_curve import nonrotating_sequence
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        seq = nonrotating_sequence(
            poly_eos, is_tab=False,
            e_min=0.1, e_max=0.8, n_pts=4,
            e_surface=0.0, p_surface=0.0, enthalpy_min=0.0,
            s_gp=s_gp, mu_arr=mu,
            P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk,
        )
        assert seq is not None
        assert seq.shape[0] == 5        # (e_c, M, M0, Re, OK)
        assert seq.shape[1] >= 2       # at least 2 valid models

    @pytest.mark.slow
    def test_sequence_mass_peaks(self, grid, legendre, kernels, poly_eos):
        """Mass reaches a maximum and then decreases for a wide density range."""
        from mr_curve import nonrotating_sequence
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        seq = nonrotating_sequence(
            poly_eos, is_tab=False,
            e_min=0.05, e_max=1.5, n_pts=8,
            e_surface=0.0, p_surface=0.0, enthalpy_min=0.0,
            s_gp=s_gp, mu_arr=mu,
            P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk,
        )
        assert seq is not None
        M_arr = seq[1]
        # Maximum should not be at the endpoints
        imax = np.argmax(M_arr)
        assert 0 < imax < len(M_arr) - 1

    @pytest.mark.slow
    def test_max_mass_poly_value(self, grid, legendre, kernels, poly_eos):
        """M_max for N=1 polytrope ≈ 0.188 code units."""
        from mr_curve import nonrotating_sequence
        s_gp, mu = grid
        P_2n, P1_2n1, sn = legendre
        fk, gk = kernels
        seq = nonrotating_sequence(
            poly_eos, is_tab=False,
            e_min=0.2, e_max=1.2, n_pts=6,
            e_surface=0.0, p_surface=0.0, enthalpy_min=0.0,
            s_gp=s_gp, mu_arr=mu,
            P_2n=P_2n, P1_2n1=P1_2n1, sn=sn, fk=fk, gk=gk,
        )
        assert seq is not None
        assert seq[1].max() == pytest.approx(0.188, rel=0.06)
