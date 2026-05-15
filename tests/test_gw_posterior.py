"""Tests for gw_posterior.py — universal relations and Bilby I/O."""
import json
import os
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Universal Λ–C relations
# ---------------------------------------------------------------------------

class TestLambdaToCompactness:
    @pytest.mark.fast
    def test_de2018_typical(self):
        """De et al. 2018: Λ=400 → C ≈ 0.17–0.20 for a typical NS."""
        from gw_posterior import lambda_to_compactness
        C = lambda_to_compactness(400.0, relation="de2018")
        assert 0.14 < C < 0.24

    @pytest.mark.fast
    def test_yy2013_typical(self):
        from gw_posterior import lambda_to_compactness
        C = lambda_to_compactness(400.0, relation="yy2013")
        assert 0.14 < C < 0.24

    @pytest.mark.fast
    def test_two_relations_agree(self):
        """Both relations agree to within ~10% for typical Λ."""
        from gw_posterior import lambda_to_compactness
        lam = np.array([100.0, 300.0, 600.0, 1000.0])
        C_de = lambda_to_compactness(lam, relation="de2018")
        C_yy = lambda_to_compactness(lam, relation="yy2013")
        assert np.allclose(C_de, C_yy, rtol=0.12), \
            "de2018 and yy2013 differ by more than 12%"

    @pytest.mark.fast
    def test_monotone_decreasing(self):
        """Larger Λ (softer EOS) → smaller C (less compact = larger R/M)."""
        from gw_posterior import lambda_to_compactness
        lam = np.array([50.0, 200.0, 500.0, 1500.0, 4000.0])
        C = lambda_to_compactness(lam)
        assert np.all(np.diff(C) < 0), "C should decrease as Λ increases"

    @pytest.mark.fast
    def test_physical_range(self):
        """C must stay in [0.05, 0.50] for any input Λ."""
        from gw_posterior import lambda_to_compactness
        lam = np.logspace(1, 4, 100)
        C = lambda_to_compactness(lam)
        assert np.all(C >= 0.05)
        assert np.all(C <= 0.50)

    @pytest.mark.fast
    def test_unknown_relation_raises(self):
        from gw_posterior import lambda_to_compactness
        with pytest.raises(ValueError, match="Unknown relation"):
            lambda_to_compactness(400.0, relation="bogus")


class TestLambdaToRadius:
    @pytest.mark.fast
    def test_typical_ns(self):
        """1.4 M☉ with Λ=400 → R ≈ 10–14 km."""
        from gw_posterior import lambda_to_radius
        R = lambda_to_radius(1.4, 400.0)
        assert 10.0 < R < 14.0, f"R = {R:.2f} km is outside 10–14 km"

    @pytest.mark.fast
    def test_larger_lambda_larger_radius(self):
        """Higher Λ (less compact) → larger R at fixed mass."""
        from gw_posterior import lambda_to_radius
        R_soft = lambda_to_radius(1.4, 800.0)
        R_stiff = lambda_to_radius(1.4, 200.0)
        assert R_soft > R_stiff

    @pytest.mark.fast
    def test_heavier_star_larger_radius_at_fixed_lambda(self):
        """
        At fixed Λ the universal relation gives C independent of mass,
        so R = GM/(c²C) ∝ M → heavier star has larger R.
        (The M-R relation slopes only emerge when the EOS is fixed and
        the mass varies, changing both M and Λ simultaneously.)
        """
        from gw_posterior import lambda_to_radius
        R_light = lambda_to_radius(1.2, 400.0)
        R_heavy = lambda_to_radius(2.0, 400.0)
        assert R_heavy > R_light

    @pytest.mark.fast
    def test_array_input(self):
        from gw_posterior import lambda_to_radius
        masses = np.array([1.2, 1.4, 1.6])
        lams   = np.array([500, 400, 250])
        R = lambda_to_radius(masses, lams)
        assert R.shape == (3,)
        assert np.all(R > 0)
        assert np.all(np.isfinite(R))

    @pytest.mark.fast
    def test_gw170817_reference(self):
        """GW170817 90% CI: R(1.4 M☉) = 10.4–13.7 km (Abbott+ 2018)."""
        from gw_posterior import lambda_to_radius
        # Use Λ=400 (central estimate from GW170817)
        R = lambda_to_radius(1.4, 400.0)
        assert 10.0 < R < 14.5, f"R = {R:.2f} km outside expected range"


# ---------------------------------------------------------------------------
# Bilby file I/O
# ---------------------------------------------------------------------------

class TestReadBilby:
    @pytest.mark.fast
    def test_reads_json(self, synthetic_posterior):
        from gw_posterior import read_bilby
        post, meta = read_bilby(synthetic_posterior)
        assert "mass_1" in post
        assert "lambda_1" in post
        assert meta["n_samples"] == 200

    @pytest.mark.fast
    def test_meta_label(self, synthetic_posterior):
        from gw_posterior import read_bilby
        _, meta = read_bilby(synthetic_posterior)
        assert meta.get("label") == "test_event"

    @pytest.mark.fast
    def test_all_samples_arrays(self, synthetic_posterior):
        from gw_posterior import read_bilby
        post, _ = read_bilby(synthetic_posterior)
        for key, val in post.items():
            assert isinstance(val, np.ndarray), f"{key} not ndarray"
            assert len(val) == 200

    @pytest.mark.fast
    def test_csv_reading(self, tmp_path):
        """CSV posterior with expected column names is read correctly."""
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        from gw_posterior import read_bilby
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({
            "mass_1": rng.uniform(1.2, 1.7, n),
            "mass_2": rng.uniform(1.0, 1.4, n),
            "lambda_1": rng.uniform(100, 800, n),
            "lambda_2": rng.uniform(150, 1200, n),
        })
        path = str(tmp_path / "post.csv")
        df.to_csv(path, index=False)
        post, meta = read_bilby(path)
        assert "mass_1" in post
        assert len(post["mass_1"]) == n

    @pytest.mark.fast
    def test_bad_extension_raises(self, tmp_path):
        from gw_posterior import read_bilby
        path = str(tmp_path / "bad.xyz")
        with open(path, "w") as f:
            f.write("dummy")
        with pytest.raises(ValueError, match="Unrecognised file extension"):
            read_bilby(path)


# ---------------------------------------------------------------------------
# Parameter extraction — edge cases
# ---------------------------------------------------------------------------

class TestExtractMassesLambdas:
    @pytest.mark.fast
    def test_chirp_mass_fallback(self, tmp_path):
        """Works when mass_1/mass_2 are absent but chirp_mass+q are present."""
        import json
        from gw_posterior import read_bilby, compute_mr_ppd
        rng = np.random.default_rng(7)
        n   = 30
        Mc  = 1.188 + rng.normal(0, 0.005, n)
        q   = np.clip(0.9 + rng.normal(0, 0.05, n), 0.5, 1.0)
        doc = {
            "label": "chirp_only",
            "posterior": {"content": {
                "chirp_mass": Mc.tolist(),
                "mass_ratio": q.tolist(),
                "lambda_1": (300 + rng.uniform(0, 200, n)).tolist(),
                "lambda_2": (450 + rng.uniform(0, 300, n)).tolist(),
            }},
        }
        path = str(tmp_path / "chirp.json")
        with open(path, "w") as f:
            json.dump(doc, f)
        post, _ = read_bilby(path)
        M, R, labels = compute_mr_ppd(post, component="1")
        assert len(M) == n
        assert np.all(M > 0)

    @pytest.mark.fast
    def test_missing_lambda_warns(self, tmp_path, capsys):
        """A warning is printed when Λ is absent for a component."""
        import json
        from gw_posterior import compute_mr_ppd
        rng = np.random.default_rng(8)
        n   = 20
        doc = {
            "posterior": {"content": {
                "mass_1": rng.uniform(1.2, 1.6, n).tolist(),
                # No lambda_1 at all
            }},
        }
        post = json.loads(json.dumps(doc))["posterior"]["content"]
        post = {k: np.array(v) for k, v in post.items()}
        with pytest.raises(ValueError):
            compute_mr_ppd(post, component="1")


# ---------------------------------------------------------------------------
# Compute PPD
# ---------------------------------------------------------------------------

class TestComputeMrPPD:
    @pytest.mark.fast
    def test_output_shapes(self, synthetic_posterior):
        from gw_posterior import read_bilby, compute_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        M, R, labels = compute_mr_ppd(post, component="both")
        assert len(M) == len(R) == len(labels)
        assert len(M) == 400    # 200 samples × 2 components

    @pytest.mark.fast
    def test_physical_radii(self, synthetic_posterior):
        """All radii should be in 5–25 km."""
        from gw_posterior import read_bilby, compute_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        M, R, labels = compute_mr_ppd(post)
        assert np.all(R > 5),  "Some radii below 5 km"
        assert np.all(R < 25), "Some radii above 25 km"

    @pytest.mark.fast
    def test_physical_masses(self, synthetic_posterior):
        from gw_posterior import read_bilby, compute_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        M, R, labels = compute_mr_ppd(post)
        assert np.all(M > 0.8)
        assert np.all(M < 3.0)

    @pytest.mark.fast
    def test_component_labels(self, synthetic_posterior):
        from gw_posterior import read_bilby, compute_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        M, R, labels = compute_mr_ppd(post, component="both")
        assert set(labels) == {"1", "2"}

    @pytest.mark.fast
    def test_thinning(self, synthetic_posterior):
        from gw_posterior import read_bilby, compute_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        M, R, _ = compute_mr_ppd(post, n_samples=50)
        assert len(M) == 100    # 50 per component

    @pytest.mark.fast
    def test_both_relations(self, synthetic_posterior):
        """de2018 and yy2013 give similar median radii (within 10%)."""
        from gw_posterior import read_bilby, compute_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        _, R_de, _ = compute_mr_ppd(post, relation="de2018")
        _, R_yy, _ = compute_mr_ppd(post, relation="yy2013")
        assert np.median(R_de) == pytest.approx(np.median(R_yy), rel=0.10)


# ---------------------------------------------------------------------------
# Plotting (no window, just confirm file written)
# ---------------------------------------------------------------------------

class TestPlotMrPPD:
    @pytest.mark.fast
    def test_pdf_written(self, synthetic_posterior, tmp_path):
        from gw_posterior import read_bilby, compute_mr_ppd, plot_mr_ppd
        post, meta = read_bilby(synthetic_posterior)
        M, R, labels = compute_mr_ppd(post, n_samples=100)
        out = str(tmp_path / "test_ppd.pdf")
        plot_mr_ppd(M, R, labels, savefig=out, event_label="test")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 10_000   # non-trivial PDF

    @pytest.mark.fast
    def test_single_credible_level(self, synthetic_posterior, tmp_path):
        from gw_posterior import read_bilby, compute_mr_ppd, plot_mr_ppd
        post, _ = read_bilby(synthetic_posterior)
        M, R, labels = compute_mr_ppd(post, n_samples=80)
        out = str(tmp_path / "ppd_one_level.pdf")
        plot_mr_ppd(M, R, labels, savefig=out,
                    credible_levels=(0.90,))
        assert os.path.exists(out)
