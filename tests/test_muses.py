"""Tests for muses_eos.py — EOS parsing and conversion (offline only)."""
import json
import os
import numpy as np
import pytest

# All tests that touch the network are marked `network` so CI can skip them
# with:  pytest -m "not network"


# ---------------------------------------------------------------------------
# parse_compose — offline, uses cached CompOSE files if present
# ---------------------------------------------------------------------------

_SK255_DIR = os.path.join(os.path.dirname(__file__), "..", "eos", "RG_SK255_")
_HAS_SK255 = all(
    os.path.exists(os.path.join(_SK255_DIR, f))
    for f in ("eos.nb", "eos.thermo")
)


class TestParseCompose:
    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_returns_dict(self):
        from muses_eos import parse_compose
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        for key in ("e", "p", "h", "n0", "n_pts", "m_ref_MeV"):
            assert key in data

    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_physical_density_range(self):
        from muses_eos import parse_compose
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        # Energy density should be in NS range: 8e13 – 8e15 g/cm³
        assert data["e"].min() > 8e13
        assert data["e"].max() < 8e15

    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_monotone_arrays(self):
        from muses_eos import parse_compose
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        assert np.all(np.diff(data["e"]) > 0), "e not monotone"
        assert np.all(np.diff(data["p"]) > 0), "p not monotone"
        assert np.all(np.diff(data["h"]) >= 0), "h not monotone"

    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_enthalpy_non_negative(self):
        from muses_eos import parse_compose
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        assert np.all(data["h"] >= 0)

    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_n_pts_reasonable(self):
        from muses_eos import parse_compose
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        assert data["n_pts"] > 30


class TestWriteRnsEos:
    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_file_written(self, tmp_path):
        from muses_eos import parse_compose, write_rns_eos
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        out = str(tmp_path / "test.eos")
        write_rns_eos(data, out)
        assert os.path.exists(out)

    @pytest.mark.fast
    @pytest.mark.skipif(not _HAS_SK255, reason="SK255 CompOSE files not present")
    def test_file_parseable_by_load_eos(self, tmp_path):
        """Written file can be re-loaded by pyRNS.load_eos."""
        from muses_eos import parse_compose, write_rns_eos
        from pyRNS import load_eos
        data = parse_compose(_SK255_DIR, n_min_fm3=0.05)
        out = str(tmp_path / "test.eos")
        write_rns_eos(data, out)
        log_e, log_p, log_h, log_n0, n_tab = load_eos(out)
        assert n_tab == data["n_pts"]
        assert np.all(np.diff(log_e) > 0)
        assert np.all(np.diff(log_p) > 0)


# ---------------------------------------------------------------------------
# Cached eos.mr files (offline)
# ---------------------------------------------------------------------------

_MR_FILES = {
    "SK255": os.path.join(_SK255_DIR, "eos.mr"),
}


class TestCachedMrData:
    @pytest.mark.fast
    @pytest.mark.skipif(
        not os.path.exists(_MR_FILES["SK255"]),
        reason="SK255 eos.mr not cached")
    def test_load_mr_data_returns_arrays(self):
        from muses_eos import _load_mr_data
        R, M, label, txt = _load_mr_data("RG(SK255)", outdir="eos/")
        assert len(R) > 10
        assert len(M) == len(R)
        assert np.all(R > 5) and np.all(R < 25)

    @pytest.mark.fast
    @pytest.mark.skipif(
        not os.path.exists(_MR_FILES["SK255"]),
        reason="SK255 eos.mr not cached")
    def test_mmax_value(self):
        from muses_eos import _load_mr_data
        R, M, *_ = _load_mr_data("RG(SK255)", outdir="eos/")
        assert M.max() == pytest.approx(2.144, rel=0.01)

    @pytest.mark.fast
    @pytest.mark.skipif(
        not os.path.exists(_MR_FILES["SK255"]),
        reason="SK255 eos.mr not cached")
    def test_plot_mr_multi_writes_pdf(self, tmp_path):
        from muses_eos import _load_mr_data, plot_mr_multi
        R, M, label, _ = _load_mr_data("RG(SK255)", outdir="eos/")
        out = str(tmp_path / "mr_test.pdf")
        plot_mr_multi([(R, M, label)], savefig=out)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 10_000


# ---------------------------------------------------------------------------
# Network-dependent tests (skipped in CI by default)
# ---------------------------------------------------------------------------

class TestNetwork:
    @pytest.mark.network
    def test_list_eos_returns_catalog(self):
        from muses_eos import list_eos
        cat = list_eos(verbose=False, refresh=True)
        assert len(cat) > 50
        has_thermo = [e for e in cat if "thermo_path" in e]
        assert len(has_thermo) > 30

    @pytest.mark.network
    def test_find_eos_by_name(self):
        from muses_eos import list_eos, find_eos
        cat = list_eos(verbose=False)
        entry = find_eos("RG(SK255)", cat)
        assert entry["id"] == 94
        assert "base_path" in entry

    @pytest.mark.network
    def test_find_eos_by_id(self):
        from muses_eos import list_eos, find_eos
        cat = list_eos(verbose=False)
        entry = find_eos(94, cat)
        assert "SK255" in entry["name"]

    @pytest.mark.network
    def test_ambiguous_name_raises(self):
        from muses_eos import list_eos, find_eos
        cat = list_eos(verbose=False)
        with pytest.raises(ValueError, match="Ambiguous"):
            find_eos("DD2", cat)

    @pytest.mark.network
    def test_missing_name_raises(self):
        from muses_eos import list_eos, find_eos
        cat = list_eos(verbose=False)
        with pytest.raises(ValueError, match="No EOS matching"):
            find_eos("COMPLETELYMADEUPEOS", cat)
