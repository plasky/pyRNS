"""
muses_eos.py  —  MUSES / CompOSE equation-of-state interface for pyRNS
=======================================================================

Downloads cold, beta-equilibrated neutron-star EOS tables from the
CompOSE database (https://compose.obspm.fr), converts them to the RNS
tabulated format, and runs either pyRNS (single model) or mr_curve
(mass-radius sequence).

Usage
-----
  # List all available cold-NS EOS
  python3 muses_eos.py list

  # Run a single-star model
  python3 muses_eos.py run --eos "RG(SK255)" --e 1e15
  python3 muses_eos.py run --eos-id 94 --e 1e15

  # Build mass-radius sequence
  python3 muses_eos.py mrseq --eos "RG(SK255)" --npts 20

  # Download EOS only (no run)
  python3 muses_eos.py fetch --eos "RG(SK255)" --outdir eos/

CompOSE EOS format (eos.thermo columns, after indices it/inb/iyq)
-----------------------------------------------------------------
  col 0  :  p         [MeV fm⁻³]   pressure
  col 1  :  s/n_B = 0               entropy per baryon (0 at T=0)
  col 2  :  (ε/n_B − m_ref)/m_ref   normalised excess energy
  col 3  :  μ_Q / m_ref             charge chemical potential
  col 4  :  μ_L / m_ref             lepton chemical potential
  col 5/6:  (μ_B − m_ref)/m_ref     baryon chemical potential (repeated)

Physical enthalpy: h = (ε + p)/n_B = μ_B  (Gibbs-Duhem, T=0)

Unit conversions
----------------
  ε [g cm⁻³]    = ε [MeV fm⁻³] × 1.7827×10¹²
  p [dyne cm⁻²] = p [MeV fm⁻³] × 1.6022×10³³
  h [cm² s⁻²]   = μ_B [MeV]    × 9.648×10¹⁷
  n₀ [cm⁻³]     = n_B [fm⁻³]   × 10³⁹
"""

import argparse
import os
import sys
import re
import urllib.request
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (match pyRNS conventions exactly)
# ---------------------------------------------------------------------------
_MeV_fm3_to_g_cm3  = 1.7827e12    # 1 MeV/fm³ → g/cm³  (= 1 MeV/fm³ / c²)
_MeV_fm3_to_dyn_cm2= 1.6022e33    # 1 MeV/fm³ → dyne/cm²
_MeV_to_cm2_s2     = 9.6478e17    # 1 MeV → cm²/s²  (= MeV / m_baryon)
_fm3_to_cm3        = 1e-39         # 1 fm⁻³ → cm⁻³  (since 1 fm = 1e-13 cm)

_COMPOSE_BASE = "https://compose.obspm.fr"

# JSON file that caches the CompOSE EOS catalog after the first fetch.
# Stored inside the package's eos/ directory so it travels with the code.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_CATALOG_CACHE = os.path.join(_PKG_DIR, "eos", "_catalog_cache.json")

# ---------------------------------------------------------------------------
# CompOSE HTTP helpers
# ---------------------------------------------------------------------------

def _fetch(url, timeout=30):
    """Fetch URL and return text content."""
    req = urllib.request.Request(url, headers={"User-Agent": "pyRNS-muses/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def list_eos(family="cold_ns", verbose=True, refresh=False):
    """
    Return a list of EOS entry dicts for available CompOSE cold-NS EOS.

    The catalog is cached to ``eos/_catalog_cache.json`` after the first
    network fetch so that subsequent calls are instant and work offline.

    Parameters
    ----------
    family  : str   (unused — kept for API compatibility)
    verbose : bool  print a formatted table of EOS names
    refresh : bool  ignore the cache and re-fetch from CompOSE
    """
    import json

    # ── load from cache if available ──────────────────────────────────────
    if not refresh and os.path.exists(_CATALOG_CACHE):
        with open(_CATALOG_CACHE) as fh:
            catalog = json.load(fh)
        if verbose:
            print(f"\n{'ID':>5}  {'Name':<45}")
            print("-" * 52)
            for e in catalog:
                if "thermo_path" in e:
                    print(f"{e['id']:>5}  {e['name']:<45}")
            n = sum(1 for e in catalog if "thermo_path" in e)
            print(f"\n{n} EOS with tabulated data  "
                  f"(cached — use --refresh to update)")
        return catalog

    # ── fetch from CompOSE ────────────────────────────────────────────────
    if verbose:
        print("Fetching EOS catalog from CompOSE …", flush=True)

    html = _fetch(_COMPOSE_BASE + "/table/")
    links = re.findall(r'hx-get="(/eos/(\d+))"', html)
    seen = set()
    catalog = []
    for path, eid in links:
        if eid in seen:
            continue
        seen.add(eid)
        catalog.append({"id": int(eid), "url_path": path})

    for entry in catalog[:200]:
        try:
            page = _fetch(_COMPOSE_BASE + entry["url_path"])
            names    = re.findall(r'<h[12][^>]*>([^<]{3,80})<', page)
            dl_paths = re.findall(r'href="(/download/[^"]+)"', page)
            entry["name"]      = names[0].strip() if names else f"EOS-{entry['id']}"
            entry["downloads"] = dl_paths
            for p in dl_paths:
                if p.endswith("eos.thermo"):
                    entry["thermo_path"] = p
                    entry["base_path"]   = p[:-len("eos.thermo")]
                    break
        except Exception:
            entry["name"]      = f"EOS-{entry['id']}"
            entry["downloads"] = []

    # ── save catalog to cache ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(_CATALOG_CACHE), exist_ok=True)
    with open(_CATALOG_CACHE, "w") as fh:
        json.dump(catalog, fh, indent=2)
    if verbose:
        print(f"  Catalog cached → {_CATALOG_CACHE}")

    if verbose:
        print(f"\n{'ID':>5}  {'Name':<45}")
        print("-" * 52)
        for e in catalog:
            if "thermo_path" in e:
                print(f"{e['id']:>5}  {e['name']:<45}")
        print(f"\n{sum(1 for e in catalog if 'thermo_path' in e)} EOS with tabulated data")

    return catalog


def find_eos(name_or_id, catalog=None):
    """Find an EOS entry by name (substring match) or numeric ID."""
    if catalog is None:
        catalog = list_eos(verbose=False)

    # Try numeric ID first
    try:
        eos_id = int(name_or_id)
        for e in catalog:
            if e["id"] == eos_id and "base_path" in e:
                return e
        raise ValueError(f"EOS id={eos_id} not found or has no tabulated data")
    except (ValueError, TypeError):
        pass

    # Substring name match
    name_lower = str(name_or_id).lower()
    matches = [e for e in catalog
               if "base_path" in e and name_lower in e.get("name", "").lower()]
    if not matches:
        raise ValueError(
            f"No EOS matching '{name_or_id}'. "
            f"Run 'python3 muses_eos.py list' to see available EOS."
        )
    if len(matches) > 1:
        names = [m["name"] for m in matches]
        raise ValueError(
            f"Ambiguous EOS name '{name_or_id}'. Matches:\n  " +
            "\n  ".join(f"[{m['id']}] {m['name']}" for m in matches)
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Download CompOSE files
# ---------------------------------------------------------------------------

def download_eos(eos_entry, outdir="eos/", force=False):
    """
    Download eos.nb, eos.t, eos.yq, eos.thermo for the given EOS entry.

    Files that already exist on disk are reused without a network request
    unless *force=True*.  Each file's status is reported as either
    "cached" or "downloaded".

    Returns the local directory where files were saved.
    """
    base_url  = _COMPOSE_BASE + eos_entry["base_path"]
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", eos_entry["name"])
    local_dir = os.path.join(outdir, safe_name)
    os.makedirs(local_dir, exist_ok=True)

    files = ["eos.t", "eos.nb", "eos.yq", "eos.thermo"]
    for fname in files:
        local_path = os.path.join(local_dir, fname)
        if os.path.exists(local_path) and not force:
            print(f"  Using cached  {fname}")
            continue
        url = base_url + fname
        try:
            content = _fetch(url)
            with open(local_path, "w") as fh:
                fh.write(content)
            print(f"  Downloaded    {fname} ({len(content):,} bytes)")
        except Exception as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    return local_dir


# ---------------------------------------------------------------------------
# Parse CompOSE files → physical arrays
# ---------------------------------------------------------------------------

def parse_compose(local_dir, n_min_fm3=1e-4, n_max_fm3=None):
    """
    Read CompOSE eos.thermo and eos.nb and return a dict with physical arrays.

    Column layout of eos.thermo (after it, inb, iyq indices):
      col 0 : p [MeV/fm³]
      col 1 : s/n_B = 0 (T = 0)
      col 2 : (ε/n_B − m_ref) / m_ref
      col 3 : μ_Q / m_ref
      col 4 : μ_L / m_ref
      col 5 : (μ_B − m_ref) / m_ref   ← baryon chemical potential

    Units
    -----
    Returned arrays are in CGS / RNS-native units:
      e   : g cm⁻³
      p   : dyne cm⁻²
      h   : cm² s⁻²   (specific enthalpy = baryon chem. potential)
      n0  : cm⁻³
    """
    # --- baryon density grid ---
    with open(os.path.join(local_dir, "eos.nb")) as fh:
        nb_lines = fh.read().strip().split("\n")
    n_nb = int(nb_lines[1])
    nb_arr = np.array([float(x.strip()) for x in nb_lines[2:2 + n_nb]])  # fm⁻³

    # --- thermodynamic quantities ---
    with open(os.path.join(local_dir, "eos.thermo")) as fh:
        thermo_lines = fh.read().strip().split("\n")

    header = thermo_lines[0].split()
    m_ref = float(header[0])   # neutron reference mass [MeV]

    rows = {}
    for line in thermo_lines[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        it, inb, iyq = int(parts[0]), int(parts[1]), int(parts[2])
        cols = [float(x) for x in parts[3:10]]   # up to 7 thermo quantities
        rows[inb] = cols

    # Build EOS table
    e_list, p_list, h_list, n0_list = [], [], [], []

    for inb in sorted(rows.keys()):
        n_B = nb_arr[inb - 1]          # fm⁻³
        if n_B < n_min_fm3:
            continue
        if n_max_fm3 is not None and n_B > n_max_fm3:
            break

        c = rows[inb]
        p_MeV  = c[0]                              # pressure [MeV/fm³]
        eps_MeV = n_B * m_ref * (1.0 + c[2])      # energy density [MeV/fm³]

        if p_MeV <= 0 or eps_MeV <= 0:
            continue

        # Convert pressure and energy density to CGS
        e_cgs  = eps_MeV * _MeV_fm3_to_g_cm3     # g/cm³
        p_cgs  = p_MeV   * _MeV_fm3_to_dyn_cm2   # dyne/cm²
        n0_cgs = n_B     / _fm3_to_cm3            # cm⁻³

        e_list.append(e_cgs); p_list.append(p_cgs)
        n0_list.append(n0_cgs)

    if len(e_list) < 5:
        raise ValueError(
            f"Too few EOS points ({len(e_list)}) above n_min={n_min_fm3} fm⁻³. "
            "Try lowering --n-min."
        )

    e_arr  = np.array(e_list)
    p_arr  = np.array(p_list)
    n0_arr = np.array(n0_list)

    # ---- Compute the RNS enthalpy by numerical integration ----------------
    # The RNS/CompOSE h [cm²/s²] is the thermodynamic integral
    #   h(p) = ∫₀ᵖ dp' / (ε(p') + p')  ×  c²
    # This equals zero at the stellar surface (p = 0) and grows monotonically
    # into the core.  For a tabulated EOS h is computed by trapezoid rule.
    #
    # Unit notes (CGS):
    #   ε in g/cm³,  p in dyne/cm² = g/(cm s²)
    #   dp / (ε c² + p) is dimensionless  →  × c² gives cm²/s²

    c_sq   = (2.9979e10) ** 2   # c² [cm²/s²]
    h_arr  = np.zeros(len(p_arr))
    # integrand: dp / (ε·c² + p)  — note ε [g/cm³] × c² [cm²/s²] = dyne/cm²
    for i in range(1, len(p_arr)):
        dp       = p_arr[i]  - p_arr[i - 1]
        eps_mid  = 0.5 * (e_arr[i] + e_arr[i - 1]) * c_sq   # dyne/cm²
        p_mid    = 0.5 * (p_arr[i] + p_arr[i - 1])
        h_arr[i] = h_arr[i - 1] + dp / (eps_mid + p_mid)

    h_arr *= c_sq          # dimensionless integral → cm²/s²

    # --- Enforce physical constraints -----------------------------------
    # Trim to the monotone region starting from the first point where
    # both p and e are increasing and h ≥ 0. This removes the low-density
    # outer crust where pressure can be non-monotone (spinodal region).
    h_arr = np.maximum(h_arr, 0.0)

    # Find the first index from which e, p, h are all simultaneously
    # monotonically increasing (march forward from the dense end)
    n = len(e_arr)
    # First ensure e is sorted; keep only points with e strictly increasing
    mono_e = np.concatenate(([True], np.diff(e_arr) > 0))
    e_arr  = e_arr[mono_e];  p_arr  = p_arr[mono_e]
    h_arr  = h_arr[mono_e];  n0_arr = n0_arr[mono_e]

    # Recompute h on the cleaned e-monotone grid
    h_arr = np.zeros(len(p_arr))
    for i in range(1, len(p_arr)):
        dp       = p_arr[i]  - p_arr[i - 1]
        eps_mid  = 0.5 * (e_arr[i] + e_arr[i - 1]) * c_sq
        p_mid    = 0.5 * (p_arr[i] + p_arr[i - 1])
        dh       = dp / max(eps_mid + p_mid, 1e-100)
        h_arr[i] = h_arr[i - 1] + max(dh, 0.0)   # never let h decrease
    h_arr *= c_sq

    # Drop points where h has not yet started increasing (crust plateau)
    first_positive = np.argmax(h_arr > 0)
    e_arr  = e_arr[first_positive:]
    p_arr  = p_arr[first_positive:]
    h_arr  = h_arr[first_positive:]
    n0_arr = n0_arr[first_positive:]

    if len(e_arr) < 5:
        raise ValueError(
            "Too few usable EOS points after filtering. "
            "Try lowering --n-min or check the EOS file."
        )

    result = {
        "e":  e_arr,
        "p":  p_arr,
        "h":  h_arr,
        "n0": n0_arr,
        "m_ref_MeV": m_ref,
        "n_pts": len(e_arr),
    }
    return result


# ---------------------------------------------------------------------------
# Write RNS tabulated EOS file
# ---------------------------------------------------------------------------

def write_rns_eos(data, outfile):
    """
    Write a tabulated EOS in RNS/pyRNS format:
      n_pts
      e[g/cm³]   p[dyne/cm²]   h[cm²/s²]   n0[cm⁻³]
      ...
    """
    n = data["n_pts"]
    with open(outfile, "w") as fh:
        fh.write(f"{n}\n")
        for i in range(n):
            fh.write(
                f"{data['e'][i]:.8e} "
                f"{data['p'][i]:.8e} "
                f"{data['h'][i]:.8e} "
                f"{data['n0'][i]:.8e}\n"
            )
    print(f"  Wrote RNS EOS file: {outfile}  ({n} points)")


# ---------------------------------------------------------------------------
# High-level: fetch → convert → run pyRNS or mr_curve
# ---------------------------------------------------------------------------

def fetch_and_convert(name_or_id, outdir="eos/", n_min=1e-4,
                      n_max=None, force=False):
    """
    Full pipeline: find EOS → download → parse → write RNS file.

    Each step is cached:
    - Catalog  : loaded from ``eos/_catalog_cache.json`` if present
    - Raw files: reused from disk if already downloaded (eos.thermo, etc.)
    - RNS file : ``<name>_rns.eos`` reused if it is newer than eos.thermo

    Pass *force=True* to bypass all caches and redo every step.

    Returns (rns_file_path, eos_name).
    """
    print(f"\n[1/3] Looking up '{name_or_id}' in CompOSE catalog …")
    catalog  = list_eos(verbose=False)
    entry    = find_eos(name_or_id, catalog)
    print(f"  Found: [{entry['id']}] {entry['name']}")

    print(f"[2/3] Fetching CompOSE raw files …")
    local_dir = download_eos(entry, outdir=outdir, force=force)

    # ── check whether the converted RNS file is already up-to-date ────────
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", entry["name"])
    rns_file  = os.path.join(local_dir, f"{safe_name}_rns.eos")
    thermo    = os.path.join(local_dir, "eos.thermo")

    rns_is_fresh = (
        os.path.exists(rns_file)
        and os.path.exists(thermo)
        and os.path.getmtime(rns_file) >= os.path.getmtime(thermo)
        and not force
    )

    if rns_is_fresh:
        print(f"[3/3] Using cached RNS EOS file: {rns_file}")
    else:
        print(f"[3/3] Converting to RNS tabulated format …")
        data = parse_compose(local_dir, n_min_fm3=n_min, n_max_fm3=n_max)
        write_rns_eos(data, rns_file)
        print(f"  {data['n_pts']} points, "
              f"ε ∈ [{data['e'].min():.2e}, {data['e'].max():.2e}] g/cm³, "
              f"p ∈ [{data['p'].min():.2e}, {data['p'].max():.2e}] dyne/cm²")

    return rns_file, entry["name"]


def run_star(name_or_id, e_center_g_cm3=1e15, outdir="eos/",
             n_min=1e-4, force=False, plot=True, savefig=None, verbose=False):
    """
    Fetch CompOSE EOS and run a single non-rotating neutron-star model
    using pyRNS.
    """
    rns_file, eos_label = fetch_and_convert(
        name_or_id, outdir=outdir, n_min=n_min, force=force
    )

    print(f"\nRunning pyRNS for {eos_label} at ε_c = {e_center_g_cm3:.2e} g/cm³ …")
    cmd = (
        f"python3 '{os.path.join(os.path.dirname(__file__), 'pyRNS.py')}'"
        f" -f '{rns_file}' -e {e_center_g_cm3}"
        f"{' --no-plot' if not plot else ''}"
        f"{' --savefig ' + repr(savefig) if savefig else ''}"
        f"{' --verbose' if verbose else ''}"
    )
    print(f"  {cmd}")
    os.system(cmd)


def _load_mr_data(name_or_id, outdir="eos/", force=False):
    """
    Fetch (or load from cache) the eos.mr table for one EOS.

    Returns
    -------
    R_arr  : np.ndarray  equatorial radii [km]
    M_arr  : np.ndarray  gravitational masses [M_☉]
    label  : str         short EOS name for plot legends
    txt_out: str         path to saved text table
    """
    catalog   = list_eos(verbose=False)
    entry     = find_eos(name_or_id, catalog)
    label     = entry["name"]
    base_url  = _COMPOSE_BASE + entry["base_path"]
    safe_name = re.sub(r"[^A-Za-z0-9_\-]", "_", label)
    local_dir = os.path.join(outdir, safe_name)
    os.makedirs(local_dir, exist_ok=True)

    mr_file = os.path.join(local_dir, "eos.mr")
    if os.path.exists(mr_file) and not force:
        print(f"  [{label}]  Using cached  eos.mr")
    else:
        try:
            content = _fetch(base_url + "eos.mr")
            with open(mr_file, "w") as fh:
                fh.write(content)
            print(f"  [{label}]  Downloaded    eos.mr ({len(content):,} bytes)")
        except Exception as exc:
            raise RuntimeError(f"Could not download eos.mr for '{label}': {exc}") from exc

    R_arr, M_arr = [], []
    with open(mr_file) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.split()
            if len(vals) >= 2:
                try:
                    R_arr.append(float(vals[0]))
                    M_arr.append(float(vals[1]))
                except ValueError:
                    pass

    if not R_arr:
        raise ValueError(f"eos.mr for '{label}' contains no valid data")

    R_arr = np.array(R_arr); M_arr = np.array(M_arr)
    imax  = np.argmax(M_arr)
    print(f"           {len(M_arr)} models  "
          f"M_max = {M_arr.max():.3f} M_☉  at R = {R_arr[imax]:.2f} km")

    txt_out = os.path.join(local_dir, f"{safe_name}_mr.txt")
    np.savetxt(txt_out,
               np.column_stack([R_arr, M_arr]),
               header="R [km]   M [M_sun]",
               fmt="%.6e")
    print(f"           Saved → {txt_out}")

    return R_arr, M_arr, label, txt_out


def plot_mr_multi(datasets, savefig, title=None):
    """
    Plot M–R curves for one or more EOS on a single figure.

    Parameters
    ----------
    datasets : list of (R_arr, M_arr, label) tuples
    savefig  : str   path for saved PDF (required)
    title    : str   optional figure title
    """
    try:
        from pyRNS import _apply_style
        _apply_style()
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    # Colour palette: use a perceptually-uniform cycle
    n     = len(datasets)
    if n == 1:
        colours = ["#1a73e8"]
    else:
        colours = [cm.tab10(i / max(n - 1, 1)) for i in range(n)]

    G = 6.6732e-8; c_light = 2.9979e10; Msun = 1.989e33

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax_mr  = axes[0]
    ax_cmp = axes[1]

    for (R_arr, M_arr, label), col in zip(datasets, colours):
        imax    = np.argmax(M_arr)
        compact = G * (M_arr * Msun) / (c_light**2 * (R_arr * 1e5))

        ax_mr.plot(R_arr, M_arr, "-", lw=1.8, color=col, label=label)
        ax_mr.plot(R_arr[imax], M_arr[imax], "*", ms=10, color=col)

        ax_cmp.plot(R_arr, compact, "-", lw=1.8, color=col, label=label)
        ax_cmp.plot(R_arr[imax], compact[imax], "*", ms=10, color=col)

    ax_mr.set_xlabel(r"$R_e$  [km]")
    ax_mr.set_ylabel(r"$M$  [$M_\odot$]")
    ax_mr.set_title("Mass–Radius diagram")
    ax_mr.legend(fontsize=8, loc="upper left")
    ax_mr.grid(True, alpha=0.25)

    ax_cmp.axhline(4 / 9, ls="--", color="gray", alpha=0.6, label="Buchdahl limit")
    ax_cmp.set_xlabel(r"$R_e$  [km]")
    ax_cmp.set_ylabel(r"Compactness  $GM/(c^2 R)$")
    ax_cmp.set_title("Compactness")
    ax_cmp.legend(fontsize=8, loc="upper right")
    ax_cmp.grid(True, alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=11)
    elif n == 1:
        fig.suptitle(f"{datasets[0][2]}  (CompOSE pre-computed M–R)", fontsize=11)
    else:
        fig.suptitle("CompOSE neutron-star EOS comparison", fontsize=11)

    fig.savefig(savefig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {savefig}")


def run_mrseq(names_or_ids, outdir="eos/", force=False, plot=True, savefig=None):
    """
    Fetch and plot pre-computed mass-radius sequences from CompOSE eos.mr.

    Accepts one EOS or a list of EOS — all curves are plotted on one figure.

    Parameters
    ----------
    names_or_ids : str | int | list
        One or more EOS names (substring match) or CompOSE numeric IDs.
    outdir       : str   directory for downloaded/cached files
    force        : bool  bypass cache and re-download
    plot         : bool  produce M–R plot (default True)
    savefig      : str   path for saved PDF; auto-generated if None
    """
    if not isinstance(names_or_ids, (list, tuple)):
        names_or_ids = [names_or_ids]

    print(f"\nFetching M–R data for {len(names_or_ids)} EOS …")
    datasets = []
    for noi in names_or_ids:
        try:
            R, M, label, _ = _load_mr_data(noi, outdir=outdir, force=force)
            datasets.append((R, M, label))
        except Exception as exc:
            print(f"  WARNING: skipping '{noi}': {exc}")

    if not datasets:
        raise RuntimeError("No valid M–R data could be loaded.")

    if not plot:
        return

    # Auto-generate savefig path
    if savefig is None:
        if len(datasets) == 1:
            safe = re.sub(r"[^A-Za-z0-9_\-]", "_", datasets[0][2])
            savefig = os.path.join(outdir, safe, f"{safe}_mr.pdf")
        else:
            safe = "_".join(re.sub(r"[^A-Za-z0-9_\-]", "_", d[2])[:12]
                            for d in datasets)
            savefig = os.path.join(outdir, f"mr_comparison_{safe}.pdf")

    plot_mr_multi(datasets, savefig=savefig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MUSES / CompOSE EOS interface for pyRNS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  list              List all available cold-NS EOS in CompOSE
  fetch             Download and convert an EOS (no run)
  run               Download EOS and run a single-star model
  mrseq             Download EOS and compute mass-radius sequence

Examples:
  python3 muses_eos.py list
  python3 muses_eos.py run   --eos "RG(SK255)" --e 1e15
  python3 muses_eos.py run   --eos-id 94 --e 5e14
  python3 muses_eos.py fetch --eos "RG(SK255)" --outdir eos/

  # Single EOS M-R sequence
  python3 muses_eos.py mrseq --eos "HS(DD2)"

  # Multiple EOS on one plot  (names or IDs, mix is fine)
  python3 muses_eos.py mrseq --eos "RG(SK255)" "HS(DD2)" "DS(CMF)-5"
  python3 muses_eos.py mrseq --eos-id 94 184 --savefig comparison.pdf
""",
    )
    parser.add_argument("command", choices=["list", "fetch", "run", "mrseq"])

    # EOS selection — mrseq accepts multiple, run/fetch accept exactly one
    parser.add_argument("--eos",    metavar="NAME",  nargs="+",
                        help="one or more EOS names (substring match); "
                             "mrseq plots all on one figure")
    parser.add_argument("--eos-id", metavar="ID",    nargs="+", type=int,
                        help="one or more CompOSE numeric EOS IDs")

    # Common options
    parser.add_argument("--outdir",  default="eos/", help="directory for downloaded files")
    parser.add_argument("--n-min",   type=float, default=1e-4,
                        help="minimum baryon density to include [fm⁻³]  (default 1e-4)")
    parser.add_argument("--n-max",   type=float, default=None,
                        help="maximum baryon density [fm⁻³]")
    parser.add_argument("--force",   action="store_true",
                        help="bypass all caches and re-download/re-convert everything")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch the EOS catalog from CompOSE (clears catalog cache)")

    # run / mrseq options
    parser.add_argument("--e",      type=float, default=1e15,
                        help="central energy density [g/cm³] for 'run'  (default 1e15)")
    parser.add_argument("--npts",   type=int,   default=25,
                        help="(unused — mrseq uses CompOSE pre-computed eos.mr)")
    parser.add_argument("--savefig", metavar="FILE",
                        help="save figure to FILE (PDF)")
    parser.add_argument("--no-plot", dest="plot", action="store_false", default=True,
                        help="suppress plots")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ── list ─────────────────────────────────────────────────────────────
    if args.command == "list":
        list_eos(verbose=True, refresh=args.refresh or args.force)
        return

    # Build EOS identifier(s) — names take priority over IDs if both given
    if args.eos and args.eos_id:
        parser.error("use --eos or --eos-id, not both")
    identifiers = args.eos or args.eos_id   # list or None
    if not identifiers:
        parser.error(f"'{args.command}' requires --eos NAME [NAME …] or --eos-id ID [ID …]")

    # ── fetch / run require exactly one EOS ──────────────────────────────
    if args.command in ("fetch", "run") and len(identifiers) > 1:
        parser.error(f"'{args.command}' accepts only one EOS; "
                     f"use 'mrseq' to process multiple EOS at once")

    single = identifiers[0]   # for fetch/run

    # ── fetch ─────────────────────────────────────────────────────────────
    if args.command == "fetch":
        fetch_and_convert(single, outdir=args.outdir,
                          n_min=args.n_min, n_max=args.n_max, force=args.force)
        return

    # ── run ───────────────────────────────────────────────────────────────
    if args.command == "run":
        run_star(single, e_center_g_cm3=args.e,
                 outdir=args.outdir, n_min=args.n_min,
                 force=args.force, plot=args.plot,
                 savefig=args.savefig, verbose=args.verbose)
        return

    # ── mrseq — one or many EOS ───────────────────────────────────────────
    if args.command == "mrseq":
        run_mrseq(identifiers,
                  outdir=args.outdir,
                  force=args.force, plot=args.plot,
                  savefig=args.savefig)


if __name__ == "__main__":
    main()
