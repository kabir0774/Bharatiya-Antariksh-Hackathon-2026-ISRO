from .common import *

# =============================================================================
#  SECTION 0  --  AUTO-SYNC: EXTRACT NEW ZIPS INTO THE DATA FOLDERS
# =============================================================================
#  How it works (plain words): every zip we extract gets remembered in a small
#  ledger file (.extracted_zips.txt) inside the data folder, as "name|size".
#  On each run: zip in the ZIP folder but not in the ledger -> extract it into
#  the data folder and add to the ledger. Zip already in the ledger -> skip.
#  Re-downloading a zip with the same name but different size counts as new
#  (size changed -> re-extract). Delete the ledger to force a full re-extract.

import zipfile

def sync_zips(zip_folder, data_folder, label):
    """Extract any not-yet-seen zips from zip_folder into data_folder."""
    if not os.path.isdir(zip_folder):
        return                                        # no staging folder -> nothing to do
    os.makedirs(data_folder, exist_ok=True)
    ledger_path = os.path.join(data_folder, ".extracted_zips.txt")
    seen = set()
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            seen = set(line.strip() for line in f if line.strip())

    zips = sorted(glob.glob(os.path.join(zip_folder, "**", "*.zip"), recursive=True))
    new_count = 0
    for zp in zips:
        key = f"{os.path.basename(zp)}|{os.path.getsize(zp)}"
        if key in seen:
            continue
        print(f"[SYNC] {label}: extracting {os.path.basename(zp)} ...")
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(data_folder)
            seen.add(key)
            new_count += 1
            with open(ledger_path, "a") as f:
                f.write(key + "\n")
        except Exception as e:
            print(f"[SYNC] {label}: FAILED on {os.path.basename(zp)}: {e} (skipping)")
    if new_count:
        print(f"[SYNC] {label}: {new_count} new zip(s) extracted into {data_folder}")
    else:
        print(f"[SYNC] {label}: no new zips ({len(zips)} already extracted)")

    # SoLEXS-only extra step: the L1 files inside arrive as .lc.gz / .pi.gz;
    # the readers want plain .lc / .pi, so gunzip anything still compressed.
    import gzip, shutil
    for gz in glob.glob(os.path.join(data_folder, "**", "*.gz"), recursive=True):
        out = gz[:-3]
        if os.path.exists(out):
            continue
        try:
            with gzip.open(gz, "rb") as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            print(f"[SYNC] {label}: gunzipped {os.path.basename(gz)}")
        except Exception as e:
            print(f"[SYNC] {label}: gunzip failed on {os.path.basename(gz)}: {e}")


# =============================================================================
#  SECTION 1  --  FIND THE FILES FOR EACH DATE
# =============================================================================

def find_solexs_dates(folder):
    """
    Return a dict: date_string(YYYYMMDD) -> {'lc': path, 'pi': path}
    We prefer SDD2 (the detector that does not saturate on bright flares).
    """
    lc_files = glob.glob(os.path.join(folder, "**", "*.lc"), recursive=True)
    result = {}
    for lc in lc_files:
        name = os.path.basename(lc)
        m = re.search(r"(\d{8})", name)
        if not m:
            continue
        date = m.group(1)
        # prefer SDD2 files
        if "SDD2" not in name and date in result:
            continue
        pi = lc[:-3] + ".pi"           # same folder, same name, .pi extension
        if not os.path.exists(pi):
            pi = None
        result[date] = {"lc": lc, "pi": pi}
    return dict(sorted(result.items()))


def find_helos_for_date(folder, date):
    """
    Find HEL1OS band-lightcurve files whose folder path contains this date.
    Returns a list of file paths (may be empty if that date has no HEL1OS data).
    We look for the CdTe lightcurve files (they hold the LC_BAND tables).
    """
    hits = glob.glob(os.path.join(folder, "**", "*lightcurve_cdte*.fits"), recursive=True)
    # HEL1OS paths carry the date both as folders (2024\06\22) and in the
    # filename (HLS_20240622...). The compact date string appears in the path,
    # so a simple substring match is reliable.
    return [h for h in hits if date in h]


def find_helos_spectra_for_date(folder, date):
    """
    Find HEL1OS Type-II spectra files (full per-channel spectra, 20 s cadence)
    for this date. Returns {'cdte': [paths], 'czt': [paths]}.
    These are what the CALDB calibration needs (band lightcurves don't have
    per-channel counts, so they can't be calibrated to W/m^2).
    """
    out = {"cdte": [], "czt": []}
    for det in out:
        hits = glob.glob(os.path.join(folder, "**", f"*spectra_{det}*.fits"), recursive=True)
        out[det] = sorted([h for h in hits if date in h])
    return out


# =============================================================================
#  SECTION 2  --  READERS (built for YOUR exact column names)
# =============================================================================

def read_solexs_lc(path):
    """
    Read a SoLEXS light-curve (.lc).  Table RATE, columns TIME + COUNTS.
    Returns (times_unix, counts) as clean float arrays (NaNs -> 0).
    """
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        cols = [c.upper() for c in d.columns.names]
        tcol = "TIME" if "TIME" in cols else d.columns.names[0]
        ccol = "COUNTS" if "COUNTS" in cols else d.columns.names[-1]
        t = np.array(d[tcol], dtype=float)
        c = np.array(d[ccol], dtype=float)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(t)
    return t[order], c[order]


def read_solexs_pi(path):
    """
    Read a SoLEXS spectrum (.pi).  Table SPECTRUM.
    Columns: TSTART, TELAPSE, SPEC_NUM, CHANNEL, COUNTS, EXPOSURE
    COUNTS is a 340-length array per row (photon counts per channel each second).
    Returns (times_unix, spectra[N,340], channels[340]) or None.
    """
    if path is None or not os.path.exists(path):
        return None
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        cols = [c.upper() for c in d.columns.names]
        tcol = "TSTART" if "TSTART" in cols else ("TIME" if "TIME" in cols else d.columns.names[0])
        t = np.array(d[tcol], dtype=float)
        spec = np.array(d["COUNTS"], dtype=float)          # shape (N, 340)
        spec = np.nan_to_num(spec, nan=0.0)
        # channels: use the first row's CHANNEL array if present, else 0..339
        if "CHANNEL" in cols:
            chan = np.array(d["CHANNEL"][0], dtype=float)
        else:
            chan = np.arange(spec.shape[1], dtype=float)
    order = np.argsort(t)
    return t[order], spec[order], chan


def find_helos_dates(folder):
    """
    Return a sorted set of YYYYMMDD strings for which any HEL1OS file exists.
    We scan lightcurve + spectra filenames/paths for compact date strings.
    """
    dates = set()
    for pattern in ("*lightcurve_*.fits", "*spectra_*.fits"):
        for p in glob.glob(os.path.join(folder, "**", pattern), recursive=True):
            for m in re.finditer(r"(20\d{6})", p):
                dates.add(m.group(1))
    return sorted(dates)


def pick_date_interactive(sol_dates, helos_dates):
    """
    Show which dates exist where, ask for ONE date, then ask HOW to run it:
      date in BOTH   -> menu: [1] combined  [2] SoLEXS-only  [3] HEL1OS-only
                              [4] both independent (two separate reports)
      SoLEXS only    -> confirm, run SoLEXS-independent
      HEL1OS only    -> confirm, run HEL1OS-independent (hard X-ray detection)
      neither        -> ask again
    Returns (date, mode) where mode is 'COMBINED'|'SOLEXS'|'HEL1OS'|'BOTH_INDEP',
    ("ALL", "COMBINED") for every date, or (None, None) to quit.
    """
    both       = sorted(set(sol_dates) & set(helos_dates))
    sol_only   = sorted(set(sol_dates) - set(helos_dates))
    helos_only = sorted(set(helos_dates) - set(sol_dates))

    print("\nAVAILABLE DATES")
    print("-" * 62)
    print(f"  BOTH SoLEXS + HEL1OS ({len(both)}):  " + (", ".join(both) if both else "none"))
    print(f"  SoLEXS only ({len(sol_only)}):        " + (", ".join(sol_only) if sol_only else "none"))
    print(f"  HEL1OS only ({len(helos_only)}):      " + (", ".join(helos_only) if helos_only else "none"))
    print("-" * 62)

    while True:
        choice = input("\nEnter ONE date (YYYYMMDD), 'all' for every date, or 'q' to quit: ").strip().lower()
        if choice == "q":
            return None, None
        if choice == "all":
            return "ALL", "COMBINED"
        if not re.fullmatch(r"\d{8}", choice):
            print("  Format is YYYYMMDD, e.g. 20240527. Try again.")
            continue

        in_sol = choice in sol_dates
        in_hel = choice in helos_dates

        if in_sol and in_hel:
            print(f"  {choice}: found in BOTH. How do you want it?")
            print("    [1] Combined master result  (SoLEXS + HEL1OS together, full physics)")
            print("    [2] SoLEXS independent      (soft X-ray only)")
            print("    [3] HEL1OS independent      (hard X-ray only)")
            print("    [4] Both independent        (two separate reports)")
            m = input("  Choose 1/2/3/4: ").strip()
            mode = {"1": "COMBINED", "2": "SOLEXS", "3": "HEL1OS", "4": "BOTH_INDEP"}.get(m)
            if mode is None:
                print("  Pick 1, 2, 3 or 4.")
                continue
            return choice, mode
        if in_sol and not in_hel:
            print(f"  {choice}: SoLEXS ONLY. No HEL1OS -> gamma, Neupert, hard W/m^2 will be N/A.")
            ok = input("  Run anyway with SoLEXS only? [yes/no]: ").strip().lower()
            if ok in ("y", "yes"):
                return choice, "SOLEXS"
            continue
        if in_hel and not in_sol:
            print(f"  {choice}: HEL1OS ONLY. Detection will run on the hard X-ray band;")
            print(f"  coronal heating + Neupert will be N/A (they need SoLEXS).")
            ok = input("  Run anyway with HEL1OS only? [yes/no]: ").strip().lower()
            if ok in ("y", "yes"):
                return choice, "HEL1OS"
            continue
        print(f"  {choice}: found in NEITHER dataset. Pick from the lists above.")


def read_helos_spectra(paths):
    """
    Read HEL1OS Type-II spectra files (SPECTRUM extension).
    Columns: SPEC_NUM, CHANNEL, COUNTS[511 or 341], STAT_ERR, ROWID,
             TSTART, TSTOP, EXPOSURE.
    KEY TIME QUIRK (same one we hit with the event lists): the TSTART/TSTOP
    COLUMNS are seconds RELATIVE to the session start; the absolute anchor is
    the TSTART KEYWORD in the header, which is an MJD. We anchor each file to
    its own header MJD, so multi-session days stitch correctly.

    Returns (times_unix, spectra[N,ch], exposures[N]) with rows time-sorted,
    or None if nothing readable.
    """
    all_t, all_spec, all_exp = [], [], []
    for path in paths:
        try:
            with fits.open(path, memmap=False) as h:
                hdu = h["SPECTRUM"] if "SPECTRUM" in [x.name for x in h] else h[1]
                anchor_mjd = float(hdu.header["TSTART"])       # header = MJD
                anchor_unix = mjd_to_unix(anchor_mjd)
                rel_t = np.array(hdu.data["TSTART"], dtype=float)   # column = rel s
                spec = np.nan_to_num(np.array(hdu.data["COUNTS"], dtype=float), nan=0.0)
                if "EXPOSURE" in [c.upper() for c in hdu.data.columns.names]:
                    exp = np.array(hdu.data["EXPOSURE"], dtype=float)
                else:
                    exp = np.full(len(rel_t), 20.0)            # 20 s default cadence
                exp = np.where(exp > 0, exp, 20.0)
                all_t.append(anchor_unix + rel_t)
                all_spec.append(spec)
                all_exp.append(exp)
        except Exception as e:
            print(f"      [warn] could not read HEL1OS spectra {os.path.basename(path)}: {e}")
    if not all_t:
        return None
    t = np.concatenate(all_t)
    spec = np.vstack(all_spec)
    exp = np.concatenate(all_exp)
    order = np.argsort(t)
    return t[order], spec[order], exp[order]


def read_helos_bands(paths):
    """
    Read HEL1OS band light-curves.  Each file has several HDUs, each an energy
    band whose NAME looks like  CDTE1_LC_BAND_20.00KEV_TO_30.00KEV.
    Columns per band: MJD, ISOT, CTR (count rate), STAT_ERR.

    Returns a list of band dicts:
        {'e_lo':20.0, 'e_hi':30.0, 't':times_unix, 'ctr':rate}
    plus a 'total' band (the widest, ~1.8-90 keV) used for the Neupert check.
    """
    bands = []
    for path in paths:
        try:
            with fits.open(path, memmap=False) as h:
                for hdu in h[1:]:
                    if not hasattr(hdu, "columns"):
                        continue
                    name = (hdu.name or "").upper()
                    m = re.search(r"([\d.]+)KEV_TO_([\d.]+)KEV", name)
                    if not m:
                        continue
                    e_lo, e_hi = float(m.group(1)), float(m.group(2))
                    cols = [c.upper() for c in hdu.columns.names]
                    if "MJD" not in cols or "CTR" not in cols:
                        continue
                    t   = mjd_to_unix(np.array(hdu.data["MJD"], dtype=float))
                    ctr = np.nan_to_num(np.array(hdu.data["CTR"], dtype=float), nan=0.0)
                    order = np.argsort(t)
                    bands.append({"e_lo": e_lo, "e_hi": e_hi,
                                  "t": t[order], "ctr": ctr[order]})
        except Exception as e:
            print(f"      [warn] could not read HEL1OS file {os.path.basename(path)}: {e}")
    return bands


# =============================================================================
#  SECTION 3  --  REBIN + BAND SUMS
# =============================================================================

def rebin(times, values, bin_sec=BIN_SEC):
    """Average a 1-second series into uniform bin_sec bins."""
    if len(times) < 2:
        return times, values
    t0, t1 = times[0], times[-1]
    n = int((t1 - t0) / bin_sec) + 1
    bt = t0 + np.arange(n) * bin_sec
    acc = np.zeros(n); cnt = np.zeros(n)
    idx = ((times - t0) / bin_sec).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(acc, idx[ok], values[ok])
    np.add.at(cnt, idx[ok], 1)
    good = cnt > 0
    acc[good] /= cnt[good]
    return bt, acc


def band_counts(spec, chan, e_lo, e_hi):
    """Sum the spectrum's channels that fall inside an energy band [e_lo, e_hi] keV."""
    energy = ch_to_keV(chan)
    mask = (energy >= e_lo) & (energy < e_hi)
    if not mask.any():
        return np.zeros(spec.shape[0])
    return spec[:, mask].sum(axis=1)


