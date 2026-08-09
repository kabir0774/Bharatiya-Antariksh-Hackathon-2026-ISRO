from .common import *

# =============================================================================
#  SECTION 1 -- BACKGROUND (same definition nowcast's detector uses, so the
#  two halves of the project agree on what "background" means)
# =============================================================================

def rolling_background(flux, window_bins=None):
    """10th-percentile rolling background, same definition as
    nowcast_physics.detect_events(), used here for FIX 2 normalization.

    Floor: an absolute 1e-6 works fine for SoLEXS (counts/s in the tens to
    thousands) but is disastrous for a low-rate series like some HEL1OS
    bands, where the 10th percentile can legitimately BE zero -- dividing
    ordinary values by 1e-6 then explodes into the millions. Floor to 1% of
    the series' own median instead (falls back to 1e-6 only if the whole
    series is zero).
    """
    window_bins = window_bins or ncp.BG_WINDOW_BINS
    s = pd.Series(flux)
    bg = s.rolling(window_bins, center=True, min_periods=window_bins // 4).quantile(0.10)
    bg = bg.ffill().bfill().values
    positive = flux[flux > 0]
    scale_floor = max(np.nanmedian(positive) * 0.01, 1e-6) if len(positive) else 1e-6
    return np.maximum(bg, scale_floor)


# =============================================================================
#  SECTION 2 -- PER-DATE DATA ASSEMBLY
# =============================================================================

def assemble_day(date, sol_paths, helos_paths):
    """
    Build everything one date needs for windowing + labeling + features:
      bt, flux            : SoLEXS total-band counts/s, 10s bins
      bg                  : rolling background of flux (FIX 2 basis)
      bandA/B/C           : SoLEXS BAND_A/B/C counts/s series (same bands
                             nowcast's coronal-heating code uses)
      helos_bt, helos_flux: HEL1OS total-band counts/s (rebinned to same grid
                             as SoLEXS where possible), or None if unavailable
      events              : ground-truth flares for this date, from the SAME
                             detector nowcasting uses (detect_events)
    """
    t_lc, c_lc = ncp.read_solexs_lc(sol_paths["lc"])
    bt, flux = ncp.rebin(t_lc, c_lc)
    bg = rolling_background(flux)
    events = ncp.detect_events(bt, flux)

    pi_data = ncp.read_solexs_pi(sol_paths.get("pi"))
    bandA = bandB = bandC = None
    if pi_data is not None:
        pt, spec, chan = pi_data
        bandA_raw = ncp.band_counts(spec, chan, *ncp.BAND_A)
        bandB_raw = ncp.band_counts(spec, chan, *ncp.BAND_B)
        bandC_raw = ncp.band_counts(spec, chan, *ncp.BAND_C)
        _, bandA = ncp.rebin(pt, bandA_raw)
        _, bandB = ncp.rebin(pt, bandB_raw)
        _, bandC = ncp.rebin(pt, bandC_raw)
        # pad/truncate to match main bt length (pi coverage can differ slightly from lc)
        bandA = _match_length(bandA, len(bt))
        bandB = _match_length(bandB, len(bt))
        bandC = _match_length(bandC, len(bt))

    helos_bt, helos_flux, helos_bands = None, None, []
    if helos_paths:
        helos_bands = ncp.read_helos_bands(helos_paths)
        hx = ncp.build_hxr_series(helos_bands)
        if hx is not None:
            h_bt, h_flux, _total = hx
            helos_bt, helos_flux = h_bt, h_flux

    return dict(bt=bt, flux=flux, bg=bg, bandA=bandA, bandB=bandB, bandC=bandC,
               helos_bt=helos_bt, helos_flux=helos_flux, helos_bands=helos_bands,
               events=events, pi_data=pi_data)


def _window_gamma(helos_bands, t0, t1):
    """
    Per-window particle-acceleration steepness (LIMITATION FIX: gamma as a
    forecasting clue). Same physics as nowcast's measure_acceleration --
    log10(mean rate / band width) vs log10(band centre energy), slope =
    -gamma -- but on just [t0, t1] and only for bands entirely >= 20 keV
    (below that, thermal glow contaminates the non-thermal slope).
    Returns gamma (float) or None if fewer than 2 usable hard bands.
    """
    energies, diffs = [], []
    for b in helos_bands or []:
        if b["e_lo"] < 20.0:
            continue
        m = (b["t"] >= t0) & (b["t"] <= t1)
        if m.sum() < 2:
            continue
        mean_rate = float(np.nanmean(b["ctr"][m]))
        if not np.isfinite(mean_rate) or mean_rate <= 0:
            continue
        energies.append(0.5 * (b["e_lo"] + b["e_hi"]))
        diffs.append(mean_rate / max(b["e_hi"] - b["e_lo"], 1e-3))
    if len(energies) < 2:
        return None
    try:
        slope = np.polyfit(np.log10(energies), np.log10(diffs), 1)[0]
    except Exception:
        return None
    gamma = -float(slope)
    return gamma if 0.5 < gamma < 12.0 else None


def _match_length(arr, n):
    if arr is None:
        return None
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])
