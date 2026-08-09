from .common import *
from .io_data import *
from .detection import *
from .physics import *
from .reports import *
from .analysis import *

# =============================================================================
#  MAIN
# =============================================================================

def process_date(date, sol, helos_paths, mode="COMBINED"):
    """
    mode: 'COMBINED' (SoLEXS detection + HEL1OS physics) or
          'SOLEXS'   (deliberately ignore HEL1OS even if present).
    """
    print(f"\n--- {date}  [{mode}] ---")
    # read SoLEXS light-curve + spectrum
    t_lc, c_lc = read_solexs_lc(sol["lc"])
    bt, flux = rebin(t_lc, c_lc)
    pi_data = read_solexs_pi(sol["pi"])
    print(f"    SoLEXS: {len(bt)} bins" + ("  + spectrum" if pi_data is not None else "  (no .pi spectrum)"))

    if mode == "SOLEXS":
        helos_paths = []                              # pretend HEL1OS doesn't exist

    # read HEL1OS bands (may be empty if this date has no HEL1OS data)
    helos_bands = read_helos_bands(helos_paths) if helos_paths else []
    if helos_bands:
        print(f"    HEL1OS: {len(helos_bands)} energy bands matched")
    else:
        print(f"    HEL1OS: none for this date (acceleration & Neupert will be N/A)")

    # read HEL1OS Type-II spectra for real hard X-ray W/m^2.
    # We now load BOTH detectors when available (not just whichever comes
    # first): CDTE covers ~8-70 keV, CZT covers ~20-150 keV per the HEL1OS
    # manual. band3 (60-150 keV) needs CZT specifically -- CDTE's own
    # calibration doesn't extend that high.
    helos_spec_cdte, helos_spec_czt = None, None
    if HEL1OS_CAL is not None and mode != "SOLEXS":
        spectra_paths = find_helos_spectra_for_date(HELOS_FOLDER, date)
        if spectra_paths["cdte"]:
            helos_spec_cdte = read_helos_spectra(spectra_paths["cdte"])
            if helos_spec_cdte is not None:
                print(f"    HEL1OS spectra: {len(helos_spec_cdte[0])} x CDTE spectra loaded")
        if spectra_paths["czt"]:
            helos_spec_czt = read_helos_spectra(spectra_paths["czt"])
            if helos_spec_czt is not None:
                print(f"    HEL1OS spectra: {len(helos_spec_czt[0])} x CZT spectra loaded")
        if helos_spec_cdte is None and helos_spec_czt is None and helos_bands:
            print(f"    HEL1OS spectra: none found -> hard W/m^2 columns will be None")

    # helos_spec/helos_cal: the ORIGINAL single-detector pick (CDTE preferred,
    # CZT fallback), kept for the existing hard_flux_calibrated()/report-plot
    # code that expects one series. Band classification below uses both
    # detectors directly instead.
    if helos_spec_cdte is not None:
        helos_spec, helos_cal = helos_spec_cdte, HEL1OS_CAL["cdte"]
    elif helos_spec_czt is not None:
        helos_spec, helos_cal = helos_spec_czt, HEL1OS_CAL["czt"]
    else:
        helos_spec, helos_cal = None, None

    # detect flares
    events = detect_events(bt, flux)
    print(f"    Flares detected: {len(events)}")

    # ---- limitation fix #1: THREE fully independent per-band detectors -----
    # Each canonical band gets its own full-day Poisson-FOCuS detector run on
    # its own raw counts, blind to the master detector and to each other.
    # Every master event is then put to a vote: how many bands discovered it
    # independently?
    triband = triband_independent_vote(mode, pi_data, helos_spec_cdte, helos_spec_czt)
    if triband["covered"]:
        per_band = ", ".join(f"{b}:{len(triband['catalogs'][b])}" for b in triband["covered"])
        print(f"    Independent band detectors ran on {len(triband['covered'])} band(s) "
              f"-- events each found on its own: {per_band}")

    rows = []
    phys = []
    for i, ev in enumerate(events, 1):
        # physics
        temp     = measure_temperature(pi_data, ev["start"], ev["end"], ev["peak"], calibrator=SOLEXS_CAL)
        qpp_soft = measure_qpp(bt, flux, ev["s_idx"], ev["e_idx"], "soft")
        accel    = measure_acceleration(helos_bands, ev["start"], ev["end"])
        neupert  = measure_neupert(bt, flux, ev["s_idx"], ev["e_idx"], helos_bands)
        neupert_lag = measure_neupert_lag(bt, flux, ev["s_idx"], ev["e_idx"], helos_bands)
        qpp_hard = None      # (hard-channel QPP would need HEL1OS fine cadence; N/A for now)

        # limitation fix #3: full forward-folded isothermal spectral fit
        # (all channels 2.8-12 keV, Cash statistic) alongside the fast proxy
        specfit = fit_isothermal_spectrum(pi_data, SOLEXS_CAL, ev["start"], ev["end"])

        # limitation fix #1: the independent tri-band vote for this event
        band_vote = vote_for_event(ev, triband) if triband["covered"] else None

        # Real CALDB-calibrated flux/energy (falls back to None if CALDB
        # isn't loaded or this event has no spectrum coverage).
        cal = radiated_energy_calibrated(pi_data, SOLEXS_CAL, ev) if pi_data is not None else None
        hcal = hard_flux_calibrated(helos_spec, helos_cal, ev)
        eband = classify_energy_band(pi_data, helos_spec_cdte, helos_spec_czt, ev)

        # Independent per-band FOCuS confirmation: did band2/band3's OWN raw
        # counts also show a genuine excursion overlapping this event, not
        # just "did they have any flux" (classify_energy_band's coverage
        # check)? Real corroborating evidence, fed into confidence below.
        band_confirms = []
        if helos_spec_czt is not None:
            t_czt, spec_czt, exp_czt = helos_spec_czt
            for lo, hi in (CANONICAL_BANDS["band2"], CANONICAL_BANDS["band3"]):
                _, cc = raw_band_counts_series(t_czt, spec_czt, HEL1OS_CAL["czt"], lo, hi)
                bc = band_focus_confirmation(t_czt, cc, ev, exposure_sec=float(np.median(exp_czt)))
                if bc is not None:
                    band_confirms.append(bc)
        elif helos_spec_cdte is not None:
            t_cd, spec_cd, exp_cd = helos_spec_cdte
            _, cc = raw_band_counts_series(t_cd, spec_cd, HEL1OS_CAL["cdte"], *CANONICAL_BANDS["band2"])
            bc = band_focus_confirmation(t_cd, cc, ev, exposure_sec=float(np.median(exp_cd)))
            if bc is not None:
                band_confirms.append(bc)
        best_band_confirm = max(band_confirms, key=lambda b: b["match_score"]) if band_confirms else None

        cross_agree = classify_cross_band_agreement(eband)

        checks, n_pass, n_flag, conf, conf_pct, coverage = run_cross_checks(
            ev, temp, qpp_soft, qpp_hard, accel, neupert, eband=eband,
            band_confirm=best_band_confirm, cross_agree=cross_agree,
            specfit=specfit, band_vote=band_vote)
        energy = radiated_energy(bt, flux, ev)

        # real error bars on the headline flux number (Poisson counting stats
        # propagated through the same ARF weights the flux itself used)
        peak_flux_err = None
        if cal is not None and pi_data is not None:
            pt, spec_sx, chan_sx = pi_data
            pk_idx = int(np.argmin(np.abs(pt - ev["peak"])))
            if abs(pt[pk_idx] - ev["peak"]) < BIN_SEC * 2:
                peak_flux_err = flux_uncertainty(spec_sx[pk_idx], SOLEXS_CAL, 1.0, 2.8, 20.0)

        # time-resolved accel: prefer CZT (widest hard-X range), CDTE fallback
        if helos_spec_czt is not None:
            accel_slices = measure_acceleration_time_resolved(helos_spec_czt, HEL1OS_CAL["czt"], ev)
        elif helos_spec_cdte is not None:
            accel_slices = measure_acceleration_time_resolved(helos_spec_cdte, HEL1OS_CAL["cdte"], ev)
        else:
            accel_slices = []

        phys.append(dict(temp=temp, accel=accel, accel_slices=accel_slices, neupert=neupert,
                         qpp_diag=qpp_diagnostics(bt, flux, ev["s_idx"], ev["e_idx"])))

        rows.append({
            "event_id":     f"{date}_SDD2_{i:03d}",
            "date":         date,
            "start_utc":    unix_to_utc(ev["start"]),
            "peak_utc":     unix_to_utc(ev["peak"]),
            "end_utc":      unix_to_utc(ev["end"]),
            "duration_s":   round(ev["end"] - ev["start"], 1),
            "peak_rate":    round(ev["peak_flux"], 2),
            "background":   round(ev["bg"], 2),
            "ratio":        round(ev["ratio"], 2),
            "sigma":        round(ev["sigma"], 1),
            "data_gap":     "YES" if ev["data_gap"] else "no",
            "class_approx": goes_class(cal["peak_flux_wm2"]) if cal else approx_class(ev["ratio"], ev["peak_flux"]),
            "class_is_real_flux": bool(cal),
            "temp_MK":         round(temp["temp_MK"], 1)        if temp  else None,
            "temp_MK_err":     round(temp["temp_MK_err"], 2)    if temp and temp.get("temp_MK_err") is not None else None,
            "temp_MK_fit":     round(specfit["T_MK_fit"], 1)    if specfit else None,
            "temp_MK_fit_err": (f"-{specfit['T_MK_err_lo']:.2f}/+{specfit['T_MK_err_hi']:.2f}"
                                if specfit else None),
            "hardness_ratio":  round(temp["hardness_ratio"], 3) if temp  else None,
            "temp_offset_s":   round(temp["temp_offset_s"], 1)  if temp  else None,
            "qpp_period_s":    round(qpp_soft, 1)               if qpp_soft else None,
            "gamma":           round(accel["gamma"], 2) if accel and accel["gamma"] else None,
            "gamma_err":       round(accel["gamma_err"], 2) if accel and accel.get("gamma_err") else None,
            "delta":           round(accel["delta"], 2) if accel and accel["delta"] else None,
            "neupert_corr":    round(neupert, 3)               if neupert is not None else None,
            "neupert_best_lag_s": round(neupert_lag["best_lag_s"], 0)   if neupert_lag else None,
            "neupert_corr_at_lag": round(neupert_lag["corr_at_best"], 3) if neupert_lag else None,
            "check_neupert":       checks["neupert"],
            "check_accel_timing":  checks["accel_timing"],
            "check_heat_timing":   checks["heat_timing"],
            "check_energy_budget": checks["energy_budget"],
            "check_qpp_witness":   checks["qpp_witness"],
            "check_class_accel":   checks["class_accel"],
            "n_pass":        n_pass,
            "n_flag":        n_flag,
            "checks_run":    coverage,
            "confidence":    conf,
            "confidence_pct": conf_pct,
            "radiated_energy_proxy": round(energy, 1),
            "peak_flux_wm2":        cal["peak_flux_wm2"]       if cal else None,
            "peak_flux_wm2_err":    peak_flux_err,
            "bg_flux_wm2":          cal["bg_flux_wm2"]         if cal else None,
            "radiated_energy_jm2":  cal["radiated_energy_jm2"] if cal else None,
            "hard_peak_flux_wm2":   hcal["hard_peak_flux_wm2"] if hcal else None,
            "hard_peak_flux_wm2_err": hcal.get("hard_peak_flux_wm2_err") if hcal else None,
            "hard_bg_flux_wm2":     hcal["hard_bg_flux_wm2"]   if hcal else None,
            "hard_energy_jm2":      hcal["hard_energy_jm2"]    if hcal else None,
            "band1_wm2_2p8_20keV":  eband["band1_wm2"],
            "band2_wm2_20_60keV":   eband["band2_wm2"],
            "band3_wm2_60_150keV":  eband["band3_wm2"],
            "dominant_energy_band": eband["dominant_band"],
            "n_bands_covered": eband["n_bands_covered"],
            "band_independently_confirmed": bool(best_band_confirm["confirmed"]) if best_band_confirm else None,
            "cross_band_class_agree": cross_agree.get("agree"),
            "independent_band_votes": band_vote["vote_str"] if band_vote else None,
            "voting_bands":          "+".join(band_vote["voting_bands"]) if band_vote and band_vote["voting_bands"] else None,
            "check_temp_fit":        checks["temp_fit_agree"],
            "check_band_vote":       checks["band_vote"],
        })
    if rows:
        multi = sum(1 for r in rows if r["n_bands_covered"] >= 2)
        print(f"    Energy-band classification: {multi}/{len(rows)} events had "
              f"real multi-band coverage (rest defaulted to whichever single "
              f"band had data -- see * in the report table)")
    # per-date report PNG
    if PLOT_REPORTS and events:
        rep_mode = "SOLEXS" if (mode == "SOLEXS" or not helos_bands) else "COMBINED"
        hx = build_hxr_series(helos_bands)
        hxr = (hx[2]["t"], hx[2]["ctr"]) if hx else None
        flux_wm2 = solexs_flux_series(bt, pi_data, SOLEXS_CAL)
        hxr_wm2 = hel1os_flux_series(helos_spec, helos_cal)

        bands = build_three_band_series(rep_mode, bt, pi_data, helos_spec_cdte, helos_spec_czt)

        try:
            make_daily_report(date, rep_mode, bt, flux, hxr, events, phys, rows, REPORT_FOLDER,
                              flux_wm2=flux_wm2, hxr_wm2=hxr_wm2,
                              band1_series=bands["band1"], band2_series=bands["band2"], band3_series=bands["band3"])
            make_event_table_png(date, rep_mode, rows, events, REPORT_FOLDER)
        except Exception as e:
            import traceback
            print(f"    [warn] report plot failed: {e}")
            traceback.print_exc()
    return rows


def process_date_hxr(date, helos_paths):
    """
    HEL1OS-ONLY mode: no SoLEXS this date, so detect flares directly on the
    widest HEL1OS hard X-ray band (same rolling-background 5-sigma detector).
    Physics available: QPP (on HXR), particle acceleration (band power law),
    hard W/m^2 (if Type-II spectra exist). NOT available without SoLEXS:
    coronal heating temperature, Neupert, soft-flux calibration.
    """
    print(f"\n--- {date}  [HEL1OS-only] ---")
    helos_bands = read_helos_bands(helos_paths)
    hx = build_hxr_series(helos_bands)
    if hx is None:
        print("    No usable HEL1OS bands. Nothing to do.")
        return []
    bt, flux, total = hx
    print(f"    HEL1OS: {len(helos_bands)} bands; detecting on "
          f"{total['e_lo']:.0f}-{total['e_hi']:.0f} keV ({len(bt)} bins)")

    helos_spec_cdte, helos_spec_czt = None, None
    if HEL1OS_CAL is not None:
        spectra_paths = find_helos_spectra_for_date(HELOS_FOLDER, date)
        if spectra_paths["cdte"]:
            helos_spec_cdte = read_helos_spectra(spectra_paths["cdte"])
            if helos_spec_cdte is not None:
                print(f"    HEL1OS spectra: {len(helos_spec_cdte[0])} x CDTE loaded")
        if spectra_paths["czt"]:
            helos_spec_czt = read_helos_spectra(spectra_paths["czt"])
            if helos_spec_czt is not None:
                print(f"    HEL1OS spectra: {len(helos_spec_czt[0])} x CZT loaded")

    if helos_spec_cdte is not None:
        helos_spec, helos_cal = helos_spec_cdte, HEL1OS_CAL["cdte"]
    elif helos_spec_czt is not None:
        helos_spec, helos_cal = helos_spec_czt, HEL1OS_CAL["czt"]
    else:
        helos_spec, helos_cal = None, None

    events = detect_events(bt, flux)
    print(f"    Flares detected: {len(events)}")

    # limitation fix #1 also applies here: independent per-band detectors
    # (band1 falls back to CDTE 9.5-20 keV in this mode; band2/band3 as usual)
    triband = triband_independent_vote("HEL1OS", None, helos_spec_cdte, helos_spec_czt)
    if triband["covered"]:
        per_band = ", ".join(f"{b}:{len(triband['catalogs'][b])}" for b in triband["covered"])
        print(f"    Independent band detectors ran on {len(triband['covered'])} band(s) "
              f"-- events each found on its own: {per_band}")

    rows, phys = [], []
    for i, ev in enumerate(events, 1):
        accel = measure_acceleration(helos_bands, ev["start"], ev["end"])
        qpp_p = measure_qpp(bt, flux, ev["s_idx"], ev["e_idx"], "hard")
        hcal = hard_flux_calibrated(helos_spec, helos_cal, ev)
        eband = classify_energy_band(None, helos_spec_cdte, helos_spec_czt, ev)
        band_vote = vote_for_event(ev, triband) if triband["covered"] else None

        band_confirms = []
        if helos_spec_czt is not None:
            t_czt, spec_czt, exp_czt = helos_spec_czt
            for lo, hi in (CANONICAL_BANDS["band2"], CANONICAL_BANDS["band3"]):
                _, cc = raw_band_counts_series(t_czt, spec_czt, HEL1OS_CAL["czt"], lo, hi)
                bc = band_focus_confirmation(t_czt, cc, ev, exposure_sec=float(np.median(exp_czt)))
                if bc is not None:
                    band_confirms.append(bc)
        best_band_confirm = max(band_confirms, key=lambda b: b["match_score"]) if band_confirms else None
        cross_agree = classify_cross_band_agreement(eband)

        if helos_spec_czt is not None:
            accel_slices = measure_acceleration_time_resolved(helos_spec_czt, HEL1OS_CAL["czt"], ev)
        elif helos_spec_cdte is not None:
            accel_slices = measure_acceleration_time_resolved(helos_spec_cdte, HEL1OS_CAL["cdte"], ev)
        else:
            accel_slices = []
        checks, n_pass, n_flag, conf, conf_pct, coverage = run_cross_checks(
            ev, None, None, qpp_p, accel, None, eband=eband,
            band_confirm=best_band_confirm, cross_agree=cross_agree,
            band_vote=band_vote)
        phys.append(dict(temp=None, accel=accel, accel_slices=accel_slices, neupert=None,
                         qpp_diag=qpp_diagnostics(bt, flux, ev["s_idx"], ev["e_idx"])))
        rows.append({
            "event_id":     f"{date}_HXR_{i:03d}",
            "date":         date,
            "start_utc":    unix_to_utc(ev["start"]),
            "peak_utc":     unix_to_utc(ev["peak"]),
            "end_utc":      unix_to_utc(ev["end"]),
            "duration_s":   round(ev["end"] - ev["start"], 1),
            "peak_rate":    round(ev["peak_flux"], 2),
            "background":   round(ev["bg"], 2),
            "ratio":        round(ev["ratio"], 2),
            "sigma":        round(ev["sigma"], 1),
            "data_gap":     "YES" if ev["data_gap"] else "no",
            "class_approx": (goes_class(hcal["hard_peak_flux_wm2"], prefix="HXR-") if hcal
                            else "HXR-event"),  # HXR-prefixed: NOT a real GOES class (different band)
            "class_is_real_flux": bool(hcal),
            "temp_MK": None, "temp_MK_err": None, "temp_MK_fit": None, "temp_MK_fit_err": None,
            "hardness_ratio": None, "temp_offset_s": None,
            "qpp_period_s":    round(qpp_p, 1) if qpp_p else None,
            "gamma":           round(accel["gamma"], 2) if accel and accel["gamma"] else None,
            "gamma_err":       round(accel["gamma_err"], 2) if accel and accel.get("gamma_err") else None,
            "delta":           round(accel["delta"], 2) if accel and accel["delta"] else None,
            "neupert_corr":    None,
            "neupert_best_lag_s": None, "neupert_corr_at_lag": None,
            "check_neupert": checks["neupert"], "check_accel_timing": checks["accel_timing"],
            "check_heat_timing": checks["heat_timing"], "check_energy_budget": checks["energy_budget"],
            "check_qpp_witness": checks["qpp_witness"], "check_class_accel": checks["class_accel"],
            "n_pass": n_pass, "n_flag": n_flag, "checks_run": coverage,
            "confidence": conf, "confidence_pct": conf_pct,
            "radiated_energy_proxy": round(radiated_energy(bt, flux, ev), 1),
            "peak_flux_wm2": None, "peak_flux_wm2_err": None, "bg_flux_wm2": None, "radiated_energy_jm2": None,
            "hard_peak_flux_wm2":   hcal["hard_peak_flux_wm2"] if hcal else None,
            "hard_peak_flux_wm2_err": hcal.get("hard_peak_flux_wm2_err") if hcal else None,
            "hard_bg_flux_wm2":     hcal["hard_bg_flux_wm2"]   if hcal else None,
            "hard_energy_jm2":      hcal["hard_energy_jm2"]    if hcal else None,
            "band1_wm2_2p8_20keV":  eband["band1_wm2"],
            "band2_wm2_20_60keV":   eband["band2_wm2"],
            "band3_wm2_60_150keV":  eband["band3_wm2"],
            "dominant_energy_band": eband["dominant_band"],
            "n_bands_covered": eband["n_bands_covered"],
            "band_independently_confirmed": bool(best_band_confirm["confirmed"]) if best_band_confirm else None,
            "cross_band_class_agree": cross_agree.get("agree"),
            "independent_band_votes": band_vote["vote_str"] if band_vote else None,
            "voting_bands":          "+".join(band_vote["voting_bands"]) if band_vote and band_vote["voting_bands"] else None,
            "check_temp_fit":        checks["temp_fit_agree"],
            "check_band_vote":       checks["band_vote"],
        })


    if PLOT_REPORTS and events:
        hxr_wm2 = hel1os_flux_series(helos_spec, helos_cal)
        bands = build_three_band_series("HEL1OS", bt, None, helos_spec_cdte, helos_spec_czt)
        try:
            make_daily_report(date, "HEL1OS", bt, flux, None, events, phys, rows, REPORT_FOLDER,
                              flux_wm2=None, hxr_wm2=hxr_wm2,
                              band1_series=bands["band1"], band2_series=bands["band2"], band3_series=bands["band3"])
            make_event_table_png(date, "HEL1OS", rows, events, REPORT_FOLDER)
        except Exception as e:
            import traceback
            print(f"    [warn] report plot failed: {e}")
            traceback.print_exc()
    return rows


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("=" * 62)
    print("  NOWCAST PHYSICS + CROSS-CHECK MODULE")
    print("  ISRO Aditya-L1 Hackathon 2026")
    print("=" * 62)

    # Step 0: pull in any freshly downloaded zips before scanning for data
    sync_zips(SOLEXS_ZIP_FOLDER, SOLEXS_FOLDER, "SoLEXS")
    sync_zips(HELOS_ZIP_FOLDER, HELOS_FOLDER, "HEL1OS")

    dates = find_solexs_dates(SOLEXS_FOLDER)
    if not dates:
        print("\n[ERROR] No SoLEXS .lc files found. Check SOLEXS_FOLDER path.")
        sys.exit(1)

    run_mode = "COMBINED"
    if ASK_DATE:
        helos_dates = find_helos_dates(HELOS_FOLDER)
        chosen, run_mode = pick_date_interactive(sorted(dates.keys()), helos_dates)
        if chosen is None:
            print("Quit. Nothing processed.")
            return
        if chosen != "ALL" and chosen in dates:
            dates = {chosen: dates[chosen]}
    else:
        chosen = "ALL"
        print(f"\nFound SoLEXS data for {len(dates)} date(s): {', '.join(dates.keys())}")

    all_rows = []

    # HEL1OS-only date: SoLEXS dict doesn't have it -> dedicated path
    if ASK_DATE and run_mode == "HEL1OS" and chosen not in dates:
        helos_paths = find_helos_for_date(HELOS_FOLDER, chosen)
        try:
            all_rows += process_date_hxr(chosen, helos_paths)
        except Exception as e:
            print(f"    [ERROR processing {chosen}] {e}")
        dates = {}                                      # skip the SoLEXS loop below

    for date, sol in dates.items():
        helos_paths = find_helos_for_date(HELOS_FOLDER, date)
        try:
            if run_mode == "COMBINED":
                all_rows += process_date(date, sol, helos_paths, mode="COMBINED")
            elif run_mode == "SOLEXS":
                all_rows += process_date(date, sol, helos_paths, mode="SOLEXS")
            elif run_mode == "HEL1OS":
                all_rows += process_date_hxr(date, helos_paths)
            elif run_mode == "BOTH_INDEP":
                all_rows += process_date(date, sol, helos_paths, mode="SOLEXS")
                all_rows += process_date_hxr(date, helos_paths)
        except Exception as e:
            print(f"    [ERROR processing {date}] {e}")

    if not all_rows:
        print("\nNo flares found. Nothing written.")
        return

    df = pd.DataFrame(all_rows)
    out = os.path.join(OUTPUT_FOLDER, "physics_catalog.csv")
    df.to_csv(out, index=False)

    # short summary to the screen
    print("\n" + "=" * 62)
    print(f"  DONE.  {len(df)} flares written to:")
    print(f"    {out}")
    print("=" * 62)
    print("\nStrongest 8 flares (by absolute peak rate):")
    cols = ["event_id","peak_utc","peak_rate","peak_flux_wm2","ratio","class_approx",
            "temp_MK","gamma","neupert_corr","checks_run","confidence_pct","data_gap"]
    show = df.sort_values("peak_rate", ascending=False).head(8)[cols]
    with pd.option_context("display.width", 220, "display.max_columns", 25):
        print(show.to_string(index=False))

    n_gap = (df["data_gap"] == "YES").sum()
    if n_gap:
        print(f"\nNote: {n_gap} event(s) sit inside data gaps (background ~0) — treat with caution.")

    # how many of each check passed
    print("\nCross-check tally (across all flares):")
    for c in ["check_neupert","check_accel_timing","check_heat_timing",
              "check_energy_budget","check_qpp_witness","check_class_accel"]:
        vc = df[c].value_counts().to_dict()
        print(f"  {c:22s}  PASS={vc.get('PASS',0):3d}  FLAG={vc.get('FLAG',0):3d}  N/A={vc.get('N/A',0):3d}")


