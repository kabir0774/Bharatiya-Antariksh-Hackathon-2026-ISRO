from .common import *
from .physics import *
from .analysis import *




def unique_report_path(out_folder, date, mode):
    """
    report_20240527_COMBINED.png -> if that exists, try _2.png, _3.png, ...
    Never overwrites a previous run's report; each run for the same date
    gets its own numbered file so you can compare runs side by side.
    """
    base = f"report_{date}_{mode}"
    candidate = os.path.join(out_folder, f"{base}.png")
    n = 2
    while os.path.exists(candidate):
        candidate = os.path.join(out_folder, f"{base}_{n}.png")
        n += 1
    return candidate


def open_file_default(path):
    """Open a file in the OS's default viewer. Never raises -- if it fails
    (headless box, missing viewer, etc.) we just skip opening it."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                                    # Windows
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])                       # macOS
        else:
            subprocess.Popen(["xdg-open", path])                   # Linux
    except Exception as e:
        print(f"    [note] couldn't auto-open the report ({e}) -- open it manually: {path}")


def _placeholder(ax, msg):
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="gray",
            bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#cccccc"))


def make_daily_report(date, mode, bt, flux, hxr, events, phys, rows, out_folder,
                       flux_wm2=None, hxr_wm2=None, band1_series=None, band2_series=None, band3_series=None):
    """
    One PNG per date, mirroring the reference layout:
      row 0 : full-day flux vs time (SXR and/or HXR), flares shaded
      row 1 : biggest-event zoom | QPP detection
      row 2 : QPP periodogram   | coronal heating (temp curve)
      row 3 : particle accel    | event catalog table
      row 4 : Neupert check
    mode : 'COMBINED' | 'SOLEXS' | 'HEL1OS'  (controls titles + placeholders)
    bt, flux : the DETECTION series (raw counts -- background/sigma/event
               boundaries are always computed on this, never on flux_wm2).
    hxr  : (t, ctr) tuple of the widest HEL1OS band, counts, for overlay.
    flux_wm2 : real CALDB-calibrated W/m^2, same length & timebase as bt,
               from solexs_flux_series(). None -> plots fall back to counts.
    hxr_wm2  : (t, wm2) tuple from hel1os_flux_series(). None -> counts.
    phys : list of per-event dicts {temp, qpp_diag, accel, neupert}
    rows : the catalog rows for the table panel

    DISPLAY vs DETECTION: flux_wm2/hxr_wm2 are used ONLY to choose what's
    drawn on the y-axis. Which seconds count as "the event" (shading, peak
    time, duration) always comes from the counts-based detector -- W/m^2 is
    a real physical unit but a noisier one to threshold on bin-by-bin, so we
    keep detection where it's most robust and calibration where it makes the
    plot mean something to a judge who knows GOES units.
    """
    t0 = bt[0]
    hrs = (bt - t0) / 3600.0
    have_solexs_flux = flux_wm2 is not None and mode != "HEL1OS"
    have_hxr_flux = hxr_wm2 is not None

    fig = plt.figure(figsize=(16, 22))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.1, 1, 1, 1, 1], hspace=0.45, wspace=0.25)
    inst = {"COMBINED": "SoLEXS + HEL1OS", "SOLEXS": "SoLEXS", "HEL1OS": "HEL1OS"}[mode]
    fig.suptitle(f"Aditya-L1 Nowcast Report  |  {inst}  |  "
                 f"{date[:4]}-{date[4:6]}-{date[6:]}", fontsize=16, fontweight="bold", y=0.995)

    # ---- row 0 : ENERGY-RESOLVED flux vs time (up to 3 real bands) -----------------------
    ax = fig.add_subplot(gs[0, :])
    BAND_LINE = {
        "band1": ("#1f77b4", "SoLEXS 2.8-20 keV (soft)"),
        "band2": ("#2ca02c", "HEL1OS 20-60 keV (medium)"),
        "band3": ("#d62728", "HEL1OS 60-150 keV (hard)"),
    }
    any_band_line = False
    if mode != "HEL1OS" and have_solexs_flux:
        col, lbl = BAND_LINE["band1"]
        ax.plot(hrs, _log_safe(flux_wm2), lw=0.8, color=col, label=lbl)
        any_band_line = True
    elif mode == "HEL1OS" and band1_series is not None:
        t1, v1 = band1_series
        ax.plot((t1 - t0) / 3600.0, _log_safe(v1), lw=0.8, color="#1f77b4",
               label="HEL1OS CDTE 9.5-20 keV (soft HXR, no SoLEXS this date)")
        any_band_line = True
    if mode != "SOLEXS" and band2_series is not None:
        t2, v2 = band2_series
        col, lbl = BAND_LINE["band2"]
        ax.plot((t2 - t0) / 3600.0, _log_safe(v2), lw=0.7, color=col, label=lbl)
        any_band_line = True
    if mode != "SOLEXS" and band3_series is not None:
        t3, v3 = band3_series
        col, lbl = BAND_LINE["band3"]
        ax.plot((t3 - t0) / 3600.0, _log_safe(v3), lw=0.7, color=col, label=lbl)
        any_band_line = True

    if not any_band_line:
        # nothing calibrated at all for this date -- fall back to the raw
        # counts detection series so the panel is never blank
        ax.plot(hrs, flux, lw=0.6, color="steelblue",
               label="counts (no CALDB calibration / no coverage this date)")

    log_axis = any_band_line
    y_unit = "W / m^2" if any_band_line else "Counts / s"

    big_i = int(np.argmax([e["peak_flux"] for e in events])) if events else None

    BAND_SHADE = {
        "band1 (2.8-20 keV, soft)":   "#f0a500",
        "band2 (20-60 keV, medium)":  "#2ca02c",
        "band3 (60-150 keV, hard)":   "#d62728",
        None:                         "#999999",
    }
    seen_bands = set()
    for ev, r in zip(events, rows):
        band = r.get("dominant_energy_band")
        col = BAND_SHADE.get(band, "#999999")
        lbl = None
        if band not in seen_bands:
            seen_bands.add(band)
            short = {"band1 (2.8-20 keV, soft)": "flare: soft (2.8-20 keV)",
                     "band2 (20-60 keV, medium)": "flare: medium (20-60 keV)",
                     "band3 (60-150 keV, hard)": "flare: hard (60-150 keV)",
                     None: "flare: unclassified"}[band]
            lbl = short
        ax.axvspan((ev["start"] - t0) / 3600.0, (ev["end"] - t0) / 3600.0,
                   color=col, alpha=0.18, label=lbl)
    if log_axis:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(utc_formatter(t0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))   # one tick per hour
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.set_xlabel("UTC Time (HH:MM)"); ax.set_ylabel(y_unit)
    ax.set_title("ENERGY-RESOLVED FLUX vs TIME  |  each colour = one real energy band  |  "
                 "shading = detected flare (colour = its dominant band)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)
    ax.grid(alpha=0.15, which="major")
    ax.tick_params(labelsize=9)

    # Arrow pointing at the single biggest flare of the day -- label sits
    # ABOVE and to the SIDE of the curve (not directly over the peak), with a
    # curved arrow pointing down to it -- matches the reference layout.
    if big_i is not None:
        ev = events[big_i]
        pk_h = (ev["peak"] - t0) / 3600.0
        y0, y1 = ax.get_ylim()
        x0, x1 = ax.get_xlim()
        xspan = x1 - x0
        # offset away from whichever edge is closer, so the box never runs
        # off the plot or sits under the upper-right legend
        offset = xspan * 0.10
        text_x = pk_h - offset if pk_h > (x0 + x1) / 2 else pk_h + offset
        text_x = min(max(text_x, x0 + xspan * 0.05), x1 - xspan * 0.05)

        if log_axis:
            new_top = y1 * 10 ** 0.45
            tip_y = y1 * 10 ** -0.05
            text_y = new_top * 10 ** -0.12
        else:
            span = y1 - y0
            new_top = y1 + span * 0.30
            tip_y = y1 * 0.98
            text_y = new_top - span * 0.06
        ax.set_ylim(y0, new_top)
        ax.annotate(f"PEAK FLARE  {rows[big_i]['class_approx']}\n{unix_to_utc(ev['peak'])[11:16]} UTC",
                   xy=(pk_h, tip_y), xytext=(text_x, text_y),
                   fontsize=8, fontweight="bold", color="#8b0000", ha="center", va="top",
                   arrowprops=dict(arrowstyle="-|>", color="#8b0000", lw=1.6,
                                   connectionstyle="arc3,rad=0.25"),
                   bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#8b0000", lw=1.2, alpha=0.95))


    # ---- row 1 left : biggest event zoom, now also 3 real bands -----------------------
    ax = fig.add_subplot(gs[1, 0])
    if big_i is not None:
        ev = events[big_i]
        pad = max(int((ev["e_idx"] - ev["s_idx"]) * 0.5), 6)
        a, b = max(ev["s_idx"] - pad, 0), min(ev["e_idx"] + pad, len(flux))
        t_lo, t_hi = bt[a], bt[min(b, len(bt) - 1)]

        ZOOM_LINE = {
            "band1": ("#1f77b4", "soft"), "band2": ("#2ca02c", "medium"), "band3": ("#d62728", "hard"),
        }
        series_map = {"band1": band1_series, "band2": band2_series, "band3": band3_series}
        peak_vals = []
        drew_any = False
        for key in ("band1", "band2", "band3"):
            s = series_map[key]
            if s is None:
                continue
            ts, vs = s
            m = (ts >= t_lo) & (ts <= t_hi)
            if m.sum() < 2:
                continue
            col, name = ZOOM_LINE[key]
            zt = (ts[m] - t0) / 3600.0
            zv = _log_safe(vs[m])
            ax.plot(zt, zv, lw=1.4, color=col, label=f"{name}", alpha=0.9)
            peak_vals.append(np.nanmax(vs[m]))
            drew_any = True

        if drew_any:
            ax.set_yscale("log")
            unit_lbl = "W/m^2"
            peak_disp = f"{max(peak_vals):.2e} W/m^2"
        else:
            # nothing calibrated covers this window -- fall back to raw counts
            zh = hrs[a:b]
            ax.plot(zh, flux[a:b], lw=1.2, color="steelblue", label="counts")
            unit_lbl = "cts/s"
            peak_disp = f"{ev['peak_flux']:.1f} cts/s"

        pk_h = (ev["peak"] - t0) / 3600.0
        ax.axvline(pk_h, ls=":", color="black", lw=1.3, alpha=0.7,
                  label=f"peak {unix_to_utc(ev['peak'])[11:16]} UTC")
        dur_min = (ev["end"] - ev["start"]) / 60.0
        ax.set_title(f"BIGGEST EVENT ZOOM  |  {rows[big_i]['class_approx']}\n"
                     f"peak={peak_disp}   dur={dur_min:.1f} min", fontweight="bold")
        ax.xaxis.set_major_formatter(utc_formatter(t0))
        ax.set_xlabel("UTC Time (HH:MM)"); ax.set_ylabel(unit_lbl)
        ax.legend(fontsize=7, framealpha=0.92)
        ax.grid(alpha=0.15, which="major")
    else:
        _placeholder(ax, "No events detected")

    # ---- row 1 right : QPP detection -------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    qd = phys[big_i]["qpp_diag"] if big_i is not None else None
    if qd is not None:
        ax.plot(qd["t_min"], qd["excess"], lw=0.8, color="darkorange", label="excess above background")
        ax.plot(qd["t_min"], qd["smooth"], lw=1.8, ls="--", color="red", label="smoothed")
        if len(qd["peaks"]):
            ax.plot(qd["t_min"][qd["peaks"]], qd["excess"][qd["peaks"]], "ko", ms=8,
                    label=f"QPP sub-peaks (n={len(qd['peaks'])})")
        per_txt = f"period = {qd['period']/60:.1f} min" if qd["period"] else "no clear period"
        ax.set_title(f"QPP DETECTION  |  {len(qd['peaks'])} sub-peaks  |  {per_txt}",
                     fontweight="bold")
        ax.xaxis.set_major_formatter(utc_formatter_minutes(t0))
        ax.set_xlabel("UTC Time (HH:MM)"); ax.set_ylabel("Excess counts/s")
        ax.legend(fontsize=7, framealpha=0.92)
        ax.grid(alpha=0.15)
    else:
        _placeholder(ax, "Event too short for\nQPP analysis")

    # ---- row 2 left : QPP periodogram -------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    if qd is not None and len(qd["power"]) > 2:
        cpm = qd["freqs"] * 60.0                    # cycles per minute
        ax.semilogy(cpm[1:], qd["power"][1:], lw=1.3, color="purple")
        if qd["period"]:
            ax.axvline(60.0 / qd["period"], ls="--", color="red", lw=1.5,
                       label=f"dom. period={qd['period']/60:.1f} min")
            ax.legend(fontsize=7, framealpha=0.92)
        ax.set_title("QPP PERIODOGRAM (FFT)  |  power at each frequency", fontweight="bold")
        ax.set_xlabel("Frequency (cycles / minute)"); ax.set_ylabel("Power (log)")
        ax.grid(alpha=0.15, which="both")
    else:
        _placeholder(ax, "No periodogram\n(event too short)")

    # ---- row 2 right : coronal heating -------------------------------------------
    ax = fig.add_subplot(gs[2, 1])
    tp = phys[big_i]["temp"] if big_i is not None else None
    if tp is not None and "curve_t" in tp:
        tmin = (tp["curve_t"] - t0) / 60.0
        ax.plot(tmin, tp["curve_T_MK"], lw=1.6, color="orangered")
        ax.axhline(tp["temp_MK"], ls=":", color="gray", lw=1.0)
        ax.set_title(f"CORONAL HEATING  |  peak T = {tp['temp_MK']:.1f} MK", fontweight="bold")
        ax.xaxis.set_major_formatter(utc_formatter_minutes(t0))
        ax.set_xlabel("UTC Time (HH:MM)"); ax.set_ylabel("Temperature (MK)")
        ax.grid(alpha=0.15)
    else:
        _placeholder(ax, "No spectral data available\n(SoLEXS data required\nfor coronal heating analysis)")

    # ---- row 3 : particle acceleration (full width) ----------------------------------------
    ax = fig.add_subplot(gs[3, :])
    slices = phys[big_i]["accel_slices"] if big_i is not None else []
    ac = phys[big_i]["accel"] if big_i is not None else None
    if slices:
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(slices)))
        for sl, col in zip(slices, colors):
            t_lbl = unix_to_utc(sl["t"])[11:16]
            ax.loglog(sl["energies"], sl["counts"], "o", color=col, ms=4, alpha=0.65,
                      mec="none", label=f"{t_lbl} UTC  g={sl['gamma']:.2f} d={sl['delta']:.2f}")
            xf = np.array([sl["energies"].min(), sl["energies"].max()])
            yf = 10 ** (sl["intercept"] + (-sl["gamma"]) * np.log10(xf))
            ax.loglog(xf, yf, "--", color=col, lw=1.8)
        ax.set_title(f"PARTICLE ACCELERATION  |  {len(slices)} time-resolved fits "
                     "(ARF area-corrected)", fontweight="bold")
        ax.set_xlabel("Energy (keV)"); ax.set_ylabel("Counts / effective area (log)")
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(alpha=0.15, which="both")
    elif ac and ac.get("energies") and len(ac["energies"]) >= 2:
        E = np.array(ac["energies"]); F = np.array(ac["diffs"])
        ax.loglog(E, F, "o", color="navy", ms=9, mec="none", label="HEL1OS band rates")
        if ac["gamma"]:
            xf = np.linspace(E.min(), E.max(), 50)
            yf = F[0] * (xf / E[0]) ** (-ac["gamma"])
            ax.loglog(xf, yf, "--", color="crimson", lw=1.8,
                      label=f"power law  gamma={ac['gamma']:.2f}  delta={ac['delta']:.2f}")
        ax.set_title("PARTICLE ACCELERATION  |  hard X-ray power-law fit "
                     "(default bands, whole event -- no per-channel spectra this date)",
                     fontweight="bold")
        ax.set_xlabel("Energy (keV)"); ax.set_ylabel("Rate / keV"); ax.legend(fontsize=8)
        ax.grid(alpha=0.15, which="both")
    else:
        _placeholder(ax, "No hard X-ray events detected\nfor spectral fitting"
                     if mode != "SOLEXS" else
                     "No HEL1OS data for this date\n(hard X-ray fitting needs HEL1OS)")

    # ---- row 4 : Neupert check ---------------------------------------------------------
    ax = fig.add_subplot(gs[4, :])
    if mode == "COMBINED" and hxr is not None:
        sxr_display = flux_wm2 if have_solexs_flux else flux
        sxr_unit = "SXR W/m^2" if have_solexs_flux else "SXR Counts/s"
        dsxr = np.gradient(sxr_display, bt)
        ax.plot(hrs, sxr_display, lw=1.0, color="steelblue", label="Soft X-ray (SXR) - thermal afterglow")
        ax2 = ax.twinx()
        if have_hxr_flux:
            ht, hw = hxr_wm2
            ax2.plot((ht - t0) / 3600.0, hw, lw=0.5, color="crimson", alpha=0.75,
                     label="Hard X-ray (HXR, real W/m^2) - particle spark")
            hxr_unit = "HXR W/m^2"
        else:
            ht, hc = hxr
            ax2.plot((ht - t0) / 3600.0, hc, lw=0.5, color="crimson", alpha=0.75,
                     label="Hard X-ray (HXR, counts) - particle spark")
            hxr_unit = "HXR counts/s (dSXR/dt scaled)"
        if have_solexs_flux:
            hxr_peak = np.nanmax(hw) if have_hxr_flux else np.nanmax(hc)
            dsxr_peak = np.nanmax(np.maximum(dsxr, 0))
            dsxr_scale = (0.5 * hxr_peak / dsxr_peak) if dsxr_peak > 0 else 1.0
        else:
            dsxr_scale = 30.0
        ax2.plot(hrs, np.maximum(dsxr, 0) * dsxr_scale, lw=1.0, ls="--", color="green",
                 label=f"d(SXR)/dt x{dsxr_scale:.2g} - Neupert prediction")
        nc = phys[big_i]["neupert"] if big_i is not None else None
        sub = f"  |  biggest-event correlation = {nc:.2f}" if nc is not None else ""
        ax.set_title("NEUPERT EFFECT CHECK\nHXR (red) should match derivative of SXR "
                     f"(green dashed) - particle acceleration driving thermal heating{sub}",
                     fontweight="bold")
        if have_solexs_flux:
            ax.set_yscale("log")
            ax.set_ylim(bottom=max(np.nanmin(sxr_display[sxr_display > 0]) * 0.5, 1e-12))
        ax.xaxis.set_major_formatter(utc_formatter(t0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))   # one tick per hour
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.set_xlabel("UTC Time (HH:MM)")
        ax.grid(alpha=0.15)
        ax.set_ylabel(sxr_unit, color="steelblue")
        ax2.set_ylabel(hxr_unit, color="crimson")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    else:
        _placeholder(ax, "Neupert check needs BOTH instruments on the same date\n"
                     "(SXR derivative vs HXR light-curve)")

    os.makedirs(out_folder, exist_ok=True)
    out = unique_report_path(out_folder, date, mode)
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"    Report -> {out}")
    if AUTO_OPEN_REPORTS:
        open_file_default(out)


def make_event_table_png(date, mode, rows, events, out_folder):
    """
    The event catalog as its OWN PNG, separate from the main report figure.
    Shows EVERY detected event (no 12-row cap -- the main report's table
    panel was overflowing into neighbouring panels at high event counts, so
    the table now lives here instead where it can be as tall as it needs).
    Figure height grows with row count so it never overflows itself.
    """
    if not rows:
        return None
    heads = ["Start (UTC)", "Peak (UTC)", "Duration", "Flux (W/m2)", "Class", "Band", "Instr."]
    band_short = {"band1 (2.8-20 keV, soft)": "soft (2.8-20)",
                 "band2 (20-60 keV, medium)": "medium (20-60)",
                 "band3 (60-150 keV, hard)": "hard (60-150)"}
    paired = sorted(zip(rows, events), key=lambda pe: pe[1]["peak_flux"], reverse=True)

    cells = []
    any_starred = False
    for r, ev in paired:
        fx = r.get("peak_flux_wm2") or r.get("hard_peak_flux_wm2")
        band_lbl = band_short.get(r.get("dominant_energy_band"), "-")
        if band_lbl != "-" and r.get("n_bands_covered", 0) < 2:
            band_lbl += "*"
            any_starred = True
        dur_min = (ev["end"] - ev["start"]) / 60.0
        cells.append([
            unix_to_utc(ev["start"])[11:19], unix_to_utc(ev["peak"])[11:19],
            f"{dur_min:.1f} min",
            f"{fx:.2e}" if fx else "-",
            r["class_approx"],
            band_lbl,
            "SXR+HXR" if mode == "COMBINED" else ("HXR" if mode == "HEL1OS" else "SXR"),
        ])

    n = len(cells)
    fig_h = max(2.5, 0.32 * n + 1.8)     # grows with row count -- never cramped
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    ax.set_title(f"Aditya-L1 Event Catalog  |  {mode}  |  {date[:4]}-{date[4:6]}-{date[6:]}  "
                f"|  all {n} detected events  (sorted by peak flux)",
                fontsize=12, fontweight="bold", pad=14)
    tbl = ax.table(cellText=cells, colLabels=heads, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.5)
    if any_starred:
        ax.text(0.5, -0.02, "* = only one energy band had real spectra coverage during "
                "this event (soft/medium/hard comparison wasn't possible, not a real result)",
                transform=ax.transAxes, ha="center", fontsize=7, color="gray", style="italic")

    os.makedirs(out_folder, exist_ok=True)
    out = unique_report_path(out_folder, date, f"{mode}_events")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"    Event table -> {out}")
    if AUTO_OPEN_REPORTS:
        open_file_default(out)
    return out
