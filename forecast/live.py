from .common import *
from .data import *
from .features import *
from .windowing import *
from .calibration import *
from .validation import *

# =============================================================================
#  MAIN
# =============================================================================

def _hhmm(unix_t):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(float(unix_t), tz=timezone.utc).strftime("%H:%M")


def _forecast_at_moment(date, day, when, model, scaler, calibrator, thr, op):
    """
    A single live-moment forecast: HARD-TRUNCATE every data stream at the
    chosen time (flux, spectra, HEL1OS bands, detected events -- nothing
    after the cutoff exists as far as the model is concerned), build the one
    window ending at that moment, and answer in plain language:

        "P% chance a significant flare starts in the next 30 minutes"

    plus a per-n-minute breakdown for n in [5..30]. The model natively
    predicts ONE number (the 30-minute probability); the n-minute rows
    spread it across the half hour assuming the risk is uniform in time
    (P_n = 1 - (1-P30)^(n/30)) -- an approximation, and labeled as one.
    Afterwards, since the file does contain the future, it reveals what
    actually happened next -- the model just never saw it.
    """
    from datetime import datetime, timezone
    try:
        hh, mm = when.split(":")
        hh, mm = int(hh), int(mm)
        # validate BEFORE using it -- "18:334" used to silently parse as
        # 18:00 + 334 minutes = 23:34 UTC and produce a real answer to the
        # WRONG moment, with nothing telling the person their typo changed
        # the question. A malformed time must fail loudly, not quietly drift.
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(f"{hh:02d}:{mm:02d} is not a valid 24-hour time")
        day0 = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp()
        cutoff = day0 + hh * 3600 + mm * 60
    except Exception as e:
        print(f"[!] Couldn't read '{when}' as a time ({e}). Use HH:MM, e.g. 14:30 "
              f"-- hours 00-23, minutes 00-59.")
        return
    bt = day["bt"]
    if cutoff < bt[0] + LOOKBACK_MIN * 60:
        print(f"[!] {when} UTC is before enough data exists on this day "
              f"(data starts {_hhmm(bt[0])} UTC; earliest usable moment is "
              f"{_hhmm(bt[0] + LOOKBACK_MIN * 60)} UTC).")
        return
    if cutoff > bt[-1]:
        print(f"[!] {when} UTC is after this day's data ends ({_hhmm(bt[-1])} UTC).")
        return

    # ---- THE TRUNCATION: the world ends at `cutoff` -------------------------
    m = bt <= cutoff
    day_cut = dict(day)
    day_cut["bt"], day_cut["flux"], day_cut["bg"] = bt[m], day["flux"][m], day["bg"][m]
    for k in ("bandA", "bandB", "bandC"):
        day_cut[k] = day[k][m] if day.get(k) is not None else None
    if day.get("helos_bt") is not None:
        hm = day["helos_bt"] <= cutoff
        day_cut["helos_bt"], day_cut["helos_flux"] = day["helos_bt"][hm], day["helos_flux"][hm]
    day_cut["helos_bands"] = [
        dict(b, t=b["t"][b["t"] <= cutoff], ctr=b["ctr"][b["t"] <= cutoff])
        for b in (day.get("helos_bands") or [])]
    if day.get("pi_data") is not None:
        pt, spec, chan = day["pi_data"]
        pm = pt <= cutoff
        day_cut["pi_data"] = (pt[pm], spec[pm], chan)
    day_cut["events"] = [e for e in day["events"] if e["start"] <= cutoff]

    wins, _ = build_windows_for_date(date, day_cut)
    wins = [w for w in wins if w["dense"] == 0]
    if not wins:
        print("[!] Not enough usable data before that moment to forecast.")
        return
    w = max(wins, key=lambda x: x["window_end_unix"])     # the window ending "now"

    p30 = float(calibrator.predict(
        model.predict_proba(scaler.transform(w["features"].reshape(1, -1)))[:, 1])[0])

    # plain-language context about the current moment -- STATE FIRST.
    # This system forecasts NEW flare starts; a flare already in progress is
    # the nowcaster's territory. Without saying which is which, "no alarm"
    # during an ongoing C-class flare reads like a lie. Ongoing-ness is
    # judged CAUSALLY: a detected start before the cutoff plus flux still
    # well above quiet right now (no peeking at when the flare ends).
    cur = day_cut["flux"][-1] / max(day_cut["bg"][-1], 1e-9)
    biggest = max((e["ratio"] for e in day_cut["events"]), default=0.0)
    last_ev = max(day_cut["events"], key=lambda e: e["start"]) if day_cut["events"] else None
    ongoing = last_ev is not None and cur >= 2.0
    print(f"\n=== FORECAST for {date} at {when} UTC "
          f"(model saw data up to this moment ONLY) ===")
    print(f"\n  RIGHT NOW: ", end="")
    if ongoing:
        print(f"A FLARE IS IN PROGRESS -- it started at {_hhmm(last_ev['start'])} UTC "
              f"and the Sun is still at {cur:.1f}x its quiet level.")
        print(f"  (Characterizing the flare happening now is the NOWCASTER's job --")
        print(f"   run nowcast_physics.py for its class, temperature and energy.")
        print(f"   THIS tool answers a different question: is ANOTHER flare coming?)")
    else:
        print(f"the Sun is at {cur:.1f}x its quiet level"
              + (f"; biggest flare so far today reached {biggest:.0f}x background."
                 if day_cut["events"] else "; no flares detected so far today."))
    print(f"\n  CHANCE A NEW SIGNIFICANT FLARE STARTS IN THE NEXT 30 MINUTES: {100*p30:.1f}%")
    is_custom = abs(thr - float(op["threshold"])) > 1e-9
    thr_label = "custom sensitivity, set for this forecast" if is_custom \
        else f"'{op['objective']}' balance point chosen at training"
    print(f"  Alarm line is {100*thr:.1f}% -> "
          + ("ALARM: a new flare looks likely. " if p30 >= thr
             else "no alarm for a NEW flare. ")
          + f"({thr_label})")
    print(f"\n  Spread across the half hour (assumes the risk is uniform in time --")
    print(f"  an approximation; the model's native answer is the 30-minute number):")
    for n in (5, 10, 15, 20, 25, 30):
        pn = 1.0 - (1.0 - p30) ** (n / 30.0)
        bar = "#" * int(round(40 * pn))
        print(f"    next {n:2d} min: {100*pn:5.1f}%  {bar}")
    print(f"\n  (Per-GOES-class probabilities need class-labeled training data --")
    print(f"   with the current dataset that split would be statistically meaningless,")
    print(f"   so this system honestly reports one 'significant flare' probability.)")

    # ---- the reveal: what actually happened next ----------------------------
    wins_full, _ = build_windows_for_date(date, day)
    sig_starts = sorted({x["flare_start"] for x in wins_full
                         if x["label"] == 1 and x["flare_start"]})
    upcoming = [fs for fs in sig_starts if cutoff < fs <= cutoff + LEAD_MIN * 60]
    # smaller flares in the same window -- below the significance bar, but a
    # human WILL see them in the nowcast catalog, so the reveal must own them
    minor = [e for e in day["events"]
             if cutoff < e["start"] <= cutoff + LEAD_MIN * 60
             and not any(abs(e["start"] - fs) < 60 for fs in sig_starts)]
    print(f"\n  THE REVEAL (the model never saw this): ", end="")
    if upcoming:
        mins = (upcoming[0] - cutoff) / 60.0
        verdict = "the forecast was RIGHT" if p30 >= thr else "the forecast MISSED it"
        print(f"a significant flare really did start {mins:.0f} minutes later, "
              f"at {_hhmm(upcoming[0])} UTC -- {verdict}.")
    else:
        nxt = [fs for fs in sig_starts if fs > cutoff]
        verdict = ("staying quiet was RIGHT" if p30 < thr
                   else "this alarm turned out to be FALSE")
        print(f"no NEW SIGNIFICANT flare started in the next 30 minutes -- {verdict}."
              + (f" (Next significant one came at {_hhmm(nxt[0])} UTC.)" if nxt else
                 " (None for the rest of the day.)"))
    if minor:
        e0 = min(minor, key=lambda e: e["start"])
        print(f"\n  FULL HONESTY: {len(minor)} smaller flare(s) DID start in that window "
              f"(first at {_hhmm(e0['start'])} UTC, reaching {e0['ratio']:.1f}x the background "
              f"at the time). This system's alarm is trained ONLY on 'significant' flares --")
        print(f"  a jump of >= {SIGNIFICANT_RATIO:.0f}x over the local background. A flare "
              f"erupting when the background is already elevated can be big in absolute class "
              f"yet a small RELATIVE jump, and it deliberately does not trigger this alarm.")


def forecast_mode():
    """
    PLAIN-LANGUAGE FORECAST REPLAY for one day of real data.

    The model only ever looks at the 30 minutes BEHIND each moment (it is
    causal by construction), so a full-day PRADAN file can be replayed as if
    it were arriving live: the code walks through the day, and at each step
    the model sees only the past. Because the file also contains what
    happened NEXT, the replay grades itself at the end -- this is a
    hindcast, the standard way operational forecasters validate.
    """
    # ---- load the trained brain ---------------------------------------------
    needed = ["best_model.pkl", "scaler.pkl", "calibrator.pkl", "operating_point.json"]
    missing = [f for f in needed if not os.path.exists(os.path.join(MODEL_OUTPUT_FOLDER, f))]
    if missing:
        print(f"\n[!] No trained model found (missing: {', '.join(missing)}).")
        print("    Run the program again and choose TRAIN first.")
        return
    model = joblib.load(os.path.join(MODEL_OUTPUT_FOLDER, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_OUTPUT_FOLDER, "scaler.pkl"))
    calibrator = joblib.load(os.path.join(MODEL_OUTPUT_FOLDER, "calibrator.pkl"))
    with open(os.path.join(MODEL_OUTPUT_FOLDER, "operating_point.json")) as fh:
        op = json.load(fh)
    if op.get("n_features") != N_FEATURES:
        print(f"\n[!] The saved model expects {op.get('n_features')} features but this code "
              f"builds {N_FEATURES}. The code changed since training -- please TRAIN again first.")
        return
    thr = float(op["threshold"])

    # ---- let the person pick sensitivity RIGHT NOW, no retrain needed -------
    # OPERATING_OBJECTIVE only controls which of 4 pre-computed points gets
    # saved at training time -- and in a given run, F1 and F2's optimal
    # thresholds can land very close together (a flat stretch in the
    # precision/recall curve), so switching objectives and retraining doesn't
    # always move the alarm line by much. This gives direct control instead:
    # pick any sensitivity, this instant, from the SAME trained model.
    all_t = op.get("all_thresholds", {})
    print(f"\nHow sensitive should the alarm be?")
    print(f"  Trained default ('{op['objective']}' balance point): {100*thr:.1f}%")
    if all_t:
        print(f"  Other pre-computed points from training: "
              + "   ".join(f"{k}={100*v:.1f}%" for k, v in all_t.items()))
    choice = input(
        "  Press Enter for the trained default, or type a lower number (e.g. 3) to catch\n"
        "  MORE flares at the cost of MORE false alarms, or a higher number for the reverse: "
        ).strip()
    if choice:
        try:
            custom = float(choice)
            if not (0 < custom < 100):
                raise ValueError
            thr = custom / 100.0
            print(f"  Using a custom alarm line: {100*thr:.1f}% "
                  f"(more sensitive than default = catches more flares, more false alarms)"
                  if thr < float(op["threshold"]) else
                  f"  Using a custom alarm line: {100*thr:.1f}% "
                  f"(less sensitive than default = fewer false alarms, may miss more flares)")
        except ValueError:
            print(f"  Couldn't read '{choice}' as a percentage -- using the trained default.")

    # ---- which day? ---------------------------------------------------------
    dates = ncp.find_solexs_dates(ncp.SOLEXS_FOLDER)
    if not dates:
        print("[!] No SoLEXS data found. Check SOLEXS_FOLDER in nowcast_physics.py.")
        return
    print("\nDays with data on this computer:")
    for d in sorted(dates):
        print(f"  {d}")
    date = input("\nWhich day should I forecast? (type it like 20240531): ").strip()
    if date not in dates:
        print(f"[!] I don't have data for '{date}'. Pick one from the list above.")
        return

    when = input(
        "Forecast from a specific moment, or replay the whole day?\n"
        "  - type a time like 14:30 (UTC) -> the model sees ONLY data up to that moment\n"
        "  - just press Enter            -> replay the whole day\n"
        "Your choice: ").strip()

    print(f"\nLoading {date}...")
    helos_paths = ncp.find_helos_for_date(ncp.HELOS_FOLDER, date)
    day = assemble_day(date, dates[date], helos_paths)

    if when:
        _forecast_at_moment(date, day, when, model, scaler, calibrator, thr, op)
        return

    print("Replaying the whole day as if it were happening live...")
    print("(The model only ever sees the past 30 minutes at each step -- the rest of")
    print(" the file is used ONLY at the end, to check how well the forecast did.)\n")
    wins, n_sig = build_windows_for_date(date, day)
    wins = [w for w in wins if w["dense"] == 0]          # natural 5-min grid only
    if not wins:
        print("[!] Not enough usable data on this day to forecast.")
        return
    wins.sort(key=lambda w: w["window_end_unix"])

    X = np.vstack([w["features"] for w in wins])
    prob = calibrator.predict(model.predict_proba(scaler.transform(X))[:, 1])
    alarm = prob >= thr

    # ---- the story of the day, in plain words -------------------------------
    _is_custom = abs(thr - float(op["threshold"])) > 1e-9
    _thr_label = "custom sensitivity, set for this forecast" if _is_custom \
        else f"set during training, '{op['objective']}' balance point"
    print(f"Alarm line: {100*thr:.1f}% ({_thr_label}).")
    print(f"Every 5 minutes the model answers one question: \"how likely is a big flare")
    print(f"in the NEXT 30 minutes?\" It stays quiet below the alarm line and warns above it.\n")
    in_alarm = False
    n_alerts = 0
    for w, p, a in zip(wins, prob, alarm):
        t = _hhmm(w["window_end_unix"])
        if a and not in_alarm:
            n_alerts += 1
            print(f"  {t} UTC  [ALARM RAISED]   chance jumped to {100*p:.1f}% -- "
                  f"a significant flare looks likely within 30 minutes")
            in_alarm = True
        elif not a and in_alarm:
            print(f"  {t} UTC  [all clear]      chance back down to {100*p:.1f}%")
            in_alarm = False
    if n_alerts == 0:
        peak = float(prob.max())
        print(f"  The model stayed quiet the whole day "
              f"(highest chance it ever saw: {100*peak:.1f}% at "
              f"{_hhmm(wins[int(np.argmax(prob))]['window_end_unix'])} UTC).")

    # ---- self-grade against what really happened ----------------------------
    print(f"\nNow the honest part -- what ACTUALLY happened on {date}:")
    sig_starts = sorted({w["flare_start"] for w in wins if w["label"] == 1 and w["flare_start"]})
    if not sig_starts:
        print(f"  No significant flares occurred. "
              + ("The model correctly stayed quiet all day. Well done."
                 if n_alerts == 0 else
                 f"The model raised {n_alerts} alarm(s) that turned out to be false. "
                 f"(False alarms on quiet days are this system's known weak spot -- "
                 f"more training days is the cure.)"))
        return
    caught = 0
    for fs in sig_starts:
        early = [w["window_end_unix"] for w, a in zip(wins, alarm)
                 if a and fs - LEAD_MIN * 60 <= w["window_end_unix"] < fs]
        if early:
            lead = (fs - min(early)) / 60.0
            caught += 1
            print(f"  Flare at {_hhmm(fs)} UTC -> CAUGHT, warned {lead:.0f} minutes before it started")
        else:
            print(f"  Flare at {_hhmm(fs)} UTC -> MISSED, no alarm was active in the 30 minutes before")
    false_alerts = 0
    for w, a in zip(wins, alarm):
        if a and not any(0 <= fs - w["window_end_unix"] <= LEAD_MIN * 60 for fs in sig_starts):
            false_alerts += 1
    print(f"\n  Scorecard: caught {caught} of {len(sig_starts)} significant flare(s); "
          f"{false_alerts} five-minute step(s) spent on false alarm.")
