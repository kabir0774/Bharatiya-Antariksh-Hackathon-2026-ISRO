from .common import *
from .data import *
from .features import *
from .windowing import *
from .evaluation import *
from .models import *
from .calibration import *
from .validation import *
from .live import *



def main():
    print("=" * 62)
    print("  FORECAST PIPELINE  --  30-minute-ahead flare prediction")
    print("  ISRO Aditya-L1 Hackathon 2026")
    print("=" * 62)
    print(f"Lookback: {LOOKBACK_MIN} min | Lead: {LEAD_MIN} min | Step: {STEP_MIN} min")
    print(f"Decay exclusion: {DECAY_EXCLUDE_MIN} min | Significant flare: >= {SIGNIFICANT_RATIO}x background\n")

    df = build_dataset()
    train_df, test_df = time_split(df)

    if train_df["label"].sum() == 0 or test_df["label"].sum() == 0:
        print("\n[WARNING] Train or test set has ZERO positive examples.")
        print("This usually means: not enough dates, or SIGNIFICANT_RATIO is too strict")
        print("for your data. The model will still train but AUC/precision/recall will")
        print("be meaningless (undefined) until this is fixed -- get more dates or lower")
        print("SIGNIFICANT_RATIO.")

    best_name, best, scaler, X_test_sc, y_test, test_df = train_models(train_df, test_df)

    # threshold sweep -- the default 0.5 threshold rarely fits a 2-3%
    # positive-rate problem well; find the actual best operating points.
    sweep, best_thresholds = sweep_thresholds(y_test, best["prob"])
    reliability_report(y_test, best["prob"])
    best_thresh = best_thresholds.get(OPERATING_OBJECTIVE, best_thresholds["f1"])
    # save the operating point so forecast mode can use the SAME alarm line
    with open(os.path.join(MODEL_OUTPUT_FOLDER, "operating_point.json"), "w") as fh:
        json.dump(dict(threshold=float(best_thresh), objective=OPERATING_OBJECTIVE,
                       all_thresholds={k: float(v) for k, v in best_thresholds.items()},
                       model_name=best_name, n_features=N_FEATURES), fh, indent=2)
    print(f"\nThreshold sweep ({best_name}, test set):")
    print(f"  {'thr':>5} {'prec':>6} {'rec':>6} {'f0.5':>6} {'f1':>6} {'f2':>6} {'tss':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    for r in sweep:
        marker = f"  <- chosen ({OPERATING_OBJECTIVE})" if r["threshold"] == best_thresh else ""
        print(f"  {r['threshold']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f} "
              f"{r['f0_5']:6.3f} {r['f1']:6.3f} {r['f2']:6.3f} {r['tss']:+6.3f} "
              f"{r['tp']:4d} {r['fp']:4d} {r['fn']:4d}{marker}")
    print("\nOperating points (the false-alarm vs missed-flare trade-off, made explicit):")
    print(f"  fewer false alarms (F0.5): threshold {best_thresholds['f0.5']:.3f}")
    print(f"  balanced           (F1):   threshold {best_thresholds['f1']:.3f}")
    print(f"  catch more flares  (F2):   threshold {best_thresholds['f2']:.3f}")
    print(f"  best skill score   (TSS):  threshold {best_thresholds['tss']:.3f}")
    print(f"  ACTIVE: OPERATING_OBJECTIVE='{OPERATING_OBJECTIVE}' -> threshold {best_thresh:.3f}. "
          f"Change OPERATING_OBJECTIVE at the top of this file to move the operating point.")

    # robustness: fold-averaged skill across every date, not just one split
    walk_forward_summary(df, best["model"], oversample=best.get("oversample", False))

    lead_times = measure_lead_time(test_df, best["prob"], threshold=best_thresh)
    if lead_times:
        print(f"\nLead time: {len(lead_times)} flares caught early, "
              f"average {np.mean(lead_times):.1f} min before start "
              f"(min {np.min(lead_times):.1f}, max {np.max(lead_times):.1f})")
    else:
        print("\nLead time: no flares in the test set were caught before they started.")

    tally = nowcast_referee_tally(test_df, best["prob"], threshold=best_thresh)
    print(f"\nNowcast-referee tally (test set, threshold={best_thresh:.3f}):")
    print(f"  Correct alarm  (real flare, we warned):     {tally['correct_alarm']}")
    print(f"  Correct quiet  (no flare, we stayed quiet): {tally['correct_quiet']}")
    print(f"  False alarm    (no flare, we warned):       {tally['false_alarm']}")
    print(f"  Missed flare   (real flare, we missed it):  {tally['missed']}")

    out_csv = os.path.join(MODEL_OUTPUT_FOLDER, "test_predictions.csv")
    test_df_out = test_df.copy()
    test_df_out["prob"] = best["prob"]
    test_df_out["alarm_default_0.5"] = best["prob"] >= ALARM_THRESHOLD
    test_df_out[f"alarm_{OPERATING_OBJECTIVE}_{best_thresh:.3f}"] = best["prob"] >= best_thresh
    test_df_out.to_csv(out_csv, index=False)
    print(f"\nTest predictions written to: {out_csv}")
    print(f"Model + scaler saved in: {MODEL_OUTPUT_FOLDER}")
    print("\nDONE.")
