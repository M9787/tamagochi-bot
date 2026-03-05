"""V10 Young+Adult Only — drop Balzak/Gran TF features, keep 5M-4H + global."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from model_training.build_labels import load_labels
from model_training.train_v10_long_oos import (
    prepare_3class_labels, align_features_labels, evaluate_at_threshold,
    select_val_threshold, save_plots, save_trade_log, save_cross_seed_equity,
    analyze_feature_groups, CLASS_NAMES, MODEL_PARAMS, SPLIT, TRADE_COOLDOWN,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ENCODED_DIR = Path(__file__).parent / "encoded_data"
RESULTS_DIR = Path(__file__).parent / "results_v10" / "young_adult_oos"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 777]
THRESHOLDS = [0.42, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SL_PCT, TP_PCT = 2.0, 4.0
MIN_ITERATIONS = 200
DROP_TFS = ['_3D', '_1D', '_12H', '_8H', '_6H']


def run():
    t_start = time.time()
    logger.info("=" * 90)
    logger.info("V10 YOUNG+ADULT ONLY — No Balzak/Gran TFs (5M/15M/30M/1H/2H/4H + global)")
    logger.info("=" * 90)

    features = pd.read_parquet(ENCODED_DIR / "feature_matrix_v10.parquet")
    all_cols = [c for c in features.columns if c != 'time']
    feature_cols = [c for c in all_cols if not any(tf in c for tf in DROP_TFS)]
    logger.info(f"Features: {len(feature_cols)} kept / {len(all_cols)} total "
                f"(dropped {len(all_cols)-len(feature_cols)} Balzak+Gran)")

    labels = load_labels()
    labels = prepare_3class_labels(labels)
    X_aligned, y_df, times = align_features_labels(features, labels)
    del features

    train_mask = (times >= SPLIT['train_start']) & (times < SPLIT['train_end'])
    val_mask = (times >= SPLIT['train_end']) & (times < SPLIT['val_end'])
    test_mask = (times >= SPLIT['embargo_end']) & (times < SPLIT['test_end'])

    X_train = X_aligned.loc[train_mask.values, feature_cols].reset_index(drop=True)
    X_val = X_aligned.loc[val_mask.values, feature_cols].reset_index(drop=True)
    X_test = X_aligned.loc[test_mask.values, feature_cols].reset_index(drop=True)
    y_train = y_df.loc[train_mask.values, 'label_3class'].values.astype(np.int8)
    y_val = y_df.loc[val_mask.values, 'label_3class'].values.astype(np.int8)
    y_test = y_df.loc[test_mask.values, 'label_3class'].values.astype(np.int8)
    test_times = times[test_mask].reset_index(drop=True)

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    seed_results = []
    all_fi = {}

    for seed in SEEDS:
        logger.info(f"\n{'='*70}")
        logger.info(f"SEED {seed}")
        seed_t0 = time.time()

        params = MODEL_PARAMS.copy()
        params['random_seed'] = seed
        model = CatBoostClassifier(**params)
        model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), use_best_model=True)

        best_iter = model.get_best_iteration()
        if best_iter < MIN_ITERATIONS:
            params_min = params.copy()
            params_min['iterations'] = MIN_ITERATIONS
            params_min.pop('early_stopping_rounds', None)
            model = CatBoostClassifier(**params_min)
            model.fit(Pool(X_train, y_train))
            best_iter = MIN_ITERATIONS

        fi = model.get_feature_importance()
        fi_sorted = sorted(zip(feature_cols, fi), key=lambda x: -x[1])
        n_used = int((np.array(fi) > 0).sum())

        logger.info(f"Best iter: {best_iter} | Features used: {n_used}/{len(feature_cols)}")
        logger.info("Top 20:")
        for fname, fval in fi_sorted[:20]:
            is_v10 = any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_']) or \
                      fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                                'dow_cos', 'is_ny_session', 'convergence_volume',
                                'crossing_atr', 'cascade_volume', 'reversal_conviction')
            tag = " [V10]" if is_v10 else ""
            logger.info(f"  {fname}: {fval:.2f}{tag}")
            if fname not in all_fi:
                all_fi[fname] = []
            all_fi[fname].append(fval)
        for fname, fval in fi_sorted[20:40]:
            if fname not in all_fi:
                all_fi[fname] = []
            all_fi[fname].append(fval)

        groups, gcounts = analyze_feature_groups(fi_sorted, feature_cols)
        logger.info("Feature groups:")
        for gn, gv in sorted(groups.items(), key=lambda x: -x[1]):
            logger.info(f"  {gn}: {gv:.2f} ({gcounts[gn]} feats)")

        # Eval
        y_proba_val = model.predict_proba(Pool(X_val))
        val_thresh, val_profit, _ = select_val_threshold(
            y_proba_val, y_val, THRESHOLDS, TRADE_COOLDOWN)

        y_proba_test = model.predict_proba(Pool(X_test))
        threshold_results = {}
        for thresh in THRESHOLDS:
            threshold_results[f"{thresh:.2f}"] = evaluate_at_threshold(
                y_proba_test, y_test, thresh, cooldown=TRADE_COOLDOWN)

        logger.info(f"\nTest results:")
        logger.info(f"  {'Thresh':>7} {'Trades':>7} {'Prec':>6} {'L_prec':>7} "
                     f"{'S_prec':>7} {'Profit':>9} {'PF':>6} {'Sharpe':>7}")
        logger.info("  " + "-" * 70)
        for thresh in THRESHOLDS:
            m = threshold_results[f"{thresh:.2f}"]
            marker = " <-VAL" if abs(thresh - val_thresh) < 0.001 else ""
            logger.info(f"  {thresh:>7.2f} {m['n_trades_simulated']:>7} "
                        f"{m['trade_precision']:>6.3f} {m['long_precision']:>7.3f} "
                        f"{m['short_precision']:>7.3f} {m['total_profit_pct']:>8.1f}% "
                        f"{m['profit_factor']:>6.2f} {m['sharpe']:>7.3f}{marker}")

        at070 = threshold_results.get('0.70', {})
        honest = threshold_results[f"{val_thresh:.2f}"]
        seed_time = time.time() - seed_t0

        auc = threshold_results[f"{THRESHOLDS[0]:.2f}"]['roc_auc_macro']
        logger.info(f"\n=> s{seed}: AUC={auc:.3f} | "
                    f"@{val_thresh:.2f}: {honest['total_profit_pct']:.0f}% | "
                    f"@0.70: {at070.get('total_profit_pct', 0):.0f}% | {seed_time:.0f}s")

        try:
            save_plots(y_proba_test, y_test, test_times, threshold_results,
                       seed, RESULTS_DIR, val_thresh)
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

        save_trade_log(y_proba_test, y_test, test_times, val_thresh,
                       seed, RESULTS_DIR, TRADE_COOLDOWN)
        model.save_model(str(RESULTS_DIR / f"s{seed}_model.cbm"))

        seed_results.append({
            'seed': seed, 'best_iteration': best_iter, 'n_features_used': n_used,
            'roc_auc_macro': auc,
            'val_selected_threshold': val_thresh,
            'val_selected_profit': honest['total_profit_pct'],
            'profit_at_070': at070.get('total_profit_pct', 0),
            'equity_at_070': at070.get('equity_curve', []),
            'val_equity_curve': honest.get('equity_curve', []),
            'feature_groups': {k: float(v) for k, v in groups.items()},
            'threshold_summary': [{
                'threshold': thresh,
                'n_trades': threshold_results[f"{thresh:.2f}"]['n_trades_simulated'],
                'trade_precision': threshold_results[f"{thresh:.2f}"]['trade_precision'],
                'profit': threshold_results[f"{thresh:.2f}"]['total_profit_pct'],
                'profit_factor': threshold_results[f"{thresh:.2f}"]['profit_factor'],
                'sharpe': threshold_results[f"{thresh:.2f}"]['sharpe'],
            } for thresh in THRESHOLDS],
        })

    # Cross-seed
    logger.info(f"\n\n{'='*90}")
    logger.info("CROSS-SEED SUMMARY — Young+Adult Only")
    logger.info(f"{'='*90}")

    aucs = [sr['roc_auc_macro'] for sr in seed_results]
    p70 = [sr['profit_at_070'] for sr in seed_results]
    honest_p = [sr['val_selected_profit'] for sr in seed_results]
    logger.info(f"AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    logger.info(f"@0.70: {[f'{p:.0f}%' for p in p70]} mean={np.mean(p70):.0f}%")
    logger.info(f"Honest: {[f'{p:.0f}%' for p in honest_p]} mean={np.mean(honest_p):.0f}%")

    logger.info(f"\nPer-threshold (mean):")
    logger.info(f"  {'Thresh':>7} {'Trades':>7} {'Prec':>6} {'Profit':>9} {'PF':>6} {'Sharpe':>7}")
    for tidx, thresh in enumerate(THRESHOLDS):
        t_avg = np.mean([sr['threshold_summary'][tidx]['n_trades'] for sr in seed_results])
        p_avg = np.mean([sr['threshold_summary'][tidx]['trade_precision'] for sr in seed_results])
        pr_avg = np.mean([sr['threshold_summary'][tidx]['profit'] for sr in seed_results])
        pf_avg = np.mean([sr['threshold_summary'][tidx]['profit_factor'] for sr in seed_results])
        sh_avg = np.mean([sr['threshold_summary'][tidx]['sharpe'] for sr in seed_results])
        logger.info(f"  {thresh:>7.2f} {t_avg:>7.0f} {p_avg:>6.3f} "
                    f"{pr_avg:>8.1f}% {pf_avg:>6.2f} {sh_avg:>7.3f}")

    fi_avg = sorted([(k, float(np.mean(v))) for k, v in all_fi.items()], key=lambda x: -x[1])
    logger.info(f"\nTop 20 features (avg):")
    for fname, fval in fi_avg[:20]:
        is_v10 = any(fname.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_']) or \
                  fname in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                            'dow_cos', 'is_ny_session', 'convergence_volume',
                            'crossing_atr', 'cascade_volume', 'reversal_conviction')
        tag = " [V10]" if is_v10 else ""
        logger.info(f"  {fname}: {fval:.2f}{tag}")

    v10_in_top20 = sum(1 for f, _ in fi_avg[:20]
                       if any(f.startswith(p) for p in ['xw_', 'xtf_', 'corr_velocity_'])
                       or f in ('xtf_corr_agreement', 'hour_sin', 'hour_cos', 'dow_sin',
                                'dow_cos', 'is_ny_session', 'convergence_volume',
                                'crossing_atr', 'cascade_volume', 'reversal_conviction'))
    logger.info(f"V10 in top-20: {v10_in_top20}/20")

    logger.info(f"\nGroup importance (avg):")
    avg_groups = {}
    for sr in seed_results:
        for gn, gv in sr['feature_groups'].items():
            if gn not in avg_groups:
                avg_groups[gn] = []
            avg_groups[gn].append(gv)
    for gn in sorted(avg_groups.keys()):
        logger.info(f"  {gn}: {np.mean(avg_groups[gn]):.2f}")

    elapsed = time.time() - t_start
    logger.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Compare vs full V10
    logger.info(f"\n{'='*90}")
    logger.info("COMPARISON: Young+Adult vs Full V10")
    logger.info(f"  Full V10 @0.70: +349% mean (508 features)")
    logger.info(f"  Y+A only @0.70: +{np.mean(p70):.0f}% mean ({len(feature_cols)} features)")
    logger.info(f"{'='*90}")

    pd.DataFrame(fi_avg, columns=['feature', 'avg_importance']).to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False)
    try:
        save_cross_seed_equity(seed_results, RESULTS_DIR)
    except:
        pass


if __name__ == "__main__":
    run()
