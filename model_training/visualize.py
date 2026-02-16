"""
Visualization: Generate ML result charts for ATR-based training with confidence thresholds.

- Per-version: equity curves at different thresholds
- Cross-version: summary heatmap (versions x thresholds)
- Confidence comparison bar charts
- Feature importance for V5
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

VERSION_COLORS = {
    'V1_Flat': '#2196F3', 'V2_Score': '#FF9800',
    'V3_Binary': '#4CAF50', 'V4_Calendar': '#E91E63',
    'V5_Unified': '#9C27B0'
}

THRESHOLD_STYLES = {
    0.0: {'linestyle': '-', 'alpha': 1.0, 'label': 'No filter'},
    0.6: {'linestyle': '--', 'alpha': 0.8, 'label': 'Conf >= 0.6'},
    0.7: {'linestyle': ':', 'alpha': 0.6, 'label': 'Conf >= 0.7'},
}


def load_results() -> dict:
    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)


def load_summary() -> pd.DataFrame:
    summary_path = RESULTS_DIR / "summary_table.csv"
    return pd.read_csv(summary_path)


def plot_equity_per_version(results: dict, save_dir: Path):
    """Plot equity curves for each version at different confidence thresholds."""
    for version_name, thresholds in results.items():
        fig, ax = plt.subplots(figsize=(14, 6))
        color = VERSION_COLORS.get(version_name, '#999')

        for thresh_str, m in sorted(thresholds.items(), key=lambda x: float(x[0])):
            thresh = float(thresh_str)
            eq = m.get('equity_curve', [0])
            style = THRESHOLD_STYLES.get(thresh, {'linestyle': '-', 'alpha': 1.0, 'label': f'@{thresh}'})
            profit = m.get('total_profit_pct', 0)
            n_trades = m.get('n_trades', 0)
            ax.plot(eq,
                    label=f"{style['label']} ({profit:.0f}%, {n_trades} trades)",
                    color=color, linestyle=style['linestyle'],
                    alpha=style['alpha'], linewidth=2)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cumulative Profit (%)')
        ax.set_title(f'Equity Curves — {version_name} (ATR SL)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"equity_{version_name}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: equity_{version_name}.png")


def plot_confidence_comparison(summary: pd.DataFrame, save_path: Path):
    """Bar chart comparing metrics at 0.0/0.6/0.7 thresholds per model."""
    metrics = ['win_rate', 'total_profit_pct', 'sharpe_ratio', 'n_trades']
    titles = ['Win Rate %', 'Total Profit %', 'Sharpe Ratio', 'Number of Trades']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    versions = sorted(summary['version'].unique())
    thresholds = sorted(summary['threshold'].unique())
    x = np.arange(len(versions))
    width = 0.25

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2][idx % 2]
        for i, thresh in enumerate(thresholds):
            t_data = summary[summary['threshold'] == thresh]
            values = []
            for v in versions:
                row = t_data[t_data['version'] == v]
                values.append(row[metric].values[0] if len(row) > 0 else 0)
            style = THRESHOLD_STYLES.get(thresh, {'label': f'@{thresh}'})
            bars = ax.bar(x + i * width, values, width,
                         label=style['label'],
                         alpha=0.85)

        ax.set_xlabel('Model Version')
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x + width * (len(thresholds) - 1) / 2)
        ax.set_xticklabels(versions, fontsize=9, rotation=15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Performance by Confidence Threshold (ATR SL)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_summary_heatmap(summary: pd.DataFrame, save_path: Path):
    """Heatmap: versions x thresholds, colored by metric."""
    metrics = ['total_profit_pct', 'win_rate', 'sharpe_ratio', 'max_drawdown_pct']
    labels = ['Profit %', 'Win Rate %', 'Sharpe', 'MaxDD %']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 2][idx % 2]

        # Convert threshold to string for pivot
        summary_copy = summary.copy()
        summary_copy['thresh_label'] = summary_copy['threshold'].apply(
            lambda t: THRESHOLD_STYLES.get(t, {}).get('label', f'@{t}')
        )
        pivot = summary_copy.pivot_table(
            index='version', columns='thresh_label',
            values=metric, aggfunc='first'
        )

        # Sort columns by threshold order
        col_order = [THRESHOLD_STYLES[t]['label'] for t in sorted(THRESHOLD_STYLES.keys())
                     if THRESHOLD_STYLES[t]['label'] in pivot.columns]
        pivot = pivot[col_order]

        cmap = 'RdYlGn' if metric != 'max_drawdown_pct' else 'RdYlGn_r'
        im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    text = f'{val:.1f}' if abs(val) < 1000 else f'{val:.0f}'
                    ax.text(j, i, text, ha='center', va='center', fontsize=10,
                            fontweight='bold' if abs(val) > 100 else 'normal')

        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        ax.set_title(label, fontsize=13, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Model x Confidence Threshold — Performance Heatmaps', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_feature_importance(results: dict, save_path: Path):
    """Plot top feature importances for V5_Unified (or best available model)."""
    # Prefer V5, fallback to any model with feature_importance
    fi = None
    model_name = None
    for vname in ['V5_Unified', 'V1_Flat', 'V2_Score', 'V3_Binary']:
        if vname in results:
            # Get feature importance from the 0.0 threshold (baseline)
            for thresh_str, m in results[vname].items():
                if 'feature_importance' in m and m['feature_importance']:
                    fi = m['feature_importance']
                    model_name = vname
                    break
        if fi:
            break

    if not fi:
        print("  No feature importance data found")
        return

    # Take top 30
    fi = fi[:30]
    names = [f[0] for f in fi]
    values = [f[1] for f in fi]

    fig, ax = plt.subplots(figsize=(12, 10))
    y_pos = np.arange(len(names))
    color = VERSION_COLORS.get(model_name, '#2196F3')
    ax.barh(y_pos, values, color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top 30 Features — {model_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_best_models(summary: pd.DataFrame, save_path: Path):
    """Highlight the best model per threshold based on Sharpe."""
    fig, ax = plt.subplots(figsize=(12, 6))

    thresholds = sorted(summary['threshold'].unique())

    best_rows = []
    for t in thresholds:
        t_data = summary[summary['threshold'] == t]
        if len(t_data) == 0:
            continue
        best = t_data.loc[t_data['sharpe_ratio'].idxmax()]
        best_rows.append(best)

    if not best_rows:
        plt.close(fig)
        return

    best_df = pd.DataFrame(best_rows)

    x_labels = [THRESHOLD_STYLES.get(t, {}).get('label', f'@{t}') for t in thresholds]
    bars = ax.bar(range(len(x_labels)), best_df['total_profit_pct'],
                  color=[VERSION_COLORS.get(v, '#999') for v in best_df['version']],
                  alpha=0.85, width=0.6)

    for i, (bar, row) in enumerate(zip(bars, best_rows)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{row['version']}\nSharpe: {row['sharpe_ratio']:.1f}\nWR: {row['win_rate']:.0f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('Total Profit %', fontsize=12)
    ax.set_title('Best Model per Confidence Threshold (by Sharpe)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def generate_all_plots():
    """Generate all visualization plots."""
    print("=" * 50)
    print("GENERATING VISUALIZATIONS")
    print("=" * 50)

    results = load_results()
    summary = load_summary()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Per-version equity curves at different thresholds
    print("\nEquity curves per version:")
    plot_equity_per_version(results, RESULTS_DIR)

    # Confidence threshold comparison
    print("\nConfidence comparison:")
    plot_confidence_comparison(summary, RESULTS_DIR / "confidence_comparison.png")

    # Summary heatmap (versions x thresholds)
    print("\nSummary heatmaps:")
    plot_summary_heatmap(summary, RESULTS_DIR / "summary_heatmap.png")

    # Feature importance
    print("\nFeature importance:")
    plot_feature_importance(results, RESULTS_DIR / "feature_importance.png")

    # Best model per threshold
    print("\nBest models:")
    plot_best_models(summary, RESULTS_DIR / "best_models.png")

    print(f"\nAll plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()
