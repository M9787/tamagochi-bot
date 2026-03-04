"""
Visualization: Generate ML result charts for CatBoost binary training.

Plots:
  1. Equity curves by confidence threshold
  2. ROC curve (binary)
  3. Confusion matrix heatmap (2x2)
  4. Confidence comparison bar charts
  5. Feature importance
  6. Win rate over time (rolling)
  7. Drawdown chart
  8. Summary metrics table
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
from sklearn.metrics import roc_curve, auc

RESULTS_DIR = Path(__file__).parent / "results"

THRESHOLD_STYLES = {
    0.5: {'linestyle': '-', 'alpha': 1.0, 'label': '@0.50', 'color': '#2196F3'},
    0.55: {'linestyle': '-', 'alpha': 0.9, 'label': '@0.55', 'color': '#00BCD4'},
    0.6: {'linestyle': '--', 'alpha': 0.85, 'label': '@0.60', 'color': '#FF9800'},
    0.65: {'linestyle': '--', 'alpha': 0.8, 'label': '@0.65', 'color': '#E91E63'},
    0.7: {'linestyle': ':', 'alpha': 0.7, 'label': '@0.70', 'color': '#4CAF50'},
}

CLASS_NAMES = ['SHORT', 'LONG']
CLASS_COLORS = ['#E91E63', '#4CAF50']


def load_results() -> dict:
    results_path = RESULTS_DIR / "training_results.json"
    with open(results_path, 'r') as f:
        return json.load(f)


def load_summary() -> pd.DataFrame:
    summary_path = RESULTS_DIR / "summary_table.csv"
    return pd.read_csv(summary_path)


def plot_equity_curves(results: dict, save_path: Path):
    """Equity curves for each confidence threshold."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for thresh_str, m in sorted(results.items(), key=lambda x: float(x[0])):
        thresh = float(thresh_str)
        eq = m.get('equity_curve', [0])
        if len(eq) <= 1:
            continue
        style = THRESHOLD_STYLES.get(thresh, {
            'linestyle': '-', 'alpha': 1.0,
            'label': f'@{thresh}', 'color': '#999'
        })
        profit = m.get('total_profit_pct', 0)
        n_trades = m.get('n_trades', 0)
        win_rate = m.get('win_rate', 0)
        ax.plot(eq,
                label=f"{style['label']} ({profit:.0f}%, {n_trades}T, WR {win_rate:.0f}%)",
                color=style['color'], linestyle=style['linestyle'],
                alpha=style['alpha'], linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Profit (%)')
    ax.set_title('Equity Curves by Confidence Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_roc_curve(results: dict, save_path: Path):
    """Binary ROC curve."""
    # Find baseline (first threshold with proba data)
    baseline = None
    for key in sorted(results.keys(), key=float):
        if 'y_pred_proba' in results[key] and 'y_test_full' in results[key]:
            baseline = results[key]
            break

    if not baseline:
        print("  No probability data for ROC curve")
        return

    y_proba = np.array(baseline['y_pred_proba'])
    y_true = np.array(baseline['y_test_full'])

    # Binary: P(LONG) = column 1
    p_long = y_proba[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, p_long)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='#2196F3', linewidth=2.5,
            label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC = 0.500)')

    # Mark optimal threshold (Youden's J)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
               label=f'Best threshold = {thresholds[best_idx]:.3f}')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve (Binary: SHORT vs LONG)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_confusion_matrix(results: dict, save_path: Path):
    """Confusion matrix heatmap (2x2)."""
    baseline = None
    for key in sorted(results.keys(), key=float):
        if 'confusion_matrix' in results[key]:
            baseline = results[key]
            break

    if not baseline or 'confusion_matrix' not in baseline:
        print("  No confusion matrix data found")
        return

    cm = np.array(baseline['confusion_matrix'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    n_classes = cm.shape[0]

    # Raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES[:n_classes], fontsize=11)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CLASS_NAMES[:n_classes], fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Counts', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Normalized (row-wise = recall per class)
    ax = axes[1]
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    cm_norm = cm.astype(float) / row_sums
    im2 = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, f'{cm_norm[i, j]:.1%}', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(CLASS_NAMES[:n_classes], fontsize=11)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(CLASS_NAMES[:n_classes], fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Normalized (Recall)', fontsize=13, fontweight='bold')
    fig.colorbar(im2, ax=ax, shrink=0.8)

    fig.suptitle('Confusion Matrix (Binary: SHORT vs LONG)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_confidence_comparison(summary: pd.DataFrame, save_path: Path):
    """Bar chart comparing key metrics across thresholds."""
    metrics = ['win_rate', 'total_profit_pct', 'sharpe_ratio', 'n_trades',
               'accuracy', 'max_drawdown_pct']
    titles = ['Win Rate %', 'Total Profit %', 'Sharpe Ratio',
              'Number of Trades', 'Accuracy', 'Max Drawdown %']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    thresholds = sorted(summary['threshold'].unique())
    x = np.arange(len(thresholds))
    x_labels = [THRESHOLD_STYLES.get(t, {}).get('label', f'@{t}') for t in thresholds]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3][idx % 3]
        if metric not in summary.columns:
            ax.set_visible(False)
            continue
        values = []
        colors = []
        for t in thresholds:
            row = summary[summary['threshold'] == t]
            values.append(row[metric].values[0] if len(row) > 0 else 0)
            colors.append(THRESHOLD_STYLES.get(t, {'color': '#999'})['color'])

        bars = ax.bar(x, values, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, values):
            fmt = f'{val:.1f}' if isinstance(val, float) else str(val)
            ypos = bar.get_height()
            va = 'bottom' if ypos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    fmt, ha='center', va=va, fontsize=8, fontweight='bold')

        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8, rotation=30)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Performance by Confidence Threshold', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_feature_importance(results: dict, save_path: Path):
    """Top feature importances."""
    fi = None
    for thresh_str, m in results.items():
        if 'feature_importance' in m and m['feature_importance']:
            fi = m['feature_importance']
            break

    if not fi:
        print("  No feature importance data found")
        return

    # Filter out zero-importance features
    fi = [(name, val) for name, val in fi if val > 0]
    fi = fi[:30]

    if not fi:
        print("  All features have zero importance")
        return

    names = [f[0] for f in fi]
    values = [f[1] for f in fi]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color='#2196F3', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {len(fi)} Features (non-zero)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_win_rate_over_time(results: dict, save_path: Path):
    """Rolling win rate over trade sequence."""
    baseline = None
    for key in sorted(results.keys(), key=float):
        if 'trades' in results[key]:
            baseline = results[key]
            break

    if not baseline or 'trades' not in baseline:
        print("  No trade data for win rate chart")
        return

    trades = baseline['trades']
    gains = [t['gain_pct'] for t in trades]
    wins = [1 if g > 0 else 0 for g in gains]

    if len(wins) < 20:
        print("  Too few trades for rolling win rate")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Rolling win rate
    window = min(100, len(wins) // 5)
    rolling_wr = pd.Series(wins).rolling(window).mean() * 100

    ax = axes[0]
    ax.plot(rolling_wr, color='#2196F3', linewidth=1.5, label=f'Rolling WR ({window} trades)')
    ax.axhline(y=33.3, color='red', linestyle='--', alpha=0.7, label='Break-even @ 2:1 RR (33.3%)')
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50%')
    ax.set_ylabel('Win Rate %')
    ax.set_title(f'Rolling Win Rate (window={window})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    # Cumulative P&L
    ax = axes[1]
    equity = np.cumsum(gains)
    ax.plot(equity, color='#4CAF50' if equity[-1] > 0 else '#E91E63', linewidth=1.5)
    ax.fill_between(range(len(equity)), equity, 0,
                    where=np.array(equity) >= 0, color='#4CAF50', alpha=0.15)
    ax.fill_between(range(len(equity)), equity, 0,
                    where=np.array(equity) < 0, color='#E91E63', alpha=0.15)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Profit (%)')
    ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_drawdown(results: dict, save_path: Path):
    """Drawdown chart from peak equity."""
    baseline = None
    for key in sorted(results.keys(), key=float):
        if 'equity_curve' in results[key] and len(results[key]['equity_curve']) > 1:
            baseline = results[key]
            break

    if not baseline:
        print("  No equity data for drawdown chart")
        return

    equity = np.array(baseline['equity_curve'])
    if len(equity) < 2:
        print("  Too few points for drawdown")
        return

    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Equity + peak
    ax = axes[0]
    ax.plot(equity, color='#2196F3', linewidth=1.5, label='Equity')
    ax.plot(peak, color='#FF9800', linewidth=1, linestyle='--', alpha=0.7, label='Peak')
    ax.set_ylabel('Cumulative Profit (%)')
    ax.set_title('Equity vs Peak', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax = axes[1]
    ax.fill_between(range(len(drawdown)), drawdown, 0, color='#E91E63', alpha=0.4)
    ax.plot(drawdown, color='#E91E63', linewidth=1)
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(f'Drawdown (Max: {drawdown.min():.0f}%)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_summary_table(summary: pd.DataFrame, save_path: Path):
    """Summary metrics table as image."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')

    display_cols = ['threshold', 'accuracy', 'accuracy_confident', 'roc_auc',
                    'n_trades', 'win_rate', 'total_profit_pct', 'sharpe_ratio',
                    'max_drawdown_pct']
    col_labels = ['Threshold', 'Accuracy', 'Conf Acc', 'ROC AUC', 'Trades',
                  'Win Rate %', 'Profit %', 'Sharpe', 'Max DD %']

    available = [c for c in display_cols if c in summary.columns]
    available_labels = [col_labels[display_cols.index(c)] for c in available]

    cell_text = []
    for _, row in summary.iterrows():
        cells = []
        for c in available:
            val = row[c]
            cells.append(f'{val:.2f}' if isinstance(val, float) else str(val))
        cell_text.append(cells)

    table = ax.table(cellText=cell_text, colLabels=available_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title('Training Results Summary (Binary: SHORT vs LONG)',
                 fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def generate_all_plots():
    """Generate all visualization plots."""
    print("=" * 50)
    print("GENERATING VISUALIZATIONS (Binary)")
    print("=" * 50)

    results = load_results()
    summary = load_summary()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n1. Equity curves:")
    plot_equity_curves(results, RESULTS_DIR / "equity_curves.png")

    print("\n2. ROC curve:")
    plot_roc_curve(results, RESULTS_DIR / "roc_curves.png")

    print("\n3. Confusion matrix:")
    plot_confusion_matrix(results, RESULTS_DIR / "confusion_matrix_viz.png")

    print("\n4. Confidence comparison:")
    plot_confidence_comparison(summary, RESULTS_DIR / "confidence_comparison.png")

    print("\n5. Feature importance:")
    plot_feature_importance(results, RESULTS_DIR / "feature_importance.png")

    print("\n6. Win rate over time:")
    plot_win_rate_over_time(results, RESULTS_DIR / "win_rate_over_time.png")

    print("\n7. Drawdown:")
    plot_drawdown(results, RESULTS_DIR / "drawdown.png")

    print("\n8. Summary table:")
    plot_summary_table(summary, RESULTS_DIR / "summary_table.png")

    print(f"\nAll 8 plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()
