"""Multi-target router: highest-prob winner with agreement gate.

Decision rules (per locked spec):

* Uniform per-target threshold: ``UNIFORM_THRESHOLD = 0.80`` -- a target
  fires only if its stacking confidence is >= 0.80 AND the argmax class
  is LONG or SHORT.
* Minimum agreement: at least 2 targets must fire on the same direction.
* Direction unanimity: all firing targets must agree on direction --
  any disagreement aborts.
* Winner selection: among the firing set, the target with the highest
  confidence wins. The router emits its ``TARGET_CONFIGS`` SL/TP/max_hold
  so the bot can pass them per-trade into ``MultiTradeManager.open_trade``.
* No cooldown is enforced anywhere -- the next FIRE signal is taken
  immediately on the next cycle (subject to ``lock_mode``).

The router always returns a ``RouterDecision`` (never ``None``) so the
data-service writer can log *why* a cycle did not fire (WARMING /
NO_FIRING / NOT_ENOUGH_TARGETS / NO_AGREEMENT) for offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from core.multitarget_predictor import MultiTargetPrediction

UNIFORM_THRESHOLD = 0.80
MIN_AGREEMENT = 2


@dataclass
class RouterDecision:
    """Outcome of one router cycle."""

    reason: str                          # FIRE | WARMING | NO_FIRING | NOT_ENOUGH_TARGETS | NO_AGREEMENT
    winner_target: str = ""              # "" if not FIRE
    direction: str = ""                  # "LONG" | "SHORT" | ""
    confidence: float = 0.0              # 0.0 if not FIRE
    sl_pct: float = 0.0
    tp_pct: float = 0.0
    max_hold_bars: int = 0
    firing_targets: list[str] = field(default_factory=list)


class MultiTargetRouter:
    """Apply uniform threshold + agreement + highest-prob winner logic."""

    def __init__(self,
                 target_configs: Mapping[str, Mapping],
                 threshold: float = UNIFORM_THRESHOLD,
                 min_agreement: int = MIN_AGREEMENT):
        self.target_configs = target_configs
        self.threshold = threshold
        self.min_agreement = min_agreement

    def route(self, pred: MultiTargetPrediction) -> RouterDecision:
        if pred.warming or any(s == "WARMING" for s in pred.signals.values()):
            return RouterDecision(reason="WARMING")

        firing: list[tuple[str, str, float]] = []
        for target, signal in pred.signals.items():
            conf = pred.confidences.get(target, 0.0)
            if signal in ("LONG", "SHORT") and conf >= self.threshold:
                firing.append((target, signal, conf))

        if not firing:
            return RouterDecision(reason="NO_FIRING")

        if len(firing) < self.min_agreement:
            return RouterDecision(
                reason="NOT_ENOUGH_TARGETS",
                firing_targets=[t for t, _, _ in firing],
            )

        directions = {d for _, d, _ in firing}
        if len(directions) > 1:
            return RouterDecision(
                reason="NO_AGREEMENT",
                firing_targets=[t for t, _, _ in firing],
            )

        winner_target, winner_dir, winner_conf = max(firing, key=lambda x: x[2])
        cfg = self.target_configs[winner_target]
        return RouterDecision(
            reason="FIRE",
            winner_target=winner_target,
            direction=winner_dir,
            confidence=winner_conf,
            sl_pct=float(cfg["sl"]),
            tp_pct=float(cfg["tp"]),
            max_hold_bars=int(cfg["max_hold"]),
            firing_targets=[t for t, _, _ in firing],
        )
