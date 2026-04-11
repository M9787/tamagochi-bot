"""Live multi-target inference: 24 base models + 8 stacking models.

The data service constructs a 518-column V10 feature row each cycle.
``MultiTargetPredictor`` runs the full multi-target flow on that row:

1. **Base inference** -- for each of 8 targets, average the
   ``predict_proba`` of 3 seeded CatBoost base models, producing 8
   3-class probability vectors.
2. **Ring buffer push** -- per-target probability vectors plus the raw
   V10 feature projection are appended to a 289-row ring buffer (288
   history rows + the current row, the size required by ``lag288`` and
   ``rolling(288)`` features in the stacking meta-feature builder).
3. **Meta feature construction** -- the buffer is fed to
   ``build_stacking_meta_features``, producing the 258-column meta
   matrix; the **last** row is the live meta vector.
4. **Stacking inference** -- each of 8 stacking models predicts on the
   live meta vector, producing per-target 3-class probabilities.
5. **Signal derivation** -- canonical
   ``argmax -> threshold(0.80) -> NO_TRADE`` per target.

Cold-start: if the persisted ring buffer state is missing or short of
289 rows, the predictor eagerly backfills by running all 24 base models
on the last 289 rows of ``features.csv`` (falling back to a slice of
``feature_matrix_v10.parquet`` shipped with the image). The cost is
~6900 CatBoost calls, paid once at first boot. Subsequent restarts
reload the persisted state in <100 ms.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from core.multitarget_feature_builder import (
    EXPECTED_META_FEATURE_COUNT,
    build_stacking_meta_features,
    load_top_raw_features_union,
)
from model_training.multitarget_config import SEEDS, TARGET_NAMES

logger = logging.getLogger(__name__)


# 288 history rows + 1 current row. shift(288) on a 289-row series at the
# last index resolves to the first row -- the only configuration that
# produces non-trivial lag288 / rolling(288) live values matching training.
RING_BUFFER_SIZE = 289

# Per-target threshold used to derive a per-target signal from stacking
# probabilities. The router applies the same value -- duplicated here so
# the predictor's ``signals`` field is meaningful in isolation (logging,
# debugging, telemetry).
SIGNAL_THRESHOLD = 0.80

CLASS_NAMES = {0: "NO_TRADE", 1: "LONG", 2: "SHORT"}


@dataclass
class MultiTargetPrediction:
    """Snapshot of one inference cycle across all 8 targets."""

    time: pd.Timestamp
    base_probs: dict[str, np.ndarray]      # T -> (3,) averaged across seeds
    stacking_probs: dict[str, np.ndarray]  # T -> (3,) from stacking head
    signals: dict[str, str]                # T -> LONG | SHORT | NO_TRADE | WARMING
    confidences: dict[str, float]          # T -> max stacking prob
    warming: bool = False                  # True if any target is still warming


class MultiTargetPredictor:
    """Stateful 24-base + 8-stacking inference engine."""

    def __init__(self,
                 models_root: Path,
                 top_raw_features_path: Path,
                 v10_feature_names: list[str],
                 state_path: Path | None = None,
                 backfill_features_path: Path | None = None,
                 targets: list[str] = TARGET_NAMES,
                 seeds: list[int] = SEEDS):
        """
        Parameters
        ----------
        models_root : directory containing ``base_models/`` and
            ``stacking/`` subdirs.
        top_raw_features_path : path to ``top_raw_features_union.json``
            (the persisted Phase A.4 dump).
        v10_feature_names : ordered list of the 518 V10 feature names
            the data service produces. Used to project incoming feature
            rows down to the raw subset and to validate input shape.
        state_path : where to persist the ring buffer between restarts.
            None means in-memory only.
        backfill_features_path : path to a parquet/csv source of
            historical V10 features to use for cold-start backfill. If
            None, the predictor starts in WARMING state and warms over
            ~24h of live cycles.
        targets : target identifiers, default ``TARGET_NAMES``.
        seeds : base-model seeds to average, default ``SEEDS``.
        """
        self.models_root = Path(models_root)
        self.top_raw_features_path = Path(top_raw_features_path)
        self.v10_feature_names = list(v10_feature_names)
        self.state_path = Path(state_path) if state_path else None
        self.backfill_features_path = Path(backfill_features_path) if backfill_features_path else None
        self.targets = list(targets)
        self.seeds = list(seeds)

        self.top_raw_features = load_top_raw_features_union(self.top_raw_features_path)
        logger.info(f"Top-raw union loaded: {len(self.top_raw_features)} features")

        missing_raw = [c for c in self.top_raw_features if c not in set(self.v10_feature_names)]
        if missing_raw:
            raise RuntimeError(
                f"top_raw_features_union references {len(missing_raw)} columns "
                f"not present in V10 feature names: {missing_raw[:5]}..."
            )
        self._raw_feature_indices = np.array(
            [self.v10_feature_names.index(c) for c in self.top_raw_features],
            dtype=np.int64,
        )

        self.base_models: dict[tuple[str, int], CatBoostClassifier] = {}
        for target in self.targets:
            for seed in self.seeds:
                path = self.models_root / "base_models" / f"base_model_{target}_s{seed}.cbm"
                if not path.exists():
                    raise FileNotFoundError(f"Missing base model: {path}")
                m = CatBoostClassifier()
                m.load_model(str(path))
                self.base_models[(target, seed)] = m
                logger.info(f"Loaded model: base {target} s{seed}")

        self.stacking_models: dict[str, CatBoostClassifier] = {}
        for target in self.targets:
            path = self.models_root / "stacking" / f"stacking_model_{target}.cbm"
            if not path.exists():
                raise FileNotFoundError(f"Missing stacking model: {path}")
            m = CatBoostClassifier()
            m.load_model(str(path))
            self.stacking_models[target] = m
            logger.info(f"Loaded model: stacking {target}")

        self._timestamps: deque[str] = deque(maxlen=RING_BUFFER_SIZE)
        self._base_buffers: dict[str, deque[np.ndarray]] = {
            t: deque(maxlen=RING_BUFFER_SIZE) for t in self.targets
        }
        self._raw_buffer: deque[np.ndarray] = deque(maxlen=RING_BUFFER_SIZE)

        if not self._load_state():
            logger.info("Multitarget state file empty/missing — attempting cold-start backfill")
            self._cold_start_backfill()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load_state(self) -> bool:
        """Load ring buffer from disk. Returns True if buffer is full."""
        if self.state_path is None or not self.state_path.exists():
            return False
        try:
            with self.state_path.open("r") as fh:
                state = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to load multitarget state ({exc}); starting empty")
            return False

        ts = state.get("timestamps", [])
        base = state.get("base_buffers", {})
        raw = state.get("raw_buffer", [])

        if not ts or len(ts) != len(raw):
            return False
        for t in self.targets:
            if t not in base or len(base[t]) != len(ts):
                return False

        self._timestamps = deque(ts, maxlen=RING_BUFFER_SIZE)
        for t in self.targets:
            self._base_buffers[t] = deque(
                (np.asarray(v, dtype=np.float64) for v in base[t]),
                maxlen=RING_BUFFER_SIZE,
            )
        self._raw_buffer = deque(
            (np.asarray(v, dtype=np.float64) for v in raw),
            maxlen=RING_BUFFER_SIZE,
        )
        logger.info(f"Multitarget state loaded: {len(self._timestamps)} rows")
        return len(self._timestamps) >= RING_BUFFER_SIZE

    def save_state(self) -> None:
        """Atomic write + 3-backup rotation, mirroring feature_state.json."""
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamps": list(self._timestamps),
            "base_buffers": {
                t: [v.tolist() for v in self._base_buffers[t]] for t in self.targets
            },
            "raw_buffer": [v.tolist() for v in self._raw_buffer],
        }
        for i in range(3, 0, -1):
            src = self.state_path.with_suffix(f".json.bak{i - 1}" if i > 1 else ".json.bak")
            dst = self.state_path.with_suffix(f".json.bak{i}")
            if src.exists():
                shutil.copy2(src, dst)
        if self.state_path.exists():
            shutil.copy2(self.state_path, self.state_path.with_suffix(".json.bak"))
        fd, tmp = tempfile.mkstemp(
            prefix=self.state_path.name + ".",
            dir=str(self.state_path.parent),
        )
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self.state_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    # ------------------------------------------------------------------
    # Cold-start
    # ------------------------------------------------------------------
    def _cold_start_backfill(self) -> None:
        """Eagerly backfill 289 rows by running base models on history."""
        if self.backfill_features_path is None or not self.backfill_features_path.exists():
            logger.warning(
                "No backfill source available; predictor will warm over ~24h of live cycles"
            )
            return

        path = self.backfill_features_path
        logger.info(f"Cold-start backfill from {path}")
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        df = df.sort_values("time").tail(RING_BUFFER_SIZE).reset_index(drop=True)

        if len(df) < RING_BUFFER_SIZE:
            logger.warning(
                f"Backfill source has only {len(df)} rows (<{RING_BUFFER_SIZE}); "
                f"populating partially -- predictor will report WARMING until full"
            )

        feature_arr = df[self.v10_feature_names].to_numpy(dtype=np.float64)
        timestamps = df["time"].astype(str).tolist()

        t0 = time.time()
        for i, (ts, row) in enumerate(zip(timestamps, feature_arr)):
            base_probs = self._run_base_models(row)
            self._timestamps.append(ts)
            for t in self.targets:
                self._base_buffers[t].append(base_probs[t])
            self._raw_buffer.append(row[self._raw_feature_indices])
            if (i + 1) % 32 == 0:
                logger.info(f"  Backfill progress: {i + 1}/{len(df)}")
        logger.info(
            f"Cold-start backfill complete: {len(df)} rows in {time.time() - t0:.1f}s"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _run_base_models(self, feature_row: np.ndarray) -> dict[str, np.ndarray]:
        """Average predict_proba across seeds for each target."""
        x = feature_row.reshape(1, -1)
        out: dict[str, np.ndarray] = {}
        for t in self.targets:
            stacked = np.zeros(3, dtype=np.float64)
            for s in self.seeds:
                stacked += self.base_models[(t, s)].predict_proba(x)[0]
            out[t] = stacked / len(self.seeds)
        return out

    def predict(self, feature_row: pd.Series, now: pd.Timestamp) -> MultiTargetPrediction:
        """Run one cycle of multi-target inference.

        Parameters
        ----------
        feature_row : a single 518-feature V10 row (Series indexed by
            ``self.v10_feature_names`` or array-like in the same order).
        now : the timestamp this row corresponds to.

        Returns
        -------
        ``MultiTargetPrediction`` with per-target base probs, stacking
        probs, signals, confidences, and a global ``warming`` flag the
        router uses to short-circuit.
        """
        if isinstance(feature_row, pd.Series):
            arr = feature_row.reindex(self.v10_feature_names).to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(feature_row, dtype=np.float64).ravel()
        if arr.shape[0] != len(self.v10_feature_names):
            raise ValueError(
                f"Feature row has {arr.shape[0]} values, expected {len(self.v10_feature_names)}"
            )

        base_probs = self._run_base_models(arr)
        ts = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)

        self._timestamps.append(str(ts))
        for t in self.targets:
            self._base_buffers[t].append(base_probs[t])
        self._raw_buffer.append(arr[self._raw_feature_indices])

        warming = len(self._timestamps) < RING_BUFFER_SIZE

        if warming:
            return MultiTargetPrediction(
                time=ts,
                base_probs=base_probs,
                stacking_probs={t: np.zeros(3) for t in self.targets},
                signals={t: "WARMING" for t in self.targets},
                confidences={t: 0.0 for t in self.targets},
                warming=True,
            )

        meta_row = self._build_live_meta_row()
        if meta_row.shape[0] != EXPECTED_META_FEATURE_COUNT:
            raise RuntimeError(
                f"Live meta row has {meta_row.shape[0]} cols, "
                f"expected {EXPECTED_META_FEATURE_COUNT} -- training/live drift"
            )

        x_meta = meta_row.reshape(1, -1)
        stacking_probs: dict[str, np.ndarray] = {}
        signals: dict[str, str] = {}
        confidences: dict[str, float] = {}
        for t in self.targets:
            probs = self.stacking_models[t].predict_proba(x_meta)[0]
            stacking_probs[t] = probs
            cls = int(np.argmax(probs))
            conf = float(probs[cls])
            confidences[t] = conf
            if cls in (1, 2) and conf >= SIGNAL_THRESHOLD:
                signals[t] = CLASS_NAMES[cls]
            else:
                signals[t] = "NO_TRADE"

        return MultiTargetPrediction(
            time=ts,
            base_probs=base_probs,
            stacking_probs=stacking_probs,
            signals=signals,
            confidences=confidences,
            warming=False,
        )

    def _build_live_meta_row(self) -> np.ndarray:
        """Reconstruct base/raw DataFrames from the ring buffers and call
        the shared meta-feature builder. Return the last row as a 1-D
        ``np.ndarray`` (no ``time`` column)."""
        times = pd.to_datetime(list(self._timestamps))
        base_data = {"time": times}
        for t in self.targets:
            arr = np.stack(list(self._base_buffers[t]))  # (289, 3)
            tl = t.lower()
            base_data[f"{tl}_prob_nt"] = arr[:, 0]
            base_data[f"{tl}_prob_long"] = arr[:, 1]
            base_data[f"{tl}_prob_short"] = arr[:, 2]
        base_df = pd.DataFrame(base_data)

        raw_arr = np.stack(list(self._raw_buffer))  # (289, len(top_raw))
        raw_data = {"time": times}
        for i, col in enumerate(self.top_raw_features):
            raw_data[col] = raw_arr[:, i]
        raw_df = pd.DataFrame(raw_data)

        meta_df = build_stacking_meta_features(
            base_df, raw_df, self.targets, self.top_raw_features
        )
        feature_cols = [c for c in meta_df.columns if c != "time"]
        return meta_df[feature_cols].iloc[-1].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        return len(self._timestamps) >= RING_BUFFER_SIZE

    def buffer_lengths(self) -> Mapping[str, int]:
        return {t: len(self._base_buffers[t]) for t in self.targets}
