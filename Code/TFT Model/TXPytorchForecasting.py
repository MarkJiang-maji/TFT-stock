# ---------------------------------------------------#
#
#   File       : TXPytorchForecasting.py
#   Description: Training / evaluating TFT on TX futures data
#
# ----------------------------------------------------#

from __future__ import annotations

import os
import sys
import datetime as dt
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
import json
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric, QuantileLoss

from data import load_data as load_tft_data

warnings.filterwarnings("ignore")  # reduce noisy warnings from libs
torch.set_float32_matmul_precision("high")  # enable tensor cores for float32 matmul when available

_CUDA_USABLE: Optional[bool] = None


def _is_cuda_usable() -> bool:
    global _CUDA_USABLE
    if _CUDA_USABLE is not None:
        return _CUDA_USABLE
    if not torch.cuda.is_available():
        _CUDA_USABLE = False
        return _CUDA_USABLE
    try:
        _ = torch.cuda.get_device_properties(0)
        _CUDA_USABLE = _ is not None
    except Exception as exc:  # pragma: no cover - best effort guard
        warnings.warn(f"CUDA detected but unusable ({exc}); falling back to CPU.")
        _CUDA_USABLE = False
    return _CUDA_USABLE


def _get_pretty_device_name() -> str:
    if not _is_cuda_usable():
        return "CPU"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "GPU (unavailable)"


if not _is_cuda_usable():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.cuda.is_available = lambda: False  # type: ignore[assignment]


class BinaryCrossEntropyLoss(MultiHorizonMetric):
    """
    BCE loss wrapper that follows the MultiHorizonMetric interface used by TFT.
    """

    def __init__(self, reduction: str = "mean", pos_weight: Optional[float] = None) -> None:
        super().__init__(reduction=reduction)
        self.pos_weight = pos_weight

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = y_pred[..., 0]
        target = target.float()
        if target.ndim > logits.ndim:
            target = target.squeeze(-1)
        pos_w: Optional[torch.Tensor] = None
        if self.pos_weight is not None:
            pos_w = torch.as_tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_w)
        return losses.unsqueeze(-1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y_pred)


class BinaryAccuracy(MultiHorizonMetric):
    """
    Classification accuracy at a fixed threshold for TFT logging_metrics.
    """

    def __init__(self, threshold: float = 0.5, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.threshold = float(threshold)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(y_pred[..., 0])
        target = target.float()
        if target.ndim > probs.ndim:
            target = target.squeeze(-1)
        correct = (probs >= self.threshold) == (target >= 0.5)
        losses = (~correct).float()
        return losses.unsqueeze(-1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y_pred)


try:  # ensure checkpoints saved under __main__ can be reloaded when imported as a module
    sys.modules["__main__"].BinaryCrossEntropyLoss = BinaryCrossEntropyLoss
    sys.modules["__main__"].BinaryAccuracy = BinaryAccuracy
except Exception:
    pass


class TrainingMetricsLogger(Callback):
    """
    Collect train/validation losses each epoch for post-training analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: List[Dict[str, Optional[float]]] = []

    @staticmethod
    def _to_float(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # type: ignore[override]
        if getattr(trainer, "sanity_checking", False):
            return
        metrics = trainer.callback_metrics
        record: Dict[str, Optional[float]] = {"epoch": float(trainer.current_epoch)}
        train_loss_epoch = metrics.get("train_loss_epoch")
        train_loss = metrics.get("train_loss")
        record["train_loss"] = (
            self._to_float(train_loss_epoch) if train_loss_epoch is not None else self._to_float(train_loss)
        )
        record["val_loss"] = self._to_float(metrics.get("val_loss"))
        try:
            optimizer = trainer.optimizers[0] if trainer.optimizers else None
            if optimizer:
                record["lr"] = float(optimizer.param_groups[0]["lr"])
        except Exception:
            pass
        for key in (
            "train_accuracy",
            "train_bce",
            "train_precision",
            "train_recall",
            "train_f1",
            "train_best_accuracy",
            "train_best_threshold",
            "train_best_f1",
            "train_best_f1_threshold",
            "train_accuracy_at_best_f1",
            "val_accuracy",
            "val_bce",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_best_accuracy",
            "val_best_threshold",
            "val_best_f1",
            "val_best_f1_threshold",
            "val_accuracy_at_best_f1",
        ):
            record[key] = self._to_float(metrics.get(key))
        self.history.append(record)


class ClassificationEpochEvaluator(Callback):
    """
    Compute per-epoch train/val accuracy + BCE (with threshold sweep) and log to TensorBoard.
    """

    def __init__(
        self,
        owner: "TFT",
        thresholds: Optional[Iterable[float]] = None,
        train_eval_batches: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.owner = owner
        self.thresholds = list(thresholds) if thresholds is not None else [x / 100 for x in range(10, 91, 1)]
        self.train_eval_batches = train_eval_batches

    def _log_scalars(self, trainer: L.Trainer, prefix: str, metrics: Dict[str, float]) -> None:
        logger = trainer.logger
        writer = getattr(logger, "experiment", None)
        if writer is None:
            return
        step = trainer.current_epoch
        for key, value in metrics.items():
            if value is None:
                continue
            writer.add_scalar(f"{prefix}/{key}", value, step)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # type: ignore[override]
        if getattr(trainer, "sanity_checking", False):
            return
        owner = self.owner
        if owner.validation is None or owner.training is None:
            return
        val_metrics = owner._evaluate_split_metrics(
            pl_module,
            split="val",
            thresholds=self.thresholds,
            trainer=trainer,
        )
        train_metrics = owner._evaluate_split_metrics(
            pl_module,
            split="train",
            thresholds=self.thresholds,
            trainer=trainer,
            max_batches=self.train_eval_batches,
        )
        for metrics, split_prefix in ((train_metrics, "train"), (val_metrics, "val")):
            if not metrics:
                continue
            metrics_prefixed = {f"{split_prefix}_{k}": v for k, v in metrics.items()}
            trainer.callback_metrics.update(
                {
                    k: torch.tensor(v, device=pl_module.device)  # type: ignore[arg-type]
                    for k, v in metrics_prefixed.items()
                    if v is not None
                }
            )
            self._log_scalars(trainer, prefix=split_prefix, metrics=metrics)


def build_task_config(task_type: str, pos_weight: Optional[float] = None) -> Dict[str, object]:
    task = (task_type or "classification").lower()
    if task == "classification":
        loss = BinaryCrossEntropyLoss(pos_weight=pos_weight)
        return {
            "task_type": task,
            "target": "target_intraday_up",
            "loss": loss,
            "output_size": 1,
        }
    if task == "regression":
        loss = QuantileLoss()
        return {
            "task_type": task,
            "target": "target_intraday_return_pct",
            "loss": loss,
            "output_size": len(loss.quantiles),
        }
    raise ValueError("task_type must be either 'classification' or 'regression'.")


class TFT:
    """
    Thin wrapper around TemporalFusionTransformer with TX-specific defaults.
    """

    def __init__(
        self,
        prediction_length: int = 5,
        encoder_length: int = 120,
        task_type: str = "classification",
        pos_weight: Optional[float] = None,
        future_horizon: Optional[int] = None,
        skip_future_rest_days: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        classification_threshold: float = 0.5,
        start_date: Optional[str] = "2015-01-01",
        end_date: Optional[str] = None,
        features_csv: Optional[str] = None,
        train_fraction: float = 0.9,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        hidden_continuous_size: int = 32,
        dropout: float = 0.1,
        lstm_layers: int = 1,
        reduce_on_plateau_patience: int = 8,
        weight_decay: float = 0.0,
        threshold_min: float = 0.1,
        threshold_max: float = 0.9,
        threshold_step: float = 0.01,
        train_eval_batches: Optional[int] = 10,
        logger_name: Optional[str] = None,
        logger_version: Optional[int] = None,
        monitor_metric: str = "val_loss",
        monitor_mode: Optional[Literal["min", "max"]] = "min",
        seed: Optional[int] = 42,
        early_stopping_patience: int = 10,
    ) -> None:
        self.prediction_length = prediction_length
        self.encoder_length = encoder_length
        self.future_horizon = future_horizon if future_horizon is not None else prediction_length
        self.skip_future_rest_days = skip_future_rest_days
        self.batch_size = batch_size
        self.num_workers = max(0, num_workers)
        self.task_config = build_task_config(task_type, pos_weight=pos_weight)
        self.task_type = self.task_config["task_type"]
        self.target_column = self.task_config["target"]
        self.classification_threshold = classification_threshold
        self.threshold_min = float(threshold_min)
        self.threshold_max = float(threshold_max)
        self.threshold_step = float(threshold_step)
        self.best_val_threshold: Optional[float] = None
        self.best_val_accuracy: Optional[float] = None
        self.best_val_f1: Optional[float] = None
        self.threshold_grid: List[float] = self._build_threshold_grid()
        self.start_date = start_date
        self.end_date = end_date
        self.features_csv = Path(features_csv).expanduser() if features_csv else None
        # 允許更大的訓練比例以手動保留極短的驗證窗口（例如僅留 60 天）
        self.train_fraction = max(0.0, min(float(train_fraction), 0.995))
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.lstm_layers = lstm_layers
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        self.weight_decay = max(0.0, float(weight_decay))
        self.train_eval_batches = train_eval_batches
        self.logger_name = logger_name
        # 預設集中到指定版本資料夾（如 version_3），可自行覆寫
        self.logger_version = logger_version
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience

        self.base_dir = Path(__file__).resolve().parent
        self.log_root = self.base_dir / "lightning_logs"
        self.run_dir = self._prepare_run_dir()
        self.artifact_dir = self.run_dir / "artifacts"

        self.training = None
        self.validation = None
        self.trainer: Optional[L.Trainer] = None
        self.model: Optional[TemporalFusionTransformer] = None
        self.metrics_tracker: Optional[TrainingMetricsLogger] = None
        self.checkpoint_callback: Optional[ModelCheckpoint] = None
        self.best_checkpoint_path: Optional[str] = None
        self.training_cutoff: Optional[int] = None
        self.training_summary: Dict[str, object] = {}

        self.dataframe: Optional[pd.DataFrame] = None
        self.future_known_inputs: Optional[pd.DataFrame] = None
        self.date_lookup: Dict[int, str] = {}
        self._historical_time_idx_keys: List[int] = []
        self._time_idx_keys: List[int] = []
        self.symbol: str = "TX"
        self.tensorboard_logger: Optional[TensorBoardLogger] = None
        self.dataset_summary: Dict[str, object] = {}
        self.observed_reals_config: List[str] = []

    def _prepare_run_dir(self) -> Path:
        """
        Pre-compute the logging/artefact directory so every output stays under
        the same lightning version folder.
        """
        if self.logger_name:
            name_root = (self.log_root / self.logger_name).resolve()
        else:
            name_root = self.log_root.resolve()
        name_root.mkdir(parents=True, exist_ok=True)
        version = self.logger_version
        if version is None:
            version = self._next_logger_version(name_root)
            self.logger_version = version
        run_dir = name_root / f"version_{version}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _next_logger_version(root: Path) -> int:
        existing: List[int] = []
        for child in root.iterdir():
            if child.is_dir() and child.name.startswith("version_"):
                try:
                    existing.append(int(child.name.split("_", 1)[1]))
                except ValueError:
                    continue
        return max(existing) + 1 if existing else 0

    # ------------------------------------------------------------------#
    # Data / model helpers
    # ------------------------------------------------------------------#
    def _build_threshold_grid(self) -> List[float]:
        """
        Build threshold candidates for classification metrics sweep.
        """
        start = max(0.0, min(1.0, self.threshold_min))
        end = max(0.0, min(1.0, self.threshold_max))
        if end < start:
            start, end = end, start
        step = max(1e-4, float(self.threshold_step))
        num_steps = int((end - start) / step) + 1
        grid = [round(start + i * step, 6) for i in range(num_steps)]
        if grid and grid[-1] < end:
            grid.append(round(end, 6))
        return grid or [0.5]

    def _effective_threshold(self) -> float:
        if self.best_val_threshold is not None:
            return float(self.best_val_threshold)
        return float(self.classification_threshold)

    def load_data_external(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        export_preview: bool = True,
        export_full: bool = True,
        features_csv: Optional[str] = None,
        export_dir: Optional[str] = None,
    ) -> None:
        """
        Prepare TimeSeriesDataSet instances via data.py.
        """
        start = start_date or self.start_date
        end = end_date or self.end_date
        feature_path = features_csv or self.features_csv
        if feature_path is not None and not isinstance(feature_path, Path):
            feature_path = Path(feature_path)
        if isinstance(feature_path, Path):
            feature_path = feature_path.expanduser()
            self.features_csv = feature_path
        export_root = Path(export_dir).expanduser() if export_dir else None
        if export_root:
            export_root.mkdir(parents=True, exist_ok=True)
        self.training, self.validation, df, builder = load_tft_data(
            start_date=start,
            end_date=end,
            target=self.target_column,
            prediction_length=self.prediction_length,
            encoder_length=self.encoder_length,
            export_preview=export_preview,
            export_full=export_full,
            future_horizon=self.future_horizon,
            skip_future_rest_days=self.skip_future_rest_days,
            task_type=self.task_type,
            features_csv=str(feature_path) if feature_path else None,
            train_fraction=self.train_fraction,
            export_dir=export_root,
            return_builder=True,
        )
        self.training_cutoff = getattr(builder, "training_cutoff", None)
        self.dataframe = df
        self.future_known_inputs = df.attrs.get("future_known_inputs")
        # Export a quick snapshot of the prepared dataset (and splits) before feeding to the model.
        # This helps verify the exact rows/features used for training/validation.
        try:
            snapshot_root = self.artifact_dir or (self.run_dir / "artifacts")
            snapshot_root.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_root / "data_snapshot.csv"
            df.to_csv(snapshot_path, index=False)
            if self.training_cutoff is not None and not df.empty:
                train_df = df[df["time_idx"] <= self.training_cutoff]
                val_df = df[df["time_idx"] > self.training_cutoff]
                train_df.to_csv(snapshot_root / "train_snapshot.csv", index=False)
                val_df.to_csv(snapshot_root / "val_snapshot.csv", index=False)
                # Masked val snapshot (removes observed price/volume features to show what the model cannot see in future)
                known_future_cols = list(getattr(self.training, "time_varying_known_reals", [])) if self.training else []
                masked_cols = ["date", "symbol", "time_idx", self.target_column]
                masked_cols.extend([c for c in known_future_cols if c not in masked_cols])
                masked_cols = [c for c in masked_cols if c in val_df.columns]
                masked_val = val_df[masked_cols].copy()
                masked_val.to_csv(snapshot_root / "val_snapshot_masked.csv", index=False)
            print(f"[data] snapshot saved -> {snapshot_root}")
        except Exception as exc:
            print(f"[data] snapshot export skipped: {exc}")
        if hasattr(self.training, "time_varying_unknown_reals"):
            self.observed_reals_config = list(getattr(self.training, "time_varying_unknown_reals", []))
        if not df.empty:
            self.symbol = str(df["symbol"].iloc[0])
            self.date_lookup = (
                df.drop_duplicates("time_idx")
                .set_index("time_idx")["date"]
                .astype(str)
                .to_dict()
            )
            self._historical_time_idx_keys = sorted(self.date_lookup)
            self._time_idx_keys = sorted(self.date_lookup)
        else:
            self.date_lookup = {}
            self._historical_time_idx_keys = []
            self._time_idx_keys = []
        if (
            isinstance(self.future_known_inputs, pd.DataFrame)
            and not self.future_known_inputs.empty
        ):
            future_lookup = (
                self.future_known_inputs.drop_duplicates("time_idx")
                .set_index("time_idx")["date"]
                .astype(str)
                .to_dict()
            )
            self.date_lookup.update(future_lookup)
            self._time_idx_keys = sorted(set(self.date_lookup))
        if df is not None and not df.empty:
            self.dataset_summary = {
                "num_rows": int(len(df)),
                "date_start": str(df["date"].min().date()),
                "date_end": str(df["date"].max().date()),
                "start_date": start,
                "end_date": end,
                "train_fraction": self.train_fraction,
                "features_csv": str(self.features_csv) if self.features_csv else None,
            }
        else:
            self.dataset_summary = {}
            self._time_idx_keys = []

    def create_tft_model(self) -> None:
        if self.training is None:
            raise RuntimeError("Call load_data_external() before creating the model.")

        monitor_metric = self.monitor_metric or "val_loss"
        monitor_mode = self.monitor_mode
        if monitor_metric == "val_best_f1":
            monitor_mode = "max"
        if monitor_mode is None:
            monitor_mode = "max" if monitor_metric.endswith(("accuracy", "f1")) else "min"
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        if self.seed is not None:
            L.seed_everything(self.seed, workers=True)
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            min_delta=1e-4,
            patience=self.early_stopping_patience,
            mode=monitor_mode,
        )
        lr_logger = LearningRateMonitor(logging_interval="epoch")
        log_root = self.log_root
        log_root.mkdir(parents=True, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=str(log_root),
            name=self.logger_name,
            version=self.logger_version,
        )
        self.tensorboard_logger = logger
        self.run_dir = Path(logger.log_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = self.run_dir / "checkpoints"
        self.artifact_dir = self.run_dir / "artifacts"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"tft-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        self.metrics_tracker = TrainingMetricsLogger()
        epoch_eval_callback = ClassificationEpochEvaluator(
            owner=self,
            thresholds=self.threshold_grid,
            train_eval_batches=self.train_eval_batches,
        )
        callbacks: List[Callback] = [
            lr_logger,
            epoch_eval_callback,
            early_stop_callback,
            self.checkpoint_callback,
            self.metrics_tracker,
        ]
        accelerator = "gpu" if _is_cuda_usable() else "cpu"

        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=1,
            precision="bf16-mixed",  # use BF16 AMP to gain speed while avoiding FP16 mask overflow
            gradient_clip_val=0.1,
            callbacks=callbacks,
            logger=logger,
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            optimizer_params=None,  # weight decay is passed separately to avoid duplicate arguments
            weight_decay=self.weight_decay,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=self.task_config["output_size"],
            loss=self.task_config["loss"],
            logging_metrics=[
                BinaryCrossEntropyLoss(pos_weight=getattr(self.task_config["loss"], "pos_weight", None)),
                BinaryAccuracy(threshold=self.classification_threshold),
            ],
            lstm_layers=self.lstm_layers,
            reduce_on_plateau_patience=self.reduce_on_plateau_patience,
        )
        print(
            f"Number of parameters in network: {self.model.size() / 1e3:.1f}k "
            f"(task={self.task_type}, target={self.target_column})"
        )
        self._record_run_config()

    def train(self) -> None:
        if self.training is None or self.validation is None or self.model is None:
            raise RuntimeError("Data/model not ready. Call load_data_external() and create_tft_model().")
        train_dataloader = self.training.to_dataloader(
            train=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_dataloader = self.validation.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.trainer.fit(self.model, train_dataloader, val_dataloader)
        self._export_training_artifacts()

    def _record_run_config(self) -> None:
        if self.run_dir is None:
            return
        splits = self._build_split_summary()
        config = {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "prediction_length": self.prediction_length,
            "encoder_length": self.encoder_length,
            "future_horizon": self.future_horizon,
            "train_fraction": self.train_fraction,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "hidden_continuous_size": self.hidden_continuous_size,
            "dropout": self.dropout,
            "lstm_layers": self.lstm_layers,
            "reduce_on_plateau_patience": self.reduce_on_plateau_patience,
            "classification_threshold": self.classification_threshold,
            "best_val_threshold": self.best_val_threshold,
            "best_val_f1": self.best_val_f1,
            "best_val_accuracy": self.best_val_accuracy,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "threshold_step": self.threshold_step,
            "train_eval_batches": self.train_eval_batches,
            "weight_decay": self.weight_decay,
            "pos_weight": getattr(self.task_config.get("loss"), "pos_weight", None) if isinstance(self.task_config, dict) else None,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "features_csv": str(self.features_csv) if self.features_csv else None,
            "device": _get_pretty_device_name(),
            "logger_version": self.logger_version,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "seed": self.seed,
            "early_stopping_patience": self.early_stopping_patience,
        }
        if splits:
            config["splits"] = splits
        if self.dataset_summary:
            config["dataset"] = self.dataset_summary
        if self.training_summary:
            config["training"] = self.training_summary
        config_path = self.run_dir / "run_config.json"
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2, ensure_ascii=False)

    def _collect_prediction_rows(
        self,
        model: TemporalFusionTransformer,
        loader,
        require_actual: bool,
    ) -> List[Dict[str, object]]:
        """
        Run the model on a dataloader without spawning new loggers and collect rows for downstream processing.
        """
        rows: List[Dict[str, object]] = []
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch_inputs = batch[0]
                else:
                    batch_inputs = batch
                batch_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch_inputs.items()}
                raw_pred = model(batch_inputs)
                predictions = getattr(raw_pred, "prediction", None)
                if predictions is None:
                    predictions = getattr(raw_pred, "output", raw_pred)
                if self.task_type == "classification":
                    predictions = torch.sigmoid(predictions)
                rows.extend(self._build_prediction_rows(predictions, batch_inputs, model, require_actual=require_actual))
        if was_training:
            model.train()
        return rows

    # ------------------------------------------------------------------#
    # Evaluation / prediction export
    # ------------------------------------------------------------------#
    def _evaluate_split_metrics(
        self,
        model: TemporalFusionTransformer,
        split: Literal["train", "val"],
        thresholds: Optional[Iterable[float]] = None,
        trainer: Optional[L.Trainer] = None,
        max_batches: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        dataset = self.training if split == "train" else self.validation
        if (
            self.training is not None
            and self.dataframe is not None
            and self.training_cutoff is not None
        ):
            if split == "val":
                dataset = TimeSeriesDataSet.from_dataset(
                    self.training,
                    self.dataframe,
                    stop_randomization=True,
                    predict=False,
                    min_prediction_idx=self.training_cutoff + 1,
                )
            elif split == "train":
                df_train = self.dataframe[self.dataframe["time_idx"] <= self.training_cutoff]
                dataset = TimeSeriesDataSet.from_dataset(
                    self.training,
                    df_train,
                    stop_randomization=True,
                    predict=False,
                    min_prediction_idx=int(df_train["time_idx"].min()) if not df_train.empty else None,
                )
        if dataset is None:
            return None
        dataloader = dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 5,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        rows: List[Dict[str, object]] = []
        thresholds = list(thresholds) if thresholds is not None else self.threshold_grid
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    batch_inputs = batch[0]
                else:
                    batch_inputs = batch
                batch_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch_inputs.items()}
                raw_pred = model(batch_inputs)
                predictions = getattr(raw_pred, "prediction", None)
                if predictions is None:
                    predictions = getattr(raw_pred, "output", raw_pred)
                if self.task_type == "classification":
                    predictions = torch.sigmoid(predictions)
                rows.extend(self._build_prediction_rows(predictions, batch_inputs, model, require_actual=True))
        if was_training:
            model.train()
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        horizon_df = df[df["horizon_step"] == 1].dropna(subset=["actual", "prob_up"])
        if split == "val" and self.training_cutoff is not None:
            horizon_df = horizon_df[horizon_df["time_idx"] > self.training_cutoff]
        if horizon_df.empty:
            return None
        horizon_df = horizon_df.drop_duplicates(subset=["time_idx", "horizon_step"], keep="last")
        actual = horizon_df["actual"].astype(int)
        prob = horizon_df["prob_up"].clip(0, 1)
        actual_arr = actual.values
        prob_arr = prob.values
        base_threshold = self._effective_threshold()
        preds_default = (prob_arr >= base_threshold).astype(int)
        accuracy = float((preds_default == actual_arr).mean())
        bce_tensor = F.binary_cross_entropy(
            torch.as_tensor(prob_arr, dtype=torch.float32),
            torch.as_tensor(actual_arr, dtype=torch.float32),
        )
        best_acc = -1.0
        best_thresh = None
        best_f1 = -1.0
        best_f1_thresh = None
        acc_at_best_f1: Optional[float] = None
        for th in thresholds:
            preds = (prob_arr >= th).astype(int)
            acc = float((preds == actual_arr).mean())
            tp_th = int(((preds == 1) & (actual_arr == 1)).sum())
            fp_th = int(((preds == 1) & (actual_arr == 0)).sum())
            fn_th = int(((preds == 0) & (actual_arr == 1)).sum())
            precision_th = tp_th / (tp_th + fp_th) if (tp_th + fp_th) > 0 else 0.0
            recall_th = tp_th / (tp_th + fn_th) if (tp_th + fn_th) > 0 else 0.0
            f1_th = (
                2 * precision_th * recall_th / (precision_th + recall_th)
                if (precision_th + recall_th) > 0
                else 0.0
            )
            if acc > best_acc:
                best_acc = acc
                best_thresh = float(th)
            if f1_th > best_f1:
                best_f1 = float(f1_th)
                best_f1_thresh = float(th)
                acc_at_best_f1 = acc
        metrics = {
            "accuracy": accuracy,
            "bce": float(bce_tensor.detach().cpu().item()),
            "positive_rate": float(actual.mean()),
            "prediction_rate": float(preds_default.mean()),
            "best_accuracy": float(best_acc) if best_acc >= 0 else None,
            "best_threshold": best_thresh,
            "best_f1": float(best_f1) if best_f1 >= 0 else None,
            "best_f1_threshold": best_f1_thresh,
            "accuracy_at_best_f1": float(acc_at_best_f1) if acc_at_best_f1 is not None else None,
            "samples": float(len(horizon_df)),
        }
        # precision / recall / f1 at the default threshold
        tp = int(((preds_default == 1) & (actual == 1)).sum())
        fp = int(((preds_default == 1) & (actual == 0)).sum())
        fn = int(((preds_default == 0) & (actual == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
        if split == "val" and best_f1_thresh is not None:
            # Keep the global best validation threshold so later evaluation matches the best checkpoint.
            is_better = (
                self.best_val_f1 is None
                or (best_f1 is not None and best_f1 > self.best_val_f1)
            )
            if is_better:
                self.best_val_threshold = best_f1_thresh
                self.best_val_accuracy = best_acc
                self.best_val_f1 = best_f1
        return metrics

    def _export_training_artifacts(self, save_dir: Optional[str] = None) -> Dict[str, Path]:
        save_root = Path(save_dir) if save_dir else self.artifact_dir
        save_root.mkdir(parents=True, exist_ok=True)
        artifacts: Dict[str, Path] = {}

        if self.checkpoint_callback and self.checkpoint_callback.best_model_path:
            best_path = Path(self.checkpoint_callback.best_model_path)
            self.best_checkpoint_path = str(best_path)
            artifacts["best_checkpoint"] = best_path
            score = self.checkpoint_callback.best_model_score
            score_val: Optional[float] = None
            if isinstance(score, torch.Tensor):
                score_val = float(score.detach().cpu().item())
            elif score is not None:
                score_val = float(score)
            score_str = f"{score_val:.6f}" if score_val is not None else "n/a"
            print(f"[train] best checkpoint -> {best_path} (val_loss={score_str})")
            # 將訓練摘要記錄下來，方便 run_config.json 回溯
            best_epoch_hint: Optional[int] = None
            try:
                best_epoch_hint = int(best_path.stem.split("-")[1])
            except Exception:
                pass
            self.training_summary = {
                "best_checkpoint": str(best_path),
                "best_monitor_score": score_val,
                "best_epoch": best_epoch_hint,
                "actual_max_epochs": self.trainer.max_epochs if self.trainer else None,
                "epochs_trained": (self.trainer.current_epoch + 1) if self.trainer else None,
                "monitor_metric": self.monitor_metric,
                "monitor_mode": self.monitor_mode,
            }

        if self.metrics_tracker and self.metrics_tracker.history:
            df = pd.DataFrame(self.metrics_tracker.history).sort_values("epoch")
            if "train_accuracy" not in df.columns:
                df["train_accuracy"] = np.nan
            if "val_accuracy" not in df.columns:
                df["val_accuracy"] = np.nan
            for col in [
                "train_precision",
                "train_recall",
                "train_f1",
                "val_precision",
                "val_recall",
                "val_f1",
                "train_best_accuracy",
                "train_best_threshold",
                "train_best_f1",
                "train_best_f1_threshold",
                "train_accuracy_at_best_f1",
                "val_best_accuracy",
                "val_best_threshold",
                "val_best_f1",
                "val_best_f1_threshold",
                "val_accuracy_at_best_f1",
            ]:
                if col not in df.columns:
                    df[col] = np.nan
            metrics_csv = save_root / "training_metrics.csv"
            df.to_csv(metrics_csv, index=False)
            artifacts["metrics_csv"] = metrics_csv

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.set_title("TFT training history")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if df["train_loss"].notna().any():
                ax.plot(df["epoch"], df["train_loss"], label="train_loss", marker="o", linewidth=1.5)
            if df["val_loss"].notna().any():
                ax.plot(df["epoch"], df["val_loss"], label="val_loss", marker="s", linewidth=1.5)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.3)
            fig_path = save_root / "training_loss_curve.png"
            fig.tight_layout()
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            artifacts["metrics_plot"] = fig_path
            if df[["train_accuracy", "val_accuracy"]].notna().any().any():
                fig_acc, ax_acc = plt.subplots(figsize=(7, 4))
                ax_acc.set_title("TFT accuracy history")
                ax_acc.set_xlabel("Epoch")
                ax_acc.set_ylabel("Accuracy")
                if df["train_accuracy"].notna().any():
                    ax_acc.plot(df["epoch"], df["train_accuracy"], label="train_acc", marker="o", linewidth=1.3)
                if df["val_accuracy"].notna().any():
                    ax_acc.plot(df["epoch"], df["val_accuracy"], label="val_acc", marker="s", linewidth=1.3)
                ax_acc.grid(True, linestyle="--", alpha=0.3)
                ax_acc.legend()
                acc_path = save_root / "training_accuracy_curve.png"
                fig_acc.tight_layout()
                fig_acc.savefig(acc_path, dpi=200)
                plt.close(fig_acc)
                artifacts["accuracy_plot"] = acc_path
            if df[["train_f1", "val_f1"]].notna().any().any():
                fig_f1, ax_f1 = plt.subplots(figsize=(7, 4))
                ax_f1.set_title("TFT F1 history")
                ax_f1.set_xlabel("Epoch")
                ax_f1.set_ylabel("F1 score")
                if df["train_f1"].notna().any():
                    ax_f1.plot(df["epoch"], df["train_f1"], label="train_f1", marker="o", linewidth=1.3)
                if df["val_f1"].notna().any():
                    ax_f1.plot(df["epoch"], df["val_f1"], label="val_f1", marker="s", linewidth=1.3)
                ax_f1.grid(True, linestyle="--", alpha=0.3)
                ax_f1.legend()
                f1_path = save_root / "training_f1_curve.png"
                fig_f1.tight_layout()
                fig_f1.savefig(f1_path, dpi=200)
                plt.close(fig_f1)
                artifacts["f1_plot"] = f1_path
            print(f"[train] saved metrics CSV -> {metrics_csv}")
            print(f"[train] saved loss curve -> {fig_path}")
        else:
            print("[train] metrics tracker empty, skipping loss export.")

        # 重新寫入 run_config.json，包含訓練摘要/seed/early stopping 等資訊
        self._record_run_config()
        return artifacts

    def _compute_split_accuracy(self, split: Literal["train", "val"]) -> Optional[float]:
        """
        Compute a single accuracy value on the train or validation slice using the best checkpoint.
        Only horizon_step=1 is considered to match the classification metrics.
        """
        if self.training is None or self.dataframe is None or self.training_cutoff is None:
            return None
        try:
            model = self._load_best_model()
        except Exception:
            model = self.model
        if model is None:
            return None

        df_source = self.dataframe.copy()
        if split == "train":
            df_source = df_source[df_source["time_idx"] <= self.training_cutoff]
        else:
            df_source = df_source[df_source["time_idx"] > self.training_cutoff]
        if df_source.empty:
            return None

        try:
            dataset = TimeSeriesDataSet.from_dataset(
                self.training,
                df_source,
                stop_randomization=True,
                predict=False,
                min_prediction_idx=int(df_source["time_idx"].min()),
            )
            loader = dataset.to_dataloader(
                train=False,
                batch_size=self.batch_size * 10,
                num_workers=self.num_workers,
            )
            prediction_result = model.predict(loader, return_x=True)
            predictions = getattr(prediction_result, "output", prediction_result)
            batch_inputs = getattr(prediction_result, "x", None) or {}
            rows = self._build_prediction_rows(predictions, batch_inputs, model, require_actual=True)
            pred_df = pd.DataFrame(rows)
            horizon_df = pred_df[pred_df["horizon_step"] == 1].dropna(subset=["actual", "prob_up"])
            if horizon_df.empty:
                return None
            horizon_df = horizon_df.drop_duplicates(subset=["time_idx", "horizon_step"], keep="last")
            horizon_df["actual"] = horizon_df["actual"].astype(int)
            threshold = self._effective_threshold()
            horizon_df["pred_label"] = (horizon_df["prob_up"] >= threshold).astype(int)
            return float((horizon_df["pred_label"] == horizon_df["actual"]).mean())
        except Exception:
            return None

    def _load_best_model(self) -> TemporalFusionTransformer:
        checkpoint_path = self.best_checkpoint_path or ""
        if self.trainer is not None:
            for cb in self.trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                    checkpoint_path = cb.best_model_path
                    self.best_checkpoint_path = checkpoint_path
                    break
        if checkpoint_path:
            print(f"[eval] loading best checkpoint: {checkpoint_path}")
            return TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        if self.model is None:
            raise RuntimeError("No trained model available. Train the model or provide a checkpoint path.")
        print("[eval] no checkpoint found, using in-memory model.")
        return self.model

    def _denormalize_targets(
        self,
        decoder_target: torch.Tensor,
        batch_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        normalizer = getattr(self.training, "target_normalizer", None)
        if normalizer is None or "target_scale" not in batch_inputs or not hasattr(normalizer, "inverse"):
            return decoder_target
        scale = torch.as_tensor(batch_inputs["target_scale"])
        return normalizer.inverse(decoder_target.unsqueeze(-1), scale.unsqueeze(1)).squeeze(-1)

    def _build_prediction_rows(
        self,
        predictions: List[torch.Tensor] | torch.Tensor,
        batch_inputs: List[Dict[str, torch.Tensor]] | Dict[str, torch.Tensor],
        model: TemporalFusionTransformer,
        require_actual: bool = True,
    ) -> List[Dict[str, object]]:
        pred_batches = predictions if isinstance(predictions, (list, tuple)) else [predictions]
        input_batches = batch_inputs if isinstance(batch_inputs, (list, tuple)) else [batch_inputs]
        rows: List[Dict[str, object]] = []
        quantiles = getattr(model.loss, "quantiles", None)

        for batch_pred, batch_x in zip(pred_batches, input_batches):
            decoder_time_idx_raw = batch_x.get("decoder_time_idx")
            if decoder_time_idx_raw is None:
                continue
            decoder_time_idx = torch.as_tensor(decoder_time_idx_raw)
            decoder_target = batch_x.get("decoder_target")
            if decoder_target is not None:
                decoder_target = torch.as_tensor(decoder_target)
                decoder_target = self._denormalize_targets(decoder_target, batch_x)
            else:
                decoder_target = torch.full(
                    decoder_time_idx.shape,
                    float("nan"),
                    dtype=torch.float32,
                    device=decoder_time_idx.device,
                )
            batch_pred = torch.as_tensor(batch_pred)
            time_steps = decoder_time_idx.size(1)
            pred_steps = batch_pred.size(1)
            steps = min(time_steps, pred_steps)
            for b in range(batch_pred.size(0)):
                for t in range(steps):
                    time_idx_val = decoder_time_idx[b, t].item()
                    if pd.isna(time_idx_val):
                        continue
                    actual_val = decoder_target[b, t].item()
                    actual_missing = pd.isna(actual_val)
                    if require_actual and actual_missing:
                        continue
                    time_idx = int(time_idx_val)
                    row: Dict[str, object] = {
                        "symbol": self.symbol,
                        "time_idx": time_idx,
                        "date": self.date_lookup.get(time_idx),
                        "horizon_step": t + 1,
                        "actual": None if actual_missing else float(actual_val),
                    }
                    pred_slice = batch_pred[b, t]
                    if self.task_type == "classification":
                        prob_tensor = pred_slice
                        if prob_tensor.ndim > 0:
                            prob_tensor = prob_tensor.reshape(-1)[0]
                        prob = float(prob_tensor.detach().cpu().item())
                        prob = max(0.0, min(1.0, prob))
                        row["prob_up"] = prob
                        threshold = self._effective_threshold()
                        row["pred_label"] = int(prob >= threshold)
                        if not actual_missing:
                            row["correct"] = int(row["pred_label"] == int(actual_val))
                    elif quantiles:
                        pred_array = pred_slice.detach().cpu().tolist()
                        # use median (0.5) if available, else central element
                        if 0.5 in quantiles:
                            median_idx = quantiles.index(0.5)
                        else:
                            median_idx = len(pred_array) // 2
                        row["prediction"] = float(pred_array[median_idx])
                        for q_idx, q in enumerate(quantiles):
                            row[f"pred_p{int(q * 100):02d}"] = float(pred_array[q_idx])
                    else:
                        row["prediction"] = float(pred_slice.detach().cpu().item())
                    rows.append(row)
        return rows

    def _summarize_classification_metrics(self, df: pd.DataFrame, save_dir: Path) -> Optional[Dict[str, object]]:
        if self.task_type != "classification":
            return None
        if not {"actual", "prob_up"}.issubset(df.columns):
            return None
        horizon_df = df[df["horizon_step"] == 1].copy()
        horizon_df = horizon_df.dropna(subset=["actual", "prob_up"])
        if horizon_df.empty:
            return None
        horizon_df["actual"] = horizon_df["actual"].astype(int)
        threshold = self._effective_threshold()
        horizon_df["pred_label"] = (horizon_df["prob_up"] >= threshold).astype(int)
        horizon_df["correct"] = (horizon_df["pred_label"] == horizon_df["actual"]).astype(int)
        summary = {
            "samples": int(len(horizon_df)),
            "overall_accuracy": float(horizon_df["correct"].mean()),
            "positive_rate": float(horizon_df["actual"].mean()),
            "prediction_rate": float(horizon_df["pred_label"].mean()),
            "threshold_used": threshold,
            "best_val_threshold": self.best_val_threshold,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_f1": self.best_val_f1,
        }
        last_window = horizon_df.sort_values("time_idx").tail(min(30, len(horizon_df)))
        summary["last_30_accuracy"] = (
            float(last_window["correct"].mean()) if not last_window.empty else None
        )
        summary["true_positives"] = int(((horizon_df["pred_label"] == 1) & (horizon_df["actual"] == 1)).sum())
        summary["false_positives"] = int(((horizon_df["pred_label"] == 1) & (horizon_df["actual"] == 0)).sum())
        summary["true_negatives"] = int(((horizon_df["pred_label"] == 0) & (horizon_df["actual"] == 0)).sum())
        summary["false_negatives"] = int(((horizon_df["pred_label"] == 0) & (horizon_df["actual"] == 1)).sum())
        # Precision / recall / F1
        tp = summary["true_positives"]
        fp = summary["false_positives"]
        fn = summary["false_negatives"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        summary["precision"] = float(precision)
        summary["recall"] = float(recall)
        summary["f1"] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # Ranking metrics (useful to debug precision/recall trade-offs)
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            summary["roc_auc"] = float(roc_auc_score(horizon_df["actual"], horizon_df["prob_up"]))
            summary["pr_auc"] = float(average_precision_score(horizon_df["actual"], horizon_df["prob_up"]))
        except Exception as exc:
            summary["roc_auc"] = None
            summary["pr_auc"] = None
            print(f"[eval] AUC metrics skipped: {exc}")
        metrics_path = save_dir / "classification_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2, ensure_ascii=False)
        print(
            "[eval] accuracy={:.3f}, last_30={:.3f}, samples={}, positives={:.1%}".format(
                summary["overall_accuracy"],
                summary["last_30_accuracy"] if summary["last_30_accuracy"] is not None else float("nan"),
                summary["samples"],
                summary["positive_rate"],
            )
        )
        return summary

    def evaluate(
        self,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        split_name: Literal["validation"] = "validation",
    ) -> Path:
        """
        Run inference on the validation set and dump predictions to CSV.
        """
        model = self._load_best_model()
        eval_dataset = self.validation
        if (
            self.training is not None
            and self.dataframe is not None
            and self.training_cutoff is not None
        ):
            eval_dataset = TimeSeriesDataSet.from_dataset(
                self.training,
                self.dataframe,
                stop_randomization=True,
                predict=False,
                min_prediction_idx=self.training_cutoff + 1,
            )
        val_loader = eval_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
        )
        rows = self._collect_prediction_rows(model, val_loader, require_actual=True)
        df = pd.DataFrame(rows).sort_values(["time_idx", "horizon_step"])
        if self.training_cutoff is not None:
            df = df[df["time_idx"] > self.training_cutoff]
        if not df.empty:
            df = df.drop_duplicates(subset=["time_idx", "horizon_step"], keep="last")

        save_path = Path(save_dir) if save_dir else self.artifact_dir
        save_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{split_name}_predictions_{self.task_type}.csv"
        output_path = save_path / filename
        try:
            df.to_csv(output_path, index=False)
            print(f"[eval] saved {len(df)} rows -> {output_path}")
        except PermissionError as exc:
            fallback = save_path / f"{output_path.stem}_{dt.datetime.now():%Y%m%d%H%M%S}.csv"
            df.to_csv(fallback, index=False)
            print(f"[eval] original file locked ({exc}); fallback saved -> {fallback}")
            output_path = fallback
        summary = self._summarize_classification_metrics(df, save_path)
        try:
            self._update_run_config_eval_metrics(summary)
        except Exception as exc:
            print(f"[eval] failed to update run_config with eval metrics: {exc}")
        return output_path

    def _update_run_config_eval_metrics(self, metrics: Optional[Dict[str, object]]) -> None:
        """
        Persist the latest evaluation summary into run_config.json for reproducibility.
        """
        if metrics is None or self.run_dir is None:
            return
        run_config_path = self.run_dir / "run_config.json"
        if not run_config_path.exists():
            return
        try:
            with run_config_path.open("r", encoding="utf-8") as fp:
                cfg = json.load(fp)
        except Exception:
            return
        cfg["last_evaluation"] = metrics
        try:
            with run_config_path.open("w", encoding="utf-8") as fp:
                json.dump(cfg, fp, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[eval] unable to write eval metrics to run_config: {exc}")

    def predict_future(
        self,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        if self.training is None or self.dataframe is None:
            raise RuntimeError("Training dataset unavailable. Load data first.")
        if self.future_known_inputs is None or self.future_known_inputs.empty:
            print("[future] 未提供 future_known_inputs，跳過未來推論。")
            return None

        model = self._load_best_model()
        historical_df = self.dataframe.copy()
        future_df = self.future_known_inputs.copy()
        observed_cols = [col for col in self.observed_reals_config if col in future_df.columns]
        if observed_cols:
            last_known_values = historical_df[observed_cols].iloc[-1]
            for col in observed_cols:
                fill_value = last_known_values[col]
                future_df[col] = future_df[col].fillna(fill_value)
            future_df[observed_cols] = future_df[observed_cols].fillna(0.0)
        if self.target_column in future_df.columns:
            future_df[self.target_column] = future_df[self.target_column].fillna(0.0)
        combined_df = pd.concat(
            [historical_df, future_df],
            ignore_index=True,
            sort=False,
        )
        combined_df = combined_df.sort_values("time_idx").reset_index(drop=True)
        future_indices = sorted(set(int(x) for x in future_df["time_idx"].tolist()))
        if not future_indices:
            print("[future] future_known_inputs 缺少 time_idx，無法推論。")
            return None
        future_start_idx = future_indices[0]
        expected_range = list(range(future_start_idx, future_start_idx + len(future_indices)))
        missing_known = [idx for idx in expected_range if idx not in future_indices]
        if missing_known:
            print(f"[future] 未來特徵缺少 time_idx：{missing_known}")
        target_window: Set[int] = set(expected_range if not missing_known else future_indices)

        try:
            forecast_dataset = TimeSeriesDataSet.from_dataset(
                self.training,
                combined_df,
                stop_randomization=True,
                predict=False,
                min_prediction_idx=future_start_idx,
            )
        except Exception as exc:
            print(f"[future] 建立推論資料集失敗：{exc}")
            return None

        forecast_loader = forecast_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
        )
        rows = self._collect_prediction_rows(model, forecast_loader, require_actual=False)

        filtered_rows: List[Dict[str, object]] = []
        seen: Set[Tuple[int, int]] = set()
        for row in rows:
            if row.get("horizon_step") != 1:
                continue
            try:
                time_idx_val = int(row.get("time_idx", -1))
            except (TypeError, ValueError):
                continue
            if time_idx_val < future_start_idx or time_idx_val not in target_window:
                continue
            row["actual"] = None
            row.pop("correct", None)
            key = (time_idx_val, 1)
            if key in seen:
                continue
            seen.add(key)
            filtered_rows.append(row)

        missing_preds = sorted(target_window - {r["time_idx"] for r in filtered_rows})
        if missing_preds:
            print(f"[future] 缺少以下 time_idx 的預測 (horizon=1): {missing_preds}")

        if not filtered_rows:
            print("[future] 推論結果為空，無法輸出。")
            return None
        df = pd.DataFrame(filtered_rows).sort_values(["time_idx", "horizon_step"])
        save_path = Path(save_dir) if save_dir else self.artifact_dir
        save_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"future_predictions_{self.task_type}.csv"
        output_path = save_path / filename
        try:
            df.to_csv(output_path, index=False)
            print(f"[future] saved {len(df)} rows -> {output_path}")
        except PermissionError as exc:
            fallback = save_path / f"{output_path.stem}_{dt.datetime.now():%Y%m%d%H%M%S}.csv"
            df.to_csv(fallback, index=False)
            print(f"[future] original file locked ({exc}); fallback saved -> {fallback}")
            output_path = fallback
        return output_path

    def _build_split_summary(self) -> Dict[str, object]:
        keys = self._historical_time_idx_keys or self._time_idx_keys
        if not keys:
            return {}
        train_cutoff = self.training_cutoff
        min_idx, max_idx = keys[0], keys[-1]

        def _first_after(idx: int) -> Optional[int]:
            for k in keys:
                if k > idx:
                    return k
            return None

        def _date(idx: Optional[int]) -> Optional[str]:
            if idx is None:
                return None
            return self.date_lookup.get(idx)

        train_max = max(k for k in keys if train_cutoff is None or k <= train_cutoff)
        val_min = _first_after(train_cutoff) if train_cutoff is not None else None
        future_min = None
        future_max = None
        if isinstance(self.future_known_inputs, pd.DataFrame) and not self.future_known_inputs.empty:
            future_min = int(self.future_known_inputs["time_idx"].min())
            future_max = int(self.future_known_inputs["time_idx"].max())

        return {
            "train": {
                "time_idx_min": int(min_idx),
                "time_idx_max": int(train_max),
                "date_start": _date(min_idx),
                "date_end": _date(train_max),
            },
            "val": {
                "time_idx_min": int(val_min) if val_min is not None else None,
                "time_idx_max": int(max_idx),
                "date_start": _date(val_min),
                "date_end": _date(max_idx),
            },
            "future": {
                "time_idx_min": int(future_min) if future_min is not None else None,
                "time_idx_max": int(future_max) if future_max is not None else None,
                "date_start": _date(future_min),
                "date_end": _date(future_max),
            },
        }


def tft() -> None:
    predictor = TFT(
        task_type="classification",
        prediction_length=1,
        encoder_length=120,
        future_horizon=30,
        batch_size=64,
        start_date="2015-01-01",
        train_fraction=0.8,
        max_epochs=90,
        learning_rate=8e-4,
        hidden_size=48,
        attention_head_size=4,
        hidden_continuous_size=24,
        dropout=0.2,
        weight_decay=5e-4,
        pos_weight=0.5,
        threshold_min=0.2,
        threshold_max=0.95,
        threshold_step=0.01,
        monitor_metric="val_best_f1",
        monitor_mode="max",
        early_stopping_patience=15,
        train_eval_batches=None,
    )
    predictor.load_data_external(export_preview=False, export_full=False)
    predictor.create_tft_model()
    predictor.train()
    predictor.evaluate()
    predictor.predict_future()


if __name__ == "__main__":
    print(
        "cuda_is_available:",
        _is_cuda_usable(),
        "| device:",
        _get_pretty_device_name(),
    )
    tft()
