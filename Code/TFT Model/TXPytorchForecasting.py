# ---------------------------------------------------#
#
#   File       : TXPytorchForecasting.py
#   Description: Training / evaluating TFT on TX futures data
#
# ----------------------------------------------------#

from __future__ import annotations

import warnings
import os
import sys
import datetime as dt
from pathlib import Path
from typing import Dict, List, Literal, Optional

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

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = y_pred[..., 0]
        target = target.float()
        if target.ndim > logits.ndim:
            target = target.squeeze(-1)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return losses.unsqueeze(-1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y_pred)


try:  # ensure checkpoints saved under __main__ can be reloaded when imported as a module
    sys.modules["__main__"].BinaryCrossEntropyLoss = BinaryCrossEntropyLoss
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
        self.history.append(record)


def build_task_config(task_type: str) -> Dict[str, object]:
    task = (task_type or "classification").lower()
    if task == "classification":
        loss = BinaryCrossEntropyLoss()
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
        future_horizon: Optional[int] = None,
        skip_future_rest_days: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        classification_threshold: float = 0.5,
        start_date: Optional[str] = "2015-01-01",
        end_date: Optional[str] = None,
        features_csv: Optional[str] = None,
        train_fraction: float = 0.8,
        max_epochs: int = 50,
        learning_rate: float = 1e-3,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        hidden_continuous_size: int = 32,
        dropout: float = 0.1,
        lstm_layers: int = 1,
        reduce_on_plateau_patience: int = 8,
        logger_name: str = "tx_tft",
    ) -> None:
        self.prediction_length = prediction_length
        self.encoder_length = encoder_length
        self.future_horizon = future_horizon if future_horizon is not None else prediction_length
        self.skip_future_rest_days = skip_future_rest_days
        self.batch_size = batch_size
        self.num_workers = max(0, num_workers)
        self.task_config = build_task_config(task_type)
        self.task_type = self.task_config["task_type"]
        self.target_column = self.task_config["target"]
        self.classification_threshold = classification_threshold
        self.start_date = start_date
        self.end_date = end_date
        self.features_csv = Path(features_csv).expanduser() if features_csv else None
        self.train_fraction = max(0.0, min(float(train_fraction), 0.95))
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.hidden_continuous_size = hidden_continuous_size
        self.dropout = dropout
        self.lstm_layers = lstm_layers
        self.reduce_on_plateau_patience = reduce_on_plateau_patience
        self.logger_name = logger_name

        self.training = None
        self.validation = None
        self.trainer: Optional[L.Trainer] = None
        self.model: Optional[TemporalFusionTransformer] = None
        self.metrics_tracker: Optional[TrainingMetricsLogger] = None
        self.checkpoint_callback: Optional[ModelCheckpoint] = None
        self.best_checkpoint_path: Optional[str] = None
        self.training_cutoff: Optional[int] = None

        self.dataframe: Optional[pd.DataFrame] = None
        self.future_known_inputs: Optional[pd.DataFrame] = None
        self.date_lookup: Dict[int, str] = {}
        self.symbol: str = "TX"
        self.tensorboard_logger: Optional[TensorBoardLogger] = None
        self.run_dir: Optional[Path] = None
        self.artifact_dir = Path("data_output")
        self.dataset_summary: Dict[str, object] = {}
        self.observed_reals_config: List[str] = []

    # ------------------------------------------------------------------#
    # Data / model helpers
    # ------------------------------------------------------------------#
    def load_data_external(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        export_preview: bool = True,
        export_full: bool = True,
        features_csv: Optional[str] = None,
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
            return_builder=True,
        )
        self.training_cutoff = getattr(builder, "training_cutoff", None)
        self.dataframe = df
        self.future_known_inputs = df.attrs.get("future_known_inputs")
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
        else:
            self.date_lookup = {}
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

    def create_tft_model(self) -> None:
        if self.training is None:
            raise RuntimeError("Call load_data_external() before creating the model.")

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            mode="min",
        )
        lr_logger = LearningRateMonitor(logging_interval="epoch")
        logger = TensorBoardLogger(save_dir="lightning_logs", name=self.logger_name)
        self.tensorboard_logger = logger
        self.run_dir = Path(logger.log_dir)
        checkpoint_dir = self.run_dir / "checkpoints"
        self.artifact_dir = self.run_dir / "artifacts"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        self.metrics_tracker = TrainingMetricsLogger()
        callbacks: List[Callback] = [lr_logger, early_stop_callback, self.checkpoint_callback, self.metrics_tracker]
        accelerator = "gpu" if _is_cuda_usable() else "cpu"

        self.trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            logger=logger,
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_continuous_size,
            output_size=self.task_config["output_size"],
            loss=self.task_config["loss"],
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
            "start_date": self.start_date,
            "end_date": self.end_date,
            "features_csv": str(self.features_csv) if self.features_csv else None,
            "device": _get_pretty_device_name(),
        }
        if self.dataset_summary:
            config["dataset"] = self.dataset_summary
        config_path = self.run_dir / "run_config.json"
        with config_path.open("w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------#
    # Evaluation / prediction export
    # ------------------------------------------------------------------#
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

        if self.metrics_tracker and self.metrics_tracker.history:
            df = pd.DataFrame(self.metrics_tracker.history).sort_values("epoch")
            train_acc = self._compute_split_accuracy("train")
            val_acc = self._compute_split_accuracy("val")
            if "train_accuracy" not in df.columns:
                df["train_accuracy"] = np.nan
            if "val_accuracy" not in df.columns:
                df["val_accuracy"] = np.nan
            if not df.empty:
                last_idx = df.index[-1]
                if train_acc is not None:
                    df.at[last_idx, "train_accuracy"] = train_acc
                if val_acc is not None:
                    df.at[last_idx, "val_accuracy"] = val_acc
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
            print(f"[train] saved metrics CSV -> {metrics_csv}")
            print(f"[train] saved loss curve -> {fig_path}")
        else:
            print("[train] metrics tracker empty, skipping loss export.")

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
            horizon_df["actual"] = horizon_df["actual"].astype(int)
            horizon_df["pred_label"] = (horizon_df["prob_up"] >= self.classification_threshold).astype(int)
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
                        row["pred_label"] = int(prob >= self.classification_threshold)
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

    def _summarize_classification_metrics(self, df: pd.DataFrame, save_dir: Path) -> None:
        if self.task_type != "classification":
            return
        if not {"actual", "prob_up"}.issubset(df.columns):
            return
        horizon_df = df[df["horizon_step"] == 1].copy()
        horizon_df = horizon_df.dropna(subset=["actual", "prob_up"])
        if horizon_df.empty:
            return
        horizon_df["actual"] = horizon_df["actual"].astype(int)
        horizon_df["pred_label"] = (horizon_df["prob_up"] >= self.classification_threshold).astype(int)
        horizon_df["correct"] = (horizon_df["pred_label"] == horizon_df["actual"]).astype(int)
        summary = {
            "samples": int(len(horizon_df)),
            "overall_accuracy": float(horizon_df["correct"].mean()),
            "positive_rate": float(horizon_df["actual"].mean()),
            "prediction_rate": float(horizon_df["pred_label"].mean()),
        }
        last_window = horizon_df.sort_values("time_idx").tail(min(30, len(horizon_df)))
        summary["last_30_accuracy"] = (
            float(last_window["correct"].mean()) if not last_window.empty else None
        )
        summary["true_positives"] = int(((horizon_df["pred_label"] == 1) & (horizon_df["actual"] == 1)).sum())
        summary["false_positives"] = int(((horizon_df["pred_label"] == 1) & (horizon_df["actual"] == 0)).sum())
        summary["true_negatives"] = int(((horizon_df["pred_label"] == 0) & (horizon_df["actual"] == 0)).sum())
        summary["false_negatives"] = int(((horizon_df["pred_label"] == 0) & (horizon_df["actual"] == 1)).sum())
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
        prediction_result = model.predict(
            val_loader,
            return_x=True,
        )
        predictions = getattr(prediction_result, "output", prediction_result)
        batch_inputs = getattr(prediction_result, "x", None) or {}
        rows = self._build_prediction_rows(predictions, batch_inputs, model, require_actual=True)
        df = pd.DataFrame(rows).sort_values(["time_idx", "horizon_step"])
        if self.training_cutoff is not None:
            df = df[df["time_idx"] > self.training_cutoff]

        save_path = Path(save_dir) if save_dir else self.artifact_dir
        if save_path is None:
            save_path = Path("data_output")
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
        self._summarize_classification_metrics(df, save_path)
        return output_path

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
        future_dataset = TimeSeriesDataSet.from_dataset(
            self.training,
            combined_df,
            stop_randomization=True,
            predict=True,
        )
        future_loader = future_dataset.to_dataloader(
            train=False,
            batch_size=self.batch_size * 10,
            num_workers=self.num_workers,
        )
        prediction_result = model.predict(future_loader, return_x=True)
        predictions = getattr(prediction_result, "output", prediction_result)
        batch_inputs = getattr(prediction_result, "x", None) or {}
        rows = self._build_prediction_rows(predictions, batch_inputs, model, require_actual=False)
        future_start_idx = int(self.future_known_inputs["time_idx"].min())
        rows = [row for row in rows if row.get("time_idx", -1) >= future_start_idx]
        if not rows:
            print("[future] 推論結果為空，無法輸出。")
            return None
        df = pd.DataFrame(rows).sort_values(["time_idx", "horizon_step"])
        save_path = Path(save_dir) if save_dir else self.artifact_dir
        if save_path is None:
            save_path = Path("data_output")
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


def tft() -> None:
    features_path = Path("data_output") / "tx_feature.csv"
    predictor = TFT(
        task_type="classification",
        prediction_length=1,
        encoder_length=120,
        future_horizon=30,
        batch_size=64,
        start_date="2015-01-01",
        features_csv=features_path,
        train_fraction=0.8,
        max_epochs=50,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        hidden_continuous_size=32,
        dropout=0.1,
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
