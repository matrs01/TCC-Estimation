from typing import Optional, Dict, Any
from catalyst import dl
import wandb
import torch.nn as nn


class WandbLoggerWithPauses(dl.WandbLogger):
    def __init__(
            self, model: nn.Module, config: Dict, project: str,
            name: Optional[str] = None,
            entity: Optional[str] = None,
    ) -> None:
        super().__init__(project, name, entity)
        wandb.watch(model)
        self.config = config

    def _log_metrics(self, metrics: Dict[str, float], step: int,
                     loader_key: str, prefix=""):
        logs = {
            f"{key}_{prefix}/{loader_key}": value for key, value in
            metrics.items()
        }
        wandb.log(logs)

    def log_metrics(
            self,
            metrics: Dict[str, Any],
            scope: str = None,
            # experiment info
            run_key: str = None,
            global_epoch_step: int = 0,
            global_batch_step: int = 0,
            global_sample_step: int = 0,
            # stage info
            stage_key: str = None,
            stage_epoch_len: int = 0,
            stage_epoch_step: int = 0,
            stage_batch_step: int = 0,
            stage_sample_step: int = 0,
            # loader info
            loader_key: str = None,
            loader_batch_len: int = 0,
            loader_sample_len: int = 0,
            loader_batch_step: int = 0,
            loader_sample_step: int = 0,
    ) -> None:
        """Logs batch and epoch metrics to wandb."""
        if scope == "batch" and (global_batch_step + 100) % self.config[
            "wandb_log_period"] == 0:
            metrics = {k: float(v) for k, v in metrics.items()}
            self._log_metrics(
                metrics=metrics, step=global_batch_step, loader_key=loader_key,
                prefix="batch"
            )
        elif scope == "epoch":
            loader_key = "_epoch_"
            per_loader_metrics = metrics[loader_key]
            self._log_metrics(
                metrics=per_loader_metrics,
                step=global_batch_step,
                loader_key=loader_key,
                prefix="epoch",
            )
