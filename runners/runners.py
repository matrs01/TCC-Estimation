from abc import ABC
from typing import Dict, Any

from catalyst import dl
from catalyst.engines import IEngine
from catalyst.typing import RunnerModel

from models.models import TCCModel


class FineTuningRunner(dl.SupervisedRunner, ABC):
    def __init__(
            self,
            config: Dict,
            model: RunnerModel = None,
            engine: IEngine = None,
            input_key: Any = "features",
            output_key: Any = "logits",
            target_key: str = "targets",
            loss_key: str = "loss",
    ):
        super().__init__(model, engine, input_key, output_key, target_key,
                         loss_key)
        self.config = config

    @property
    def stages(self):
        return ["train_freezed", "train_unfreezed"]

    def get_stage_len(self, stage: str) -> int:
        if stage == "train_freezed":
            return self.config["freeze_len"]
        return self._num_epochs - self.config["freeze_len"]

    def get_model(self, stage: str):
        model = self._model

        if stage == "train_freezed":
            model.freeze_extractor()
        else:
            model.unfreeze_extractor()

        return model

    def log_metrics(self, *args, **kwargs) -> None:
        """Logs batch, loader and epoch metrics to available loggers."""
        for logger in self._loggers.values():
            logger.log_metrics(
                *args,
                **kwargs,
                # experiment info
                run_key=self.run_key,
                global_sample_step=self.global_sample_step,
                global_batch_step=self.global_batch_step,
                global_epoch_step=self.global_epoch_step,
                # stage info
                stage_key=self.stage_key,
                stage_epoch_len=self.stage_epoch_len,
                stage_epoch_step=self.stage_epoch_step,
                stage_batch_step=self.stage_batch_step,
                stage_sample_step=self.stage_sample_step,
                # loader info
                loader_key=self.loader_key,
                loader_batch_len=self.loader_batch_len,
                loader_sample_len=self.loader_sample_len,
                loader_batch_step=self.loader_batch_step,
                loader_sample_step=self.loader_sample_step,
            )
