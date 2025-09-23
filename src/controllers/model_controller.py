import os
from typing import Union
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader
from controllers.base_controller import BaseController
from data_processing.annot_loading import AnnotationLoader
from utils.model_utils import train, test
from utils.logging_utils import setup_logger
from enums import ModelMode, OptimizerEnum, LossFNEnum, ModelBaseline
from stores.baselines.providers import B1CustomDataset, B1Model


class ModelController(BaseController):
    def __init__(
        self, DatasetClass, baseline_number: int, verbose: bool = True
    ) -> None:
        super().__init__()
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=self.app_settings.PATH_LOGS,
            log_to_console=verbose,
            use_tqdm=True,
        )

        try:
            self.model = self.load_model(baseline_number)
        except ValueError as exc:
            self.logger.exception(str(exc))
            raise exc

        self.baseline_number = baseline_number
        self.baseline_config = self.load_config(self.baseline_number)
        self.volleyball_data = AnnotationLoader(verbose=True).load_pkl_version(
            verbose=True
        )
        self.DatasetClass = DatasetClass
        self.loss_fn = self.load_loss_fn()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()

        print(f"\n[Baseline-{self.baseline_number} Configuration]")
        for k, v in vars(self.baseline_config).items():
            print(f"  - {k}: {v}")
        print(f"  - Scheduler: {type(self.scheduler).__name__}")

    def train(self):
        train_dataset = self.DatasetClass(
            volleyball_data=self.volleyball_data, mode=ModelMode.TRAIN, verbose=True
        )
        valid_dataset = self.DatasetClass(
            volleyball_data=self.volleyball_data,
            mode=ModelMode.VALIDATION,
            verbose=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.baseline_config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=self.baseline_config.NUM_WORKERS,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.baseline_config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.baseline_config.NUM_WORKERS,
        )
        train(
            model=self.model,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.baseline_config.TRAIN_EPOCHS,
            baseline_path=self.baseline_config.PATH_MODEL,
            num_classes=self.app_settings.NUM_ACTIVITY_LABELS,
            verbose=True,
        )

    def test(self):
        test_dataset = self.DatasetClass(
            volleyball_data=self.volleyball_data, mode=ModelMode.TEST, verbose=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.baseline_config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.baseline_config.NUM_WORKERS,
        )
        test(
            model=self.model,
            test_loader=test_loader,
            loss_fn=self.loss_fn,
            num_classes=self.app_settings.NUM_ACTIVITY_LABELS,
            verbose=True,
        )

    def inference(self, x):
        pass

    def load_model(self, baseline_number: int) -> nn.Module:
        baseline_model = {
            ModelBaseline.BASELINE1.value: B1Model(),
            ModelBaseline.BASELINE2.value: None,
            ModelBaseline.BASELINE3.value: None,
            ModelBaseline.BASELINE4.value: None,
            ModelBaseline.BASELINE5.value: None,
            ModelBaseline.BASELINE6.value: None,
            ModelBaseline.BASELINE7.value: None,
            ModelBaseline.BASELINE8.value: None,
        }

        model: Union[nn.Module, None] = baseline_model.get(baseline_number, None)
        if model is None:
            raise ValueError(
                f"The Model for the given baseline number ({baseline_number}) not found."
            )

        return model

    def load_optimizer(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=self.baseline_config.LR,
            weight_decay=self.baseline_config.WEIGHT_DECAY,
        )
        if self.baseline_config.OPTIMIZER == OptimizerEnum.ADAMW.value:
            optimizer = torch.optim.AdamW(
                params=self.model.parameters(),
                lr=self.baseline_config.LR,
                weight_decay=self.baseline_config.WEIGHT_DECAY,
            )

        return optimizer

    def load_loss_fn(self) -> nn.Module:
        loss_fn = nn.CrossEntropyLoss()
        if self.baseline_config.LOSS_FN == LossFNEnum.BCE_LOSS.value:
            loss_fn = nn.BCELoss()

        return loss_fn

    def load_scheduler(
        self,
    ) -> (
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        """Load a learning rate scheduler for the optimizer.

        Returns:
            torch.optim.lr_scheduler: Scheduler object controlling LR updates.
        """
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=3,
        )

        return scheduler

    def load_config(self, baseline_number: int) -> SimpleNamespace:
        """
        Load configuration for the given baseline into a dictionary with general keys.

        Args:
            baseline_number (int): The baseline number (e.g., 1 for baseline-1).

        Returns:
            dict: A dictionary of baseline configuration values
                with general keys (e.g., TRAIN_EPOCHS instead of B1_TRAIN_EPOCHS).
        """
        settings = self.app_settings

        prefix = f"B{baseline_number}_"
        config = {}

        for field_name, value in settings.__dict__.items():
            if field_name.startswith(prefix):
                general_key = field_name.replace(prefix, "", 1)
                config[general_key] = value

        return SimpleNamespace(**config)


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")
    train_controller = ModelController(B1CustomDataset, baseline_number=10)
    train_controller.train()


if __name__ == "__main__":
    main()
