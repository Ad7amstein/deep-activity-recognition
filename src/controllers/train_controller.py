import os
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader
from controllers.base_controller import BaseController
from data_processing.annot_loading import AnnotationLoader
from utils.model_utils import train, test
from utils.stream_utils import log_stream
from enums import ModelMode, OptimizerEnum, LossFNEnum
from stores.baselines.providers import B1CustomDataset, B1Model


class TrainController(BaseController):
    def __init__(self, DatasetClass, model: nn.Module, baseline_number: int) -> None:
        super().__init__()
        self.model = model
        self.baseline_number = baseline_number
        self.baseline_config = self.load_config(self.baseline_number)
        self.volleyball_data = AnnotationLoader(verbose=True).load_pkl_version(
            verbose=True
        )
        self.DatasetClass = DatasetClass
        self.loss_fn = self.load_loss_fn()
        self.optimizer = self.load_optimizer()

        print(f"\n[Baseline-{self.baseline_number} Configuration]")
        for k, v in vars(self.baseline_config).items():
            print(f"  - {k}: {v}")

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
            verbose=True
        )

    def load_optimizer(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.baseline_config.LR, weight_decay=self.baseline_config.WEIGHT_DECAY)
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
    log_stream(log_file="b1", prog="train", verbose=True, overwrite=True)
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")
    train_controller = TrainController(B1CustomDataset, B1Model(verbose=True), baseline_number=1)
    train_controller.train()


if __name__ == "__main__":
    main()
