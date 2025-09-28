import os
from typing import Union, Type, Optional
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from controllers.base_controller import BaseController
from data_processing.annot_loading import AnnotationLoader
from utils.model_utils import train, test, plot_results
from utils.logging_utils import setup_logger
from enums import ModelMode, OptimizerEnum, LossFNEnum, ModelBaseline
from stores.baselines.providers import B1CustomDataset, B1Model


class ModelController(BaseController):
    def __init__(self, baseline_number: int, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=self.app_settings.PATH_LOGS_DIR,
            log_to_console=self.verbose,
            use_tqdm=True,
        )
        self.baseline_number = baseline_number
        self.baseline_config = self.load_config(self.baseline_number)

        try:
            self.model = self.load_model(self.baseline_number)
        except ValueError as exc:
            self.logger.exception(str(exc))
            raise exc

        self.volleyball_data = AnnotationLoader(verbose=True).load_pkl_version(
            verbose=True
        )
        try:
            self.dataset_class = self.load_dataset_class(self.baseline_number)
        except ValueError as exc:
            self.logger.exception(str(exc))
            raise exc

        self.loss_fn = self.load_loss_fn()
        self.optimizer = self.load_optimizer()
        self.scheduler = self.load_scheduler()

        self.logger.info("Baseline-%s Configuration:", self.baseline_number)
        for k, v in vars(self.baseline_config).items():
            self.logger.info("  - %s: %s", k, v)
        self.logger.info("  - Scheduler: %s", type(self.scheduler).__name__)

    def train(self, verbose: Optional[bool] = None):
        verbose = self.verbose if verbose is None else verbose
        train_dataset = self.dataset_class(
            volleyball_data=self.volleyball_data, mode=ModelMode.TRAIN, verbose=verbose  # type: ignore
        )
        valid_dataset = self.dataset_class(
            volleyball_data=self.volleyball_data,  # type: ignore
            mode=ModelMode.VALIDATION,  # type: ignore
            verbose=verbose,  # type: ignore
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

        train_results = train(
            model=self.model,
            train_dataloader=train_loader,
            valid_dataloader=valid_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=self.baseline_config.TRAIN_EPOCHS,
            baseline_path=os.path.join(
                self.model.__class__.__name__, self.baseline_config.EXPERIMENT_NUM
            ),
            num_classes=self.app_settings.NUM_ACTIVITY_LABELS,
            verbose=verbose,
        )

        plot_results(
            results=train_results,
            save_path=os.path.join(
                self.app_settings.PATH_ASSETS,
                self.model.__class__.__name__,
                self.get_experiment_path(),
                self.app_settings.PATH_METRICS,
            ),
        )

    def test(self, verbose: Optional[bool] = None):
        verbose = self.verbose if verbose is None else verbose
        test_dataset = self.dataset_class(
            volleyball_data=self.volleyball_data, mode=ModelMode.TEST, verbose=verbose  # type: ignore
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
            verbose=verbose,
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

    def load_dataset_class(self, baseline_number: int) -> Type[Dataset]:
        baseline_dataset = {
            ModelBaseline.BASELINE1.value: B1CustomDataset,
            ModelBaseline.BASELINE2.value: None,
            ModelBaseline.BASELINE3.value: None,
            ModelBaseline.BASELINE4.value: None,
            ModelBaseline.BASELINE5.value: None,
            ModelBaseline.BASELINE6.value: None,
            ModelBaseline.BASELINE7.value: None,
            ModelBaseline.BASELINE8.value: None,
        }

        dataset_class: Union[Type[Dataset], None] = baseline_dataset.get(
            baseline_number, None
        )
        if dataset_class is None:
            raise ValueError(
                f"The Dataset Class for the given baseline number ({baseline_number}) not found."
            )

        return dataset_class

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

        if len(list(config.keys())) == 0:
            self.logger.warning(
                "The Config for the given baseline number (%s) not found.",
                baseline_number,
            )

        return SimpleNamespace(**config)

    def get_experiment_path(self) -> str:
        return f"exp_{self.baseline_config.EXPERIMENT_NUM}"


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")
    train_controller = ModelController(baseline_number=10)
    train_controller.train()


if __name__ == "__main__":
    main()
