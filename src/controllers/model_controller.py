"""
Model controller module for managing model training, testing, inference, and configuration.
"""

import os
from typing import Union, Type, Optional
from types import SimpleNamespace
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from controllers.base_controller import BaseController
from data_processing.annot_loading import AnnotationLoader
from utils.model_utils import train, test
from utils.plotting_utils import plot_results
from utils.logging_utils import setup_logger
from models.enums import ModelMode, OptimizerEnum, LossFNEnum, ModelBaseline
from stores.baselines.providers import B1CustomDataset, B1Model


class ModelController(BaseController):
    """Controller for managing model lifecycle (train, test, inference) and configuration.

    Attributes:
        verbose (bool): Whether to enable verbose logging.
        logger (logging.Logger): Configured logger for the module.
        baseline_number (int): Identifier for the chosen model baseline.
        baseline_config (SimpleNamespace): Configuration parameters for the baseline.
        model (torch.nn.Module): Loaded baseline model.
        dataset_class (Type[Dataset]): Dataset class associated with the baseline.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        optimizer (torch.optim.Optimizer): Optimizer used for training.
        loss_fn (torch.nn.Module): Loss function used in training/testing.
        volleyball_data (Any): Dataset annotations loaded for the task.
    """

    def __init__(self, baseline_number: int, mode: str, verbose: bool = True) -> None:
        """Initialize the ModelController.

        Args:
            baseline_number (int): The baseline number to load (e.g., 1 for Baseline-1).
            mode (str): Mode of operation ("train", "test", or "inference").
            verbose (bool, optional): Whether to enable verbose logging. Defaults to True.

        Raises:
            ValueError: If the model or dataset class for the baseline is not found.
        """

        super().__init__()
        self.verbose = verbose
        self.logger = setup_logger(
            log_file=__file__,
            log_dir=self.app_settings.PATH_LOGS,
            log_to_console=self.verbose,
            use_tqdm=True,
        )
        self.logger.info("Initializing ModelController Module...")
        self.baseline_number = baseline_number
        self.baseline_config = self.load_config(self.baseline_number)

        try:
            self.model = self.load_model(self.baseline_number)
        except ValueError as exc:
            self.logger.exception(str(exc))
            raise exc

        if not mode == ModelMode.INFERENCE.value:
            self.volleyball_data = AnnotationLoader(verbose=verbose).load_pkl_version(
                verbose=verbose
            )
            try:
                self.dataset_class = self.load_dataset_class(self.baseline_number)
            except ValueError as exc:
                self.logger.exception(str(exc))
                raise exc
            self.scheduler = self.load_scheduler()
            self.optimizer = self.load_optimizer()
            self.loss_fn = self.load_loss_fn()

        config_str = [f"Baseline-{self.baseline_number} Configuration:"]
        for k, v in vars(self.baseline_config).items():
            config_str.append(f"  - {k}: {v}")

        if not mode == ModelMode.INFERENCE.value:
            config_str.append(f"  - Scheduler: {type(self.scheduler).__name__}")

        self.logger.info("\n".join(config_str))

    def train(self, verbose: Optional[bool] = None) -> None:
        """Train the model on training and validation datasets.

        Args:
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                If None, falls back to the instance-level verbosity setting. Defaults to None.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info("Setup training")
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
                str(self.model.__class__.__name__),
                str(self.baseline_config.EXPERIMENT_NUM),
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

    def test(self, verbose: Optional[bool] = None) -> None:
        """Evaluate the model on the test dataset.

        Args:
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                If None, falls back to the instance-level verbosity setting.
                Defaults to None.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info("Setup Testing")
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

    def inference(self, x: torch.Tensor) -> dict:
        """Run inference on a given input tensor.

        Args:
            x (torch.Tensor): Input tensor or array-like object to infer on.

        Returns:
            dict: A dictionary containing:
                - "preds" (list[int]): Predicted class indices.
                - "probs" (list[list[float]]): Corresponding prediction probabilities.

        Raises:
            ValueError: If the input `x` is None.
        """

        if x is None:
            raise ValueError("Input `x` for inference must not be None.")

        # ensure model on appropriate device
        device = (
            next(self.model.parameters()).device
            if any(p is not None for p in self.model.parameters())
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(device)

        # prepare input tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.to(dtype=torch.float32)

        if x.dim() == 1:
            x = x.unsqueeze(0)  # add batch dim

        x = x.to(device)

        if self.verbose:
            self.logger.info("Running inference on input with shape %s", tuple(x.shape))

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(x)
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            # compute probabilities depending on loss / task type
            if (
                getattr(self.baseline_config, "LOSS_FN", None)
                == LossFNEnum.BCE_LOSS.value
            ):
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
            else:
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1, keepdim=False)

        return {"preds": preds.cpu().tolist(), "probs": probs.cpu().tolist()}

    def load_model(
        self, baseline_number: int, verbose: Optional[bool] = None
    ) -> nn.Module:
        """Load the model corresponding to the specified baseline.

        Args:
            baseline_number (int): Baseline number to load.
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                Defaults to None.

        Returns:
            nn.Module: The loaded baseline model.

        Raises:
            ValueError: If the baseline model is not found.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info("Loading model for baseline %s", str(baseline_number))
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

    def load_dataset_class(
        self, baseline_number: int, verbose: Optional[bool] = None
    ) -> Type[Dataset]:
        """Load the dataset class corresponding to the specified baseline.

        Args:
            baseline_number (int): Baseline number to load.
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                Defaults to None.

        Returns:
            Type[Dataset]: The dataset class associated with the baseline.

        Raises:
            ValueError: If the dataset class for the baseline is not found.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info(
                "Loading dataset_class for baseline %s", str(baseline_number)
            )
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

    def load_optimizer(self, verbose: Optional[bool] = None) -> torch.optim.Optimizer:
        """Load the optimizer defined in the baseline configuration.

        Args:
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                Defaults to None.

        Returns:
            torch.optim.Optimizer: Configured optimizer instance.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info(
                "Loading optimizer: %s", str(self.baseline_config.OPTIMIZER)
            )
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

    def load_loss_fn(self, verbose: Optional[bool] = None) -> nn.Module:
        """Load the loss function defined in the baseline configuration.

        Args:
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                Defaults to None.

        Returns:
            nn.Module: Configured loss function instance.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info(
                "Loading Loss Function: %s", str(self.baseline_config.LOSS_FN)
            )
        loss_fn = nn.CrossEntropyLoss()
        if self.baseline_config.LOSS_FN == LossFNEnum.BCE_LOSS.value:
            loss_fn = nn.BCELoss()

        return loss_fn

    def load_scheduler(
        self, verbose: Optional[bool] = None
    ) -> (
        torch.optim.lr_scheduler._LRScheduler
        | torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        """Load a learning rate scheduler for the optimizer.

        Args:
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                Defaults to None.

        Returns:
            torch.optim.lr_scheduler: Scheduler object controlling learning rate updates.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info("Loading Scheduler: %s", "ReduceLROnPlateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=3,
        )

        return scheduler

    def load_config(
        self, baseline_number: int, verbose: Optional[bool] = None
    ) -> SimpleNamespace:
        """Load configuration for the given baseline.

        Args:
            baseline_number (int): Baseline number to load.
            verbose (Optional[bool], optional): Whether to enable verbose logging.
                Defaults to None.

        Returns:
            SimpleNamespace: Namespace containing baseline configuration values.

        Warns:
            UserWarning: If no configuration is found for the given baseline.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info(
                "Loading Model Config for baseline %s", self.baseline_number
            )
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

    def get_experiment_path(self, verbose: Optional[bool] = None) -> str:
        """Get the experiment path identifier for the current baseline.

        Args:
            verbose (Optional[bool], optional): Whether to enable verbose logging. Defaults to None.

        Returns:
            str: Experiment path string (e.g., "exp_1").
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info(
                "Getting Experiment Path for baseline %s", self.baseline_number
            )
        return f"exp_{str(self.baseline_config.EXPERIMENT_NUM)}"


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")
    train_controller = ModelController(baseline_number=1, mode="train")
    train_controller.train()


if __name__ == "__main__":
    main()
