"""Training and evaluation utilities for PyTorch models."""

import os
import time
from datetime import timedelta
from datetime import datetime
from typing import Dict, Any, Optional
import json
from tqdm import tqdm
import torch
from torch import nn
from torchmetrics import Metric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    ConfusionMatrix,
)
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger
from models.enums import ModelResults
from controllers.base_controller import BaseController

app_settings = get_settings()
logger = setup_logger(
    logger_name=__name__,
    log_file=__file__,
    log_dir=os.path.join(app_settings.PATH_LOGS, BaseController.get_baseline_root()),
    log_to_console=True,
    use_tqdm=True,
    file_mode="a",
)


# time out the experience
def get_train_time(start: float, end: float) -> float:
    """Returns the elapsed training time.

    Args:
        start (float): Start time of computation (e.g., from timeit).
        end (float): End time of computation.

    Returns:
        float: Time difference in seconds between start and end.
    """

    total_time = end - start
    return total_time


def train_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    acc_fn: Metric,
    device: torch.device,
    verbose: bool = False,
):
    """Performs a single training epoch step.

    Args:
        model (nn.Module): Model to be trained.
        data_loader (DataLoader): DataLoader for training data.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        acc_fn (nn.Module): Accuracy metric function.
        device (torch.device): Device to perform training on.
        verbose (bool, optional): If True, shows batch progress. Defaults to False.

    Returns:
        Tuple[float, float]: Average training loss and accuracy for the epoch.
    """

    model.to(device)
    acc_fn.to(device)
    acc_fn.reset()
    model.train()
    train_loss, train_acc = 0, 0

    for batch_idx, (x, y) in tqdm(
        enumerate(data_loader),
        desc="Train Batches",
        disable=not verbose,
        unit="batch",
    ):
        x, y = x.to(device), y.to(device)

        y_logits = model(x).squeeze(dim=1)
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_labels = torch.argmax(y_pred_probs, dim=1)

        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        train_acc = acc_fn(y_pred_labels, y).item() * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and batch_idx % 100 == 0:
            logger.info(
                "".join(
                    [
                        f"\t[BATCH {batch_idx}/{len(data_loader)}] ",
                        f"Loss: {loss.item():.4f} | Acc: {train_acc:.2f}%",
                    ]
                )
            )

    train_loss /= len(data_loader)
    train_acc = acc_fn.compute().item()

    return train_loss, train_acc


def test_step(
    model: nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    acc_fn: Metric,
    precision_fn: Metric,
    recall_fn: Metric,
    f1_score_fn: Metric,
    confusion_matrix_fn: Metric,
    verbose: bool = False,
):
    """Evaluates the model on a validation/test set for one epoch.

    Args:
        model (nn.Module): Model to be evaluated.
        device (torch.device): Device to perform evaluation on.
        data_loader (DataLoader): DataLoader for evaluation data.
        loss_fn (nn.Module): Loss function.
        acc_fn (nn.Module): Accuracy metric function.
        precision_fn (nn.Module): Precision metric function.
        recall_fn (nn.Module): Recall metric function.
        f1_score_fn (nn.Module): F1-score metric function.
        verbose (bool, optional): If True, shows batch progress. Defaults to False.

    Returns:
        Tuple[float, float, float, float, float]:
            Average test loss, accuracy, precision, recall, and F1-score.
    """

    model.to(device)
    acc_fn.to(device)
    precision_fn.to(device)
    recall_fn.to(device)
    f1_score_fn.to(device)
    confusion_matrix_fn.to(device)

    acc_fn.reset()
    precision_fn.reset()
    recall_fn.reset()
    f1_score_fn.reset()
    confusion_matrix_fn.reset()

    test_loss = 0

    model.eval()
    with torch.inference_mode():
        for _, (x, y) in tqdm(
            enumerate(data_loader),
            desc="Test Batches",
            disable=not verbose,
            unit="batch",
        ):
            x, y = x.to(device), y.to(device)

            test_pred_logits = model(x).squeeze(dim=1)
            test_pred_probs = torch.softmax(test_pred_logits, dim=1)
            test_pred_labels = torch.argmax(test_pred_probs, dim=1)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            acc_fn.update(test_pred_labels, y)
            precision_fn.update(test_pred_labels, y)
            recall_fn.update(test_pred_labels, y)
            f1_score_fn.update(test_pred_labels, y)
            confusion_matrix_fn.update(test_pred_labels, y)

        test_loss /= len(data_loader)
        test_acc = acc_fn.compute().item()
        test_precision = precision_fn.compute().item()
        test_recall = recall_fn.compute().item()
        test_f1_score = f1_score_fn.compute().item()
        test_confmat = confusion_matrix_fn.compute().cpu().numpy()

    return test_loss, test_acc, test_precision, test_recall, test_f1_score, test_confmat


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epochs: int = app_settings.EPOCHS,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    baseline_path: str = "baseline",
    verbose: bool = False,
) -> dict[str, Any]:
    """Runs the full training loop with evaluation and checkpointing.

    Args:
        model (nn.Module): Model to be trained.
        train_dataloader (DataLoader): DataLoader for training data.
        valid_dataloader (DataLoader): DataLoader for validation data.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        epochs (int, optional): Number of training epochs. Defaults to app_settings.EPOCHS.
        device (torch.device, optional): Device for training.
            Defaults to CUDA if available, else CPU.
        verbose (bool, optional): If True, prints progress. Defaults to False.

    Returns:
        dict: Dictionary containing lists of metrics (loss, accuracy, precision,
        recall, F1-score, confusion-matrix) for all epochs.
    """

    if verbose:
        logger.info("Training Started and in Progress...")

    acc_fn = MulticlassAccuracy(num_classes=num_classes)
    precision_fn = MulticlassPrecision(num_classes=num_classes)
    recall_fn = MulticlassRecall(num_classes=num_classes)
    f1_score_fn = MulticlassF1Score(num_classes=num_classes)
    confusion_matrix_fn = ConfusionMatrix(
        num_classes=num_classes, task="multiclass"
    )
    best_val_loss = float("inf")
    best_val_acc = float("-inf")

    results = {
        ModelResults.TRAIN_LOSS.value: [],
        ModelResults.TRAIN_ACCURACY.value: [],
        ModelResults.TEST_LOSS.value: [],
        ModelResults.TEST_ACCURACY.value: [],
        ModelResults.TEST_PRECISION.value: [],
        ModelResults.TEST_RECALL.value: [],
        ModelResults.TEST_F1_SCORE.value: [],
        ModelResults.TIME_PER_EPOCH.value: [],
        ModelResults.CONFUSION_MATRIX.value: None,
        ModelResults.TOTAL_TRAIN_TIME.value: None,
    }

    start_time = time.time()
    for epoch in tqdm(
        range(epochs), desc="Train Epochs", disable=not verbose, unit="Epoch"
    ):
        if verbose:
            logger.info("Epoch %s/%s", str(epoch + 1), str(epochs))
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        (
            test_loss,
            test_acc,
            test_precision,
            test_recall,
            test_f1_score,
            test_confmat,
        ) = test_step(
            model=model,
            device=device,
            data_loader=valid_dataloader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            precision_fn=precision_fn,
            recall_fn=recall_fn,
            f1_score_fn=f1_score_fn,
            confusion_matrix_fn=confusion_matrix_fn,
            verbose=True,
        )

        if scheduler:
            scheduler.step(test_loss)

        if verbose:
            logger.info(
                "".join(
                    [
                        f"Epoch: {epoch + 1} | ",
                        f"Train loss: {train_loss:.4f} | ",
                        f"Train acc: {100*train_acc:.2f}% | ",
                        f"Eval loss: {test_loss:.4f} | ",
                        f"Eval acc: {100*test_acc:.2f}%",
                    ]
                )
            )

        epoch_end_time = time.time()
        epoch_total_time = get_train_time(start_time, epoch_end_time)
        results[ModelResults.TRAIN_LOSS.value].append(train_loss)
        results[ModelResults.TRAIN_ACCURACY.value].append(train_acc)
        results[ModelResults.TEST_LOSS.value].append(test_loss)
        results[ModelResults.TEST_ACCURACY.value].append(test_acc)
        results[ModelResults.TEST_PRECISION.value].append(test_precision)
        results[ModelResults.TEST_RECALL.value].append(test_recall)
        results[ModelResults.TEST_F1_SCORE.value].append(test_f1_score)
        results[ModelResults.TIME_PER_EPOCH.value].append(epoch_total_time)
        results[ModelResults.CONFUSION_MATRIX.value] = test_confmat

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results": results,
        }

        # Save each 2 epocsh and ensure last checkpoint is saved
        if epoch % 2 == 0 or epoch == epochs - 1:
            save_checkpoint(
                save_path=os.path.join(
                    app_settings.PATH_MODELS,
                    baseline_path,
                    app_settings.PATH_MODELS_CHECKPOINTS,
                    "epochs",
                ),
                file_name=f"{model.__class__.__name__}_epoch_{epoch+1}",
                checkpoint=checkpoint,
                verbose=True,
            )

        # Save the last epoch as the main model
        if epoch == epochs - 1:
            save_checkpoint(
                save_path=app_settings.PATH_MODELS,
                file_name=f"{model.__class__.__name__}",
                checkpoint=checkpoint,
            )

        # Save the best loss
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            save_checkpoint(
                save_path=os.path.join(
                    app_settings.PATH_MODELS,
                    baseline_path,
                    app_settings.PATH_MODELS_CHECKPOINTS,
                    "best",
                ),
                file_name=f"{model.__class__.__name__}_best_loss",
                checkpoint=checkpoint,
                verbose=True,
            )

        # Save the best accuracy
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            save_checkpoint(
                save_path=os.path.join(
                    app_settings.PATH_MODELS,
                    baseline_path,
                    app_settings.PATH_MODELS_CHECKPOINTS,
                    "best",
                ),
                file_name=f"{model.__class__.__name__}_best_acc",
                checkpoint=checkpoint,
                verbose=True,
            )

    end_time = time.time()
    total_train_time = get_train_time(start_time, end_time)
    if verbose:
        logger.info(
            "Total Train time on %s: %s",
            device,
            str(timedelta(seconds=int(total_train_time))),
        )

    results[ModelResults.TOTAL_TRAIN_TIME.value] = total_train_time
    return results


def save_checkpoint(
    save_path, file_name, checkpoint: Dict[str, Any], verbose: bool = False
) -> None:
    """Saves a model checkpoint to disk.

    Args:
        save_path (str): Directory path to save the checkpoint.
        file_name (str): Base name of the checkpoint file.
        checkpoint (Dict[str, Any]): Dictionary containing epoch, model state,
            optimizer state, and training results.
        verbose (bool, optional): If True, prints confirmation message. Defaults to False.
    """

    os.makedirs(save_path, exist_ok=True)
    torch.save(
        checkpoint,
        os.path.join(save_path, f"{file_name}.pth"),
    )
    if verbose:
        logger.info("Checkpoint %s saved in %s", file_name, save_path)


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    num_classes: int,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    verbose: bool = False,
):
    """Runs evaluation on the test dataset and prints performance metrics.

    Args:
        model (nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for test dataset.
        loss_fn (nn.Module): Loss function.
        device (torch.device, optional): Device for evaluation. Defaults to CUDA if available.
        acc_fn (nn.Module, optional): Accuracy metric function.
        precision_fn (nn.Module, optional): Precision metric function.
        recall_fn (nn.Module, optional): Recall metric function.
        f1_score_fn (nn.Module, optional): F1-score metric function.
        verbose (bool, optional): If True, prints batch progress. Defaults to False.

    Returns:
        dict: Dictionary containing test loss, accuracy, precision, recall, and F1-score.
    """
    if verbose:
        logger.info("Testing Started and in Progress...")

    acc_fn = MulticlassAccuracy(num_classes=num_classes)
    precision_fn = MulticlassPrecision(num_classes=num_classes)
    recall_fn = MulticlassRecall(num_classes=num_classes)
    f1_score_fn = MulticlassF1Score(num_classes=num_classes)
    confusion_matrix_fn = ConfusionMatrix(
        num_classes=num_classes, task="multiclass"
    )

    test_loss, test_acc, test_precision, test_recall, test_f1_score, test_confmat = (
        test_step(
            model=model,
            device=device,
            data_loader=test_loader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            precision_fn=precision_fn,
            recall_fn=recall_fn,
            f1_score_fn=f1_score_fn,
            confusion_matrix_fn=confusion_matrix_fn,
            verbose=verbose,
        )
    )

    results = {
        ModelResults.TEST_LOSS.value: test_loss,
        ModelResults.TEST_ACCURACY.value: test_acc,
        ModelResults.TEST_PRECISION.value: test_precision,
        ModelResults.TEST_RECALL.value: test_recall,
        ModelResults.TEST_F1_SCORE.value: test_f1_score,
        ModelResults.CONFUSION_MATRIX.value: test_confmat,
    }

    if verbose:
        logger.info(
            "".join(
                [
                    f"Test Results -> Loss: {test_loss:.4f} | ",
                    f"Acc: {100*test_acc:.2f}% | ",
                    f"Precision: {100*test_precision:.2f}% | ",
                    f"Recall: {100*test_recall:.2f}% | ",
                    f"F1: {100*test_f1_score:.2f}%",
                ]
            )
        )

    # prepare a directory similar to where metrics/plots are stored
    save_dir = os.path.join(
        app_settings.PATH_ASSETS, model.__class__.__name__, app_settings.PATH_METRICS
    )
    os.makedirs(save_dir, exist_ok=True)

    # make a JSON-serializable copy of results
    serializable = {}
    for k, v in results.items():
        try:
            # numpy arrays and many tensor-like objects expose tolist()
            if hasattr(v, "tolist"):
                serializable[k] = v.tolist()
            else:
                # convert numeric scalars to native Python types
                if isinstance(v, (int, float)):
                    serializable[k] = v
                else:
                    serializable[k] = v
        except (AttributeError, TypeError, ValueError):
            # fallback to string representation for unsupported or unexpected types
            serializable[k] = str(v)

    # add metadata (timestamp, device, model name)
    meta = {
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "model": model.__class__.__name__,
        "device": str(device),
    }
    output = {"meta": meta, "results": serializable}

    file_name = f"test_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    if verbose:
        logger.info("Test results saved to %s", file_path)

    return results


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
