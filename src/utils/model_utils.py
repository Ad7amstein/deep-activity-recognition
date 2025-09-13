"""Training and evaluation utilities for PyTorch models."""

import os
from typing import Dict, Any
from tqdm import tqdm
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
from utils.config_utils import get_settings
from enums import ModelResults

app_settings = get_settings()


# time out the experience
def print_train_time(start: float, end: float, device: torch.device) -> float:
    """Prints and returns the elapsed training time.

    Args:
        start (float): Start time of computation (e.g., from timeit).
        end (float): End time of computation.
        device (torch.device): Device on which the computation is running.

    Returns:
        float: Time difference in seconds between start and end.
    """

    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def train_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    acc_fn: nn.Module,
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
    model.train()
    train_loss, train_acc = 0, 0

    for _, (x, y) in tqdm(
        enumerate(data_loader),
        desc="Train Batches",
        disable=not verbose,
        unit="batch",
    ):
        x, y = x.to(device), y.to(device)

        y_logits = model(x).squeeze(dim=1)
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_labels = torch.round(torch.max(y_pred_probs, dim=1).values)

        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        acc = acc_fn(y_pred_labels, y)
        train_acc += acc.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    acc_fn: nn.Module,
    precision_fn: nn.Module,
    recall_fn: nn.Module,
    f1_score_fn: nn.Module,
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

    test_loss, test_acc = 0, 0
    test_precision, test_recall, test_f1_score = 0, 0, 0

    model.eval()
    with torch.inference_mode():
        for _, (x, y) in tqdm(
            enumerate(data_loader),
            desc="Test Batches",
            disable=not verbose,
            unit="batch",
        ):
            x, y = x.to(device), y.to(device)
            y = y.type(torch.float)

            test_pred_logits = model(x).squeeze(dim=1)
            test_pred_labels = torch.round(torch.sigmoid(test_pred_logits))

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_acc += acc_fn(test_pred_labels, y).item()

            precision = precision_fn(test_pred_labels, y)
            test_precision += precision.item()
            recall = recall_fn(test_pred_labels, y)
            test_recall += recall.item()
            f1_score = f1_score_fn(test_pred_labels, y)
            test_f1_score += f1_score.item()

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        test_precision /= len(data_loader)
        test_recall /= len(data_loader)
        test_f1_score /= len(data_loader)
    return test_loss, test_acc, test_precision, test_recall, test_f1_score


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = app_settings.EPOCHS,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    num_classes=app_settings.NUM_ACTIVITY_LABELS,
    verbose: bool = False,
):
    """Runs the full training loop with evaluation and checkpointing.

    Args:
        model (nn.Module): Model to be trained.
        train_dataloader (DataLoader): DataLoader for training data.
        test_dataloader (DataLoader): DataLoader for test data.
        loss_fn (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model parameters.
        epochs (int, optional): Number of training epochs. Defaults to app_settings.EPOCHS.
        device (torch.device, optional): Device for training.
            Defaults to CUDA if available, else CPU.
        verbose (bool, optional): If True, prints progress. Defaults to False.

    Returns:
        dict: Dictionary containing lists of metrics (loss, accuracy, precision,
        recall, F1-score) for all epochs.
    """

    acc_fn = MulticlassAccuracy(num_classes=num_classes)
    precision_fn=MulticlassPrecision(num_classes=num_classes)
    recall_fn = MulticlassRecall(num_classes=app_settings.NUM_ACTIVITY_LABELS)
    f1_score_fn = MulticlassF1Score(num_classes=app_settings.NUM_ACTIVITY_LABELS)

    results = {
        ModelResults.TRAIN_LOSS.value: [],
        ModelResults.TRAIN_ACCURACY.value: [],
        ModelResults.TEST_LOSS.value: [],
        ModelResults.TEST_ACCURACY.value: [],
        ModelResults.TEST_PRECISION.value: [],
        ModelResults.TEST_RECALL.value: [],
        ModelResults.TEST_F1_SCORE.value: [],
    }

    for epoch in tqdm(
        range(epochs), desc="Train Epochs", disable=not verbose, unit="Epoch", position=0
    ):
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_dataloader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            optimizer=optimizer,
            device=device,
            verbose=False,
        )

        test_loss, test_acc, test_precision, test_recall, test_f1_score = test_step(
            model=model,
            device=device,
            data_loader=test_dataloader,
            loss_fn=loss_fn,
            acc_fn=acc_fn,
            precision_fn=precision_fn,
            recall_fn=recall_fn,
            f1_score_fn=f1_score_fn,
            verbose=False,
        )

        if verbose:
            print(
                "".join(
                    [
                        f"Epoch: {epoch} | ",
                        f"Train loss: {train_loss:.4f} | ",
                        f"Train acc: {100*train_acc:.2f}% | ",
                        f"Eval loss: {test_loss:.4f} | ",
                        f"Eval acc: {100*test_acc:.2f}%",
                    ]
                )
            )

        results[ModelResults.TRAIN_LOSS.value].append(train_loss)
        results[ModelResults.TRAIN_ACCURACY.value].append(train_acc)
        results[ModelResults.TEST_ACCURACY.value].append(test_loss)
        results[ModelResults.TEST_ACCURACY.value].append(test_acc)
        results[ModelResults.TEST_PRECISION.value].append(test_precision)
        results[ModelResults.TEST_RECALL.value].append(test_recall)
        results[ModelResults.TEST_F1_SCORE.value].append(test_f1_score)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "results": results,
        }

        save_checkpoint(
            save_path=app_settings.PATH_MODELS_CHECKPOINTS,
            file_name=model.__class__.__name__,
            checkpoint=checkpoint,
            verbose=True,
        )

    save_checkpoint(
        save_path=app_settings.PATH_MODELS_CHECKPOINTS,
        file_name=model.__class__.__name__,
        checkpoint=checkpoint,
    )

    return results


def save_checkpoint(
    save_path, file_name, checkpoint: Dict[str, Any], verbose: bool = False
):
    """Saves a model checkpoint to disk.

    Args:
        save_path (str): Directory path to save the checkpoint.
        file_name (str): Base name of the checkpoint file.
        checkpoint (Dict[str, Any]): Dictionary containing epoch, model state,
            optimizer state, and training results.
        verbose (bool, optional): If True, prints confirmation message. Defaults to False.
    """

    full_save_path = os.path.join(
        save_path, f"{file_name}_epoch_{checkpoint['epoch'] + 1}.pth"
    )
    torch.save(checkpoint, full_save_path)
    if verbose:
        print(f"Checkpoint saved for epoch {checkpoint['epoch'] + 1}\n")


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
    acc_fn: nn.Module = MulticlassAccuracy(
        num_classes=app_settings.NUM_ACTIVITY_LABELS
    ),
    precision_fn: nn.Module = MulticlassPrecision(
        num_classes=app_settings.NUM_ACTIVITY_LABELS
    ),
    recall_fn: nn.Module = MulticlassRecall(
        num_classes=app_settings.NUM_ACTIVITY_LABELS
    ),
    f1_score_fn: nn.Module = MulticlassF1Score(
        num_classes=app_settings.NUM_ACTIVITY_LABELS
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

    test_loss, test_acc, test_precision, test_recall, test_f1_score = test_step(
        model=model,
        device=device,
        data_loader=test_loader,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        precision_fn=precision_fn,
        recall_fn=recall_fn,
        f1_score_fn=f1_score_fn,
        verbose=verbose,
    )

    results = {
        ModelResults.TEST_LOSS.value: test_loss,
        ModelResults.TEST_ACCURACY.value: test_acc,
        ModelResults.TEST_PRECISION.value: test_precision,
        ModelResults.TEST_RECALL.value: test_recall,
        ModelResults.TEST_F1_SCORE.value: test_f1_score,
    }

    if verbose:
        print(
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

    return results


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
