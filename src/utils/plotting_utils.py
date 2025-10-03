"""
Visualization utilities for training and evaluation metrics.
"""

import os
from matplotlib import pyplot as plt
import seaborn as sns
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger
from models.enums import ModelResults
from models.enums import activity_category2label_dct

app_settings = get_settings()
logger = setup_logger(
    logger_name=__name__,
    log_file=__file__,
    log_dir=app_settings.PATH_LOGS,
    log_to_console=True,
    use_tqdm=True,
    file_mode="a",
)


def plot_results(results: dict, save_path: str, verbose: bool = True):
    """Plot and save training/evaluation results including confusion matrix,
    metric curves, and comparison plots.

    Args:
        results (dict):
            Dictionary mapping metric names (e.g., "train_loss", "test_accuracy",
            "confusion_matrix") to their corresponding values (lists, arrays, or matrices).
        save_path (str):
            Directory path where all plots will be saved.
            The function will create the directory if it does not exist.
        verbose (bool, optional):
            If True, logs the plotting process. Defaults to True.
    """

    if verbose:
        logger.info("Plotting Results...")
    os.makedirs(save_path, exist_ok=True)

    for name, vals in results.items():
        if name == ModelResults.CONFUSION_MATRIX.value:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                vals,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=True,
                ax=ax,
                xticklabels=list(activity_category2label_dct.keys()),
                yticklabels=list(activity_category2label_dct.keys()),
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.tight_layout()
            fig.savefig(os.path.join(save_path, "confusion_matrix.png"), dpi=300)
            plt.close(fig)
            continue
        if name in [
            ModelResults.TEST_ACCURACY.value,
            ModelResults.TRAIN_ACCURACY.value,
            ModelResults.TEST_PRECISION.value,
            ModelResults.TEST_RECALL.value,
            ModelResults.TEST_F1_SCORE.value,
        ]:
            plot_vals = [v * 100 for v in vals]
            ylabel = "Percentage (%)"
        elif name in [ModelResults.TEST_LOSS.value, ModelResults.TRAIN_LOSS.value]:
            plot_vals = vals
            ylabel = "Loss"
        elif name in [ModelResults.TIME_PER_EPOCH.value]:
            plot_vals = vals
            ylabel = "Elapsed Time (seconds)"
        elif name in [ModelResults.TOTAL_TRAIN_TIME.value]:
            continue
        else:
            plot_vals = vals
            ylabel = "Value"
        fig, ax = plt.subplots()
        ax.plot(range(1, len(plot_vals) + 1), plot_vals, marker="o")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        fig.savefig(os.path.join(save_path, f"{name}_plot.png"), dpi=300)
        plt.close(fig)

    # Configuration for the four comparison plots
    plot_configs = [
        {
            "title": "Train Loss vs. Train Accuracy",
            "filename": "train_loss_vs_accuracy.png",
            "metric1_key": ModelResults.TRAIN_LOSS.value,
            "metric1_label": "Train Loss",
            "metric1_ylabel": "Loss",
            "metric1_color": "tab:red",
            "metric1_scale": 1.0,
            "metric2_key": ModelResults.TRAIN_ACCURACY.value,
            "metric2_label": "Train Accuracy",
            "metric2_ylabel": "Accuracy (%)",
            "metric2_color": "tab:blue",
            "metric2_scale": 100.0,
            "dual_axis": True,
        },
        {
            "title": "Test Loss vs. Test Accuracy",
            "filename": "test_loss_vs_accuracy.png",
            "metric1_key": ModelResults.TEST_LOSS.value,
            "metric1_label": "Test Loss",
            "metric1_ylabel": "Loss",
            "metric1_color": "tab:red",
            "metric1_scale": 1.0,
            "metric2_key": ModelResults.TEST_ACCURACY.value,
            "metric2_label": "Test Accuracy",
            "metric2_ylabel": "Accuracy (%)",
            "metric2_color": "tab:blue",
            "metric2_scale": 100.0,
            "dual_axis": True,
        },
        {
            "title": "Train Loss vs. Test Loss",
            "filename": "train_vs_test_loss.png",
            "metric1_key": ModelResults.TRAIN_LOSS.value,
            "metric1_label": "Train Loss",
            "metric1_ylabel": "Loss",
            "metric1_color": "tab:blue",
            "metric1_scale": 1.0,
            "metric2_key": ModelResults.TEST_LOSS.value,
            "metric2_label": "Test Loss",
            "metric2_ylabel": "Loss",
            "metric2_color": "tab:orange",
            "metric2_scale": 1.0,
            "dual_axis": False,
        },
        {
            "title": "Train Accuracy vs. Test Accuracy",
            "filename": "train_vs_test_accuracy.png",
            "metric1_key": ModelResults.TRAIN_ACCURACY.value,
            "metric1_label": "Train Accuracy",
            "metric1_ylabel": "Accuracy (%)",
            "metric1_color": "tab:blue",
            "metric1_scale": 100.0,
            "metric2_key": ModelResults.TEST_ACCURACY.value,
            "metric2_label": "Test Accuracy",
            "metric2_ylabel": "Accuracy (%)",
            "metric2_color": "tab:orange",
            "metric2_scale": 100.0,
            "dual_axis": False,
        },
    ]
    for config in plot_configs:
        plot_xy_comparison(results=results, plot_config=config, save_path=save_path)


def plot_xy_comparison(results: dict, plot_config: dict, save_path: str) -> None:
    """
    Plot and compare two metrics (e.g., loss vs. accuracy) on either a single axis
    or dual y-axes.

    Args:
        results (dict): Dictionary containing metric values, where each key maps
            to a list or array of values (e.g., {"train_loss": [...], "train_acc": [...]}).
        plot_config (dict): Configuration for plotting. Expected keys include:
            - "title" (str): Title of the plot.
            - "filename" (str): Output filename for the saved figure.
            - "metric1_key" (str): Key in `results` for the first metric.
            - "metric1_label" (str): Label for the first metric line.
            - "metric1_ylabel" (str): Y-axis label for the first metric.
            - "metric1_color" (str): Color for the first metric line.
            - "metric1_scale" (float): Scaling factor for the first metric.
            - "metric2_key" (str): Key in `results` for the second metric.
            - "metric2_label" (str): Label for the second metric line.
            - "metric2_ylabel" (str): Y-axis label for the second metric.
            - "metric2_color" (str): Color for the second metric line.
            - "metric2_scale" (float): Scaling factor for the second metric.
            - "dual_axis" (bool): Whether to plot the second metric on a separate y-axis.
        save_path (str): Directory path where the plot will be saved.

    Returns:
        None: The function saves the plot as a PNG file and closes the figure.
    """

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(plot_config["metric1_ylabel"], color=plot_config["metric1_color"])
    ax1.plot(
        range(1, len(results[plot_config["metric1_key"]]) + 1),
        [v * plot_config["metric1_scale"] for v in results[plot_config["metric1_key"]]],
        marker="o",
        color=plot_config["metric1_color"],
        label=plot_config["metric1_label"],
    )
    ax1.tick_params(axis="y", labelcolor=plot_config["metric1_color"])
    ax1.grid(True)

    if plot_config["dual_axis"]:
        ax2 = ax1.twinx()
        ax2.set_ylabel(
            plot_config["metric2_ylabel"], color=plot_config["metric2_color"]
        )
        ax2.plot(
            range(1, len(results[plot_config["metric2_key"]]) + 1),
            [
                v * plot_config["metric2_scale"]
                for v in results[plot_config["metric2_key"]]
            ],
            marker="s",
            color=plot_config["metric2_color"],
            label=plot_config["metric2_label"],
        )
        ax2.tick_params(axis="y", labelcolor=plot_config["metric2_color"])
    else:
        ax1.plot(
            range(1, len(results[plot_config["metric2_key"]]) + 1),
            [
                v * plot_config["metric2_scale"]
                for v in results[plot_config["metric2_key"]]
            ],
            marker="s",
            color=plot_config["metric2_color"],
            label=plot_config["metric2_label"],
        )

    fig.suptitle(plot_config["title"])
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, plot_config["filename"]), dpi=300)
    plt.close(fig)


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
