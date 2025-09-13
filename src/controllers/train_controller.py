import os
from typing import Type
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from controllers.base_controller import BaseController
from data_processing.annot_loading import AnnotationLoader
from utils.model_utils import train
from utils.stream_utils import log_stream
from enums.model import ModelMode
from enums.baselines import B1Enum
from stores.baselines.providers import B1CustomDataset, B1Model


class TrainController(BaseController):
    def __init__(self, DatasetClass, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.volleyball_data = AnnotationLoader(verbose=True).load_pkl_version(
            verbose=True
        )
        self.DatasetClass = DatasetClass
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=model.parameters(), lr=B1Enum.LR.value)

    def train(self):
        train_dataset = self.DatasetClass(
            volleyball_data=self.volleyball_data, mode=ModelMode.TRAIN, verbose=True
        )
        test_dataset = self.DatasetClass(
            volleyball_data=self.volleyball_data, mode=ModelMode.TEST, verbose=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=B1Enum.BATCH_SIZE.value, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=B1Enum.BATCH_SIZE.value, shuffle=False
        )
        train(
            model=self.model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            epochs=B1Enum.TRAIN_EPOCHS.value,
            verbose=True,
        )


def main():
    """Entry Point for the Program."""
    log_stream(log_file="b1", prog="train", verbose=True, overwrite=True)
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")
    train_controller = TrainController(B1CustomDataset, B1Model())
    train_controller.train()


if __name__ == "__main__":
    main()
