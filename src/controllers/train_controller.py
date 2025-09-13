import os
from typing import Type
from torch import nn
from torch.utils.data import Dataset
from controllers.base_controller import BaseController
from utils.model_utils import train
from data_processing import AnnotationLoader


class TrainController(BaseController):
    def __init__(self, DatasetClass: Type[Dataset]) -> None:
        super().__init__()
        self.volleyball_data = AnnotationLoader(verbose=True).load_pkl_version()
        self.DatasetClass = DatasetClass

    def train(self, model: nn.Module):
        train_dataset = self.DatasetClass()


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
