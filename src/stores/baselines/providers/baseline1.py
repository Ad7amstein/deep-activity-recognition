"""Baseline 1 dataset and model for volleyball activity recognition."""

import os
from typing import Tuple, List, Optional
import cv2 as cv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from models import VolleyballData
from models.enums import ActivityEnum, ModelMode
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger
from data_processing.annot_loading import AnnotationLoader

app_settings = get_settings()


class B1CustomDataset(Dataset):
    """Custom dataset for volleyball activity recognition.

    Attributes:
        verbose (bool): Flag to enable logging output during initialization and usage.
        logger (logging.Logger): Configured logger instance for this dataset.
        mode (str): Operational mode, typically `train`, `val`, or `test`.
        transform (transforms.Compose): Image transformation pipeline applied to input frames.
        num_right_frames (int): Number of frames to include to the right of the reference frame.
        num_left_frames (int): Number of frames to include to the left of the reference frame.
        volleyball_data (VolleyballData): Object containing volleyball-specific data and metadata.
        activity_category2label_dct (Dict[str, int]):
            Mapping from activity category names to integer labels.
        dataset (List[Tuple[str, int]]):
            List of tuples containing image paths and their corresponding labels.
    """

    def __init__(
        self,
        volleyball_data: VolleyballData,
        img_shape: Tuple[int, int] = (
            app_settings.B1_FEATURES_SHAPE_0,
            app_settings.B1_FEATURES_SHAPE_1,
        ),
        num_right_frames: int = app_settings.B1_RIGHT_FRAMES,
        num_left_frames: int = app_settings.B1_LEFT_FRAMES,
        mode: str = app_settings.MODEL_MODE,
        verbose: bool = False,
    ) -> None:
        """Initialize the B1CustomDataset.

        Args:
            volleyball_data (VolleyballData):
                Dataset object containing volleyball activity annotations and metadata.
            img_shape (Tuple[int, int], optional):
                Target image shape (height, width) after preprocessing.
                Defaults to values from application settings.
            num_right_frames (int, optional):
                Number of frames to include to the right of the reference frame.
                Defaults to `app_settings.B1_RIGHT_FRAMES`.
            num_left_frames (int, optional):
                Number of frames to include to the left of the reference frame.
                Defaults to `app_settings.B1_LEFT_FRAMES`.
            mode (str, optional): Dataset mode (`train`, `val`, or `test`).
                Determines which transformations are applied. Defaults to `app_settings.MODEL_MODE`.
            verbose (bool, optional): If True, enable detailed logging during initialization.
                Defaults to False.
        """

        super().__init__()
        self.verbose = verbose
        self.logger = setup_logger(
            logger_name=__name__,
            log_file=__file__,
            log_dir=app_settings.PATH_LOGS,
            log_to_console=self.verbose,
            use_tqdm=True,
        )
        self.mode = mode
        if self.verbose:
            self.logger.info(
                "Initializing Baseline 1 Custom Dataset (mode=%s)...", self.mode
            )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop(img_shape),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.mode == ModelMode.TRAIN:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(img_shape),
                    transforms.RandomChoice(
                        [
                            transforms.ColorJitter(
                                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                            ),
                            transforms.RandomGrayscale(p=1.0),
                            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                        ]
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.num_right_frames = num_right_frames
        self.num_left_frames = num_left_frames
        self.volleyball_data = volleyball_data
        self.activity_category2label_dct = {
            member.category: member.label for member in ActivityEnum
        }
        self.dataset = self.load_dataset(verbose=self.verbose)

        if self.verbose:
            self.logger.info(
                "\n".join(
                    [
                        "Dataset Configuration:",
                        f"  - Mode: {self.mode}",
                        f"  - Image shape: {img_shape}",
                        f"  - Num right frames: {self.num_right_frames}",
                        f"  - Num left frames: {self.num_left_frames}",
                        f"  - Dataset size (after init): {len(self.dataset)}",
                        "  - Transforms:",
                        "\n".join(f"    * {t}" for t in self.transform.transforms),
                    ]
                )
            )

    def load_dataset(self, verbose: Optional[bool] = None) -> List[Tuple[str, int]]:
        """Load image paths and labels from volleyball annotations.

        Args:
            verbose (Optional[bool]): If True, display progress logs.
                If None, uses the class-level `self.verbose`.

        Returns:
            List[Tuple[str, int]]: List of tuples containing (image_path, label).
        """

        verbose = self.verbose if not verbose else verbose
        if verbose:
            self.logger.info("Loading Image Paths and Labels...")

        dataset = []
        for video_id, video_annot in tqdm(
            self.volleyball_data.items(),
            desc="Loading Dataset",
            unit="video",
        ):
            if (
                (
                    self.mode == ModelMode.TRAIN
                    and video_id not in app_settings.TRAIN_IDS
                )
                or (
                    self.mode == ModelMode.TEST
                    and video_id not in app_settings.TEST_IDS
                )
                or (
                    self.mode == ModelMode.VALIDATION
                    and video_id not in app_settings.VALIDATION_IDS
                )
            ):
                continue

            video_path = os.path.join(
                app_settings.PATH_DATA_ROOT,
                app_settings.PATH_VIDEOS_ROOT,
                str(video_id),
            )

            for clip_id, clip_annot in tqdm(
                video_annot.items(),
                desc=f"Loading V-{video_id} Clips",
                unit="clip",
                disable=not verbose,
            ):
                clip_path = os.path.join(video_path, str(clip_id))
                clip_category = clip_annot["category"]
                img_files = os.listdir(clip_path)
                num_files = len(img_files)
                mid_idx = num_files // 2
                img_files = img_files[
                    mid_idx - self.num_left_frames : mid_idx + self.num_right_frames + 1
                ]

                for img_file in img_files:
                    img_path = os.path.join(clip_path, img_file)
                    dataset.append(
                        (img_path, self.activity_category2label_dct[clip_category])
                    )

        return dataset

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """Retrieve an image and its label from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: A tuple of (image, label).
                - image: Preprocessed image tensor.
                - label: Integer class label.

        Raises:
            IndexError: If index is out of range for the dataset.
            RuntimeError: If the image cannot be read.
        """

        try:
            img_path, label = self.dataset[index]
        except IndexError as exc:
            self.logger.exception(str(exc))
            raise IndexError(
                f"Index {index} out of range for dataset with length {len(self.dataset)}"
            ) from exc

        img = cv.imread(img_path)  # type: ignore pylint: disable=[E1101]
        if img is None:
            self.logger.exception("Failed to read image at %s", img_path)
            raise RuntimeError(f"Failed to read image at {img_path!r}")

        if self.transform:
            img = self.transform(img)

        return img, label  # type: ignore

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of dataset elements.
        """

        return len(self.dataset)


class B1Model(nn.Module):
    """Baseline 1 model using a ResNet-50 backbone.

    Attributes:
        verbose (bool): Flag to enable detailed logging during initialization.
        logger (logging.Logger): Configured logger instance for model logging.
        device (str): Device used for training/inference (`"cuda"` if available, else `"cpu"`).
        in_features (Tuple[int, int]): Input feature shape (height, width).
        num_classes (int): Number of output classes for classification.
        model (nn.Module): The underlying ResNet-50 model (modified).
        extract_features (bool): If True, the model outputs features instead of classification logits.
    """

    def __init__(
        self,
        extract_features: bool = False,
        verbose: bool = False,
        in_features: Tuple[int, int] = (
            app_settings.B1_FEATURES_SHAPE_0,
            app_settings.B1_FEATURES_SHAPE_1,
        ),
        num_classes: int = app_settings.NUM_ACTIVITY_LABELS,
    ) -> None:
        """Initialize the Baseline-1 ResNet-50 model.

        Args:
            extract_features (bool, optional):
                If True, use the model as a feature extractor instead of a classifier.
                Defaults to False.
            verbose (bool, optional):
                If True, enable detailed logging during initialization.
                Defaults to False.
            in_features (Tuple[int, int], optional):
                Input feature shape (height, width).
                Defaults to `(app_settings.B1_FEATURES_SHAPE_0, app_settings.B1_FEATURES_SHAPE_1)`.
            num_classes (int, optional):
                Number of output classes for classification.
                Defaults to `app_settings.NUM_ACTIVITY_LABELS`.
        """

        super().__init__()
        self.verbose = verbose
        self.logger = setup_logger(
            logger_name=__name__,
            log_file=__file__,
            log_dir=app_settings.PATH_LOGS,
            log_to_console=self.verbose,
            use_tqdm=True,
        )
        if self.verbose:
            self.logger.info("[INFO] Initializing Baseline-1 Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_features = in_features
        self.num_classes = num_classes
        self.model = self.prepare_model(torch.device(self.device))
        self.extract_features = extract_features

        if self.verbose:
            self.logger.info(
                "\n".join(
                    [
                        "Model Configuration:",
                        f"  - Device: {self.device}",
                        f"  - Input features shape: {self.in_features}",
                        f"  - Number of classes: {self.num_classes}",
                        f"  - Extract features mode: {self.extract_features}",
                        "  - Backbone: ResNet-50 (pretrained)",
                    ]
                )
            )

    def prepare_model(
        self, device: torch.device, verbose: Optional[bool] = None
    ) -> nn.Module:
        """Prepare a ResNet-50 model for classification.

        Args:
            device (torch.device): Device to load the model on.
            verbose (Optional[bool], optional):
                If True, enable detailed logging during preparation.
                If None, defaults to the class-level `self.verbose`.

        Returns:
            nn.Module: The modified ResNet-50 model.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            self.logger.info("Preparing Baseline-1 Model...")
        resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=verbose)
        num_features = resnet_model.fc.in_features
        resnet_model.fc = torch.nn.Linear(
            in_features=num_features, out_features=self.num_classes
        )
        if app_settings.B1_FREEZE_BACKBONE:
            for param in resnet_model.parameters():
                param.requires_grad = False

        for param in resnet_model.fc.parameters():
            param.requires_grad = True

        return resnet_model.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output. If `extract_features=True`,
            returns extracted features, otherwise classification logits.
        """
        if self.extract_features:
            model_feature_extractor = nn.Sequential(*(list(self.model.children())[:-1]))
            model_feature_extractor.to(self.device)
            model_feature_extractor.eval()
            return model_feature_extractor(x)
        return self.model(x)


def main():
    """Entry point for the program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")

    volleyball_data = AnnotationLoader(verbose=True).load_pkl_version(verbose=True)
    b1_dataset = B1CustomDataset(volleyball_data=volleyball_data)
    print(b1_dataset[0])


if __name__ == "__main__":
    main()
