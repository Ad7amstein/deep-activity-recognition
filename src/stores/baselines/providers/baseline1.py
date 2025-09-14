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
from pydantic_models import VolleyballData
from utils.config_utils import get_settings
from enums import activity_category2label_dct, ModelMode
from data_processing.annot_loading import AnnotationLoader

app_settings = get_settings()


class B1CustomDataset(Dataset):
    """Custom dataset for volleyball activity recognition."""

    def __init__(
        self,
        volleyball_data: VolleyballData,
        img_shape: Tuple[int, int] = (app_settings.B1_FEATURES_SHAPE_0, app_settings.B1_FEATURES_SHAPE_1),
        num_right_frames: int = app_settings.B1_RIGHT_FRAMES,
        num_left_frames: int = app_settings.B1_LEFT_FRAMES,
        mode: str = app_settings.MODEL_MODE,
        verbose: bool = False,
    ) -> None:

        super().__init__()
        self.verbose = verbose
        self.mode = mode
        if self.verbose:
            print(
                f"[INFO] Initializing Baseline 1 Custom Dataset (mode={self.mode})..."
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
        self.num_right_frames = num_right_frames
        self.num_left_frames = num_left_frames
        self.volleyball_data = volleyball_data
        self.activity_category_label_dct = activity_category2label_dct
        self.dataset = self.load_img_paths_lables(verbose=self.verbose)

        if self.verbose:
            print("[CONFIG] Dataset Configuration:")
            print(f"  - Mode: {self.mode}")
            print(f"  - Image shape: {img_shape}")
            print(f"  - Num right frames: {self.num_right_frames}")
            print(f"  - Num left frames: {self.num_left_frames}")
            print(f"  - Dataset size (after init): {len(self.dataset)}")

    def load_img_paths_lables(
        self, verbose: Optional[bool] = None
    ) -> List[Tuple[str, int]]:
        """Load image paths and labels from volleyball annotations.

        Args:
            verbose (Optional[bool]): If True, display progress logs.
                If None, uses the class-level `self.verbose`.

        Returns:
            List[Tuple[str, int]]: List of tuples containing (image_path, label).
        """

        verbose = self.verbose if not verbose else verbose
        if verbose:
            print("[INFO] Loading Image Paths and Labels...")
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
                    mid_idx - self.num_left_frames - 1 : mid_idx + self.num_right_frames
                ]

                for img_file in img_files:
                    img_path = os.path.join(clip_path, img_file)
                    dataset.append(
                        (img_path, self.activity_category_label_dct[clip_category])
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
            raise IndexError(
                f"Index {index} out of range for dataset with length {len(self.dataset)}"
            ) from exc

        img = cv.imread(img_path)  # type: ignore pylint: disable=[E1101]
        if img is None:
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
    """Baseline 1 model using ResNet-50 backbone."""

    def __init__(
        self,
        extract_features: bool = False,
        verbose: bool = False,
        in_features: Tuple[int, int] = (app_settings.B1_FEATURES_SHAPE_0, app_settings.B1_FEATURES_SHAPE_1),
        num_classes: int = app_settings.NUM_ACTIVITY_LABELS,
    ) -> None:
        """Initialize the baseline model.

        Args:
            extract_features (bool, optional): If True, use model as a feature
                extractor instead of classifier. Defaults to False.
            verbose (bool, optional): If True, print initialization messages.
                Defaults to False.
            in_features (tuple, optional): Input shape of features. Defaults to (224, 224).
            out_shape (int, optional): Output shape (number of classes).
                Defaults to 8.
        """

        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("[INFO] Initializing Baseline-1 Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_features = in_features
        self.num_classes = num_classes
        self.model = self.prepare_model(torch.device(self.device))
        self.extract_features = extract_features

        if self.verbose:
            print("[CONFIG] Model Configuration:")
            print(f"  - Device: {self.device}")
            print(f"  - Input features shape: {self.in_features}")
            print(f"  - Number of classes: {self.num_classes}")
            print(f"  - Extract features mode: {self.extract_features}")
            print(f"  - Backbone: ResNet-50 (pretrained)")

    def prepare_model(
        self, device: torch.device, verbose: Optional[bool] = None
    ) -> nn.Module:
        """Prepare a ResNet-50 model for classification.

        Args:
            device (torch.device): Device to load the model on.

        Returns:
            nn.Module: The modified ResNet-50 model.
        """

        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print("[INFO] Preparing Baseline-1 Model...")
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
