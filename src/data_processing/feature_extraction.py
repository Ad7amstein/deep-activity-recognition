"""
Feature Extraction Module for Volleyball Dataset.
"""

import os
from typing import Optional, Callable
import cv2 as cv
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms, models
from models import (  # pylint: disable=[W0611]
    VolleyballData,
    TrackingData,
    ClipAnnotation,
)
from data_processing.annot_loading import AnnotationLoader
from utils.config_utils import get_settings
from utils.logging_utils import setup_logger
from controllers.base_controller import BaseController

app_settings = get_settings()


class FeatureExtractor:
    """
    A class to handle feature extraction from volleyball dataset videos.

    Attributes:
        verbose (bool): Whether to print progress information.
        logger (logging.Logger): Logger instance for recording events and debugging information.
        model (torch.nn.Module or None): The deep learning model used for feature extraction.
        transform (callable or None): Data transformation pipeline applied to input frames.
        device (str): The device used for computation ("cuda" if available, otherwise "cpu").
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the FeatureExtractor.

        Args:
            verbose (bool): Whether to print progress information. Defaults to False.
        """

        self.verbose = verbose
        self.logger = setup_logger(
            logger_name=__name__,
            log_file=__file__,
            log_dir=os.path.join(app_settings.PATH_LOGS, BaseController.get_baseline_root()),
            log_to_console=self.verbose,
            use_tqdm=True,
        )
        self.logger.info("Initializing Feature Extractor Module...")
        self.model = None
        self.transform = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_model(
        self,
        model: Callable[..., nn.Module],
        img_level: bool = False,
        last_layer: int = -1,
        verbose: Optional[bool] = None,
    ) -> tuple[nn.Module, transforms.Compose]:
        """
        Prepare the CNN model and preprocessing transforms.

        Args:
            model (Callable[..., nn.Module]): A callable that returns a pretrained model
                (e.g., torchvision.models.resnet50).
            img_level (bool): If True, prepares transforms for full-frame extraction.
                Otherwise, prepares transforms for cropped bounding boxes.
            last_layer (int): Index of the last layer to keep from the model.
                Defaults to -1 (use all layers except final classifier).
            verbose (Optional[bool]): Override for verbosity. Defaults to class-level setting.

        Returns:
            tuple[nn.Module, transforms.Compose]: The prepared model and preprocessing transform.
        """

        verbose = self.verbose if not verbose else verbose
        if verbose:
            self.logger.info("Preparing Model and Preprocessor...")

        if self.transform is None:
            if img_level:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        self.model = nn.Sequential(*(list(model.children())[:last_layer]))
        self.model.to(self.device)
        self.model.eval()

        return self.model, self.transform

    def extract_features(
        self,
        videos_root: str,
        output_file: str,
        volleyball_data: VolleyballData,
        img_level: bool = False,
        verbose: Optional[bool] = None,
    ):
        """
        Extract features from the volleyball dataset videos.

        Args:
            videos_root (str): Root directory containing volleyball videos.
            output_file (str): Path to the output `.npy` file where features will be saved.
            volleyball_data (VolleyballData): Parsed dataset annotations.
            img_level (bool): If True, extract features at full-frame level.
                If False, extract features from bounding boxes.
            verbose (Optional[bool]): Override for verbosity. Defaults to class-level setting.
        """

        verbose = self.verbose if not verbose else verbose
        if verbose:
            self.logger.info("Extracting Features from Volleyball Dataset...")

        with torch.inference_mode():
            for video_id, video_annot_dct in tqdm(
                volleyball_data.items(),
                desc="Processing Videos",
                disable=not verbose,
                unit="video",
            ):
                video_path = os.path.join(videos_root, str(video_id))
                for clip_id, clip_annot_dct in tqdm(
                    video_annot_dct.items(),
                    desc="Processing Clips",
                    disable=not verbose,
                    unit="clip",
                ):
                    clip_path = os.path.join(video_path, str(clip_id))
                    for frame_id, boxes_info in clip_annot_dct["tracking_annot_dct"][
                        1
                    ].items():
                        img_path = os.path.join(clip_path, f"{frame_id}.jpg")
                        img_rgb = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)  # type: ignore # pylint: disable=[E1101]
                        preprocessed_imgs = []

                        if img_level:
                            preprocessed_imgs.append(self.transform(img_rgb))  # type: ignore
                        else:
                            for box_info in boxes_info:
                                x1, y1, x2, y2 = box_info.box
                                cropped_img = img_rgb[y1:y2, x1:x2]
                                preprocessed_imgs.append(
                                    self.transform(cropped_img).unsqueeze(0)  # type: ignore
                                )

                        preprocessed_imgs = torch.cat(preprocessed_imgs)
                        preprocessed_imgs = preprocessed_imgs.to(self.device)
                        if img_level:
                            preprocessed_imgs = preprocessed_imgs.unsqueeze(0)

                        extracted_features = self.model(preprocessed_imgs)  # type: ignore
                        extracted_features = extracted_features.view(
                            extracted_features.shape[0], -1
                        )
                        np.save(output_file, extracted_features.cpu().detach().numpy())

                        # Free GPU memory
                        del preprocessed_imgs, extracted_features
                        torch.cuda.empty_cache()


def main():
    """Entry Point for the Program."""

    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")
    feature_extractor = FeatureExtractor()
    _, _ = feature_extractor.prepare_model(model=models.resnet50(pretrained=True))
    annot_loader = AnnotationLoader(verbose=True)
    videos_root = os.path.join(
        app_settings.PATH_DATA_ROOT, app_settings.PATH_VIDEOS_ROOT
    )
    feature_extractor.extract_features(
        videos_root,
        "out.npy",
        annot_loader.load_pkl_version(verbose=True),
        img_level=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
