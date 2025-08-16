import os
from typing import Tuple, List, Optional
import cv2 as cv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from pydantic_models import VolleyballData
from utils.config_utils import get_settings
from enums import activity_category2label_dct
from data_processing.annot_loading import AnnotationLoader

app_settings = get_settings()


class B1CustomDataset(Dataset):
    def __init__(
        self,
        volleyball_data: VolleyballData,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("[INFO] Initializing Baseline 1 Custome Dataset...")
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
        self.volleyball_data = volleyball_data
        self.activity_category_label_dct = activity_category2label_dct
        self.dataset = self.load_img_paths_lables(verbose=self.verbose)

    def load_img_paths_lables(
        self, verbose: Optional[bool] = None
    ) -> List[Tuple[str, int]]:
        verbose = self.verbose if not verbose else verbose
        if verbose:
            print("[INFO] Loading Image Paths and Labels...")
        dataset = []
        for video_id, video_annot in tqdm(
            self.volleyball_data.items(),
            desc="Loading Dataset",
            unit="video",
            disable=not verbose,
        ):
            video_path = os.path.join(
                app_settings.PATH_DATA_ROOT,
                app_settings.PATH_VIDEOS_ROOT,
                str(video_id),
            )
            for clip_id, clip_annot in tqdm(
                video_annot.items(),
                desc="Loading Clips",
                unit="clip",
                disable=not verbose,
            ):
                clip_path = os.path.join(video_path, str(clip_id))
                clip_category = clip_annot["category"]
                for img_file in os.listdir(clip_path):
                    img_path = os.path.join(clip_path, img_file)
                    dataset.append(
                        (img_path, self.activity_category_label_dct[clip_category])
                    )

        return dataset

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        """
        Retrieve an image and its label from the dataset.
        Args:
            index (int): Index of the item to retrieve from the underlying dataset.
        Returns:
            Tuple[torch.Tensor, int]: A tuple of (image, label). `image` is the loaded image
                (after applying `self.transform` if present) and is typically a torch.Tensor.
                `label` is an integer class label.
        Raises:
            IndexError: If `index` is out of range for `self.dataset`.
            RuntimeError: If the image file cannot be read (e.g., `cv.imread` returns None).
        """

        try:
            img_path, label = self.dataset[index]
        except IndexError as exc:
            raise IndexError(
                f"Index {index} out of range for dataset with length {len(self.dataset)}"
            ) from exc

        img = cv.imread(img_path)  # type: ignore # pylint: disable=[E1101]
        if img is None:
            raise RuntimeError(f"Failed to read image at {img_path!r}")

        if self.transform:
            img = self.transform(img)

        return img, label  # type: ignore

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        Returns:
            int: The total number of elements in the underlying dataset.
        """

        return len(self.dataset)


class B1Moedl(nn.Module):
    def __init__(self, extract_features: bool = False, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("[INFO] Initializing Baseline-1 Model...")
        self.model_classifier = resnet50(pretrained=True)
        self.extract_features = extract_features
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        if self.extract_features:
            model_feature_extractor = nn.Sequential(*(list(self.model_classifier.children())[:-1]))
            model_feature_extractor.to(self.device)
            model_feature_extractor.eval()
            return model_feature_extractor(x)
        return self.model_classifier(x)


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")

    volleyball_data = AnnotationLoader(verbose=True).load_pkl_version()
    b1_dataset = B1CustomDataset(volleyball_data=volleyball_data)
    print(b1_dataset[0])


if __name__ == "__main__":
    main()
