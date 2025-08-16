import os
from typing import Tuple, List
import cv2 as cv
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pydantic_models import VolleyballData
from utils.config_utils import get_settings
from enums import activity_category_label_dct
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
        self.activity_category_label_dct = activity_category_label_dct
        self.dataset = self.load_img_paths_lables()

    def load_img_paths_lables(self) -> List[Tuple[str, int]]:
        img_paths = []
        for video_id, video_annot in tqdm(
            self.volleyball_data.items(),
            desc="Loading Dataset",
            unit="video",
            disable=not self.verbose,
        ):
            video_path = os.path.join(
                app_settings.PATH_DATA_ROOT,
                app_settings.PATH_VIDEOS_ROOT,
                str(video_id),
            )
            for clip_id, clip_annot in tqdm(
                video_annot.items(), desc="Loading Clips", unit="clip"
            ):
                clip_path = os.path.join(video_path, str(clip_id))
                clip_category = clip_annot["category"]
                for img_file in os.listdir(clip_path):
                    img_path = os.path.join(clip_path, img_file)
                    img_paths.append(
                        (img_path, self.activity_category_label_dct[clip_category])
                    )

        return img_paths

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img_path, label = self.dataset[index]
        img = cv.imread(img_path)  # type: ignore # pylint: disable=[E1101]
        if self.transform:
            img = self.transform(img)

        return img, label  # type: ignore

    def __len__(self) -> int:
        return len(self.dataset)


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")

    volleyball_data = AnnotationLoader().load_volleyball_dataset()
    b1_dataset = B1CustomDataset(volleyball_data=volleyball_data)
    print(b1_dataset[0])


if __name__ == "__main__":
    main()
