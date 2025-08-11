import os
from typing import Optional, Callable
import cv2 as cv
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms, models
from pydantic_models.data_repr import VolleyballData


class FeatureExtractor:
    def __init__(self, verbose: bool = False) -> None:
        self.model = None
        self.transform = None
        self.verbose = verbose

    def prepare_model(
        self,
        model: Callable[..., nn.Module],
        img_level: bool = False,
        last_layer: int = -1,
        verbose: Optional[bool] = None,
    ) -> tuple[nn.Module, transforms.Compose]:
        verbose = self.verbose if not verbose else verbose
        if verbose:
            print("[INFO] Preparing Model and Preprocessor...")
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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = nn.Sequential(*(list(model.children()[:last_layer])))
        self.model.to(device)
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
        verbose = self.verbose if not verbose else verbose
        if verbose:
            print("[INFO] Extracting Features from Volleyball Dataset...")
        for video_id, video_annot_dct in tqdm(
            volleyball_data.items(),
            desc="Processing Videos",
            disable=not verbose,
            unit="video",
        ):
            video_path = os.path.join(videos_root, str(video_id))
            for clip_id, clip_annot_dct in video_annot_dct.items():
                clip_path = os.path.join(video_path, str(clip_id))
                for frame_id, boxes_info in clip_annot_dct["tracking_annot_dct"][
                    1
                ].items():
                    img_path = os.path.join(clip_path, f"{frame_id}.jpg")
                    img_rgb = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
                    preprocessed_imgs = []

                    if img_level:
                        preprocessed_imgs.append(self.transform(img_rgb)) # type: ignore
                    else:
                        for box_info in boxes_info:
                            x1, y1, x2, y2 = box_info.box
                            cropped_img = img_rgb[x1:x2, y1:y2]
                            preprocessed_imgs.append(
                                self.transform(cropped_img).unsqueeze(0) # type: ignore
                            )
                    preprocessed_imgs = torch.cat(preprocessed_imgs)
                    extracted_fetures = self.model(preprocessed_imgs) # type: ignore
                    np.save(output_file, extracted_fetures.numpy())


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )
    feature_extractor = FeatureExtractor()
    model, preprocessor = feature_extractor.prepare_model(
        model=models.resnet50(pretrained=True)
    )


if __name__ == "__main__":
    main()
