import os
from typing import Optional
import torch
from torch import nn
from torchvision import transforms, models


class FeatureExtractor:
    def __init__(self, verbose: bool = False) -> None:
        self.model = None
        self.transform = None
        self.verbose = verbose

    def prepare_model(
        self,
        model: nn.Module,
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
