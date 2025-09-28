import os

import os
from fastapi import APIRouter, Request
from fastapi import UploadFile
from controllers.model_controller import ModelController
import io
from PIL import Image
import torch
from torchvision import transforms

ai_router = APIRouter(prefix="/api/v1/ai", tags=["api_v1"])


@ai_router.post("/inference")
def inference(request: Request, image: UploadFile):
    baseline_number = 1
    model_controller = ModelController(
        baseline_number=baseline_number, verbose=True, mode="inference"
    )
    image.file.seek(0)
    contents = image.file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0)
    result = model_controller.inference(tensor)
    return {"baseline_number": baseline_number, "result": result}


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
