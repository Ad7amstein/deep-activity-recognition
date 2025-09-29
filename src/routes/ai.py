import os
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from controllers import ModelController, ProjectController
from routes.schemas import InferenceRequest
from models.enums import ResponseSignalEnum, activity_label2category_dct

ai_router = APIRouter(prefix="/api/v1/ai", tags=["api_v1"])


@ai_router.post("/inference/{project_id}")
def inference(request: Request, project_id: int, inference_request: InferenceRequest):
    app_settings = request.app.state.app_settings
    project_controller = ProjectController()
    model_controller = ModelController(
        baseline_number=inference_request.baseline_number,
        verbose=True,
        mode="inference",
    )
    project_path = project_controller.get_project_path(project_id)
    img_file_path = os.path.join(project_path, inference_request.image_filename)
    if not os.path.exists(img_file_path):
        return JSONResponse(
            content={"signal": ResponseSignalEnum.FILE_ID_ERROR.value},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    img = Image.open(img_file_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(
                (app_settings.B1_FEATURES_SHAPE_0, app_settings.B1_FEATURES_SHAPE_1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0)
    result = model_controller.inference(tensor)
    return {
        "baseline_number": inference_request.baseline_number,
        "image_filename": inference_request.image_filename,
        "category": activity_label2category_dct[result["preds"][0]],
        "model_result": result,
    }


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
