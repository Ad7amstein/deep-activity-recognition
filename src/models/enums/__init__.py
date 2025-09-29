from models.enums.activity import ActivityEnum
from models.enums.model import ModelResults, ModelMode, ModelBaseline
from models.enums.train import OptimizerEnum, LossFNEnum
from models.enums.response_enums import ResponseSignalEnum

activity_category2label_dct = {
    ActivityEnum.RIGHT_SET_CATEGORY.value: ActivityEnum.RIGHT_SET_LABEL.value,
    ActivityEnum.RIGHT_PASS_CATEGORY.value: ActivityEnum.RIGHT_PASS_LABEL.value,
    ActivityEnum.RIGHT_WINPOINT_CATEGORY.value: ActivityEnum.RIGHT_WINPOINT_LABEL.value,
    ActivityEnum.RIGHT_SPIKE_CATEGORY.value: ActivityEnum.RIGHT_SPIKE_LABEL.value,
    ActivityEnum.LEFT_SET_CATEGORY.value: ActivityEnum.LEFT_SET_LABEL.value,
    ActivityEnum.LEFT_PASS_CATEGORY.value: ActivityEnum.LEFT_PASS_LABEL.value,
    ActivityEnum.LEFT_WINPOINT_CATEGORY.value: ActivityEnum.LEFT_WINPOINT_LABEL.value,
    ActivityEnum.LEFT_SPIKE_CATEGORY.value: ActivityEnum.LEFT_SPIKE_LABEL.value,
}

activity_label2category_dct = {
    ActivityEnum.RIGHT_SET_LABEL.value: ActivityEnum.RIGHT_SET_CATEGORY.value,
    ActivityEnum.RIGHT_PASS_LABEL.value: ActivityEnum.RIGHT_PASS_CATEGORY.value,
    ActivityEnum.RIGHT_WINPOINT_LABEL.value: ActivityEnum.RIGHT_WINPOINT_CATEGORY.value,
    ActivityEnum.RIGHT_SPIKE_LABEL.value: ActivityEnum.RIGHT_SPIKE_CATEGORY.value,
    ActivityEnum.LEFT_SET_LABEL.value: ActivityEnum.LEFT_SET_CATEGORY.value,
    ActivityEnum.LEFT_PASS_LABEL.value: ActivityEnum.LEFT_PASS_CATEGORY.value,
    ActivityEnum.LEFT_WINPOINT_LABEL.value: ActivityEnum.LEFT_WINPOINT_CATEGORY.value,
    ActivityEnum.LEFT_SPIKE_LABEL.value: ActivityEnum.LEFT_SPIKE_CATEGORY.value,
}
