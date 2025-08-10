import os
import pickle
from typing import Dict, List, NamedTuple, TypedDict, Optional
import cv2 as cv
from tqdm import tqdm
from data_processing.box_info import BoxInfo
from utils.config_utils import load_config

CONFIG = load_config()


class TrackingData(NamedTuple):
    player_boxes: Dict[int, List[BoxInfo]]
    frame_boxes: Dict[int, List[BoxInfo]]


class ClipAnnotation(TypedDict):
    category: str
    frame_boxes_dct: Dict[int, List[BoxInfo]]


VolleyballData = Dict[int, Dict[int, ClipAnnotation]]


class AnnotationLoader:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        if self.verbose:
            print("\n[INFO] Initializing Annotation Loader Object...")

    def vis_clip(
        self,
        annot_file: str,
        clip_path: str,
        frame_fmt: str = "jpg",
        verbose: Optional[bool] = None,
    ) -> None:
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[INFO] Visualizing clip {clip_path} with annotations...")
        AnnotationLoader.check_file(annot_file)
        if not isinstance(clip_path, str):
            raise TypeError(f"clip must be a string, got {type(clip_path).__name__}")
        if not os.path.isdir(clip_path):
            raise ValueError(f"{clip_path} does not exist or is not a directory.")

        _, frame_boxes_dict = self.load_tracking_annot(annot_file)

        for frame_id, boxes_info in frame_boxes_dict.items():
            img_path = os.path.join(clip_path, f"{frame_id}.{frame_fmt}")
            img = cv.imread(img_path)
            clip_name = "/".join(clip_path.split("\\")[-2:])

            for box_info in boxes_info:
                cv.rectangle(img, box_info.box[:2], box_info.box[2:], (0, 0, 255), 2)
                cv.putText(
                    img,
                    box_info.category,
                    box_info.box[:2],
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv.imshow(f"Tracking Annotation for clip {clip_name}", img)
            cv.waitKey(200)

        cv.destroyAllWindows()

    def load_tracking_annot(
        self, annot_file: str, verbose: Optional[bool] = None
    ) -> TrackingData:
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[INFO] Loading tracking annotation from file {annot_file}")
        AnnotationLoader.check_file(annot_file)

        with open(annot_file, mode="r", encoding="utf-8") as file:
            lines = file.readlines()
            player_boxes_dct = {player_id: [] for player_id in range(12)}
            frame_boxes_dct = {}

            for i, line in enumerate(
                tqdm(
                    lines,
                    desc="Parsing annotation lines",
                    disable=not verbose,
                    unit="line",
                )
            ):
                box_info = BoxInfo(line)
                if box_info.player_id < 12:
                    player_boxes_dct[box_info.player_id].append(box_info)
                else:
                    print(
                        f"[WARNING] Skipping line {i}. Player ID ({box_info.player_id}) > 11."
                    )

            for boxes_info in tqdm(
                player_boxes_dct.values(),
                desc="Parsing Player Boxes",
                disable=not verbose,
                unit="player",
            ):
                boxes_info = boxes_info[5:-5]
                for box_info in boxes_info:
                    if box_info.frame_id not in frame_boxes_dct:
                        frame_boxes_dct[box_info.frame_id] = []
                    frame_boxes_dct[box_info.frame_id].append(box_info)

            return TrackingData(
                player_boxes=player_boxes_dct, frame_boxes=frame_boxes_dct
            )

    @staticmethod
    def check_file(annot_file: str) -> None:
        if not isinstance(annot_file, str):
            raise TypeError(
                f"annot_file must be a string, got {type(annot_file).__name__}"
            )
        if not os.path.isfile(annot_file):
            raise ValueError(
                f"{annot_file} does not exist or is not a file. It should be a text (.txt) file."
            )

    def load_video_annot(
        self,
        video_annot_file: str,
        tracking_annot_root: str,
        verbose: Optional[bool] = None,
    ) -> dict:
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[INFO] Loading Video Annotations from {video_annot_file}")
        clip_category_dct = {}
        AnnotationLoader.check_file(video_annot_file)

        with open(video_annot_file, mode="r", encoding="utf-8") as file:
            lines = file.readlines()
            for _, line in enumerate(
                tqdm(
                    lines,
                    desc="Parsing annotation lines",
                    disable=not verbose,
                    unit="clip",
                )
            ):
                clip_id, category = line.split()[:2]
                clip_id = int(clip_id.split(".")[0])
                cur_clip_annot_file = os.path.join(
                    tracking_annot_root,
                    os.path.basename(os.path.dirname(video_annot_file)),
                    str(clip_id),
                    f"{clip_id}.txt",
                )
                clip_category_dct[clip_id] = {
                    "category": category,
                    "tracking_annot_dct": self.load_tracking_annot(
                        cur_clip_annot_file, verbose=False
                    ),
                }

        return clip_category_dct

    def load_volleyball_dataset(
        self, videos_root: str, tracking_annot_root: str, verbose: Optional[bool] = None
    ) -> VolleyballData:
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print(f"[INFO] Loading Volleyball Dataset From {videos_root}...")
        video_annot_dct = {}
        for video_dir in tqdm(
            os.listdir(videos_root),
            desc="Processing Videos",
            disable=not verbose,
            unit="video",
        ):
            video_path = os.path.join(videos_root, video_dir)
            if not os.path.isdir(video_path):
                print(
                    f"[WARNING] {video_path} doesn't exist or it not a directory. Skipping..."
                )
                continue
            video_annot_file = os.path.join(video_path, "annotations.txt")
            video_id = int(video_dir)
            video_annot_dct[video_id] = self.load_video_annot(
                video_annot_file, tracking_annot_root, verbose=False
            )

        return video_annot_dct

    def save_pkl_version(
        self, data: Optional[VolleyballData] = None, verbose: Optional[bool] = None
    ) -> None:
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            print("[INFO] Saving Pickle Volleyball Dataset Version...")
        videos_root = os.path.join(
            CONFIG["PATH"]["data_root"], CONFIG["PATH"]["videos"]
        )
        tracking_annot_root = os.path.join(
            CONFIG["PATH"]["data_root"], CONFIG["PATH"]["track_annot"]
        )
        save_path = os.path.join(CONFIG["PATH"]["data_root"], "volleyball_dataset.pkl")

        volleyball_data = (
            data
            if data is not None
            else self.load_volleyball_dataset(videos_root, tracking_annot_root)
        )

        if os.path.exists(save_path):
            print(
                " ".join(
                    [
                        f"[WARNING] The save path '{save_path}' already exists",
                        "and will be overwritten with the new dataset pickle file.",
                    ]
                )
            )

        with open(save_path, mode="wb") as file:
            pickle.dump(volleyball_data, file)

    def load_pkl_version(self, verbose: Optional[bool] = None) -> VolleyballData:
        verbose = self.verbose if verbose is None else verbose
        data_path = os.path.join(CONFIG["PATH"]["data_root"], "volleyball_dataset.pkl")
        if verbose:
            print(f"[INFO] Loading Data from pickle file: {data_path}")

        try:
            with open(data_path, "rb") as file:
                loaded_data = pickle.load(file)
                return loaded_data
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"[ERROR]: The file '{data_path}' was not found."
            ) from exc

    def __repr__(self) -> str:
        return f"{__class__.__name__}(verbose=False)"


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.\n")

    annot_loader = AnnotationLoader(verbose=True)

    test_clip = r"7\38025"
    annot_file = os.path.join(
        CONFIG["PATH"]["data_root"],
        CONFIG["PATH"]["track_annot"],
        test_clip,
        f"{os.path.basename(test_clip)}.txt",
    )
    clip_dir = os.path.join(
        CONFIG["PATH"]["data_root"], CONFIG["PATH"]["videos"], test_clip
    )

    annot_loader.vis_clip(annot_file, clip_dir)


if __name__ == "__main__":
    main()
