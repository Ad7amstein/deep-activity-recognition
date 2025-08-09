import os
from data_processing.box_info import BoxInfo
from utils.config_utils import load_config

CONFIG = load_config()


class AnnotationLoader:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        if self.verbose:
            print("\n[INFO] Initializing Annotation Loader Object...")

    def load_tracking_annot(self, annot_file: str):
        if self.verbose:
            print(f"[INFO] Loading tracking annotation from file {annot_file}")
        if not isinstance(annot_file, str):
            raise TypeError(
                f"annot_file must be a string, got {type(annot_file).__name__}"
            )
        if not os.path.isfile(annot_file):
            raise ValueError(
                f"{annot_file} does not exist or is not a file. It should be a text (.txt) file."
            )

        with open(annot_file, mode="r", encoding="utf-8") as file:
            player_boxes_dct = {player_id: [] for player_id in range(12)}
            frame_boxes_dct = {}

            for i, line in enumerate(file.readlines()):
                box_info = BoxInfo(line)
                if box_info.player_id < 12:
                    player_boxes_dct[box_info.player_id].append(box_info)
                else:
                    print(
                        f"[WARNING] Skipping line {i}. Player ID ({box_info.player_id}) > 11."
                    )

            for boxes_info in player_boxes_dct.values():
                boxes_info = boxes_info[5:-5]
                for box_info in boxes_info:
                    if box_info.frame_id not in frame_boxes_dct:
                        frame_boxes_dct[box_info.frame_id] = []
                    frame_boxes_dct[box_info.frame_id].append(box_info)

            return player_boxes_dct, frame_boxes_dct

    def __repr__(self) -> str:
        return f"{__class__.__name__}()"


def main():
    """Entry Point for the Program."""
    print(f"Welcome from `{os.path.basename(__file__).split(".")[0]}` Module.\n")

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

    player_boxes_dct, frame_boxes_dct = annot_loader.load_tracking_annot(annot_file)

    print(annot_loader)

    print(f"\nFirst 5 Players:\n{list(player_boxes_dct.items())[:5]}")
    print(f"\nFirst 5 Frames:\n{list(frame_boxes_dct.items())[:5]}")

    print(f"Number of Players: {len(player_boxes_dct)}")
    print(f"Number of Original Frames: {len(player_boxes_dct[0])}")
    print(f"Number of Selected Frames: {len(frame_boxes_dct)}")


if __name__ == "__main__":
    main()
