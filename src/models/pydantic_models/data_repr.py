"""
Data structures for volleyball tracking and clip annotations.
"""

import os
from typing import Dict, List, NamedTuple, TypedDict
from data_processing.box_info import BoxInfo


class TrackingData(NamedTuple):
    """
    TrackingData stores bounding box information for players and frames.

    Attributes:
        player_boxes (Dict[int, List[BoxInfo]]): A mapping from player IDs to
            lists of BoxInfo objects representing bounding boxes associated with each player.
        frame_boxes (Dict[int, List[BoxInfo]]): A mapping from frame indices to
            lists of BoxInfo objects representing bounding boxes detected in each frame.
    """

    player_boxes: Dict[int, List[BoxInfo]]
    frame_boxes: Dict[int, List[BoxInfo]]


class ClipAnnotation(TypedDict):
    """
    A TypedDict representing the annotation for a video clip.

    Attributes:
        category (str): The category label for the clip.
        tracking_annot_dct (TrackingData): The tracking annotation data associated with the clip.
    """

    category: str
    tracking_annot_dct: TrackingData


VideoAnnotation = Dict[int, ClipAnnotation]

VolleyballData = Dict[int, VideoAnnotation]


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
