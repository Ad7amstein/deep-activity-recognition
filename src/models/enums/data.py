"""Enumeration of activity categories, actions and labels for classification."""

import os
from enum import Enum


class ActivityEnum(Enum):
    """Enumeration of activity categories and their corresponding numeric labels.

    Attributes:
        RIGHT_SET (tuple): Represents a right-side set activity with label 0.
        RIGHT_SPIKE (tuple): Represents a right-side spike activity with label 1.
        RIGHT_PASS (tuple): Represents a right-side pass activity with label 2.
        RIGHT_WINPOINT (tuple): Represents a right-side winning point activity with label 3.

        LEFT_SET (tuple): Represents a left-side set activity with label 4.
        LEFT_SPIKE (tuple): Represents a left-side spike activity with label 5.
        LEFT_PASS (tuple): Represents a left-side pass activity with label 6.
        LEFT_WINPOINT (tuple): Represents a left-side winning point activity with label 7.
    """

    RIGHT_SET = ("r_set", 0)
    RIGHT_SPIKE = ("r_spike", 1)
    RIGHT_PASS = ("r_pass", 2)
    RIGHT_WINPOINT = ("r_winpoint", 3)

    LEFT_SET = ("l_set", 4)
    LEFT_SPIKE = ("l_spike", 5)
    LEFT_PASS = ("l_pass", 6)
    LEFT_WINPOINT = ("l_winpoint", 7)

    def __init__(self, category: str, label: int) -> None:
        """Initialize an activity enumeration instance.

        Args:
            category (str): The string identifier for the activity (e.g., "r_set", "l_spike").
            label (int): The numeric label assigned to the activity.
        """

        self.category = category
        self.label = label

    def __str__(self) -> str:
        """Return a human-readable string representation of the activity.

        Returns:
            str: A formatted string showing the enum member name, category, and label.
        """
        return f"{self.name} (category='{self.category}', label={self.label})"


class ActionEnum(Enum):
    """
    Enumeration of volleyball player actions with both category (string) and label (int).

    Attributes:
        STANDING ("standing", 0): Player is standing still.
        BLOCKING ("blocking", 1): Player is attempting to block an opponent's shot.
        DIGGING ("digging", 2): Player performs a defensive dig.
        WAITING ("waiting", 3): Player is idle or waiting for the play.
        FALLING ("falling", 4): Player is in the process of falling.
        MOVING ("moving", 5): Player is moving across the court.
        SPIKING ("spiking", 6): Player is attempting an attacking spike.
        SETTING ("setting", 7): Player is setting the ball for a teammate.
        JUMPING ("jumping", 8): Player is jumping without a specific action.
    """

    STANDING = ("standing", 0)
    BLOCKING = ("blocking", 1)
    DIGGING = ("digging", 2)
    WAITING = ("waiting", 3)
    FALLING = ("falling", 4)
    MOVING = ("moving", 5)
    SPIKING = ("spiking", 6)
    SETTING = ("setting", 7)
    JUMPING = ("jumping", 8)

    def __init__(self, category: str, label: int):
        """
        Initialize an ActionEnum member with a category string and numeric label.

        Args:
            category (str): The human-readable name of the action
                (e.g., "standing", "spiking").
            label (int): The unique integer identifier assigned to the action.
        """

        self.category = category
        self.label = label

    def __str__(self) -> str:
        """Return a human-readable string representation of the action.

        Returns:
            str: A formatted string showing the enum member name, category, and label.
        """
        return f"{self.name} (category='{self.category}', label={self.label})"


def main():
    """Entry Point for the Program."""
    print(
        f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module. Nothing to do ^_____^!"
    )


if __name__ == "__main__":
    main()
