from __future__ import annotations

from typing import Any


def get_part_of_day(hour: int) -> str:
    """
    Get the part of day based on hour.

    Args:
        hour: Hour of day (0-23).

    Returns:
        Part of day string: morning, afternoon, evening, night, or late_night.
    """
    return (
        "morning"
        if 7 <= hour <= 11
        else "afternoon"
        if 12 <= hour <= 16
        else "evening"
        if 17 <= hour <= 20
        else "night"
        if 21 <= hour <= 23
        else "late_night"
    )


def debug_print(title: str, message: Any) -> None:
    """
    Print a debug message with formatted header.

    Args:
        title: Title for the debug section.
        message: Message content to print.
    """
    print("----------------------------------------------------------------")
    print(f"----------------------{title}--------------------------")
    print(message)
    print("----------------------------------------------------------------")
