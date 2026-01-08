from __future__ import annotations


def get_part_of_day(hour: int) -> str:
    """Get the part of day based on the hour.

    Args:
        hour: Hour of the day (0-23)

    Returns:
        String representing the part of day: morning, afternoon, evening, night, or late_night
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


def debug_print(title: str, message: object) -> None:
    """Print a debug message with a title header.

    Args:
        title: Title for the debug message
        message: Message content to print
    """
    print("----------------------------------------------------------------")
    print(f"----------------------{title}--------------------------")
    print(message)
    print("----------------------------------------------------------------")
