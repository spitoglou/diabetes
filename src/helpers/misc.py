def get_part_of_day(hour):
    return (
        "morning" if 7 <= hour <= 11
        else
        "afternoon" if 12 <= hour <= 16
        else
        "evening" if 16 <= hour <= 20
        else
        "night" if 21 <= hour <= 23
        else
        "late night"
    )
