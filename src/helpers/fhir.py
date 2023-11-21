from pprint import pprint
import json


def create_fhir_json_from_reading(
    reading: dict,
    value_unit: str = "mg/dL",
    status: str = "final",
    coding_system: str = "http://terminology.hl7.org/CodeSystem/observation-category",
    codind_code: str = "vital-signs",
    code_text: str = "Blood Glucose Concentration",
    device_display_name: str = "Software Simulator",
    device_note_text: str = "for Stavros Pitoglou PhD thesis",
    verbose: bool = False,
):
    fhir_dict = {
        "status": status,
        "category": [{"coding": [{"system": coding_system, "code": codind_code}]}],
        "code": {"text": code_text},
        "subject": {"identifier": reading["patient"]},
        "effectiveDateTime": reading["time"],
        "valueQuantity": {"value": reading["value"], "unit": value_unit},
        "device": {"displayName": device_display_name, "note": device_note_text},
    }
    # pprint(fhir_dict)
    fhir_json = json.dumps(fhir_dict)
    if verbose:
        pprint(fhir_json)
    return fhir_json


if __name__ == "__main__":
    create_fhir_json_from_reading(
        {
            "timestamp": 1638840420.0,
            "time": "2021-12-07T01:27:00+00:00",
            "value": 104.0,
            "patient": "559",
        },
        verbose=True
    )
