"""FHIR data format utilities.

This module provides functions for creating FHIR-compliant JSON representations
of glucose readings for interoperability with healthcare systems.
"""

from __future__ import annotations

import json
from pprint import pprint
from typing import Any


def create_fhir_json_from_reading(
    reading: dict[str, Any],
    value_unit: str = "mg/dL",
    status: str = "final",
    coding_system: str = "http://terminology.hl7.org/CodeSystem/observation-category",
    codind_code: str = "vital-signs",
    code_text: str = "Blood Glucose Concentration",
    device_display_name: str = "Software Simulator",
    device_note_text: str = "for Stavros Pitoglou PhD thesis",
    verbose: bool = False,
) -> str:
    """
    Create a FHIR JSON representation of a glucose reading.

    Args:
        reading: Dictionary with patient, time, and value keys.
        value_unit: Unit of measurement (default mg/dL).
        status: FHIR observation status.
        coding_system: FHIR coding system URL.
        codind_code: FHIR coding code.
        code_text: Human-readable code text.
        device_display_name: Device display name.
        device_note_text: Device note.
        verbose: If True, print the resulting JSON.

    Returns:
        FHIR-formatted JSON string.
    """
    fhir_dict: dict[str, Any] = {
        "status": status,
        "category": [{"coding": [{"system": coding_system, "code": codind_code}]}],
        "code": {"text": code_text},
        "subject": {"identifier": reading["patient"]},
        "effectiveDateTime": reading["time"],
        "valueQuantity": {"value": reading["value"], "unit": value_unit},
        "device": {"displayName": device_display_name, "note": device_note_text},
    }
    # pprint(fhir_dict)
    fhir_json: str = json.dumps(fhir_dict)
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
        verbose=True,
    )
