"""
╔═╗┌─┐┬─┐┬  ┬┌─┐┬─┐
╚═╗├┤ ├┬┘└┐┌┘├┤ ├┬┘
╚═╝└─┘┴└─ └┘ └─┘┴└─
Author: Stavros Pitoglou
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings
from src.helpers.misc import debug_print
from src.mongo import MongoDB

DEBUG = settings.DEBUG

app = FastAPI(
    title="Diabetes BGC Prediction API",
    description="Blood Glucose Concentration prediction service using CGM data",
    version="1.0.0",
)


# =============================================================================
# Pydantic Models
# =============================================================================


class FhirCoding(BaseModel):
    """FHIR Coding element."""

    system: str
    code: str


class FhirCategory(BaseModel):
    """FHIR Category element."""

    coding: list[FhirCoding]


class FhirCode(BaseModel):
    """FHIR Code element."""

    text: str


class FhirSubject(BaseModel):
    """FHIR Subject element."""

    identifier: str


class FhirValueQuantity(BaseModel):
    """FHIR ValueQuantity element."""

    value: float
    unit: str = "mg/dL"


class FhirDevice(BaseModel):
    """FHIR Device element."""

    displayName: str
    note: str


class FhirObservation(BaseModel):
    """FHIR Observation resource for blood glucose readings."""

    status: str
    category: list[FhirCategory]
    code: FhirCode
    subject: FhirSubject
    effectiveDateTime: str
    valueQuantity: FhirValueQuantity
    device: FhirDevice


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status: healthy or unhealthy")
    timestamp: str = Field(..., description="ISO format timestamp")
    database: str = Field(..., description="Database connection status")
    version: str = Field(..., description="API version")
    patient_id: str = Field(..., description="Currently configured patient ID")


class ReadingResponse(BaseModel):
    """Response model for successful reading submission."""

    message: str
    record_id: str


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str


# =============================================================================
# Database Connection
# =============================================================================

dbms = MongoDB()


# =============================================================================
# Exception Handler
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)},
    )


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Root endpoint - service availability check."""
    return {"message": "The server is up and running!!!"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring and load balancers.

    Returns:
        HealthResponse with service status, database connectivity, and configuration info.
    """
    db_status = "unknown"
    try:
        dbms.client.admin.command("ping")
        db_status = "connected"
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        timestamp=datetime.utcnow().isoformat() + "Z",
        database=db_status,
        version="1.0.0",
        patient_id=settings.OHIO_ID,
    )


@app.post(
    "/bg/reading",
    response_model=ReadingResponse,
    responses={500: {"model": ErrorResponse}},
)
async def post_reading(json_payload: FhirObservation) -> ReadingResponse:
    """Receive CGM readings in FHIR format and store in database.

    Args:
        json_payload: FHIR Observation object containing blood glucose reading

    Returns:
        ReadingResponse with success message and database record ID

    Raises:
        HTTPException: If database insertion fails
    """
    payload_dict = json_payload.model_dump()

    if DEBUG:
        debug_print("JSON payload", json_payload)
        debug_print("JSON payload converted to python dictionary", payload_dict)

    try:
        db = dbms.client[settings.DATABASE]
        cgm_db = db[f"measurements_{payload_dict['subject']['identifier']}"]
        rec_id = cgm_db.insert_one(payload_dict).inserted_id
        logger.success(f"Inserted record: {rec_id}")

        return ReadingResponse(
            message="Success",
            record_id=str(rec_id),
        )
    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
