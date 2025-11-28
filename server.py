"""
╔═╗┌─┐┬─┐┬  ┬┌─┐┬─┐
╚═╗├┤ ├┬┘└┐┌┘├┤ ├┬┘
╚═╝└─┘┴└─ └┘ └─┘┴└─
Author: Stavros Pitoglou

FastAPI server for receiving CGM glucose measurements.
"""

from typing import Any, Dict, List

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.config import Config, get_config
from src.logging_config import setup_logging
from src.mongo import MongoDB
from src.repositories.measurement_repository import MeasurementRepository

config = get_config()
setup_logging(level=config.log_level)

logger.info(f"Logging initialized with level={config.log_level}, debug={config.debug}")


# Pydantic models for request validation
class ValueQuantity(BaseModel):
    """Blood glucose value with unit."""

    value: float = Field(..., gt=0, description="Blood glucose value in mg/dL")
    unit: str = Field(default="mg/dL")


class SubjectIdentifier(BaseModel):
    """Patient/subject identifier."""

    identifier: str = Field(..., min_length=1)


class FhirObservation(BaseModel):
    """FHIR Observation resource for glucose measurement."""

    status: str
    category: List[Dict[str, Any]]
    code: Dict[str, Any]
    subject: SubjectIdentifier
    effectiveDateTime: str
    valueQuantity: ValueQuantity
    device: Dict[str, Any]


class InsertResponse(BaseModel):
    """Response model for successful insert."""

    message: str
    record_id: str


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str


# Dependency injection
def get_config_dep() -> Config:
    """Dependency for configuration."""
    return get_config()


def get_mongo(config: Config = Depends(get_config_dep)) -> MongoDB:
    """Dependency for MongoDB connection."""
    return MongoDB(config)


def get_measurement_repo(
    mongo: MongoDB = Depends(get_mongo),
    config: Config = Depends(get_config_dep),
) -> MeasurementRepository:
    """Dependency for measurement repository."""
    db = mongo.get_database()
    return MeasurementRepository(db, config)


# FastAPI app
app = FastAPI(
    title="Diabetes CGM Server",
    description="API for receiving CGM glucose measurements in FHIR format",
    version="2.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Verify database connection on startup."""
    try:
        config = get_config()
        mongo = MongoDB(config)
        if mongo.ping():
            logger.info("Server started, MongoDB connection verified")
        else:
            logger.warning("Server started but MongoDB ping failed")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB on startup: {e}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "The server is up and running!", "status": "healthy"}


@app.get("/health")
async def health_check(config: Config = Depends(get_config_dep)):
    """Detailed health check endpoint."""
    try:
        mongo = MongoDB(config)
        db_healthy = mongo.ping()
    except Exception:
        db_healthy = False

    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected",
    }


@app.post(
    "/bg/reading",
    response_model=InsertResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid measurement data"},
        500: {"model": ErrorResponse, "description": "Database error"},
    },
)
async def post_reading(
    json_payload: FhirObservation,
    repo: MeasurementRepository = Depends(get_measurement_repo),
    config: Config = Depends(get_config_dep),
):
    """
    Receive a CGM glucose measurement in FHIR format.

    Args:
        json_payload: FHIR Observation object with glucose measurement.

    Returns:
        Success message with inserted record ID.
    """
    payload_dict = json_payload.model_dump()
    patient_id = payload_dict["subject"]["identifier"]

    if config.debug:
        logger.debug(f"Received measurement for patient {patient_id}")
        logger.debug(f"Payload: {payload_dict}")

    try:
        record_id = repo.save(patient_id, payload_dict)
        logger.success(f"Saved measurement {record_id} for patient {patient_id}")
        return InsertResponse(
            message="Success",
            record_id=record_id,
        )
    except ValueError as e:
        logger.warning(f"Invalid patient ID: {patient_id}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to save measurement: {e}")
        raise HTTPException(status_code=500, detail="Database error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
