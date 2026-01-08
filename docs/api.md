# API Reference

This document describes the REST API endpoints provided by the Diabetes BGC Prediction Server.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement appropriate authentication mechanisms.

## Endpoints

### Root

#### `GET /`

Service availability check.

**Response:**
```json
{
  "message": "The server is up and running!!!"
}
```

**Status Codes:**
- `200 OK`: Service is running

---

### Health Check

#### `GET /health`

Health check endpoint for monitoring and load balancers. Returns service status, database connectivity, and configuration information.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-08T12:00:00.000Z",
  "database": "connected",
  "version": "1.0.0",
  "patient_id": "559"
}
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy` (all systems operational) or `degraded` (database issues) |
| `timestamp` | string | ISO 8601 timestamp |
| `database` | string | `connected`, `disconnected`, or `unknown` |
| `version` | string | API version |
| `patient_id` | string | Currently configured patient ID |

**Status Codes:**
- `200 OK`: Health check completed (may still be degraded)

---

### Blood Glucose Reading

#### `POST /bg/reading`

Receive CGM readings in FHIR format and store in MongoDB.

**Request Body (FHIR Observation):**
```json
{
  "status": "final",
  "category": [
    {
      "coding": [
        {
          "system": "http://terminology.hl7.org/CodeSystem/observation-category",
          "code": "vital-signs"
        }
      ]
    }
  ],
  "code": {
    "text": "Blood Glucose Concentration"
  },
  "subject": {
    "identifier": "559"
  },
  "effectiveDateTime": "2024-01-08T12:00:00Z",
  "valueQuantity": {
    "value": 120.5,
    "unit": "mg/dL"
  },
  "device": {
    "displayName": "CGM Device",
    "note": "Continuous Glucose Monitor"
  }
}
```

**Request Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | string | Yes | Observation status (typically `final`) |
| `category` | array | Yes | FHIR category coding |
| `code` | object | Yes | Observation code with `text` field |
| `subject.identifier` | string | Yes | Patient identifier |
| `effectiveDateTime` | string | Yes | ISO 8601 timestamp of reading |
| `valueQuantity.value` | float | Yes | Blood glucose value |
| `valueQuantity.unit` | string | No | Unit (default: `mg/dL`) |
| `device` | object | Yes | Device information |

**Response (Success):**
```json
{
  "message": "Success",
  "record_id": "507f1f77bcf86cd799439011"
}
```

**Response (Error):**
```json
{
  "error": "Internal Server Error",
  "detail": "Database connection failed"
}
```

**Status Codes:**
- `200 OK`: Reading stored successfully
- `422 Unprocessable Entity`: Invalid FHIR payload
- `500 Internal Server Error`: Database or server error

---

## Data Models

### FhirObservation

The server accepts FHIR R4 Observation resources for blood glucose readings.

```python
class FhirObservation(BaseModel):
    status: str
    category: list[FhirCategory]
    code: FhirCode
    subject: FhirSubject
    effectiveDateTime: str
    valueQuantity: FhirValueQuantity
    device: FhirDevice
```

### HealthResponse

```python
class HealthResponse(BaseModel):
    status: str       # healthy | degraded
    timestamp: str    # ISO 8601
    database: str     # connected | disconnected | unknown
    version: str      # API version
    patient_id: str   # Configured patient ID
```

---

## Example Usage

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Post a blood glucose reading
curl -X POST http://localhost:8000/bg/reading \
  -H "Content-Type: application/json" \
  -d '{
    "status": "final",
    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
    "code": {"text": "Blood Glucose Concentration"},
    "subject": {"identifier": "559"},
    "effectiveDateTime": "2024-01-08T12:00:00Z",
    "valueQuantity": {"value": 120.5, "unit": "mg/dL"},
    "device": {"displayName": "CGM Device", "note": "Test"}
  }'
```

### Python (requests)

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Post reading
reading = {
    "status": "final",
    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
    "code": {"text": "Blood Glucose Concentration"},
    "subject": {"identifier": "559"},
    "effectiveDateTime": "2024-01-08T12:00:00Z",
    "valueQuantity": {"value": 120.5, "unit": "mg/dL"},
    "device": {"displayName": "CGM Device", "note": "Test"}
}
response = requests.post("http://localhost:8000/bg/reading", json=reading)
print(response.json())
```

### Python (httpx async)

```python
import httpx
import asyncio

async def post_reading():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/health")
        return response.json()

result = asyncio.run(post_reading())
print(result)
```

---

## OpenAPI Documentation

When the server is running, interactive API documentation is available:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Error Type",
  "detail": "Detailed error message"
}
```

Common error scenarios:
- **Invalid JSON**: Returns `422` with validation details
- **Database unavailable**: Returns `500` with connection error
- **Missing required fields**: Returns `422` with field validation errors

---

## Rate Limiting

No rate limiting is currently implemented. For production deployments, consider adding rate limiting middleware.

---

## CORS

CORS is not configured by default. For cross-origin requests, configure CORS middleware:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```
