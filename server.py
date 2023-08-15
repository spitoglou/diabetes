"""
╔═╗┌─┐┬─┐┬  ┬┌─┐┬─┐
╚═╗├┤ ├┬┘└┐┌┘├┤ ├┬┘
╚═╝└─┘┴└─ └┘ └─┘┴└─
Author: Stavros Pitoglou
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from db import MyDatabase
from loguru import logger

app = FastAPI()
dbms = MyDatabase("sqlite", dbname="mydb.sqlite")


class Bg_value(BaseModel):
    time: str
    patient: str
    value: float


@app.get("/")
async def root():
    return {"message": "It Works!!!"}


@app.post("/bg/reading")
async def post_reading(value: Bg_value):
    logger.debug(value)
    query = f"""INSERT INTO bg_values(patient_id, timestamp, type, value)
    VALUES ('{value.patient}','{value.time}', 'bg', {value.value})
    """
    dbms.execute_query(query)
    return value


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
