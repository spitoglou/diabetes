"""
╔═╗┌─┐┬─┐┬  ┬┌─┐┬─┐
╚═╗├┤ ├┬┘└┐┌┘├┤ ├┬┘
╚═╝└─┘┴└─ └┘ └─┘┴└─
Author: Stavros Pitoglou
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.mongo import MongoDB
from loguru import logger
import json
import config.server_config as conf

DEBUG = conf.DEBUG

app = FastAPI()
dbms = MongoDB()
dbms.ping()

class Fhir(BaseModel):
    status: str
    category: list
    code: dict
    subject: dict
    effectiveDateTime: str
    valueQuantity: dict
    device: dict
    

@app.get("/")
async def root():
    return {"message": "It Works!!!"}


@app.post("/bg/reading")
async def post_reading(json_payload:Fhir):
    if DEBUG:
        logger.debug(json_payload)
        logger.debug(json_payload.__dict__)
    db = dbms.client[conf.DATABASE]
    cgm_db = db[conf.COLLECTION]
    rec_id = cgm_db.insert_one(json_payload.__dict__).inserted_id
    return {'message': f'Success [Record Id : {rec_id}]'}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
