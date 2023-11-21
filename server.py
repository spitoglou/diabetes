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
from src.helpers.misc import debug_print

DEBUG = conf.DEBUG

app = FastAPI()

# Σύνδεση με τη ΒΔ και δοκιμή
dbms = MongoDB()
dbms.ping()

# Ορισμός μοντέλου δεδομένων του αντικειμένου FHIR
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
    """Endpoint Δοκιμής διαθεσιμότητας της υπηρεσίας
    """    
    return {"message": "The server is up and running!!!"}


@app.post("/bg/reading")
async def post_reading(json_payload:Fhir):
    """Endpoint υποδοχής μετρήσεων CGM σε μορφή αντικειμένων FHIR και
        ενημέρωση της Βάσης Δεδομένων 

    Args:
        json_payload (Fhir): Αντικέιμενο FHIR

    Returns:
        JSON : Σε περίπτωση επιτυχίας -> Μήνυμα επιτυχούς εκτέλεσης με το id εισαγωγής στη ΒΔ
    """    
    payload_dict = json_payload.__dict__
    if DEBUG:
        debug_print('JSON payload', json_payload)
        debug_print('JSON payload converted to python dictionary', payload_dict)
    
    # Επιλογή ΒΔ βάσει παραμετροποίησης, 
    # ορισμός συλλογής εγγράφων ως "measurements_[αναγνωριστικό υποκειμένου]" και
    # εισαγωγή εγγραφής στη ΒΔ
    db = dbms.client[conf.DATABASE]
    cgm_db = db[f'measurements_{payload_dict["subject"]["identifier"]}']
    rec_id = cgm_db.insert_one(payload_dict).inserted_id
    logger.success(rec_id)
    
    return {'message': f'Success [Record Id : {rec_id}]'}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
