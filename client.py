"""
┌─┐┬  ┬┌─┐┌┐┌┌┬┐
│  │  │├┤ │││ │
└─┘┴─┘┴└─┘┘└┘ ┴
Author: Stavros Pitoglou
"""
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider
from src.helpers.fhir import create_fhir_json_from_reading
from time import sleep
import requests
from loguru import logger
import json
from datetime import datetime
from config.simulation_config import OHIO_ID, INTERVAL


def stream_data(send_to_service: bool = True, verbose: bool = False):
    """Συνάρτηση ανάκτησης, προετοιμασίας και αποστολής σειράς μετρήσεων
        για το simulation και τη δοκιμή γεννήτριας μετρήσεων CGM

    Args:
        send_to_service (bool, optional): Διακόπτης τελικής αποστολής στο service. Defaults to True.
        verbose (bool, optional): Διακόπτης εκτεταμένων μηνυμάτων εκτέλεσης. Defaults to False.
    """    
    
    # Ορισμός της μεθόδου streaming από τον αντίστοιχο provider
    provider = OhioBgcProvider(ohio_no=OHIO_ID)
    stream = provider.simulate_glucose_stream()
    try:
        # Εκτέλεση ατέρμονου βρόχου έως την ακύρωση από το χρήστη
        while True:  
            # ανάκτηση επόμενης μέτρησης
            values = next(stream)  
            logger.info(values) if verbose else ...
            # κλήση μεθόδου μετατροπής της μέτρησης σε αντικέιμενο FHIR
            payload = create_fhir_json_from_reading(values)  
            logger.info(payload) if verbose else ...
            # αποστολή στο RESTful endpoint του service (εφόσον είναι ενεργοποιημένη)
            if send_to_service:
                r = requests.post("http://localhost:8000/bg/reading", data=payload)
                logger.info(r.status_code) if verbose else ...
                logger.info(r.text) if verbose else ...
                if r.status_code != 200:
                    logger.warning(r.text)
                logger.success(values)
            sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Interrupted by the user")


if __name__ == "__main__":
    stream_data(send_to_service=True, verbose=False)
