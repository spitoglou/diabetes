"""
┌─┐┬  ┬┌─┐┌┐┌┌┬┐
│  │  │├┤ │││ │
└─┘┴─┘┴└─┘┘└┘ ┴
Author: Stavros Pitoglou
"""
from ohio_data import Ohio_Dataset_xml
from time import sleep
import requests
from loguru import logger
import json

# logger.disable('')

if __name__ == "__main__":
    ds = Ohio_Dataset_xml()
    stream = ds.simulate_glucose_stream()
    try:
        while True:
            values = next(stream)
            # logger.info(values)
            r = requests.post(
                "http://localhost:8000/bg/reading", data=json.dumps(values))
            logger.info(r.status_code)
            if r.status_code != 200:
                logger.warning(r.text)
            sleep(1)
    except KeyboardInterrupt:
        print('Interrupted by the user')