import itertools

from loguru import logger

from src.helpers.experiment import Experiment

patients = [559, 563, 570, 575, 588, 591]
windows = [6, 12]
horizons = [1, 6, 12]
neptune = False

patients = [563]
windows = [12]
horizons = [12]

if __name__ == "__main__":
    for patient, window, horizon in itertools.product(patients, windows, horizons):
        exp = Experiment(patient, window, horizon, enable_neptune=neptune)
        try:
            exp.run_experiment()
        except Exception as e:
            logger.exception(e)
            if neptune:
                exp.neptune.stop()
