from src.helpers.experiment import create_tsfresh_dataframe
import itertools
patients = [559, 563, 570, 575, 588, 591]
windows = [6, 12]
horizons = [1, 6, 12]

for patient, window, horizon in itertools.product(patients, windows, horizons):
    parameters = {
        'ohio_no': patient,
        'scope': 'train',
        'train_ds_size': 0,
        'window_size': window,
        'prediction_horizon': horizon,
        'minimal_features': False,
    }
    create_tsfresh_dataframe(parameters)

    parameters = {
        'ohio_no': patient,
        'scope': 'test',
        'train_ds_size': 0,
        'window_size': window,
        'prediction_horizon': horizon,
        'minimal_features': False,
    }
    create_tsfresh_dataframe(parameters)