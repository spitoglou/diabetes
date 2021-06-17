from src.helpers.experiment import create_tsfresh_dataframe
from src.helpers.diabetes.cega import clarke_error_grid
from pycaret.regression import setup, create_model, compare_models, predict_model
from loguru import logger
import warnings
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parameters = {
        'ohio_no': 559,
        'scope': 'train',
        'train_ds_size': 3000,
        'window_size': 6,
        'prediction_horizon': 1,
        'minimal_features': False,
    }

    df2 = create_tsfresh_dataframe(parameters)
    df3 = df2.drop(columns=['start', 'end', 'start_time', 'end_time'])
    df3

    exp_reg = setup(df3,
                    target='label',
                    feature_selection=True,
                    html=False,
                    silent=True
                    )

    best3 = compare_models(
        exclude=['catboost', 'xgboost'],
        sort='RMSE',
        n_select=3,
        # verbose=False
    )
    logger.info(best3)

    model = create_model('et', verbose=False)
    print(model)
    pd = predict_model(model)

    (plot, res) = clarke_error_grid(pd['label'], pd['Label'], 'Test')
    logger.info(res)
    plot.show()
    # test_parameters = {
    #     'ohio_no': 559,
    #     'scope': 'test',
    #     'train_ds_size': 100000,
    #     'window_size': 6,
    #     'prediction_horizon': 1,
    #     'minimal_features': False,
    # }
    # df4 = create_tsfresh_dataframe(test_parameters)
    # df6 = df4.drop(columns=['start', 'end', 'start_time', 'end_time'])
    # pd2 = predict_model(model, data=df6)
    # clarke_error_grid(pd2['label'], pd2['Label'], 'Test')
