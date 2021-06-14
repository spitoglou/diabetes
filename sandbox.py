import pandas as pd
from tqdm import trange
from tsfresh import extract_features
from tsfresh.feature_extraction import (ComprehensiveFCParameters,
                                        MinimalFCParameters)
import matplotlib.pyplot as plt
from lxml import objectify
from datetime import datetime
from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider


def slice_df(dataframe, index, number):
    return dataframe[index:index + number]


def ohio_dataframe(scope='train', ohio_no='559', trunc=100):
    source_file = 'data/ohio/{0}/{1}-ws-{0}ing.xml'.format(scope, ohio_no)
    # path = 'data/ohio/train/559-ws-training.xml'
    xml = objectify.parse(open(source_file))
    root = xml.getroot()
    glucose_levels_xml = root.getchildren()[0]
    base_time_string = glucose_levels_xml.getchildren()[0].attrib['ts']
    base_time = datetime.strptime(base_time_string, '%d-%m-%Y %H:%M:%S')
    # print(base_time)
    data_array = []
    for glucose_event in glucose_levels_xml.getchildren():
        # print(glucose_event.attrib)
        dtime = datetime.strptime(
            glucose_event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        delta = dtime - base_time
        array_time = (abs(delta.days) * 24 +
                      round(((dtime - base_time).seconds) / 3600, 2))
        array_value = int(glucose_event.attrib['value'])
        data_array.append([array_time, array_value])
    df = pd.DataFrame(data=data_array, columns=['time', 'bg_value'])
    df = df[:trunc]
    df['id'] = 'a'
    # df.plot('time', 'bg_value')
    # plt.show()
    return df


def aida_dataframe(aida_no=6359):
    source_file = 'data/aida/glucose' + str(aida_no) + '.dat'
    df = pd.read_table(source_file, header=None, names=['time', 'bg_value'])
    df['id'] = 'a'
    # df.plot('time', 'bg_value')
    # plt.show()
    return df


def create_feature_dataframe(initial_df, size,
                             window=1,
                             chunks='auto',
                             hide_progressbars=True,
                             minimal_features=True,
                             plot_chunks=False):
    """Creates Feaure Dataframe using tsfresh

    Parameters:
        initial_df (pandas dataframe): Initial Dataframe containing the time-series (pandas)
        size (int): the size of the time "chunks" that will be used for feature extraction
        chunks (int or 'auto)': # of chunks ('auto' for automatic calculation based on the dataframe row count)
        window (int): the prediction window
        hide_progressbars (bool): if tsfresh progress bars are to be hidden
        minimal_features (bool): if True tsfresh MinimalFCParameters() will be used
        plot_chunks (bool): if plots for every time chunk are to be plotted

    Returns:
        Feature dataframe (pandas)
    """

    if chunks == 'auto':
        chunks = initial_df.shape[0] - size + 1 - window

    default_fc_parameters = ComprehensiveFCParameters()
    if minimal_features is True:
        default_fc_parameters = MinimalFCParameters()

    master = extract_features(initial_df, column_id="id", column_sort="time",
                              default_fc_parameters=default_fc_parameters,
                              disable_progressbar=hide_progressbars)
    master['start'] = 'all'
    master['end'] = 'all'
    master['start_time'] = ' '
    master['end_time'] = ' '
    for i in trange(chunks):
        chunk = slice_df(initial_df, i, size)
        if plot_chunks:
            chunk.plot('time', 'bg_value')
            plt.show()
        # display(chunk)
        chunk_features = extract_features(chunk,
                                          column_id="id",
                                          column_sort="time",
                                          default_fc_parameters=default_fc_parameters,
                                          disable_progressbar=hide_progressbars)
        chunk_features['start'] = i
        chunk_features['end'] = i + size - 1
        chunk_features['start_time'] = initial_df.loc[i].time
        chunk_features['end_time'] = initial_df.loc[i + size - 1].time

        # display(chunk_features)
        master = pd.concat([master, chunk_features])

    # print(master)
    master.reset_index(inplace=True)
    # master.drop(['id'], axis=1, inplace=True)
    master.drop([0], inplace=True)
    master.reset_index(inplace=True)
    return master


def create_target_array(initial_df, size, window=1, chunks='auto'):
    """Creates target vector

    Parameters:
        initial_df (pandas dataframe): Initial Dataframe containing the time-series (pandas)
        size (int): the size of the time "chunks" that will be used for feature extraction
        chunks (int or 'auto)': # of chunks ('auto' for automatic calculation based on the dataframe row count)
        window (int): the prediction window

    Returns:
        Target Values (pandas series)
    """

    if chunks == 'auto':
        chunks = initial_df.shape[0] - size + 1 - window
    array = [initial_df.loc[i + size - 1 +
                            window].bg_value for i in range(chunks)]
    return pd.Series(array)


if __name__ == '__main__':
    # print('test')
    # df = ohio_dataframe()
    # print(df)
    provider = OhioBgcProvider()
    df2 = provider.tsfresh_dataframe(trunc=12)
    print(df2)

    master = create_feature_dataframe(df2, 10)
    print(master)

    print(create_target_array(df2, 10))
