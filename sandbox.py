import pandas as pd
# from tqdm import trange
# from tsfresh import extract_features
# from tsfresh.feature_extraction import (ComprehensiveFCParameters,
# MinimalFCParameters)
# import matplotlib.pyplot as plt
# from lxml import objectify
# from datetime import datetime
# from src.bgc_providers.ohio_bgc_provider import OhioBgcProvider


def slice_df(dataframe, index, number):
    return dataframe[index:index + number]


def aida_dataframe(aida_no=6359):
    source_file = 'data/aida/glucose' + str(aida_no) + '.dat'
    df = pd.read_table(source_file, header=None, names=['time', 'bg_value'])
    df['id'] = 'a'
    # df.plot('time', 'bg_value')
    # plt.show()
    return df


if __name__ == '__main__':
    print('test')
    # df = ohio_dataframe()
    # print(df)
    # provider = OhioBgcProvider()
    # df2 = provider.tsfresh_dataframe(trunc=12)
    # print(df2)

    # master = create_feature_dataframe(df2, 10)
    # print(master)

    # print(create_target_array(df2, 10))
