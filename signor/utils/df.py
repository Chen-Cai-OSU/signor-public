import pandas as pd
import sys
from sklearn import preprocessing
from signor.monitor.probe import summary
from signor.monitor.time import timefunc
import matplotlib.pyplot as plt
from signor.utils.nan import nan1

def drop_all_zero_cols(df):
    """ http://bit.ly/3cAU1MP """
    assert isinstance(df, pd.DataFrame)
    return df.loc[:, (df != 0).any(axis=0)]

@timefunc
def norm_df(df, cols = ['score'], saveto=None):
    """ http://bit.ly/2xb9v9Y
        Normalize each columns of df
    :param saveto: the directory to which save the norm
    """
    assert isinstance(df, pd.DataFrame)
    n_col = len(df.columns)
    # Create x, where x the 'scores' column's values as floats
    obj_cols = [col for col, type in df.dtypes.items() if str(type)=='object'] # filter out object
    filter_cols = [col for col in cols if col not in obj_cols]
    x = df[filter_cols].values.astype(float)


    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df[filter_cols] = pd.DataFrame(x_scaled)
    assert len(df.columns) == n_col

    # save the magnitute of each feature
    if saveto is not None:
        nan_num = np.sum(x == nan1, axis=0).tolist()
        min_ = min_max_scaler.data_min_
        max_ = min_max_scaler.data_max_
        mag_dict = {}
        assert len(max_) == len(min_) == len(filter_cols) == len(nan_num)
        for i, name in enumerate(filter_cols):
            mag_dict[name] = [min_[i], max_[i], max_[i] - min_[i], nan_num[i]]
        mag_df = pd.DataFrame(mag_dict, index=['min', 'max', 'magnitute', 'nan_num'])
        print(mag_df)
        mag_df.to_csv(saveto)

    return df

def load_norm():
    """ load norm for df feature.
        Use it for calculating MAE.
    """
    idx2name = {0:'band_gap', 1:'elasticity.G_VRH', 2:'elasticity.K_VRH', 3:'final_energy_per_atom', 4:'elasticity.poisson_ratio',
    5: 'formation_energy_per_atom', 6: 'efermi'}

    dir = '/home/cai.507/Documents/DeepLearning/Signor/signor/graph/cgcnn/data_/material-data/norm_bucket_False_pretrain_False.csv'
    df = pd.read_csv(dir)
    print(df)
    df = df.loc[2, ]
    # print(dict(df)) # {'Unnamed: 0': 'magnitute', 'band_gap': 17.9023, 'elasticity.G_VRH': 13783.0, 'elasticity.K_VRH': 591.0, 'final_energy_per_atom': 14.315670673125, 'elasticity.poisson_ratio': 221.76, 'formation_energy_per_atom': 9.973883708340516, 'efermi': 33.47712653}
    # df = df.loc[:, ~df.columns.str.match('Unnamed')] # https://bit.ly/3bhCwj2
    # print(df)
    idx2mag = dict()
    for k, v in idx2name.items():
        if '_VRH' in v:
            idx2mag[k] = np.log10(dict(df)[v])
        else:
            idx2mag[k] = dict(df)[v]
    # print(idx2mag)
    return idx2mag

def renorm(mae):
    assert isinstance(mae, list)
    assert len(mae) == 7
    idx2mag = load_norm()

    mae_renorm = []
    for i in range(7):
        tmp = mae[i] * idx2mag[i]
        from signor.format.format import pf
        tmp = pf(tmp, precision=3)
        mae_renorm.append(tmp)
    return mae_renorm

def test(df):
    assert 'task_id' in df.columns
    assert 'task_ids' in df.columns

    for idx, row in df.iterrows():
        assert row['task_id'] in row['task_ids'], f'row {idx} is unexpected.'

    exit()

import numpy as np
def example_df():
    # d = {'task_id': [1, 2], 'task_ids': [[1, 2], [2, 3]]}
    data = {'score2': [234, 24, 14, 27, -74, 46, 73, -18, 59, 160], 'a': list(range(10)), 'c': ['a'] * 10, 'score1': np.random.random(10).tolist()}
    df = pd.DataFrame(data)
    return df

def range_lookup(df, subplots=False, **kwargs):
    cols = df.columns
    df[cols].plot(kind="density", subplots=subplots, sharex=False, **kwargs)
    plt.show()

from colorama import Fore, Back, Style

def color_red_green(val):
    if val < 0:
        color = Fore.GREEN
    else:
        color = Fore.RED
    return color + str('{0:.2%}'.format(val)) + Style.RESET_ALL


if __name__ == '__main__':
    load_norm()

    exit()
    df = example_df()
    df = norm_df(df, cols=df.columns)
    summary(df)
    exit()

    df = example_df()
    range_lookup(df)
    sys.exit()

    dfs = example_df()
    dfs["score"] = dfs["score"].apply(color_red_green)
    print(dfs)
    exit()

