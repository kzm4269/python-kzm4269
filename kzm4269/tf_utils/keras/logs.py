import re

import pandas as pd


def loss_names(columns, ignore_total_loss=True):
    columns = [
        re.sub(r'^val(_|$)', '', re.sub(r'_loss$', '', col))
        for col in columns if
        re.search(r'_loss$', col) or not ignore_total_loss and col == 'loss']
    return list(dict.fromkeys(col for col in columns if col).keys())


def load_logs(filename, index='epoch', include_batch_logs=False):
    df = pd.read_csv(filename)

    if include_batch_logs:
        df = df[~np.isnan(df['size'])]
        index = 'nsamples'
    else:
        df = df[np.isnan(df['size'])]

    # for compatibility
    if index == 'nsamples' and 'total_sample' in df:
        index = 'total_sample'
    if index == 'nbatches' and 'total_batch' in df:
        index = 'total_batches'

    return df.set_index(index)
