import pandas as pd
import numpy as np


def preprocessing(train: pd.DataFrame, test: pd.DataFrame):
    # Drop all rows have price <= 0
    train = train[train.price > 0.0].reset_index(drop=True)

    # Get number of train rows
    nrow_train = train.shape[0]

    # Get
    y_train = np.log1p(train["price"])
    merge = pd.concat([train, test])

    del train
    del test
    # return train, test
