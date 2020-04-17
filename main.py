import os
import multiprocessing as mp
import pandas as pd

import preprocessing as pre

input_dir = './input'
if __name__ == '__main__':
    mp.set_start_method('forkserver', True)

    train = pd.read_table(os.path.join(input_dir, 'train.tsv'),
                          engine='c',
                          dtype={'item_condition_id': 'category',
                                 'shipping': 'category'})

    test = pd.read_table(os.path.join(input_dir, 'test.tsv'),

                         engine='c',
                         dtype={'item_condition_id': 'category',
                                'shipping': 'category'})

    print('Train shape: ', train.shape)
    print('Test shape: ', test.shape)
    pre.preprocessing(train, test)
    print(train.shape)