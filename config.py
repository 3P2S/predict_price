import os
import sys
import logging

N_CORES = 4
DEBUG_N = int(os.environ.get('DEBUG_N', 10))
# TEST_SIZE is used to simulate larger test size for the second stage
TEST_SIZE = int(os.environ.get('TEST_SIZE', 1))
VALIDATION_SIZE = float(os.environ.get('VALIDATION_SIZE', 0.05))
DUMP_DATASET = int(os.environ.get('DUMP_DATASET', 0))
USE_CACHED_DATASET = int(os.environ.get('USE_CACHED_DATASET', 0))
HANDLE_TEST = int(os.environ.get('HANDLE_TEST', 1))
DEBUG = DEBUG_N > 0
TEST_CHUNK = 350000
MIN_PRICE = 3
MAX_PRICE = 10000
MIN_PRICE_PRED = 3
MAX_PRICE_PRED = 2000
MEAN_LOG_PRICE = 2.9806
UNK = 'unk'

ITEM_DESCRIPTION_MAX_LENGTH = 1050
NAME_MAX_LENGTH = 100
BRAND_NAME_MAX_LENGTH = 100


