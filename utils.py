import psutil
import pandas as pd
import numpy as np
import time
from functools import wraps
from config import logger
import re
from functools import lru_cache
from string import digits
import config
from sklearn.linear_model import Lasso
from functools import partial
from multiprocessing.pool import ThreadPool
from config import *
import unidecode
import pickle
import multiprocessing as mp
from load_data import *
from nltk import PorterStemmer

stemmer = PorterStemmer()


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        logger.info('Starting {}'.format(self.message))
        self.start_clock = time.clock()
        self.start_time = time.time()

    def __exit__(self, *args):
        self.end_clock = time.clock()
        self.end_time = time.time()
        self.interval_clock = self.end_clock - self.start_clock
        self.interval_time = self.end_time - self.start_time
        template = "Finished {}. Took {:.2f} seconds, CPU time {:2f}, " \
                   "effectiveness {:.2f}"

        logger.info(template.format(self.message,
                                    self.interval_time,
                                    self.interval_clock,
                                    self.interval_clock / self.interval_time))


def make_submission(idx, preds, save_as):
    submission = pd.DataFrame({
        "test_id": idx,
        'price': preds
    }, columns=['test_id', 'price'])

    submission.to_csv(save_as, index=False)


def memory_info():
    process = psutil.Process()
    info = process.memory_info()
    return (f'process {process.pid}: RSS {info.rss:,} {info}; '
            f'system: {psutil.virtual_memory()}')


def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y) - np.log1p(y0), 2)))


def log_time(fn, name):
    @wraps(fn)
    def decorator(*args, **kwargs):
        logger.info(f'[{name}] << starting {fn.__name__}')
        t0 = time.time()

        try:
            return fn(*args, **kwargs)
        finally:
            dt = time.time() - t0
            logger.info(f'[{name}] >> finished {fn.__name__} in {dt:.2f} s, '
                        f'{memory_info()}')

    return decorator


_white_spaces = re.compile(r"\s\s+")
regex_double_dash = re.compile('[-]{2,}')


@lru_cache(maxsize=10000)
def word_to_charset(word):
    return ''.join(sorted(list(set(word))))


@lru_cache(maxsize=1000000)
def stem_word(word):
    return stemmer.stem(word)


def clean_text(text, tokenizer, hashchars=False):
    text = str(text).lower()
    text = _white_spaces.sub(" ", text)
    text = unidecode.unidecode(text)
    text = text.replace(' -', ' ').replace('- ', ' ').replace(' - ', ' ')
    text = regex_double_dash.sub(' ', text)
    # text = re.sub(r'\b([a-z0-9.]+)([\&\-])([a-z0-9.]+)\b', '\\1\\2\\3 \\1 \\3 \\1\\3', text, re.DOTALL)
    # text = re.sub(r'\b([a-z]+)([0-9]+)\b', '\\1\\2 \\1 \\2', text, re.DOTALL)
    # text = re.sub(r'([0-9]+)%', '\\1% \\1 percent', text, re.DOTALL)
    text = text.replace(':)', 'smiley').replace('(:', 'smiley').replace(':-)', 'smiley')
    tokens = tokenizer(str(text).lower())
    if hashchars:
        tokens = [word_to_charset(t) for t in tokens]
    return "".join(map(stem_word, tokens)).lower()


def extract_year(text):
    text = str(text)
    matches = [int(year) for year in re.findall('[0-9]{4}', text)
               if int(year) >= 1970 and int(year) <= 2018]
    if matches:
        return max(matches)
    else:
        return 0


def trim_description(text):
    if text and isinstance(text, str):
        return text[:config.ITEM_DESCRIPTION_MAX_LENGTH]
    else:
        return text


def trim_name(text):
    if text and isinstance(text, str):
        return text[:config.NAME_MAX_LENGTH]
    else:
        return text


def trim_brand_name(text):
    if text and isinstance(text, str):
        return text[:config.BRAND_NAME_MAX_LENGTH]
    else:
        return text


def fit_one(est, X, y):
    print("fitting y min={} max={}".format(y.min(), y.max()))
    return est.fit(X, y)


def predict_one(est, X):
    yhat = est.predict(X)
    print("predicting y min={} max={}".format(yhat.min(), yhat.max()))
    return yhat


def predict_models(X, fitted_models, vectorizer=None, parallel='thread'):
    if vectorizer:
        # TODO: parallelize this
        with Timer('Transforming data'):
            X = vectorizer.transform(X)
    predict_one_ = partial(predict_one, X=X)
    preds = map_parallel(predict_one_, fitted_models, parallel)
    return np.expm1(np.vstack(preds).T)


def fit_models(X_tr, y_tr, models, parallel='thread'):
    y_tr = np.log1p(y_tr)
    fit_one_ = partial(fit_one, X=X_tr, y=y_tr)
    return map_parallel(fit_one_, models, parallel)


def map_parallel(fn, lst, parallel, max_processes=4):
    if parallel == 'thread':
        with ThreadPool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel == 'mp':
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=max_processes) as pool:
            return pool.map(fn, lst)
    elif parallel is None:
        return list(map(fn, lst))
    else:
        raise ValueError(f'unexpected parallel value: {parallel}')


def predict_models_test_batches(models, vectorizer, parallel='thread'):
    chunk_preds = []
    test_idx = []
    for df in load_test_iter():
        test_idx.append(df.test_id.values)
        print("Predicting batch {} {}".format(df.test_id.min(), df.test_id.max()))
        chunk_preds.append(predict_models(df, models, vectorizer=vectorizer, parallel=parallel))
    predictions = np.vstack(chunk_preds)
    test_idx = np.concatenate(test_idx)
    return test_idx, predictions


def fit_transform_vectorizer(vectorizer):
    df_tr, df_va = load_train_validation()
    y_tr = df_tr.price.values
    y_va = df_va.price.values
    X_tr = vectorizer.fit_transform(df_tr, y_tr)
    X_va = vectorizer.transform(df_va)
    return X_tr, y_tr, X_va, y_va, vectorizer


def fit_validate(models, vectorizer, name=None,
                 fit_parallel='thread', predict_parallel='thread'):
    cached_path = 'data_{}.pkl'.format(name)
    if USE_CACHED_DATASET:
        assert name is not None
        with open(cached_path, 'rb') as f:
            X_tr, y_tr, X_va, y_va, fitted_vectorizer = pickle.load(f)
        if DEBUG_N:
            X_tr, y_tr = X_tr[:DEBUG_N], y_tr[:DEBUG_N]
    else:
        X_tr, y_tr, X_va, y_va, fitted_vectorizer = fit_transform_vectorizer(vectorizer)
    if DUMP_DATASET:
        assert name is not None
        with open(cached_path, 'wb') as f:
            pickle.dump((X_tr, y_tr, X_va, y_va, fitted_vectorizer), f)
    fitted_models = fit_models(X_tr, y_tr, models, parallel=fit_parallel)
    y_va_preds = predict_models(X_va, fitted_models, parallel=predict_parallel)
    return fitted_vectorizer, fitted_models, y_va, y_va_preds


def merge_predictions(X_tr, y_tr, X_te=None, est=None, verbose=True):
    if est is None:
        est = Lasso(alpha=0.0001, precompute=True, max_iter=1000,
                    positive=True, random_state=9999, selection='random')
    est.fit(np.log1p(X_tr), np.log1p(y_tr))
    if hasattr(est, 'intercept_') and verbose:
        logger.info('merge_predictions = \n{:+.4f}\n{}'.format(
            est.intercept_,
            '\n'.join('{:+.4f} * {}'.format(coef, i) for i, coef in
                      zip(range(X_tr.shape[0]), est.coef_))))
    return (np.expm1(est.predict(np.log1p(X_tr))),
            np.expm1(est.predict(np.log1p(X_te))) if X_te is not None else None)
