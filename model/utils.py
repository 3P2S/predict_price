import numpy as np
from config import *
import pickle
from functools import partial
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from sklearn.linear_model import Lasso

from load_data import load_test_iter, load_train_validation


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


def fit_validate(models, vectorizer, name=None, fit_parallel='thread', predict_parallel='thread'):
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
    return (np.expm1(est.predict(np.log1p(X_tr))),
            np.expm1(est.predict(np.log1p(X_te))) if X_te is not None else None)


def fit_one(est, X, y):
    print("fitting y min={} max={}".format(y.min(), y.max()))
    return est.fit(X, y)


def predict_one(est, X):
    yhat = est.predict(X)
    print("predicting y min={} max={}".format(yhat.min(), yhat.max()))
    return yhat


def predict_models(X, fitted_models, vectorizer=None, parallel='thread'):
    if vectorizer:
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
