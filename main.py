import os
import joblib
import traceback

from preprocess import *
from config import *
from utils import rmsle
from model.model import RegressionHuber, prelu, RegressionClf
from model import utils
import load_data


def define_models_1(n_jobs, seed):
    h0 = 192  # the same to make training take the same time
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu)
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_1(n_jobs=n_jobs)


def define_models_2(n_jobs, seed):
    h0 = 192  # the same to make training take the same time
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu, binary_X=True),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu, binary_X=True),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 12, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu)
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_2(n_jobs=n_jobs)


def define_models_3(n_jobs, seed):
    h0 = 128  # reduced from 192 due to kaggle slowdown
    n_epoch = 3 if TEST_SIZE == 1 else 1
    models = [
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu, binary_X=True),
        RegressionHuber(n_hidden=(h0, 64, 32), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.1e-5,
                        actfunc=prelu),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu, binary_X=True),
        RegressionClf(n_hidden=(h0, 64), n_epoch=n_epoch, batch_size=2 ** 11, learning_rate=0.4e-2, reg_l2=0.05e-5,
                      actfunc=prelu),
    ]
    for i, model in enumerate(models, seed * 100):
        model.seed = i
    return models, prepare_vectorizer_3(n_jobs)


def main(name, action, arg_map, fit_parallel='thread', predict_parallel='thread'):
    prefix = lambda r: '{}_{}s'.format(name, r)

    if action in ("1", "2", "3"):
        model_round = int(action)
        models, vectorizer = arg_map[model_round]
        vectorizer, fitted_models, y_va, y_va_preds = utils.fit_validate(models,
                                                                         vectorizer,
                                                                         name=model_round,
                                                                         fit_parallel=fit_parallel,
                                                                         predict_parallel=predict_parallel)
        joblib.dump(y_va_preds, "{}_va_preds.pkl".format(prefix(model_round)), compress=3)
        if HANDLE_TEST:
            test_idx, y_te_preds = utils.predict_models_test_batches(fitted_models, vectorizer,
                                                                     parallel=predict_parallel)
            joblib.dump(y_te_preds, "{}_te_preds.pkl".format(prefix(model_round)), compress=3)
            joblib.dump(test_idx, "test_idx.pkl", compress=3)
        joblib.dump(y_va, "y_va.pkl", compress=3)
        for i in range(y_va_preds.shape[1]):
            print("Model {} rmsle {:.4f}".format(i, rmsle(y_va_preds[:, i], y_va)))
        print("Model mean rmsle {:.4f}".format(rmsle(y_va_preds.mean(axis=1), y_va)))

    elif action == "merge":
        va_preds = []
        te_preds = []
        for model_round in ("1", "2", "3"):
            try:
                va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
                if HANDLE_TEST:
                    te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
            except Exception as e:
                print(f'Warning: error loading round {model_round}: {e}')
                traceback.print_exc()
        va_preds = np.hstack(va_preds).clip(MIN_PRICE_PRED, MAX_PRICE_PRED)
        if HANDLE_TEST:
            te_preds = np.hstack(te_preds).clip(MIN_PRICE_PRED, MAX_PRICE_PRED)
        else:
            te_preds = None
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = utils.merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        print("Stacking rmsle", rmsle(y_va, va_preds_merged))
        if HANDLE_TEST:
            test_idx = joblib.load("test_idx.pkl")
            make_submission(test_idx, te_preds_merged, 'submission_merged.csv')

    elif action == "merge_describe":
        va_preds = []
        te_preds = []
        for model_round in ("1", "2", "3"):
            va_preds.append(joblib.load("{}_va_preds.pkl".format(prefix(model_round))))
            te_preds.append(joblib.load("{}_te_preds.pkl".format(prefix(model_round))))
        va_preds = np.hstack(va_preds)
        te_preds = np.hstack(te_preds)
        _, df_va = load_data.load_train_validation()
        y_va = joblib.load("y_va.pkl")
        va_preds_merged, te_preds_merged = utils.merge_predictions(X_tr=va_preds, y_tr=y_va, X_te=te_preds)
        print("Stacking rmsle", rmsle(y_va, va_preds_merged))
        df_va['preds'] = va_preds_merged
        df_va['err'] = (np.log1p(df_va['preds']) - np.log1p(df_va['price'])) ** 2
        df_va.sort_values('err', ascending=False).to_csv('validation_preds.csv', index=False)


if __name__ == '__main__':
    main(
        'tf',
        sys.argv[1],
        {
            1: define_models_1(n_jobs=4, seed=1),
            2: define_models_2(n_jobs=4, seed=2),
            3: define_models_3(n_jobs=4, seed=3),
        },
    )
