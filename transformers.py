import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from utils import *


class SparseMatrixOptimize(BaseEstimator, TransformerMixin):
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def fit(self, X, *args):
        return self

    def transform(self, X):
        return csr_matrix(X, dtype=self.dtype)


class FillEmpty(BaseEstimator, TransformerMixin):

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X['name'].fillna('unk', inplace=True)
        X['item_condition_id'] = X['item_condition_id'].fillna('unk')
        X['category_name'].fillna('unk', inplace=True)
        X['brand_name'].fillna('unk', inplace=True)
        X['shipping'].fillna(0, inplace=True)
        X['item_description'].fillna('unk', inplace=True)
        return X


class FastTokenizer():
    def __init__(self):
        self._default_word_chars = \
            u"-&" \
            u"0123456789" \
            u"ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
            u"abcdefghijklmnopqrstuvwxyz" \
            u"ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞß" \
            u"àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ" \
            u"ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğ" \
            u"ĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁł" \
            u"ńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧ" \
            u"ŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ" \
            u"ΑΒΓΔΕΖΗΘΙΚΛΜΝΟΠΡΣΤΥΦΧΨΩΪΫ" \
            u"άέήίΰαβγδεζηθικλμνξοπρςστυφχψω"
        self._default_word_chars_set = set(self._default_word_chars)

        self._default_white_space_set = {'\t', '\n', ' '}

    def __call__(self, text):
        token = []
        for char in text:
            if len(token) == 0:
                token.append(char)
                continue
            if self._merge_with_prev(token, char):
                token[-1] = token[-1] + char
            else:
                token.append(char)

    def _merge_with_prev(self, tokens, ch):
        return (ch in self._default_word_chars_set and tokens[-1][-1] in self._default_word_chars_set) or \
               (ch in self._default_white_space_set and tokens[-1][-1] in self._default_white_space_set)


class PreprocessDataPJ(BaseEstimator, TransformerMixin):
    def __init__(self, n_jobs=4, hash_chars=False, stem=True):
        self.n_jobs = n_jobs
        self.hash_chars = hash_chars
        self.stem = stem

    def fit(self, X):
        return self

    def transform(self, X):
        tokenizer = FastTokenizer()
        clean_text_ = partial(clean_text, tokenizer=tokenizer, hashchars=self.hash_chars)
        X['item_condition_id'] = X['item_condition_id'].fillna('UNK').astype(str)
        X['shipping'] = X['shipping'].astype(str)
        X['item_description'][X['item_description'] == 'No description yet'] = 'unk'
        X['item_description'] = X['item_description'].fillna('').astype(str)
        X['name'] = X['name'].fillna('').astype(str)

        # trim
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        if self.stem:
            with Pool(4) as pool:
                X['name_clean'] = pool.map(clean_text_, tqdm(X['name'], mininterval=2), chunksize=1000)
                X['desc_clean'] = pool.map(clean_text_, tqdm(X['item_description'], mininterval=2), chunksize=1000)
                X['brand_name_clean'] = pool.map(clean_text_, tqdm(X['brand_name'], mininterval=2), chunksize=1000)
                X['category_name_clean'] = pool.map(clean_text_, tqdm(X['category_name'], mininterval=2),
                                                    chunksize=1000)

        # handle category_name
        cat_def = ['unk', 'unk', 'unk']
        X['no_cat'] = X['category_name'].isnull().map(int)
        X['cat_split'] = X['category_name'].fillna('/'.join(cat_def)).map(lambda x: x.split('/'))
        X['cat_1'] = X['cat_split'].map(lambda x: x[0] if isinstance(x, list) and len(x) >= 1 else cat_def).str.lower()
        X['cat_2'] = X['cat_split'].map(lambda x: x[1] if isinstance(x, list) and len(x) >= 2 else cat_def).str.lower()
        X['cat_3'] = X['cat_split'].map(lambda x: x[2] if isinstance(x, list) and len(x) >= 3 else cat_def).str.lower()
        X['is_bundle'] = (X['item_description'].str.find('bundl') >= 0).map(int)

        return X


class PandasSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, dtype=None, inverse=False, return_vector=True):
        self.dtype = dtype
        self.columns = columns
        self.inverse = inverse
        self.return_vector = return_vector

        if isinstance(self.columns, str):
            self.columns = [self.columns]

    def check_condition(self, x, col):
        cond = (self.dtype is not None and x[col].dtype == self.dtype) or \
               (self.columns is not None and col in self.columns)
        return self.inverse ^ cond

    def fit(self, X, y=None):
        return self

    def _check_if_all_columns_present(self, x):
        if not self.inverse and self.columns is not None:
            missing_columns = set(self.columns) - set(x.columns)
            if len(missing_columns) > 0:
                missing_columns_ = ','.join(col for col in missing_columns)
                raise KeyError('Keys are missing in the record: %s' %
                               missing_columns_)

    def transform(self, x):
        # check if x is a pandas DataFrame
        if not isinstance(x, pd.DataFrame):
            raise KeyError('Input is not a pandas DataFrame')

        selected_cols = []
        for col in self.columns:
            if self.check_condition(x, col):
                selected_cols.append(col)

        # if the column was selected and inversed = False make sure the column is in the DataFrame
        self._check_if_all_columns_present(x)

        # if only 1 column is returned return a vector instead of a dataframe
        if len(selected_cols) == 1 and self.return_vector:
            return list(x[selected_cols[0]])
        else:
            return x[selected_cols]


class ConcatTexts(BaseEstimator, TransformerMixin):
    def __init__(self, columns, use_separators=True, output_col='text_concat'):
        self.columns = columns
        self.use_separators = use_separators
        self.output_col = output_col

    def fit(self, X, *args):
        return self

    def transform(self, X):
        X[self.output_col] = ''

        if self.use_separators:
            for i, col in enumerate(self.columns):
                X[self.output_col] += ' cs00{} '.format(i)
                X[self.output_col] += X[col]
        else:
            for col in self.columns:
                X[self.output_col] += X[col]

        return X


class PandasToRecords(BaseEstimator, TransformerMixin):
    def fit(self, X, *args):
        return self

    def transform(self, X):
        return X.to_dict(orient='records')


class ReportShape(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info('=' * 30)
        logger.info("Matrix shape {} min {} max {}".format(X.shape, X.min(), X.max()))
        logger.info('=' * 30)
        return X


class SanitizeSparseMatrix(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        self.datamax = np.nanmax(X.data)
        return self

    def transform(self, X):
        X.data[np.isnan(X.data)] = 0
        X.data = X.data.clip(0, self.datamax)
        return X


class PreprocessDataKL(BaseEstimator, TransformerMixin):

    def __init__(self, num_brands, repl_patterns):
        self.num_brands = num_brands
        self.repl_patterns = repl_patterns

    def fit(self, X, y):
        self.pop_brands = X['brand_name'].value_counts().index[:self.num_brands]
        return self

    def transform(self, X):
        # fill missing values
        X['category_name'] = X['category_name'].fillna('unk').map(str)
        X['brand_name'] = X['brand_name'].fillna('unk').map(str)
        X['item_description'] = X['item_description'].fillna('').map(str)
        X['name'] = X['name'].fillna('').map(str)

        # trim
        X['item_description'] = X['item_description'].map(trim_description)
        X['name'] = X['name'].map(trim_name)
        X['brand_name'] = X['brand_name'].map(trim_brand_name)

        X.loc[~X['brand_name'].isin(self.pop_brands), 'brand_name'] = 'Other'
        X['category_name_l1'] = X['category_name'].str.split('/').apply(lambda x: x[0])
        X['category_name_l1s'] = \
            X['category_name'].str.split('/').apply(
                lambda x: x[0] if x[0] != 'Women' else '/'.join(x[:2]))
        X['category_name_l2'] = \
            X['category_name'].str.split('/').apply(lambda x: '/'.join(x[:2]))
        for pat, repl in self.repl_patterns:
            X['item_description'] = X['item_description'].str.replace(
                pat, repl, flags=re.IGNORECASE)

        no_description = X['item_description'] == 'No description yet'
        X.loc[no_description, 'item_description'] = ''
        X['no_description'] = no_description.astype(str)
        X['item_condition_id'] = X['item_condition_id'].map(str)
        X['shipping'] = X['shipping'].map(str)

        return X