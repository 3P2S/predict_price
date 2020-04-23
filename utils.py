import psutil
import numpy as np
import pandas as pd
import time
import re
from functools import lru_cache
import config
import unidecode
from nltk import PorterStemmer
from string import digits

stemmer = PorterStemmer()
digits_set = set(digits)


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_clock = time.clock()
        self.start_time = time.time()

    def __exit__(self, *args):
        self.end_clock = time.clock()
        self.end_time = time.time()
        self.interval_clock = self.end_clock - self.start_clock
        self.interval_time = self.end_time - self.start_time
        template = "Finished {}. Took {:.2f} seconds, CPU time {:2f}, " \
                   "effectiveness {:.2f}"


class FastTokenizer():
    _default_word_chars = \
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

    _default_word_chars_set = set(_default_word_chars)

    _default_white_space_set = set(['\t', '\n', ' '])

    def __call__(self, text: str):
        tokens = []
        for ch in text:
            if len(tokens) == 0:
                tokens.append(ch)
                continue
            if self._merge_with_prev(tokens, ch):
                tokens[-1] = tokens[-1] + ch
            else:
                tokens.append(ch)
        return tokens

    def _merge_with_prev(self, tokens, ch):
        return (ch in self._default_word_chars_set and tokens[-1][-1] in self._default_word_chars_set) or \
               (ch in self._default_white_space_set and tokens[-1][-1] in self._default_white_space_set)


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


def has_digit(text):
    try:
        return any(c in digits_set for c in text)
    except:
        return False


def try_float(t):
    try:
        return float(t)
    except:
        return 0
