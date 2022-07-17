from itertools import chain
from typing import List, Tuple, Dict

from nltk import FreqDist

DIVIDER = ' '
TOKENS_KEY = 'cleaned_tokens'
CATEGORIES = [
    'religion',
    'sports', 'drink',
    'financial', 'health', 'literature',
    'social networks', 'food', 'history',
    'animals', 'news', 'science', 'movies',
    'music', 'games', 'computer',
    'football', 'basketball', 'volleyball',
    'university', 'national', 'politics'
]
OTHERS = 'others'


def get_keywords(tokens: List[List[str]], count: int = 20) -> List[Tuple]:
    words = get_words(tokens)
    return FreqDist(words).most_common(count)


def get_content(tokens: List[List[str]]):
    return DIVIDER.join(get_sentences(tokens))


def get_contents(data: List, key: str):
    return [get_content(doc[key]) for doc in data]


def get_sentences(tokens: List[List[str]]):
    return [DIVIDER.join(words) for words in tokens]


def get_words(tokens: List[List[str]]):
    return list(chain(*tokens))


def get_doc_words(doc: Dict, key: str):
    return get_words(doc[key])
