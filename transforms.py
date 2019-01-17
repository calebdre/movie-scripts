import re, cgi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import numpy as np
from collections import Counter
from functools import reduce

unk_token = "<unk>"
pad_token = "<pad>"

def reduce_to_tokens(data):
    def reducer(accum, x):
        accum += x
        return accum
    
    return reduce(reducer, data.values, [])
    
def lower(data):
    return data.str.lower().str.strip()

def word_normalize(data):
    def fn(x):
        x = re.sub(r"([.!?])", r" \1", x)
        x = re.sub(r"[^a-zA-Z.!?]+", r" ", x)
        return x
    
    return data.apply(fn)

def strip_tags(data):
    tag_re = re.compile(r'(<!--.*?-->|<[^>]*>)')
    def strip(x):
        # https://stackoverflow.com/a/19730306/2229572
        no_tags = tag_re.sub('', x)
        escaped = cgi.escape(no_tags)
        return escaped

    return data.apply(strip)

# https://stackoverflow.com/a/17320458/2229572
def stem(data):
    stemmer = PorterStemmer()
    return data.apply(lambda x: stemmer.stem(x))

def tokenize(data):
    return data.apply(lambda x: nltk.word_tokenize(x))

def remove_stopwords(data):
    stpwds = set(stopwords.words("english"))
    def remove(x):
        return [token for token in x if token not in stpwds]
    
    return data.apply(remove)

def most_common_k(data, k = 6000, skip_top_n = 10):
    all_tokens = reduce_to_tokens(data)
    counts = Counter(all_tokens)
    most_common = counts.most_common(k + skip_top_n)
    allowed_words = set([w_c[0] for w_c in most_common[skip_top_n:]])
    
    def remove(x):
        return [token if token in allowed_words else unk_token for token in x]
    
    return data.apply(remove)

def to_idxs(data):
    all_tokens = reduce_to_tokens(data)
    all_tokens.append(pad_token)
    if unk_token not in all_tokens:
        all_tokens.append(unk_token)
    
    all_tokens = set(all_tokens)
        
    token_idx = dict(zip(all_tokens, list(range(len(all_tokens)))))
    def to_idx(x):
        return [token_idx[token] for token in x]
    
    return data.apply(to_idx), token_idx, list(all_tokens)
    
def normalize(data):
    data = np.array(data.values)
    return (data - data.mean()) / data.std()