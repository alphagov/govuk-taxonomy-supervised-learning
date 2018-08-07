# coding: utf-8

from keras.preprocessing.text import Tokenizer

import json
from collections import OrderedDict


def create_and_save_tokenizer(data, num_words, outfilename):
    tokenizer = Tokenizer(oov_token='UNK', num_words=num_words+1)
    tokenizer.fit_on_texts(data)
    tokenizer.word_index = {e: i for e, i in tokenizer.word_index.items() if i <= num_words}
    tokenizer.word_index[tokenizer.oov_token] = num_words + 1

    tokenizer_dict = {
        "word_counts": list(tokenizer.word_counts.items()),
        "word_docs": tokenizer.word_docs,
        "word_index": tokenizer.word_index,
        "document_count": tokenizer.document_count,
        "index_docs": tokenizer.index_docs
    }

    with open(outfilename, 'w') as outfile:
        json.dump(tokenizer_dict, outfile)


def load_tokenizer_from_file(filename):
    
    tokenizer = Tokenizer()

    with open(filename, 'r') as infile:
        tokenizer_data = json.load(infile)

    tokenizer.word_counts = OrderedDict(tokenizer_data['word_counts'])
    tokenizer.word_docs = tokenizer_data['word_docs']
    tokenizer.word_index = tokenizer_data['word_index']
    tokenizer.document_count = tokenizer_data['document_count']
    tokenizer.index_docs = tokenizer_data['index_docs']

    return tokenizer
