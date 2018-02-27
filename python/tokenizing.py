from keras.preprocessing.text import Tokenizer

import json
from collections import OrderedDict

# def tokenize(local_tokenizer,input_data, =True):
# # apply tokenizer to our text data
#     data = []
#     local_tokenizer.fit_on_texts(input_data)
# # list of word indexes, where the word of rank i in the dataset (starting at 1) has index i
#     sequences = local_tokenizer.texts_to_sequences(input_data)
#     word_index = local_tokenizer.word_index  
#     print('Found %s unique tokens.' % len(word_index))
#     if :
#         data = pad_sequences(sequences, maxlen= MAX_SEQUENCE_LENGTH)
#     else:
#         data = local_tokenizer.sequences_to_matrix(sequences)
#     return data,word_index

def create_and_save_tokenizer(data, num_words, outfilename):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(data)

    tokenizer_dict = {
        "word_counts" : list(tokenizer.word_counts.items()), 
        "word_docs" : tokenizer.word_docs,
        "word_index" : tokenizer.word_index, 
        "document_count" : tokenizer.document_count
    }

    with open(outfilename, 'w') as outfile:
        json.dump(tokenizer_dict, outfile)


text= ["chris is here here here and next to me", "Ellie is also here isn't she", "Felisia is at home"]

create_and_save_tokenizer(
    text, 
    num_words=10, 
    outfilename="test_tokenizer.json"
)


def load_tokenizer_from_file(filename):
    
    tokenizer = Tokenizer()

    with open(filename, 'r') as infile:
        tokenizer_data = json.load(infile)

    tokenizer.word_counts = OrderedDict(tokenizer_data['word_counts'])
    tokenizer.word_docs = tokenizer_data['word_docs']
    tokenizer.word_index = tokenizer_data['word_index']
    tokenizer.document_count = tokenizer_data['document_count']

    return tokenizer

loaded_tokenizer = load_tokenizer_from_file("test_tokenizer.json")

print(loaded_tokenizer)

print(loaded_tokenizer.texts_to_sequences(text))


# collections.OrderedDict is nothing but a dict, which remembers the order in which the elements are included in it. So you can create one with its constructor like this

# [OrderedDict((k, d[k](v)) for (k, v) in l.iteritems()) for l in L]