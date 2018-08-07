
# coding: utf-8

# ## Convolutional NN to classify govuk content to level2 taxons

# Based on:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

# ### Load requirements and data
import os

from tokenizing import load_tokenizer_from_file
from algorithm_functions import f1, to_file, WeightedBinaryCrossEntropy

import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import (Embedding, Input, Dense, Dropout, 
                          Activation, Conv1D, MaxPooling1D, Flatten, concatenate, Reshape)
from keras.models import Model, Sequential
from sklearn.metrics import precision_recall_fscore_support, classification_report



DATADIR = os.getenv('DATADIR')
print('algorithm running on data extracted from content store on {}'.format(DATADIR))


# ## Hyperparameters
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100  # keras embedding layer output_dim = Dimension of the dense embedding
P_THRESHOLD = 0.5  # Threshold for probability of being assigned to class
POS_RATIO = 0.5  # ratio of positive to negative for each class in weighted binary cross entropy loss function
NUM_WORDS = 20000  # keras tokenizer num_words: None or int. Maximum number of words to work with
# (if set, tokenization will be restricted to the top num_words most common words in the dataset).


# ### Read in data
train = np.load(os.path.join(DATADIR, 'train_arrays.npz'))

x_train = train['x']
meta_train = train['meta'].all().todense()
title_train = train['title'].all().todense()
desc_train = train['desc'].all().todense()
y_train = train['y'].all().todense()

print('x_train.shape = {}'.format(x_train.shape))
print('meta_train.shape = {}'.format(meta_train.shape))
print('title_train.shape = {}'.format(title_train.shape))
print('desc_train.shape = {}'.format(desc_train.shape))
print('y_train.shape = {}'.format(y_train.shape))


dev = np.load(os.path.join(DATADIR, 'dev_arrays.npz'))

x_dev = dev['x']
meta_dev = dev['meta'].all().todense()
title_dev = dev['title'].all().todense()
desc_dev = dev['desc'].all().todense()
y_dev = dev['y'].all().todense()

print('x_dev.shape = {}'.format(x_dev.shape))
print('meta_dev.shape = {}'.format(meta_dev.shape))
print('title_dev.shape = {}'.format(title_dev.shape))
print('desc_dev.shape = {}'.format(desc_dev.shape))
print('y_dev.shape = {}'.format(y_dev.shape))

test = np.load(os.path.join(DATADIR, 'test_arrays.npz'))

x_test = test['x']
meta_test = test['meta'].all().todense()
title_test = test['title'].all().todense()
desc_test = test['desc'].all().todense()
y_test = test['y'].all().todense()

print('x_test.shape = {}'.format(x_test.shape))
print('meta_test.shape = {}'.format(meta_test.shape))
print('title_test.shape = {}'.format(title_test.shape))
print('desc_test.shape = {}'.format(desc_test.shape))
print('y_test.shape = {}'.format(y_test.shape))

tokenizer_combined_text = load_tokenizer_from_file(os.path.join(DATADIR, "combined_text_tokenizer.json"))

embedding_layer = Embedding(len(tokenizer_combined_text.word_index) + 1, 
                            EMBEDDING_DIM, 
                            input_length=MAX_SEQUENCE_LENGTH)

# ### 1. Create model

NB_CLASSES = y_train.shape[1]
NB_METAVARS = meta_train.shape[1]

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='wordindex') #MAX_SEQUENCE_LENGTH
embedded_sequences = embedding_layer(sequence_input)
x = Dropout(0.2, name = 'dropout_embedded')(embedded_sequences)

x = Conv1D(128, 5, activation='relu', name = 'conv0')(x)

x = MaxPooling1D(5, name = 'max_pool0')(x)

x = Dropout(0.5, name = 'dropout0')(x)

x = Conv1D(128, 5, activation='relu', name = 'conv1')(x)

x = MaxPooling1D(5 , name = 'max_pool1')(x)

x = Conv1D(128, 5, activation='relu', name = 'conv2')(x)

x = MaxPooling1D(35, name = 'global_max_pool')(x)  # global max pooling

x = Flatten()(x) #reduce dimensions from 3 to 2; convert to vector + FULLYCONNECTED

meta_input = Input(shape=(NB_METAVARS,), name='meta')
meta_hidden = Dense(128, activation='relu', name = 'hidden_meta')(meta_input)
meta_hidden = Dropout(0.2, name = 'dropout_meta')(meta_hidden)


title_input = Input(shape=(title_train.shape[1],), name='titles')
title_hidden = Dense(128, activation='relu', name = 'hidden_title')(title_input)
title_hidden = Dropout(0.2, name = 'dropout_title')(title_hidden)

desc_input = Input(shape=(desc_train.shape[1],), name='descs')
desc_hidden = Dense(128, activation='relu', name = 'hidden_desc')(desc_input)
desc_hidden = Dropout(0.2, name = 'dropout_desc')(desc_hidden)

concatenated = concatenate([meta_hidden, title_hidden, desc_hidden, x])

x = Dense(400, activation='relu', name = 'fully_connected0')(concatenated)

x = Dropout(0.2, name = 'dropout1')(x)

x = Dense(NB_CLASSES, activation='sigmoid', name = 'fully_connected1')(x)

# # The Model class turns an input tensor and output tensor into a model
# This creates Keras model instance, will use this instance to train/test the model.
model = Model(inputs=[meta_input, title_input, desc_input, sequence_input], outputs=x)


print(model.summary())

#
# CHECKPOINT_PATH = os.path.join(DATADIR, 'model_checkpoint.hdf5')
#
# cp = ModelCheckpoint(
#                      filepath = CHECKPOINT_PATH,
#                      monitor='val_loss',
#                      verbose=0,
#                      save_best_only=False,
#                      save_weights_only=False,
#                      mode='auto',
#                      period=1
#                     )

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# ### 2. compile model

model.compile(loss=WeightedBinaryCrossEntropy(POS_RATIO),
              optimizer='rmsprop',
              metrics=['binary_accuracy', f1])

# ### 3. Train model

history = model.fit(
    {'meta': meta_train, 'titles': title_train, 'descs': desc_train, 'wordindex': x_train},
    y_train, 
    validation_data=([meta_dev, title_dev, desc_dev, x_dev], y_dev), 
    epochs=10, batch_size=128, callbacks=[early_stopping]
)


# history_dict = history.history
# history_dict.keys()
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
#
#
#
# plt.plot(range(len(loss_values)), loss_values, 'bo', label='Training loss')
# plt.plot(range(len(val_loss_values)), val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()
#
# f1_values = history_dict['f1']
# val_f1_values = history_dict['val_f1']
#
# plt.plot(range(len(f1_values)), f1_values, 'bo', label='Training f1')
# plt.plot(range(len(val_f1_values)), val_f1_values, 'b', label='Validation f1')
# plt.title('Training and validation batch-level f1-micro')
# plt.xlabel('Epochs')
# plt.ylabel('F1-micro')
# plt.legend()
#
# plt.show()


# ### 4. Save results arrays

# Train
y_prob = model.predict([meta_train, title_train, desc_train, x_train])
to_file(y_prob, "train_results", y_train)

y_pred = y_prob.copy()
y_pred[y_pred >= P_THRESHOLD] = 1
y_pred[y_pred < P_THRESHOLD] = 0

print('train micro: {}'.format(precision_recall_fscore_support(y_train, y_pred, average='micro', sample_weight=None)))
print('train macro: {}'.format(precision_recall_fscore_support(y_train, y_pred, average='macro', sample_weight=None)))
print('train weightedmacro: {}'.format(precision_recall_fscore_support(y_train, y_pred, average='weighted', sample_weight=None)))

# Dev
y_prob_dev = model.predict([meta_dev, title_dev, desc_dev, x_dev])
to_file(y_prob_dev, "dev_results", y_train)

y_pred_dev = y_prob_dev.copy()
y_pred_dev[y_pred_dev >= P_THRESHOLD] = 1
y_pred_dev[y_pred_dev < P_THRESHOLD] = 0

print('dev micro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average='micro', sample_weight=None)))
print('dev macro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average='macro', sample_weight=None)))
print('dev weightedmacro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average='weighted', sample_weight=None)))


print('dev weightedmacro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average=None, sample_weight=None)))

to_file(y_train, "true_train", y_train)
to_file(y_dev, "true_dev", y_train)
