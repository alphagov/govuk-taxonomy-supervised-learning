
# coding: utf-8

# ## Convolutional NN to classify govuk content to level2 taxons

# Based on:
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
import os
MODEL_NAME = os.getenv('EXPERIMENT_NAME')
DATADIR = os.getenv('DATADIR')
COMET_API_KEY = os.getenv("COMET_API_KEY")
print('algorithm running on data extracted from content store on {}'.format(DATADIR))

# ### Load requirements and data
from comet_ml import Experiment
from tokenizing import load_tokenizer_from_file
from algorithm_functions import f1, to_file, WeightedBinaryCrossEntropy

import numpy as np
import pandas as pd

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import (Embedding, Input, Dense, Dropout, 
                          Activation, Conv1D, MaxPooling1D, Flatten, concatenate, Reshape)
from keras.models import Model, Sequential
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import json

from scipy import sparse

experiment = Experiment(api_key=COMET_API_KEY, project_name='govuk_taxonomy_levelagnostic')
experiment.set_name(MODEL_NAME)

def sparse_generator(X_data, y_data=None, batch_size=128):
    samples_per_epoch = X_data[0].all().shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        dense_x_data = []
        for i, x in enumerate(X_data):
            if i == 3:
                X_batch = x[index_batch,:]
            else:
                x = x.all()
                X_batch = np.array(x[index_batch,:].todense())
            
            dense_x_data.append(X_batch)
            
        y_batch = np.array(y_data[index_batch])
        counter += 1
        yield dense_x_data,y_batch
       
        if (counter > number_of_batches):
            counter=0

# ## Hyperparameters
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100  # keras embedding layer output_dim = Dimension of the dense embedding
P_THRESHOLD = 0.5  # Threshold for probability of being assigned to class
POS_RATIO = 0.5  # ratio of positive to negative for each class in weighted binary cross entropy loss function
NUM_WORDS = 20000  # keras tokenizer num_words: None or int. Maximum number of words to work with
# (if set, tokenization will be restricted to the top num_words most common words in the dataset).
BATCH_SIZE= 128


# ### Read in data
train = np.load(os.path.join(DATADIR, 'level_agnostic_train_arrays.npz'))

x_train = train['x']
meta_train = train['meta']
title_train = train['title']
desc_train = train['desc']
y_train = train['y'].all().todense()

print('x_train.shape = {}'.format(x_train))
print('meta_train.all().shape = {}'.format(meta_train.all().shape))
print('title_train.all().shape = {}'.format(title_train.all().shape))
print('desc_train.all().shape = {}'.format(desc_train.all().shape))
print('y_train.shape = {}'.format(y_train.shape))


dev = np.load(os.path.join(DATADIR, 'level_agnostic_dev_arrays.npz'))

x_dev = dev['x']
meta_dev = dev['meta']
title_dev = dev['title']
desc_dev = dev['desc']
y_dev = dev['y'].all().todense()

print('x_dev.shape = {}'.format(type(x_dev)))
print('meta_dev.shape = {}'.format(meta_dev.shape))
print('title_dev.shape = {}'.format(title_dev.shape))
print('desc_dev.shape = {}'.format(desc_dev.shape))
print('y_dev.shape = {}'.format(y_dev.shape))

# test = np.load(os.path.join(DATADIR, 'level_agnostic_test_arrays.npz'))

# x_test = test['x']
# meta_test = test['meta'].all().todense()
# title_test = test['title'].all().todense()
# desc_test = test['desc'].all().todense()
# y_test = test['y'].all().todense()

# print('x_test.shape = {}'.format(x_test.shape))
# print('meta_test.shape = {}'.format(meta_test.shape))
# print('title_test.shape = {}'.format(title_test.shape))
# print('desc_test.shape = {}'.format(desc_test.shape))
# print('y_test.shape = {}'.format(y_test.shape))

tokenizer_combined_text = load_tokenizer_from_file(os.path.join(DATADIR, "combined_text_tokenizer.json"))

embedding_layer = Embedding(len(tokenizer_combined_text.word_index) + 1, 
                            EMBEDDING_DIM, 
                            input_length=MAX_SEQUENCE_LENGTH)

# ### 1. Create model

NB_CLASSES = y_train.shape[1]
NB_METAVARS = meta_train.all().todense().shape[1]

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


title_input = Input(shape=(title_train.all().shape[1],), name='titles')
title_hidden = Dense(128, activation='relu', name = 'hidden_title')(title_input)
title_hidden = Dropout(0.2, name = 'dropout_title')(title_hidden)

desc_input = Input(shape=(desc_train.all().shape[1],), name='descs')
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

# tb = TensorBoard(log_dir='./logs', histogram_freq=3, write_graph=False, embeddings_freq=1, embeddings_layer_names=['embedded_sequences'], embeddings_metadata='combined_text_word_index.csv')

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# ### 2. compile model

model.compile(loss=WeightedBinaryCrossEntropy(POS_RATIO),
              optimizer='rmsprop',
              metrics=['binary_accuracy', f1])

### 3. Train model
history = model.fit_generator(generator=sparse_generator([meta_train,
                                                          title_train,
                                                          desc_train,
                                                          x_train],
                                                         y_train,
                                                         batch_size=BATCH_SIZE),
                              steps_per_epoch=x_train.shape[0]/BATCH_SIZE,
                              verbose=1,
                              validation_data=sparse_generator([meta_dev,
                                                                title_dev,
                                                                desc_dev,
                                                                x_dev],
                                                               y_dev,
                                                               batch_size=BATCH_SIZE),
                              validation_steps=x_dev.shape[0]/BATCH_SIZE,  epochs=10, callbacks=[early_stopping])

# history = model.fit(
#     {'meta': meta_train, 'titles': title_train, 'descs': desc_train, 'wordindex': x_train},
#     y_train, 
#     validation_data=([meta_dev, title_dev, desc_dev, x_dev], y_dev), 
#     epochs=10, batch_size=128, callbacks=[early_stopping], verbose=2
# )


# history_dict = history.history
# history_dict.keys()
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']


# plt.plot(range(len(loss_values)), loss_values, 'bo', label='Training loss')
# plt.plot(range(len(val_loss_values)), val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# experiment.log_figure(figure_name='loss', figure=plt)

# plt.savefig(os.path.join(DATADIR, 'loss.png'))
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
with experiment.train():
    y_prob = model.predict_generator(generator=sparse_generator([meta_train, title_train, desc_train, x_train], y_train),
                                     steps=x_train.shape[0]/BATCH_SIZE)
    to_file(y_prob, "train_results", y_train)

    y_pred = y_prob.copy()
    y_pred[y_pred >= P_THRESHOLD] = 1
    y_pred[y_pred < P_THRESHOLD] = 0

    print('train micro: {}'.format(precision_recall_fscore_support(y_train, y_pred, average='micro', sample_weight=None)))
    print('train macro: {}'.format(precision_recall_fscore_support(y_train, y_pred, average='macro', sample_weight=None)))
    print('train weightedmacro: {}'.format(precision_recall_fscore_support(y_train, y_pred, average='weighted', sample_weight=None)))

    train_metrics = {
        "train_micro":f1_score(y_train, y_pred, average='micro', sample_weight=None),
        "train_macro":f1_score(y_train, y_pred, average='macro', sample_weight=None),
        "train_weighted_macro":f1_score(y_train, y_pred, average='weighted', sample_weight=None)
        }

    experiment.log_multiple_metrics(train_metrics)

# Dev
with experiment.validate():
    y_prob_dev = model.predict_generator(generator=sparse_generator([meta_dev, title_dev, desc_dev, x_dev], y_dev),
                                         steps=x_dev.shape[0]/BATCH_SIZE)
    to_file(y_prob_dev, "dev_results", y_train)

    y_pred_dev = y_prob_dev.copy()
    y_pred_dev[y_pred_dev >= P_THRESHOLD] = 1
    y_pred_dev[y_pred_dev < P_THRESHOLD] = 0

    print('dev micro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average='micro', sample_weight=None)))
    print('dev macro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average='macro', sample_weight=None)))
    print('dev weightedmacro: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average='weighted', sample_weight=None)))


    print('dev unweighted F1: {}'.format(precision_recall_fscore_support(y_dev, y_pred_dev, average=None, sample_weight=None)))

    dev_metrics = {
        "dev_micro":f1_score(y_dev, y_pred_dev, average='micro', sample_weight=None),
        "dev_macro":f1_score(y_dev, y_pred_dev, average='macro', sample_weight=None),
        "dev_weighted_macro":f1_score(y_dev, y_pred_dev, average='weighted', sample_weight=None)
        }

    experiment.log_multiple_metrics(dev_metrics)

print('loading taxon_codes and labels_index')
taxon_codes = pd.Series(np.load(os.path.join(DATADIR, 'levelagnostic_taxon_codes.npy')))

with open(os.path.join(DATADIR, "agnostictaxon_labels_index.json"), 'r') as f:
    labels_index = json.load(f, object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v for k, v in d.items()})

with open(os.path.join(DATADIR, "agnostictaxon_id_index.json"), 'r') as f:
    taxonid_index = json.load(f, object_hook=lambda d: {int(k): [int(i) for i in v] if isinstance(v, list) else v for k, v in d.items()})

print('creating plotting_metrics dataframe')
dev_f1s = pd.Series(f1_score(y_dev, y_pred_dev, average=None, sample_weight=None))

plotting_metrics = pd.concat([pd.concat([pd.DataFrame(np.sum(y_train, axis=0)),
                              pd.DataFrame(np.sum(y_dev, axis=0))]).transpose(),
                              taxon_codes,
                              dev_f1s], axis=1)
      
plotting_metrics.columns = ['train_support', 'dev_support', 'taxon_code', 'dev_f1']
plotting_metrics['taxon_label'] = plotting_metrics['taxon_code'].map(labels_index)
plotting_metrics['taxon_id'] = plotting_metrics['taxon_code'].map(taxonid_index)
plotting_metrics = plotting_metrics.sort_values('dev_f1', ascending=False)
plotting_metrics.to_csv(os.path.join(DATADIR, MODEL_NAME+'plotting_metrics.csv'))
print('logging plotting_metrics dataframe to html table')
experiment.log_html(plotting_metrics.to_html())
# dev_f1 = plotting_metrics.sort_values('dev_f1', ascending=False).plot(x='taxon_label',
#                                                                       y='dev_f1',
#                                                                       kind = 'barh',
#                                                                       figsize=(10,30),
#                                                                       legend=False,
#                                                                       title='F1 score by taxon',
#                                                                       color='#2B8CC4')

# dev_f1, ax = plt.subplots(figsize=(10, 30))	
# ax.barh(plotting_metrics['taxon_label'].values, plotting_metrics['dev_f1'].values, color='#2B8CC4')
# experiment.log_figure(figure_name='dev_f1_scores', figure=dev_f1)
# plt.close()

# train_support = plotting_metrics.sort_values('dev_f1', ascending=False).plot(x='taxon_label',
#                                                                              y='train_support',
#                                                                              kind = 'barh',
#                                                                              figsize=(10,30),
#                                                                              legend=False,
#                                                                              title='F1 score by taxon',
#                                                                              color='#4C2C92').get_figure()

# dev_support = plotting_metrics.sort_values('dev_f1', ascending=False).plot(x='taxon_label',
#                                                                            y='dev_support',
#                                                                            kind = 'barh',
#                                                                            figsize=(10,30),
#                                                                            legend=False,
#                                                                            title='F1 score by taxon',
#                                                                            color='#00823B').get_figure()
                             


to_file(y_train, "true_train", y_train)
to_file(y_dev, "true_dev", y_train)

print('saving model')
model.save(os.path.join(DATADIR, MODEL_NAME))

# experiment.log_figure(figure_name='dev_support', figure=train_support)
# experiment.log_figure(figure_name='train_support', figure=dev_support)

print('logging experiment parameters')
params={
    "max_sequence_length":MAX_SEQUENCE_LENGTH,
    "embedding_dim":EMBEDDING_DIM,
    "p_threshold":P_THRESHOLD,
    "pos_ratio":POS_RATIO,
    "num_words":NUM_WORDS,
    "datadir":DATADIR,
    "metadata":os.getenv('METADATA_LIST'),
    "Data_since":os.getenv('SINCE_THRESHOLD')
        }

experiment.log_multiple_params(params)

experiment.log_other("datadir", DATADIR)
experiment.log_other("metadata", os.getenv('METADATA_LIST'))
experiment.log_other("data_since", os.getenv('SINCE_THRESHOLD'))
experiment.log_dataset_hash(x_train)
