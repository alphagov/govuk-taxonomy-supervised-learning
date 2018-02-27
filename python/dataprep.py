import pandas as pd
import os
import tokenizing
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, MinMaxScaler


DATADIR = os.getenv('DATADIR')

labelled_level2 = pd.read_csv(
	os.path.join(DATADIR, 'labelled_level2.csv.gz'), 
	dtype=object, 
	compression='gzip'
)

# Create World taxon in case any items not identified 
# through doc type in clean_content are still present
labelled_level2.loc[labelled_level2['level1taxon'] == 'World', 'level2taxon'] = 'world_level1'




# **** TAXONS TO CATEGORICAL -> DICT **********
# *********************************************

# creating categorical variable for level2taxons from values
labelled_level2['level2taxon'] = labelled_level2['level2taxon'].astype('category')

#Add 1 because of zero-indexing to get 1-number of level2taxons as numerical targets
labelled_level2['level2taxon_code'] = labelled_level2.level2taxon.astype('category').cat.codes + 1

#create dictionary of taxon category code to string label for use in model evaluation
labels_index = dict(zip((labelled_level2['level2taxon_code']), 
	labelled_level2['level2taxon']))



# ***** NEW COLUMNS FREQUENCY COUNTS **********
# *********************************************

#count the number of content items per taxon into new column
labelled_level2['num_content_per_taxon'] = labelled_level2.groupby(["level2taxon"])['level2taxon'].transform("count")

print('Number of unique level2taxons: {}'.format(labelled_level2.level2taxon.nunique()))

#count the number of taxons per content item into new column
labelled_level2['num_taxon_per_content'] = labelled_level2.groupby(["content_id"])['content_id'].transform("count")

# **** RESHAPE data long -> wide by taxon *******
# ***********************************************

# reshape to wide per taxon and keep the combined text so indexing is consistent when splitting X from Y

multilabel = (labelled_level2.pivot_table(index=['content_id',
												 'combined_text',
												 'title',
												 'description'
												 ],
										  columns='level2taxon_code',
										  values='num_taxon_per_content')
			  )

print('labelled_level2 shape: {}'.format(labelled_level2.shape))
print('pivot table shape (no duplicates): {} '.format(multilabel.shape))

# ******* Metadata ***************
# ********************************

#extract content_id index to df
meta_df = pd.DataFrame(balanced_df['content_id'])
meta_varlist = ['document_type',
		'first_published_at',
		'publishing_app',
		'primary_publishing_organisation']

for meta_var in meta_varlist:
	meta_df[meta_var] = meta_df['content_id'].map(
		dict(zip(labelled_level2['content_id'], labelled_level2[meta_var])))

# convert nans to empty strings for labelencoder types
meta_df = meta_df.replace(np.nan, '', regex=True)


def to_cat_to_hot(metavar):
	"""one hot encode each metavar"""


encoder = LabelEncoder()
metavar_cat = column + "_cat"  # get categorical codes into new column
meta_df[metavar_cat] = encoder.fit_transform(meta_df[metavar])
tf.cast(meta_df[metavar_cat], tf.float32)

return to_categorical(meta_df[metavar_cat])

dict_of_onehot_encodings = {}
for metavar in meta_varlist:
	if metavar != "first_published_at":
		print(metavar)
		dict_of_onehot_encodings[metavar] = to_cat_to_hot(metavar)

# First_published_at:
# Convert to timestamp, then scale between 0 and 1 so same weight as binary vars
meta_df['first_published_at'] = pd.to_datetime(meta_df['first_published_at'])
first_published = np.array(meta1['first_published_at']).reshape(meta1['first_published_at'].shape[0], 1)

scaler = MinMaxScaler()
first_published_scaled = scaler.fit_transform(first_published)

last_year = np.where(
	(np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
	< np.timedelta64(1, 'Y'), 1, 0)

last_2years = np.where(
	(np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
	< np.timedelta64(2, 'Y'), 1, 0)

last_5years = np.where(
	(np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
	< np.timedelta64(5, 'Y'), 1, 0)

olderthan5 = np.where(
	(np.datetime64('today', 'D') - first_published).astype('timedelta64[Y]')
	> np.timedelta64(5, 'Y'), 1, 0)

meta = np.concatenate((dict_of_encodings['document_type'],
					   dict_of_encodings['primary_publishing_organisation'],
					   dict_of_encodings['publishing_app'],
					   first_published_scaled,
					   last_year,
					   last_2years,
					   last_5years,
					   olderthan5),
					  axis=1)

# **** TOKENIZE TEXT ********************
# ************************************

# Load tokenizers, fitted on both labelled and unlabelled data from file
# created in clean_content.py

tokenizer_combined_text = load_tokenizer_from_file("combined_text_tokenizer.json")
tokenizer_title = load_tokenizer_from_file("title_tokenizer.json")
tokenizer_description = load_tokenizer_from_file("description_tokenizer.json")

# Prepare combined text data for input into embedding layer

combined_text_sequences = tokenizer_combined_text.texts_to_sequences(
	balanced_df['combined_text']
)

combined_text_sequences_padded = pad_sequences(
	combined_text_sequences,
	maxlen=MAX_SEQUENCE_LENGTH,
	padding='post', truncating='post'
)

# prepare title and description matrices, 
# which are one-hot encoded for the 10,000 most common words
# to be fed in after the flatten layer (through fully connected layers)

title_sequences = tokenizer_title.texts_to_sequences(
	balanced_df['title']
)

title_onehot = tokenizer_title.sequences_to_matrix(title_sequences)

description_sequences = tokenizer_title.texts_to_sequences(
	balanced_df['description']
)

description_onehot = tokenizer_title.sequences_to_matrix(description_sequences)

# ******* TRAIN/DEV/TEST SPLIT DATA ****************
# **************************************************

# - Training data = 80%
# - Development data = 10%
# - Test data = 10%
nb_samples_before_resampling =
nb_samples_after_resampling = balanced_df.shape[0]
nb_extra_resamples = nb_samples_after_resampling - nb_samples_before_resampling


def split(data_to_split, split_indices):
	"""split data along axis=0 (rows) at indices designated in split_indices"""


list_of_split_data_subsets = []
for (start, end) in split_indices:
	list_of_split_data_subsets.append(data_to_split[start:end])
return tuple(list_of_split_data_subsets)
