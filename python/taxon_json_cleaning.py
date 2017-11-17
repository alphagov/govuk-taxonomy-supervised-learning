"""Extract taxon hierarchy from taxons.json
"""
# coding: utf-8

import json
import io
import os
import logging
import logging.config
import numpy as np
import pandas as pd

# Get data file locations

DATADIR = os.getenv('DATADIR')
DATADIR = os.getenv('DATADIR')
FILENAME = 'taxons_example.json'
# Convert to uri to satisfy pd.read_json
DATAPATH = os.path.join(DATADIR, FILENAME)
# Assert that the file exists

assert os.path.exists(DATAPATH), "File doesn't exist"

DATAPATH = pathlib.Path(DATAPATH).as_uri()


print('Data is being loaded from %s.' % DATAPATH)
# Setup pipeline logging

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('pipeline')

logger.info('Importing data from %s.', 'data/taxons.json')

taxons = pd.read_json(
    'file:///data/taxons.json',
    orient='table',
    typ='frame',
    dtype=True,
    convert_axes=True,
    convert_dates=True,
    keep_default_dates=True,
    numpy=False,
    precise_float=False,
    date_unit=None
)

logger.info('%s taxons loaded', taxons.shape[0])

taxons_notnan = taxons.where((pd.notnull(taxons)), None)
child_dict = dict(zip(taxons_notnan['content_id'], taxons_notnan['parent_content_id']))

# Define a function that recursively generates the child -> tree taxon path

def ancestors(parent_content_id):
    """
    Recursively generate child -> taxon path
    """
    parent = child_dict[parent_content_id]

    if parent is None:
        out = []
    else:
        out = [parent] + ancestors(parent)
    return out

# Create dictionary containg taxon id (content_id) and the taxon path 
# within taxonomy

logger.info('Iterating through child_dict')

dict_taxonpath = {
    parent_content_id: ancestors(parent_content_id)
    for parent_content_id in child_dict.keys()
}

# Convert dict to pandas dataframe
logger.info('Convert child_dict to pandas dataframe.')


df_taxonpath = pd.DataFrame(
    dict_taxonpath.items(), 
    columns=['content_id', 'taxonpath']
)

logger.info('The longest taxonpath is %s.', max(df_taxonpath['taxonpath'].map(len)))


#Get this taxonpath list into separate columns per taxon, within the taxon path, reverse the order of the list so Left = higher up tree

df_split_taxonpath_to_cols = pd.concat([df_taxonpath['content_id'], df_taxonpath['taxonpath'].apply(pd.Series).loc[:,::-1]], axis = 1)
df_split_taxonpath_to_cols.columns = ['content_id', 'level1', 'level2', 'level3', 'level4'] #hard coded so think about what to do if taxonomy gets deeper


# In[ ]:


# Move non empty cells to left in grouped columns pandas: https://stackoverflow.com/questions/39361839/move-non-empty-cells-to-left-in-grouped-columns-pandas/39362818#39362818
# pushna Pushes all null values to the end of the series
# coltype Uses regex to extract the non-numeric prefix from all column names

def pushna(s):
    notnull = s[s.notnull()]
    isnull = s[s.isnull()]
    values = notnull.append(isnull).values
    return pd.Series(values, s.index)



coltype = df_split_taxonpath_to_cols.columns.to_series().str.extract(r'(\D*)', expand=False)

df_split_taxonpath_ordered = df_split_taxonpath_to_cols.groupby(coltype, axis=1).apply(lambda df_split_taxonpath_to_cols: df_split_taxonpath_to_cols.apply(pushna, axis=1))


# In[ ]:


#lookup dictionary to get titles from the content_ids.
#Although content_ids remain gold standard data quality for manipulation, titles will be used during descriptivve analysis so both will be retained for now

id_tit_dict = dict(zip(df_notnan['content_id'], df_notnan['title']))


# In[ ]:


#Pull titles into the content_id df, using the lookup dict
#hard coded so think about what to do if taxonomy gets deeper

df_split_taxonpath_ordered['contenttitle'] = df_split_taxonpath_ordered['content_id'].map(id_tit_dict)
df_split_taxonpath_ordered['level1taxon'] = df_split_taxonpath_ordered['level1'].map(id_tit_dict)
df_split_taxonpath_ordered['level2taxon'] = df_split_taxonpath_ordered['level2'].map(id_tit_dict)
df_split_taxonpath_ordered['level3taxon'] = df_split_taxonpath_ordered['level3'].map(id_tit_dict)
df_split_taxonpath_ordered['level4taxon'] = df_split_taxonpath_ordered['level4'].map(id_tit_dict)


# In[ ]:


# use merge to get the base path

df_taxons = pd.merge(
    df, 
    df_split_taxonpath_ordered, 
    how = 'left', 
    on = 'content_id', 
    indicator = True
)


# In[21]:


list(df_taxons.columns.values)


# In[22]:


df_taxons.drop(['parent_content_id', 'contenttitle', '_merge', 'level1_x'], axis=1, inplace=True)
df_taxons.rename(columns={'title': 'taxon_name', 'level1_y': 'level1tax_id', 'level2': 'level2tax_id', 'level3': 'level3tax_id', 'level4': 'level4tax_id'}, inplace=True)


# In[241]:


list(df_taxons.columns.values)


# ## Exploring the top levels

# In[ ]:


# Check how many top level taxons there are. 
# A top level taxon should have no parent_content_id

df['level1'] = df.where(df['parent_content_id'].isnull()).loc[:,'title']
df['level1'].value_counts()


# In[ ]:


len(set(df['level1']))


# In[246]:


level1 = df[df['parent_content_id'].isnull()]
level1.drop(['base_path', 'parent_content_id', 'level1'], axis=1, inplace=True)
level1


# In[247]:


level1.shape


# ## Import CONTENT data

# In[248]:


# Upload content.json from 'Data science/data'
# This is copied from the colaboratory template

file_id = '1kBHRUkOVHppGRro79Qkz-_k_3Id4BOXq' # This id relates to content.json

request = drive_service.files().get_media(fileId=file_id)
downloaded = io.BytesIO()
downloader = MediaIoBaseDownload(downloaded, request)
done = False
while done is False:
  # _ is a placeholder for a progress object that we ignore.
  # (Our file is small, so we skip reporting progress.)
  _, done = downloader.next_chunk()
    
downloaded.seek(0)


# In[ ]:


# Downloaded json is now a character string

json_string = downloaded.read()

#download the taxon data from content store for all links which are taxons
df_content = pd.read_json(
    json_string, 
    orient='table', 
    typ='frame', 
    dtype=True, 
    convert_axes=True, 
    convert_dates=True, 
    keep_default_dates=True, 
    numpy=False, 
    precise_float=False, 
    date_unit=None
)


# In[258]:


df_content.shape


# In[251]:


df_content.head()


# In[252]:


list(df_content.columns.values)


# In[254]:


type(df_content['details'][0])


# In[255]:


df_content['details'][0]


# In[256]:


df_content['details'][100]


# In[257]:


df_content['details'][10000]


# ## Plan
# - Sort out text variable, will be a concatenation of title, description and details->body
# - Get long by taxon
# - many-to-one merge
# - clean up column names
# 
# - check missings
# - check dupes
# 
# - Do some QA of merges and reshapes. e.g. shapes before/after
# 
# - Export in format for EDA
# 
