
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
