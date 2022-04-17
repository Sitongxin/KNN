
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report


# In[2]:


df_train = pd.read_csv('train.csv')[['text', 'label']]
df_val = pd.read_csv('val.csv')[['text', 'label']]
df_test = pd.read_csv('test.csv')[['text', 'label']]


# In[3]:


vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df_train.text.values)
sequences = tokenizer.texts_to_sequences(df_train.text.values)
X_train = pad_sequences(sequences, maxlen=100)


# In[4]:


sequences = tokenizer.texts_to_sequences(df_test.text.values)
X_test = pad_sequences(sequences, maxlen=100)


# In[5]:


sequences = tokenizer.texts_to_sequences(df_val.text.values)
X_val = pad_sequences(sequences, maxlen=100)


# In[6]:


y_train = df_train.drop(['text'], axis=1).values
y_test = df_test.drop(['text'], axis=1).values
y_val = df_val.drop(['text'], axis=1).values


# svm

# In[7]:


from sklearn import svm


# In[8]:


clf = svm.SVC()
clf.fit(X_train, y_train.ravel())


# In[9]:


y_pred = clf.predict(X_test)


# In[10]:


print(classification_report(y_test.ravel(), y_pred))


# In[11]:


svm_report = pd.DataFrame(classification_report(y_test.ravel(), y_pred, output_dict=True)).T
svm_report.to_csv('report_svm.csv')


# 随机森林

# In[12]:


from sklearn.ensemble import RandomForestClassifier


# In[13]:


rfc = RandomForestClassifier(max_depth=20, random_state=0)
rfc.fit(X_train, y_train.ravel())


# In[15]:


y_pred = rfc.predict(X_test)
print(classification_report(y_test.ravel(), y_pred))


# In[16]:


Forest_report = pd.DataFrame(classification_report(y_test.ravel(), y_pred, output_dict=True)).T
Forest_report.to_csv('report_Forest.csv')

