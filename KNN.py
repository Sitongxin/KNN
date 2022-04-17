
# coding: utf-8

# ## 安装

# In[1]:


import re
import os
import shutil
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.metrics import classification_report
# from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

# tf.get_logger().setLevel('ERROR')


# # 数据处理

# In[2]:


df_train = pd.read_csv('train.csv')[['text', 'label']]
df_val = pd.read_csv('val.csv')[['text', 'label']]
df_test = pd.read_csv('test.csv')[['text', 'label']]#读取数据


# In[3]:


df_train.head()


# In[4]:


vocabulary_size = 20000#字典大小为20000 
tokenizer = Tokenizer(num_words= vocabulary_size)#对统计单词出现数量后选择次数多的前20000个单词

tokenizer.fit_on_texts(df_train.text.values)# 通过文档列表更新tokenizer的词典
sequences = tokenizer.texts_to_sequences(df_train.text.values)#将train文档中列表转换为向量


# In[5]:


X_train = pad_sequences(sequences, maxlen=100)#使用pad_sequences()使各序列长度为100


# In[6]:


sequences = tokenizer.texts_to_sequences(df_test.text.values)#将text文档中列表转换为向量
X_test = pad_sequences(sequences, maxlen=100)#使用pad_sequences()使各序列长度为100


# In[7]:


sequences = tokenizer.texts_to_sequences(df_val.text.values)#将val文档中列表转换为向量
X_val = pad_sequences(sequences, maxlen=100)


# In[8]:


y_train = df_train.drop(['text'],axis=1).values
y_test = df_test.drop(['text'],axis=1).values
y_val = df_val.drop(['text'],axis=1).values#将‘text’对应的列标签沿着水平的方向依次删掉


# # KNN 模型

# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[10]:


knn = KNeighborsClassifier(n_neighbors=10)


# In[11]:


knn.fit(X_train,y_train.ravel())#fit函数 使用X作为训练数据，y作为目标值来拟合模型


# In[12]:


y_pred = knn.predict(X_test)
y_pred
#预测新样本


# In[13]:


print(classification_report(y_test.ravel(),y_pred))# 使用classification_report函数进行模型评价


# In[14]:


knn_report = pd.DataFrame(classification_report(y_test.ravel(),y_pred,output_dict=True)).T
knn_report


# In[ ]:


knn_report.to_csv('report_knn.csv')#把classification_report输出到csv文件

