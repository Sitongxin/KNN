
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report


# In[2]:


df_train = pd.read_csv('train.csv')[['text', 'label']]
df_val = pd.read_csv('val.csv')[['text', 'label']]
df_test = pd.read_csv('test.csv')[['text', 'label']]
#读取数据


# In[3]:


vocabulary_size = 20000#字典大小为20000 
tokenizer = Tokenizer(num_words=vocabulary_size)#对统计单词出现数量后选择次数多的前20000个单词，后面的单词都不做处理。
tokenizer.fit_on_texts(df_train.text.values)# 通过文档列表更新tokenizer的词典。
sequences = tokenizer.texts_to_sequences(df_train.text.values)#将train文档中列表转换为向量
X_train = pad_sequences(sequences, maxlen=100)#使用pad_sequences()使各序列长度为100


# In[4]:


sequences = tokenizer.texts_to_sequences(df_test.text.values)#将test文档中列表转换为向量
X_test = pad_sequences(sequences, maxlen=100)#使用pad_sequences()使各序列长度为100


# In[5]:


sequences = tokenizer.texts_to_sequences(df_val.text.values)#将test文档中列表转换为向量
X_val = pad_sequences(sequences, maxlen=100)#使用pad_sequences()使各序列长度为100


# In[6]:


y_train = df_train.drop(['text'], axis=1).values
y_test = df_test.drop(['text'], axis=1).values
y_val = df_val.drop(['text'], axis=1).values#将‘text’对应的列标签沿着水平的方向依次删掉


# In[7]:


from sklearn.ensemble import RandomForestClassifier


# In[8]:


rfc = RandomForestClassifier(max_depth=20, random_state=0)
rfc.fit(X_train, y_train.ravel())
#调试训练集的随机森林


# In[11]:


y_pred = rfc.predict(X_test)
print(classification_report(y_test.ravel(), y_pred))
#预测测试集结果


# In[12]:


Forest_report = pd.DataFrame(classification_report(y_test.ravel(), y_pred, output_dict=True)).T
Forest_report.to_csv('report_Forest.csv')
#生成字典类型的分类报告 

