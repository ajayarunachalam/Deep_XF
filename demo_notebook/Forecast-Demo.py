#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import io
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")


# In[2]:


from deep_xf.main import *
from deep_xf.dpp import *
from deep_xf.forecast_ml import *
from deep_xf.forecast_ml_extension import *
from deep_xf.stats import *
from deep_xf.utility import *
from deep_xf.denoise import *
from deep_xf.similarity import *


# In[3]:


df = pd.read_csv('../data/PJME_hourly.csv')


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.head(2)


# In[7]:


df.tail(2)


# In[8]:


df


# In[9]:


# set variables
ts, fc = Forecast.set_variable(ts='Datetime', fc='PJME_MW')


# In[10]:


# get variables
model_df, orig_df = Helper.get_variable(df, ts, fc)


# In[11]:


model_df.shape


# In[12]:


model_df.head(1)


# In[13]:


orig_df.shape


# In[14]:


orig_df.head(1)


# # EDA

# In[15]:


ExploratoryDataAnalysis.plot_dataset(df=model_df,title='PJM East (PJME) Region: estimated energy consumption in Megawatts (MW)')


# # Feature Engg

# In[16]:


df_full_features = Features.generate_date_time_features_hour(model_df, ['hour','month','day','day_of_week','week_of_year'])


# In[17]:


df_full_features.columns


# In[18]:


# generating cyclic features
df_full_features = Features.generate_cyclic_features(df_full_features, 'hour', 24, 0)
df_full_features = Features.generate_cyclic_features(df_full_features, 'day_of_week', 7, 0)
df_full_features = Features.generate_cyclic_features(df_full_features, 'month', 12, 1)
df_full_features = Features.generate_cyclic_features(df_full_features, 'week_of_year', 52, 0)


# In[19]:


df_full_features.head(1)


# In[20]:


df_full_features = Features.generate_other_related_features(df=df_full_features)


# In[21]:


df_full_features.head(1)


# In[22]:


df_full_features.shape


# In[23]:


select_model, select_user_path, select_scaler, forecast_window = Forecast.set_model_config(select_model='rnn', select_user_path='./forecast_MY/', select_scaler='minmax', forecast_window=1)


# In[24]:


hidden_dim, layer_dim, batch_size, dropout, n_epochs, learning_rate, weight_decay = Forecast.hyperparameter_config(hidden_dim=64, layer_dim = 3, batch_size=64, dropout = 0.2,                                n_epochs = 30, learning_rate = 1e-3, weight_decay = 1e-6)


# In[25]:


opt, scaler = Forecast.train(df=df_full_features, target_col='value', split_ratio=0.2, select_model=select_model,              select_scaler=select_scaler, forecast_window=forecast_window, batch_size=batch_size,            hidden_dim=hidden_dim, layer_dim=layer_dim,dropout=dropout,              n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay)


# In[26]:


forecasted_data, ff_full_features, ff_full_features_ = Forecast.forecast(model_df, ts, fc, opt, scaler, period=25, fq='h', select_scaler=select_scaler,)


# In[31]:


Helper.explainable_forecast(df_full_features, ff_full_features_, fc, specific_prediction_sample_to_explain=df_full_features.shape[0]+2, input_label_index_value=0, num_labels=1)

