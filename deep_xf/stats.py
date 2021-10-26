#! /usr/bin/env python

"""
@author: Ajay Arunachalam
Created on: 25/10/2021
Goal: Explainable Nowcasting with Dynamic Factor Model based on EM algorithm
Version: 0.0.1
"""

import statsmodels.api as sm
import pandas as pd
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt

class EMModel():
	'''
	Expectation–maximization algorithm
	'''
	def __init__(self, df, model, FORECAST_PERIOD):
		super(EMModel, self).__init__()
		self.df = df
		self.fp = FORECAST_PERIOD
		self.mod = model
	def predict(self):
		self.mod = sm.tsa.DynamicFactorMQ(self.df)
		print(self.mod.summary())
		self.res = self.mod.fit(disp=5)
		#print(self.res.forecast(steps=self.fp))
		pred = self.res.forecast(steps=self.fp)
		return pd.DataFrame(pred)

	def get_stats_model(model):
		models = {
			"em": EMModel,

		}
		return models.get(model.lower())


	def make_future_df(df, ts, period:int, fq:str):
		if str(fq) == 'h' or str(fq) == 'H':
			#last_date = df[ts].max()
			last_date = df.index.max()
			dates = pd.date_range(start=last_date, periods=period + 1, freq=fq) # Add 1 incase last date is included
			dates = dates[dates > last_date]  # Drop start if equals last_date
			dates = dates[:period]  # Return correct number of periods
			ff = pd.DataFrame({ts:dates})
			ff.set_index(ts, inplace=True)
		elif str(fq) == 'm' or str(fq) == 'M':
			last_date = df.index.max()
			dates = pd.date_range(start=last_date, periods=period + 1, freq=fq) # Add 1 incase last date is included
			dates = dates[dates > last_date]  # Drop start if equals last_date
			dates = dates[1:period+1]  # Return correct number of periods
			ff = pd.DataFrame({ts:dates})
			ff.set_index(ts, inplace=True)
		return ff

	def plot_nowcast(ff, fc, FORECAST_PERIOD):
		# Create the plotly figure
		actual = go.Scatter(
		  x=ff.index[:-FORECAST_PERIOD],
		  y=ff.iloc[:-FORECAST_PERIOD,0],
		  mode = 'markers',
		  marker = {
			'color': '#fffaef',
			'size': 1,
			'line': {
			  'color': '#000000',
			  'width': .15
			}
		  },
		  name = 'Actual'
		)

		nowcast = go.Scatter(
		x=ff.index[-FORECAST_PERIOD:],
		  y=ff.iloc[-FORECAST_PERIOD:,0],
		  mode = 'lines',
		  marker = {
		'color': '#3bbed7'
		  },
		  line = {
			'width': 3
		  },
		  name = 'Nowcast',
		)

		data = [actual, nowcast]

		layout = dict(
			title="Nowcast vs Historical Values for the dataset",
				xaxis=dict(title="Time", ticklen=5, zeroline=False),
				yaxis=dict(title=f'{fc}', ticklen=5, zeroline=False),
		)

		fig = dict(data=data, layout=layout)
		iplot(fig)



	def nowcast(df, ts, fc, period:int, fq:str, forecast_window, select_model):
		ff_df = EMModel.make_future_df(df, ts, period, fq)
		model = EMModel.get_stats_model(str(select_model)) # 'em'

		em = EMModel(df=df, model=model, FORECAST_PERIOD=forecast_window)
		em_data = em.predict()
		merged_data = pd.concat([ff_df.reset_index(), em_data.reset_index()], axis=1)
		merged_data = merged_data.drop(columns=['index'])
		merged_data = merged_data.set_index(ts)
		#print(f' Merged: {merged_data}')
			
		nowcast_full_data = pd.concat([df.reset_index(), merged_data.reset_index()], axis=0)
		nowcast_full_data = nowcast_full_data.set_index(ts)
		nowcast_full_data = nowcast_full_data.rename(columns={'value':fc})
		#print(f'Nowcast Merged: {nowcast_full_data}')
		EMModel.plot_nowcast(nowcast_full_data, fc, FORECAST_PERIOD=period)
		nowcast_full_data.to_csv('./nowcast_full_data.csv', encoding='utf-8')

		return nowcast_full_data, merged_data

	def explainable_nowcast(df, ff, fc, specific_prediction_sample_to_explain:int, input_label_index_value, num_labels:int):

		"""
		Understand, interpret, and trust the results of the deep learning models at individual/samples level and multiple columns
		"""
		import shap
		import numpy as np
		import pandas as pd
		from keras.models import Sequential
		from keras.layers import Dense
		import ipywidgets as widgets

		data = pd.concat([df, ff], axis=0)
		data.rename(columns={"value":fc}, inplace=True)

		def X_y_split(df, target_col):
			y = df[[target_col]]
			X = df.drop(columns=[target_col])
			return X, y

		X, Y = X_y_split(data, fc)

		# Get the number of inputs and outputs from the dataset
		n_inputs, n_outputs = X.shape[1], Y.shape[1]

		def get_model(n_inputs, n_outputs):
			model_nn = Sequential()
			model_nn.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
			model_nn.add(Dense(n_outputs, kernel_initializer='he_uniform'))
			model_nn.compile(loss='mae', optimizer='adam')
			return model_nn

		model_nn = get_model(n_inputs, n_outputs)

		# model_nn.fit(X.iloc[10:,:].values, Y, epochs=30)

		# model_nn.evaluate(x = X.iloc[10:,:].values, y = Y)

		model_nn.fit(X.iloc[:df.shape[0],:].values, Y.iloc[:df.shape[0]], epochs=30)

		model_nn.evaluate(x = X.iloc[:df.shape[0],:].values, y = Y.iloc[:df.shape[0]])

		XpredictInputData = X.iloc[specific_prediction_sample_to_explain,:] # X[specific_prediction_sample_to_explain,:]

		if (XpredictInputData.ndim == 1):
			XpredictInputData = np.array([XpredictInputData])

		print(model_nn.predict(XpredictInputData)) # 0:1

		'''
		Here we take the Keras model trained above and explain why it makes different predictions on individual samples.

		Set the explainer using the Kernel Explainer (Model agnostic explainer method form SHAP).
		'''
		explainer = shap.KernelExplainer(model = model_nn.predict, data = X.head(50), link = "identity") # data = X[0:50]

		'''
		Get the Shapley value for a single example.
		'''
		# Set the index of the specific example to explain

		shap_value_single = explainer.shap_values(X = X.iloc[specific_prediction_sample_to_explain,:], nsamples = 100)  # X[specific_prediction_sample_to_explain,:]

		'''
		Display the details of the single example
		'''
		print(X.iloc[specific_prediction_sample_to_explain,:]) 
		'''
		Choose the label/output/target to run individual explanations on:

		Note: The dropdown menu can easily be replaced by manually setting the index on the label to explain.
		'''
		# Create the list of all labels for the drop down list
		#label_cols = ['window_diff_0', 'window_diff_1', 'window_diff_2', 'window_diff_3', 'window_diff_4', 'window_diff_5', 'window_diff_6']

		#label_cols = ['window_diff_'+str(i) for i in range(num_labels)]
		label_cols = [f'{fc}_'+str(i) for i in range(num_labels)]
		#print(label_cols)
		df_labels = pd.DataFrame(data = Y, columns = label_cols)
		#df_labels.to_csv('./y_labels.csv')
		list_of_labels = df_labels.columns.to_list() # Y.columns.to_list()

		# Create a list of tuples so that the index of the label is what is returned
		tuple_of_labels = list(zip(list_of_labels, range(len(list_of_labels))))

		# Create a widget for the labels and then display the widget
		current_label = widgets.Dropdown(options=tuple_of_labels,
									  value=input_label_index_value,
									  description='Select Label:'
									  )

		# Display the dropdown list (Note: access index value with 'current_label.value')
		print(current_label)
		#Dropdown(description='Select Label:', options=(('labels_01', 0), ('labels_02', 1), ('labels_03', 2), etc

		'''
		Plot the force plot for a single example and a single label/output/target
		'''
		print(f'Current label Shown: {list_of_labels[current_label.value]}')

		# print the JS visualization code to the notebook
		shap.initjs()

		shap.force_plot(base_value = explainer.expected_value[current_label.value],
						shap_values = shap_value_single[current_label.value], 
						features = X.iloc[specific_prediction_sample_to_explain,:] # X_idx:X_idx+1
						)

		'''
		Create the summary plot for a specific output/label/target.
		'''
		# Note: We are limiting to the first 50 training examples since it takes time to calculate the full number of sampels
		shap_values = explainer.shap_values(X = X.iloc[0:50,:], nsamples = 100) # X[0:50,:]

		print(f'Current Label Shown: {list_of_labels[current_label.value]}\n')

		# print the JS visualization code to the notebook
		shap.initjs()

		shap.summary_plot(shap_values = shap_values[current_label.value],
				  features = X.iloc[0:50,:],
				  plot_type="bar", # X[0:50,:]
				  show=False
				  )

		plt.savefig('./nowcast_model_summary_plot.png', dpi=150, bbox_inches='tight')

		'''
		Force Plot for the first 50 individual examples.
		'''
		print(f'Current Label Shown: {list_of_labels[current_label.value]}\n')

		# print the JS visualization code to the notebook
		shap.initjs()

		shap.force_plot(base_value = explainer.expected_value[current_label.value],
						shap_values = shap_values[current_label.value], 
						features = X.iloc[0:50,:] # X[0:50,:]
						)