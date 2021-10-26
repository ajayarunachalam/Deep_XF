#! /usr/bin/env python

"""
@author: Ajay Arunachalam
Created on: 06/10/2021
Helper and Utility functions for forecasting model pipeline
Version: 0.0.3
"""
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot
import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .forecast_ml import *
from .forecast_ml_extension import *
from .stats import *
import seaborn as sns
from random import *
get_ipython().run_line_magic('matplotlib', 'inline')

class Helper:

	def get_variable(df, timestamp_col, forecasting_col):
		orig_df = df.copy(deep=True)
		df = df.set_index([timestamp_col])
		df = df.rename(columns={forecasting_col: 'value'})

		df.index = pd.to_datetime(df.index)
		if not df.index.is_monotonic:
			df = df.sort_index()
		return df, orig_df


	def predictor_outcome_split(df, target_col):
		#y = df.iloc[:,target_col]
		#X = df.iloc[:,target_col+1:df.shape[1]]
		y = df[[target_col]]
		X = df.drop(columns=[target_col])
		return X, y

	def train_val_test_split(df, target_col, test_ratio):
		val_ratio = test_ratio / (1 - test_ratio)
		#val_ratio = 0.25 #test_ratio / (1 - test_ratio)
		X, y = Helper.predictor_outcome_split(df, target_col)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
		print("Train-Val-Test Split")
		print(f'Predictors: Train-{(X_train.shape)}, Val-{(X_val.shape)}, Test-{(X_test.shape)}')
		print(f'Response: Train-{(y_train.shape)}, Val-{(y_val.shape)}, Test-{(y_test.shape)}')
		return X_train, X_val, X_test, y_train, y_val, y_test

	def get_scaler(scaler):
		scalers = {
		"minmax": MinMaxScaler,
		"standard": StandardScaler,
		"maxabs": MaxAbsScaler,
		"robust": RobustScaler,
		}
		return scalers.get(scaler.lower())()

	def apply_transformation(X_train, X_val, X_test, y_train, y_val, y_test, scaler):

		def get_scaler(scaler):
			scalers = {
				"minmax": MinMaxScaler,
				"standard": StandardScaler,
				"maxabs": MaxAbsScaler,
				"robust": RobustScaler,
			}
			return scalers.get(scaler.lower())()

		scaler = get_scaler(scaler)
		X_train_arr = scaler.fit_transform(X_train)
		X_val_arr = scaler.transform(X_val)
		X_test_arr = scaler.transform(X_test)

		y_train_arr = scaler.fit_transform(y_train)
		y_val_arr = scaler.transform(y_val)
		y_test_arr = scaler.transform(y_test)

		#X_full_arr = np.concatenate(X_train_arr, X_val_arr, X_test_arr)
		#y_full_arr = np.concatenate(y_train_arr, y_val_arr, y_test_arr)
		return X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr, scaler

	def apply_transformation_forecast(X, scaler):
		scaler = Helper.get_scaler(scaler)
		X_arr = scaler.fit_transform(X)
		return X_arr

	def prepare_pytorch_data_forecast_df(X_arr):
		unseen_loader = torch.Tensor(X_arr)
		print(f'Tensor size: {unseen_loader.size(0)}')
		return unseen_loader

	def prepare_pytorch_data(X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr, batch_size):
		import torch
		from torch.utils.data import TensorDataset, DataLoader
		train_features = torch.Tensor(X_train_arr)
		train_targets = torch.Tensor(y_train_arr)
		val_features = torch.Tensor(X_val_arr)
		val_targets = torch.Tensor(y_val_arr)
		test_features = torch.Tensor(X_test_arr)
		test_targets = torch.Tensor(y_test_arr)
		
		train = TensorDataset(train_features, train_targets)
		val = TensorDataset(val_features, val_targets)
		test = TensorDataset(test_features, test_targets)
		
		train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
		val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
		test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
		test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
		
		return train_loader, val_loader, test_loader, test_loader_one

	def get_stats_model(model):
		models = {
			"em": Stats_Models.EMModel,

		}
		return models.get(model.lower())

	def get_model(model, model_params):
		models = {
			"rnn": Forecasting_Models.RNNModel,
			"lstm": Forecasting_Models.LSTMModel,
			"gru": Forecasting_Models.GRUModel,
			"cnn": Forecasting_Models.CNNModel,
			"deepcnn": Forecasting_Models.DeepCNNModel,
			
		}
		return models.get(model.lower())(**model_params)

	def inverse_transform(scaler, df, columns):
		for col in columns:
			df[col] = scaler.inverse_transform(df[col])
		return df


	def format_predictions(predictions, values, df_test, scaler):
		vals = np.concatenate(values, axis=0).ravel()
		preds = np.concatenate(predictions, axis=0).ravel()
		df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
		df_result = df_result.sort_index()
		df_result = Helper.inverse_transform(scaler, df_result, [["value", "prediction"]])
		return df_result

	def forecast_bias():
		pass

	def smape(actual, prediction):
		dividend= np.abs(np.array(actual) - np.array(prediction))
		denominator = np.array(actual) + np.array(prediction)
		return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=denominator!=0, casting='unsafe'))

	def calculate_metrics(df):
		metrics_one = {'mae' : mean_absolute_error(df.value, df.prediction),
						  'rmse' : mean_squared_error(df.value, df.prediction) ** 0.5,
						   'fc_bias' : (np.sum(np.array(df.value) - np.array(df.prediction)) * 1.0/len(df.value)),
						   'mape' : np.mean((np.abs((df.value-df.prediction) / df.value)) * 100) ,
					}
		
		print("Mean Absolute Error (MAE):       ", metrics_one["mae"])
		print("Root Mean Squared Error (RMSE):   ", metrics_one["rmse"])
		print("Forecast bias:   		  ", metrics_one["fc_bias"])
		print("Mean Absolute Percentage Error (MAPE):   ", metrics_one["mape"])
	

		metrics_two = {'r2' : r2_score(df.value, df.prediction),
						  'evs': explained_variance_score(df.value, df.prediction),
						  'rmsre' : np.sqrt(np.mean(((df.value-df.prediction)/df.value)**2)),
						  'smape': Helper.smape(df.value, df.prediction),
						  'r': np.corrcoef(df.value, df.prediction)[0, 1],}

		print("R^2 Score:                 ", metrics_two["r2"])
		print("Explained Variance Score:  ", metrics_two["evs"])
		print("Root Mean Squared Relative Error (RMSRE): ", metrics_two["rmsre"])
		print("Symmetric Mean Absolute Percentage Error (sMAPE):   ", metrics_two["smape"])
		print("pcc coefficient: ", metrics_two["r"])

		return metrics_one, metrics_two

	def plot_metrics(metrics_one, metrics_two):
		# Visualize Metrics as bar plot with sns
		data_metrics_one = pd.DataFrame.from_dict(metrics_one.items())

		fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
		axes = sns.barplot(data = data_metrics_one, x = 0, y=1, hue=0)
		axes.set(xlabel = 'Metrics', ylabel='Value', title='Model Peformance wrt. their corresponding metrics')
		fig.savefig('./metric_plot_1.png')

		data_metrics_two = pd.DataFrame.from_dict(metrics_two.items())

		fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
		axes = sns.barplot(data = data_metrics_two, x = 0, y=1, hue=0)
		axes.set(xlabel = 'Metrics', ylabel='Value', title='Model Peformance wrt. their corresponding metrics')
		fig.savefig('./metric_plot_2.png')

	# def make_future_df(df, ts, period:int, fq:str):
	# 	last_date = df[ts].max()
	# 	dates = pd.date_range(start=last_date, periods=period + 1, freq=fq)
	# 	dates = dates[dates > last_date]  # Drop start if equals last_date
	# 	dates = dates[:period]  # Return correct number of periods 
	# 	return pd.Dataframe({ts:dates})

	def make_future_df(df, ts, period:int, fq:str): #ts
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

	def make_future_dataframe(self, periods, freq='D', include_history=True):
		
		"""Simulate the trend using the extrapolated generative model.
		Parameters
		----------
		periods: Int number of periods to forecast forward.
		freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.
		include_history: Boolean to include the historical dates in the data
			frame for predictions.
		Returns
		-------
		pd.Dataframe that extends forward from the end of self.history for the
		requested number of periods.
		"""
		if self.history_dates is None:
			raise Exception('Model has not been fit.')
		last_date = self.history_dates.max()
		dates = pd.date_range(
			start=last_date,
			periods=periods + 1,  # An extra in case we include start
			freq=freq)
		dates = dates[dates > last_date]  # Drop start if equals last_date
		dates = dates[:periods]  # Return correct number of periods

		if include_history:
			dates = np.concatenate((np.array(self.history_dates), dates))

		return pd.DataFrame({'ds': dates})


	# def forecast_window_inference(predictions, df, scaler):
	# 	preds = np.concatenate(predictions, axis=0).ravel()
	# 	df_result = pd.DataFrame(data={"prediction": preds}, index=df_test.head(len(df)).index)
	# 	df_result = df_result.sort_index()
	# 	df_result = Helper.inverse_transform(scaler, df_result, [["prediction"]])
	# 	return df_result

	def forecast_window_inference(predictions, df, scaler):
		preds = np.concatenate(predictions, axis=0).ravel()
		df_result = pd.DataFrame(data={"value": preds}, index=df.head(len(preds)).index)
		df_result = df_result.sort_index()
		df_result = Helper.inverse_transform(scaler, df_result, [["value"]])
		return df_result

	def save_final_data(df, ff, ts, fc):
		forecasted_data = pd.concat([df.reset_index(), ff.reset_index()], axis=0)
		forecasted_data.set_index(ts, inplace=True)
		forecasted_data = forecasted_data.rename(columns={'value':fc})
		fig = forecasted_data.iloc[df.shape[0]:].plot().get_figure()
		plt.tight_layout()
		fig.savefig("./model_forecast.png", dpi=150, bbox_inches='tight')
		forecasted_data.to_csv('./model_full_data.csv', encoding='utf-8')
		return forecasted_data

	
	def compare_ml_models_and_plot():
		# Install regressormetricgraphplot package from terminal or notebook
		'''
		Terminal: 
		$ pip install regressormetricgraphplot
		 OR
		$ git clone https://github.com/ajayarunachalam/RegressorMetricGraphPlot
		$ cd RegressorMetricGraphPlot
		$ python setup.py install
		Notebook:
		!git clone https://github.com/ajayarunachalam/RegressorMetricGraphPlot.git
		cd RegressorMetricGraphPlot/
		Just replace the line 'from CompareModels import *' with 'from regressormetricgraphplot import CompareModels' 
		'''
		# Now, let us check how machine learning algorithms perform on this dataset in comparison to the build neural network
		from regressioncomparemetricplot import CompareModels

		# Linear Regression 

		# Fitting training set to linear regression model
		lr = LinearRegression(n_jobs=-1)
		lr.fit(X_train, y_train)

		# Predicting the house price
		y_pred = lr.predict(X_test)

		# Metrics
		print(f'R2_nd_RMSE LR MODEL: {CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred)}')

		plot = CompareModels()

		plot.add(model_name='Linear Regression', y_test=y_test, y_pred=y_pred)

		plot.show(figsize=(10, 5))

		# Fitting Random Forest model to the dataset
		rfr = RandomForestRegressor(n_estimators=10, random_state=10, n_jobs=-1)
		rfr.fit(X_train, y_train)

		# Predicting the outcome
		y_pred = rfr.predict(X_test)

		print(f'R2_nd_RMSE RF MODEL: {CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred)}')

		plot.add('Random Forest', y_test, y_pred)

		plot.show(figsize=(10, 5))

		xgb = XGBRegressor(n_jobs=4, silent=False, objective='reg:linear',
					   max_depth=3, random_state=10, n_estimators=100,
					   learning_rate=0.3, verbose=True)

		xgb.fit(X_train, y_train)

		# Predicting the outcome
		y_pred = xgb.predict(X_test)

		print(f'R2_nd_RMSE XGB MODEL: {CompareModels.R2AndRMSE(y_test=y_test, y_pred=y_pred)}')

		plot.add('XGBoost', y_test, y_pred)

		plot.show(figsize=(10, 5))

	def build_baseline_model(df, test_ratio, target_col):
		X, y = Helper.predictor_outcome_split(df, target_col)
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=test_ratio, shuffle=False
		)
		model = LinearRegression()
		model.fit(X_train, y_train)
		prediction = model.predict(X_test)

		result = pd.DataFrame(y_test)
		result["prediction"] = prediction
		result = result.sort_index()

		return result

	def plot_predictions(df_result, df_baseline):
		# Create the plotly figure
		data = []
		value = go.Scatter(
			x=df_result.index,
			y=df_result.value,
			mode="lines",
			name="actual values",
			marker=dict(),
			text=df_result.index,
			line=dict(color="rgba(0,0,0, 0.3)"),
		)
		data.append(value)

		baseline = go.Scatter(
			x=df_baseline.index,
			y=df_baseline.prediction,
			mode="lines",
			line={"dash": "dot"},
			name='linear regression',
			marker=dict(),
			text=df_baseline.index,
			opacity=0.8,
		)
		data.append(baseline)
		
		prediction = go.Scatter(
			x=df_result.index,
			y=df_result.prediction,
			mode="lines",
			line={"dash": "dot"},
			name='deep model predictions',
			marker=dict(),
			text=df_result.index,
			opacity=0.8,
		)
		data.append(prediction)
		
		layout = dict(
			title="Predictions vs Actual Values for the dataset",
			xaxis=dict(title="Time", ticklen=5, zeroline=False),
			yaxis=dict(title="Value", ticklen=5, zeroline=False),
		)

		fig = dict(data=data, layout=layout)
		iplot(fig)

	def plot_forecast(ff_result, fc):
		# Create the plotly figure
		data = []
		forecast = go.Scatter(
			x=ff_result.index,
			y=ff_result.value,
			mode="lines",
			line={"dash": "dot"},
			name='Forecasted Results',
			marker=dict(),
			text=ff_result.index,
			opacity=0.8,
			
		)
		data.append(forecast)
		
		layout = dict(
			title="Forecast for the user provided time period",
			xaxis=dict(title="Time", ticklen=5, zeroline=False),
			yaxis=dict(title=f"{fc}", ticklen=5, zeroline=False),
		)

		fig = dict(data=data, layout=layout)
		iplot(fig)

	def plot_nowcast(ff, fc, FORECAST_PERIOD):
		# Create the plotly figure
		actual = go.Scatter(
		  x=ff.index[:-FORECAST_PERIOD],
		  y=ff.iloc[:-FORECAST_PERIOD,0],
		  mode = 'markers',
		  marker = {
			'color': '#fffaef',
			'size': 3,
			'line': {
			  'color': '#000000',
			  'width': .75
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
				yaxis=dict(title="Value", ticklen=5, zeroline=False),
		)

		fig = dict(data=data, layout=layout)
		iplot(fig)



	def explainable_forecast(df, ff, fc, specific_prediction_sample_to_explain:int, input_label_index_value, num_labels:int):

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

		X, Y = Helper.predictor_outcome_split(data, fc)

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

		plt.savefig('./forecast_model_summary_plot.png', dpi=150, bbox_inches='tight')

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
		

class Optimization:

	"""Optimization is a helper class that allows training, validation, prediction.

	Optimization is a helper class that takes model, loss function, optimizer function
	learning scheduler (optional), early stopping (optional) as inputs. In return, it
	provides a framework to train and validate the models, and to predict future values
	based on the models.

	Attributes:
		model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
		loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
		optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
		train_losses (list[float]): The loss values from the training
		val_losses (list[float]): The loss values from the validation
		last_epoch (int): The number of epochs that the models is trained
	"""
	def __init__(self, device, model, loss_fn, optimizer):

		"""
		Args:
			model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
			loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
			optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
		"""
		self.device = device
		self.model = model
		self.loss_fn = loss_fn
		self.optimizer = optimizer
		self.train_losses = []
		self.val_losses = []
		

	
	def train_step(self, x, y):
		"""The method train_step completes one step of training.

		Given the features (x) and the target values (y) tensors, the method completes
		one step of the training. First, it activates the train mode to enable back prop.
		After generating predicted values (yhat) by doing forward propagation, it calculates
		the losses by using the loss function. Then, it computes the gradients by doing
		back propagation and updates the weights by calling step() function.

		Args:
			x (torch.Tensor): Tensor for features to train one step
			y (torch.Tensor): Tensor for target values to calculate losses

		"""
		# Sets model to train mode
		self.model.train().to(self.device)

		# Makes predictions
		yhat = self.model(x).to(self.device)

		# Computes loss
		loss = self.loss_fn(y, yhat)

		# Computes gradients
		loss.backward()

		# Updates parameters and zeroes gradients
		self.optimizer.step()
		self.optimizer.zero_grad()

		# Returns the loss
		return loss.item()

	def train(self, train_loader, val_loader, batch_size, n_epochs, n_features):
		"""The method train performs the model training

		The method takes DataLoaders for training and validation datasets, batch size for
		mini-batch training, number of epochs to train, and number of features as inputs.
		Then, it carries out the training by iteratively calling the method train_step for
		n_epochs times. If early stopping is enabled, then it  checks the stopping condition
		to decide whether the training needs to halt before n_epochs steps. Finally, it saves
		the model in a designated file path.

		Args:
			train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
			val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
			batch_size (int): Batch size for mini-batch training
			n_epochs (int): Number of epochs, i.e., train steps, to train
			n_features (int): Number of feature columns

		"""
		os.makedirs('./model_path/', exist_ok=True)
		path = "./model_path/"
		#os.path.join(path,“filename.pth”)
		#model_path = f'{path}deployed_model/{self.model}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
		#model_path = f'{path}deployed_model/_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'

		for epoch in range(1, n_epochs + 1):
			batch_losses = []
			for x_batch, y_batch in train_loader:
				x_batch = x_batch.view([batch_size, -1, n_features]).to(self.device)
				y_batch = y_batch.to(self.device)
				loss = self.train_step(x_batch, y_batch)
				batch_losses.append(loss)
			training_loss = np.mean(batch_losses)
			self.train_losses.append(training_loss)

			with torch.no_grad():
				batch_val_losses = []
				for x_val, y_val in val_loader:
					x_val = x_val.view([batch_size, -1, n_features]).to(self.device)
					y_val = y_val.to(self.device)
					self.model.eval()
					yhat = self.model(x_val)
					val_loss = self.loss_fn(y_val, yhat).item()
					batch_val_losses.append(val_loss)
				validation_loss = np.mean(batch_val_losses)
				self.val_losses.append(validation_loss)

			if (epoch <= n_epochs) | (epoch % 50 == 0):
				print(
					f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
				)

		#torch.save(self.model.state_dict(), model_path)


	def evaluate(self, test_loader, batch_size=1, n_features=1, ):
		"""The method evaluate performs the model evaluation

		The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
		and number of features as inputs. Similar to the model validation, it iteratively
		predicts the target values and calculates losses. Then, it returns two lists that
		hold the predictions and the actual values.

		Note:
			This method assumes that the prediction from the previous step is available at
			the time of the prediction, and only does one-step prediction into the future.

		Args:
			test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
			batch_size (int): Batch size for mini-batch training
			n_features (int): Number of feature columns

		Returns:
			list[float]: The values predicted by the model
			list[float]: The actual values in the test set.

		"""
		with torch.no_grad():
			predictions = []
			values = []
			for x_test, y_test in test_loader:
				x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
				y_test = y_test.to(self.device)
				self.model.eval()
				yhat = self.model(x_test)
				#predictions.append(yhat.to(self.device).detach().numpy())
				predictions.append(yhat.cpu().detach().numpy())
				#values.append(y_test.to(self.device).detach().numpy())
				values.append(y_test.cpu().detach().numpy())

		return predictions, values

	def predict(self, forecast_loader, batch_size=1, n_features=1, ):
		"""The method performs the model forecasting

		The method takes DataLoaders for the forecast dataset, batch size for mini-batch testing,
		and number of features as inputs. Similar to the model validation, it iteratively
		predicts the target values and calculates losses. Then, it returns two lists that
		hold the predictions and the actual values.

		Note:
			This method assumes that the prediction from the previous step is available at
			the time of the prediction, and only does one-step prediction into the future.

		Args:
			forecast_loader (torch.utils.data.DataLoader): DataLoader that stores future forecast data
			batch_size (int): Batch size for mini-batch training
			n_features (int): Number of feature columns

		Returns:
			list[float]: The values predicted by the model

		"""
		with torch.no_grad():
			predictions = []
			for x in forecast_loader:
				x = x.view([batch_size, -1, n_features]).to(self.device)
				self.model.eval()
				yhat = self.model(x)
				#predictions.append(yhat.to(self.device).detach().numpy())
				predictions.append(yhat.cpu().detach().numpy())
				
		return predictions

	# def predict(self, test_loader, batch_size,  n_features=1,):
	# 	"""
	# 	Window-step prediction into the future 
	# 	"""
	# 	#self.model.eval()
	# 	#for i in range(future_prediction):

	# 	with torch.no_grad():

	# 		predictions_forecast_window = list()
	# 		for i in range(future_prediction):


	# 	# 	for x_test, y_test in test_loader:
	# 	# 		x_test = x_test.view([batch_size, -1, n_features]).to(self.device)
	# 	# 		self.model.eval()
	# 	# 		yhat = self.model(x_test)
	# 	# 		predictions_forecast_window.append(yhat.to(self.device).detach().numpy())
	# 	# return predictions_forecast_window


	def forecast_torch(self, full_loader,  n_steps:int, batch_size=1, n_features=1,):

		predictions = []

		for x, y in full_loader:

			x = torch.roll(x, shifts=1, dims=2)
			x[..., -1, 0] = torch.Tensor(np.array(y)).item(0)
			with torch.no_grad():

				self.model.eval()
				for _ in range(n_steps):
					x = x.view([batch_size, -1, n_features]).to(device)
					yhat = self.model(x)
					yhat = yhat.to(device).detach().numpy()
					x = torch.roll(x, shifts=1, dims=2)
					x[..., -1, 0] = yhat.item(0)
					predictions.append(yhat)

		return predictions


	def forecast(self, full_loader,  n_steps:int, batch_size=1, n_features=1,):
		predictions = []
		values = []
		for x, y in full_loader:
			for _ in range(n_steps):
				x = x.view([batch_size, -1, n_features]).to(self.device)
				y = y.to(self.device)
				self.model.eval()
				yhat = self.model(x)
				#yhat = yhat.to(self.device).detach().numpy()
				x = torch.roll(x, shifts=1, dims=2)
				x[..., -1, 0] = yhat.item() #0
				predictions.append(yhat.to(self.device).detach().numpy())
				values.append(y.to(self.device).detach().numpy())

		return predictions, values


	def plot_losses(self):
		"""The method plots the calculated loss values for training and validation
		"""
		plt.plot(self.train_losses, label="Training loss")
		plt.plot(self.val_losses, label="Validation loss")
		plt.legend()
		plt.title("Losses")
		plt.show()
		plt.close()

