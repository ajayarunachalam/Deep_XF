#! /usr/bin/env python

"""
@author: Ajay Arunachalam
Created on: 25/10/2021
Training the forecasting and Nowcasting model
Version: 0.0.1
"""


import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from .utility import *
from .dpp import *
#from .forecast_ml import *
from .forecast_ml_extension import *
from .denoise import *
from .similarity import *
from .gnn_layer import *
from .stats import *


class Forecast:

	ts = globals()
	fc = globals()

	select_model = globals() # Possible values ['rnn','lstm', 'gru', 'em', etc]
	select_user_path = globals() # Provide user_path './forecast_folder/'
	select_scaler = globals() # Possible values ['minmax','standard','maxabs','robust']
	forecast_window = globals() # no. of timesteps/points to be used for the forecasting model and nowcasting period

	hidden_dim = globals()
	layer_dim = globals()
	batch_size = globals()
	dropout = globals()
	n_epochs = globals()
	learning_rate = globals()
	weight_decay = globals()


	def set_variable(**kwargs):
		for key, value in kwargs.items():
			print("{0} = {1}" .format(key,value))

		ts = list(kwargs.values())[0]
		fc = list(kwargs.values())[1]

		return ts, fc

		assert ts == ts
		assert fc == fc

	def set_model_config(**kwargs):

		for key, value in kwargs.items():
			print("{0} = {1}" .format(key,value))

		select_model = list(kwargs.values())[0]
		select_user_path = list(kwargs.values())[1]
		select_scaler = list(kwargs.values())[2]
		forecast_window = list(kwargs.values())[3]

		return select_model, select_user_path, select_scaler, forecast_window

		assert select_model == select_model
		assert select_user_path == select_user_path
		assert select_scaler == select_scaler
		assert forecast_window ==  forecast_window

	def hyperparameter_config(**kwargs):

		for key, value in kwargs.items():
			print("{0} = {1}" .format(key,value))

		hidden_dim = list(kwargs.values())[0]
		layer_dim = list(kwargs.values())[1]
		batch_size = list(kwargs.values())[2]
		dropout = list(kwargs.values())[3]
		n_epochs = list(kwargs.values())[4]
		learning_rate = list(kwargs.values())[5] 
		weight_decay = list(kwargs.values())[6]

		return hidden_dim, layer_dim, batch_size, dropout, n_epochs, learning_rate, weight_decay

		assert hidden_dim == hidden_dim
		assert layer_dim == layer_dim
		assert batch_size == batch_size
		assert dropout == dropout
		assert n_epochs == n_epochs
		assert learning_rate == learning_rate
		assert weight_decay == weight_decay

	def forecast(df, ts, fc, opt, scaler, period:int, fq:str, select_scaler=select_scaler, ):

		ff_df = Helper.make_future_df(df, ts, period, fq)

		print(f'Forecast period dataframe: {ff_df.index}')

		#print(f'Forecast period dataframe: {ff_df.index.hour}')

		#cols=['hour','month','day','day_of_week','week_of_year']

		if str(fq)=='h' or str(fq)=='H':

			ff_full_features = Features.generate_date_time_features_hour(ff_df, ['hour','month','day','day_of_week','week_of_year'])
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'hour', 24, 0)
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'day_of_week', 7, 0)
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'month', 12, 1)
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'week_of_year', 52, 0)

			ff_full_features = Features.generate_other_related_features(df=ff_full_features)

		elif str(fq)=='m' or str(fq)=='M':

			ff_full_features = Features.generate_date_time_features_month(ff_df, ['month','day_of_week','week_of_year'])
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'day_of_week', 7, 0)
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'month', 12, 1)
			ff_full_features = Features.generate_cyclic_features(ff_full_features, 'week_of_year', 52, 0)

			ff_full_features = Features.generate_other_related_features(df=ff_full_features)


		X = ff_full_features
		
		input_dim = len(X.columns)
		#X, y = Helper.predictor_outcome_split(df, target_col)
		X_arr = Helper.apply_transformation_forecast(X, select_scaler)

		unseen_loader = Helper.prepare_pytorch_data_forecast_df(X_arr)

		predictions = opt.predict(
			unseen_loader,
			batch_size=1,
			n_features=input_dim
		)

		ff_result = Helper.forecast_window_inference(predictions, ff_df, scaler)
		print(f'Forecast period predictions: {ff_result}')

		Helper.plot_forecast(ff_result, fc)

		forecasted_data = Helper.save_final_data(df, ff_result, ts, fc)
		ff_full_features_ = pd.concat([ff_result, ff_full_features], axis=1)
		return forecasted_data, ff_full_features, ff_full_features_


	def train(df, target_col, split_ratio:float, select_model=select_model, select_scaler=select_scaler, forecast_window=forecast_window, hidden_dim=hidden_dim, layer_dim=layer_dim, batch_size=batch_size,dropout=dropout, n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay):
		from torch.utils.data import TensorDataset, DataLoader

		X_train, X_val, X_test, y_train, y_val, y_test = Helper.train_val_test_split(df, target_col, split_ratio) #'value', 0.2
		X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr, scaler = Helper.apply_transformation(X_train, X_val, X_test, y_train, y_val, y_test, select_scaler)
		
		train_loader, val_loader, test_loader, test_loader_one = Helper.prepare_pytorch_data(X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr, batch_size=batch_size)
		'''

		scaler = Helper.get_scaler(str(select_scaler)) #'minmax'
		X_train_arr = scaler.fit_transform(X_train)
		X_val_arr = scaler.transform(X_val)
		X_test_arr = scaler.transform(X_test)

		y_train_arr = scaler.fit_transform(y_train)
		y_val_arr = scaler.transform(y_val)
		y_test_arr = scaler.transform(y_test)

		#batch_size = 64

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
		'''

		input_dim = len(X_train.columns)
		# output_dim = 1
		# hidden_dim = 64
		# layer_dim = 3
		# batch_size = 64
		# dropout = 0.2
		# n_epochs = 5
		# learning_rate = 1e-3
		# weight_decay = 1e-6

		output_dim = 1
		hidden_dim = hidden_dim
		layer_dim = layer_dim
		batch_size = batch_size
		dropout = dropout
		n_epochs = n_epochs
		learning_rate = learning_rate
		weight_decay = weight_decay


		model_params = {'input_dim': input_dim,
						'hidden_dim' : hidden_dim,
						'layer_dim' : layer_dim,
						'output_dim' : output_dim,
						'dropout_prob' : dropout}

		model = Helper.get_model(str(select_model), model_params) # 'lstm'

		loss_fn = nn.MSELoss(reduction="mean")
		optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
		device = "cuda" if torch.cuda.is_available() else "cpu"
		print(f"{device}" " is available.")


		opt = Optimization(device=device, model=model, loss_fn=loss_fn, optimizer=optimizer)
		opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
		opt.plot_losses()

		predictions, values = opt.evaluate(
			test_loader_one,
			batch_size=1,
			n_features=input_dim
		)

		#scaler = Helper.get_scaler(select_scaler)

		df_result = Helper.format_predictions(predictions, values, X_test, scaler)
		print(f'Forecast testset predictions: {df_result}')

		result_metrics,  key_metrics = Helper.calculate_metrics(df_result)
		print(f'Model Evaluations: {result_metrics}')

		print(f'Model Evaluations: {key_metrics}')

		Helper.plot_metrics(result_metrics, key_metrics)

		df_baseline = Helper.build_baseline_model(df, split_ratio, target_col) #df_feature, 0.2, 'value'
		baseline_metrics = Helper.calculate_metrics(df_baseline)

		Helper.plot_predictions(df_result, df_baseline)

		return opt, scaler


	






