#! /usr/bin/env python

"""
@author: Ajay Arunachalam
Created on: 23/10/2021
Goal: Deep Explainable Forecasting with State-Of-the-Networks for Time-series data - RNN, LSTM, GRU, BiRNN, SNN, GNN, Transformers, GAN, FFNN, etc
Version: 0.0.5
"""

import torch
import torch.nn as nn
import statsmodels.api as sm
from .gnn_layer import *


class Forecasting_Models:


	class RNNModel(nn.Module):

		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			"""The __init__ method that initiates an RNN instance.

			Args:
				input_dim (int): The number of nodes in the input layer
				hidden_dim (int): The number of nodes in each layer
				layer_dim (int): The number of layers in the network
				output_dim (int): The number of nodes in the output layer
				dropout_prob (float): The probability of nodes being dropped out

			"""
			super(Forecasting_Models.RNNModel, self).__init__()

			# Defining the number of layers and the nodes in each layer
			self.hidden_dim = hidden_dim
			self.layer_dim = layer_dim

			# RNN layers
			self.rnn = nn.RNN(
				input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
			)
			# Fully connected layer
			self.fc = nn.Linear(hidden_dim, output_dim)

		def forward(self, x):
			"""The forward method takes input tensor x and does forward propagation

			Args:
				x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

			Returns:
				torch.Tensor: The output tensor of the shape (batch size, output_dim)

			"""
			# Initializing hidden state for first input with zeros
			h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

			# Forward propagation by passing in the input and hidden state into the model
			out, h0 = self.rnn(x, h0.detach())

			# Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
			# so that it can fit into the fully connected layer
			out = out[:, -1, :]

			# Convert the final state to our desired output shape (batch_size, output_dim)
			out = self.fc(out)
			return out


	class LSTMModel(nn.Module):
		"""LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

		   LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
		   It has only two methods, namely init() and forward(). While the init()
		   method initiates the model with the given input parameters, the forward()
		   method defines how the forward propagation needs to be calculated.
		   Since PyTorch automatically defines back propagation, there is no need
		   to define back propagation method.

		   Attributes:
			   hidden_dim (int): The number of nodes in each layer
			   layer_dim (str): The number of layers in the network
			   lstm (nn.LSTM): The LSTM model constructed with the input parameters.
			   fc (nn.Linear): The fully connected layer to convert the final state
							   of LSTMs to our desired output shape.

		"""
		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			"""The __init__ method that initiates a LSTM instance.

			Args:
				input_dim (int): The number of nodes in the input layer
				hidden_dim (int): The number of nodes in each layer
				layer_dim (int): The number of layers in the network
				output_dim (int): The number of nodes in the output layer
				dropout_prob (float): The probability of nodes being dropped out

			"""
			super(Forecasting_Models.LSTMModel, self).__init__()

			# Defining the number of layers and the nodes in each layer
			self.hidden_dim = hidden_dim
			self.layer_dim = layer_dim

			# LSTM layers
			self.lstm = nn.LSTM(
				input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
			)

			# Fully connected layer
			self.fc = nn.Linear(hidden_dim, output_dim)

		def forward(self, x):
			"""The forward method takes input tensor x and does forward propagation

			Args:
				x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

			Returns:
				torch.Tensor: The output tensor of the shape (batch size, output_dim)

			"""
			# Initializing hidden state for first input with zeros
			h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

			# Initializing cell state for first input with zeros
			c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

			# We need to detach as we are doing truncated backpropagation through time (BPTT)
			# If we don't, we'll backprop all the way to the start even after going through another batch
			# Forward propagation by passing in the input, hidden state, and cell state into the model
			out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

			# Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
			# so that it can fit into the fully connected layer
			out = out[:, -1, :]

			# Convert the final state to our desired output shape (batch_size, output_dim)
			out = self.fc(out)

			return out

	class GRUModel(nn.Module):
		"""GRUModel class extends nn.Module class and works as a constructor for GRUs.

		   GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
		   It has only two methods, namely init() and forward(). While the init()
		   method initiates the model with the given input parameters, the forward()
		   method defines how the forward propagation needs to be calculated.
		   Since PyTorch automatically defines back propagation, there is no need
		   to define back propagation method.

		   Attributes:
			   hidden_dim (int): The number of nodes in each layer
			   layer_dim (str): The number of layers in the network
			   gru (nn.GRU): The GRU model constructed with the input parameters.
			   fc (nn.Linear): The fully connected layer to convert the final state
							   of GRUs to our desired output shape.

		"""
		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			"""The __init__ method that initiates a GRU instance.

			Args:
				input_dim (int): The number of nodes in the input layer
				hidden_dim (int): The number of nodes in each layer
				layer_dim (int): The number of layers in the network
				output_dim (int): The number of nodes in the output layer
				dropout_prob (float): The probability of nodes being dropped out

			"""
			super(Forecasting_Models.GRUModel, self).__init__()

			# Defining the number of layers and the nodes in each layer
			self.layer_dim = layer_dim
			self.hidden_dim = hidden_dim

			# GRU layers
			self.gru = nn.GRU(
				input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
			)

			# Fully connected layer
			self.fc = nn.Linear(hidden_dim, output_dim)

		def forward(self, x):
			"""The forward method takes input tensor x and does forward propagation

			Args:
				x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

			Returns:
				torch.Tensor: The output tensor of the shape (batch size, output_dim)

			"""
			# Initializing hidden state for first input with zeros
			h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

			# Forward propagation by passing in the input and hidden state into the model
			out, _ = self.gru(x, h0.detach())

			# Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
			# so that it can fit into the fully connected layer
			out = out[:, -1, :]

			# Convert the final state to our desired output shape (batch_size, output_dim)
			out = self.fc(out)

			return out

	
	# Bidirectional recurrent neural network (many-to-one)
	class BiRNNModel(nn.Module):

		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			super(Forecasting_Models.BiRNNModel, self).__init__()
			self.hidden_dim = hidden_dim
			self.layer_dim = layer_dim
			self.lstm = nn.LSTM(input_size, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=dropout_prob)
			self.fc = nn.Linear(hidden_dim*2, output_dim)  # 2 for bidirection
		
		def forward(self, x):
			# Set initial states
			h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() #.to(device) # 2 for bidirection 
			c0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() #.to(device)
			
			# Forward propagate LSTM
			out, _ = self.lstm(x, (h0.detach(), c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
			
			# Decode the hidden state of the last time step
			out = self.fc(out[:, -1, :])
			return out

	class BiGRUModel(nn.Module):
		"""BiGRUModel class extends nn.Module class and works as a constructor for GRUs.

		   BiGRUModel class initiates a GRU module based on PyTorch's nn.Module class.
		   It has only two methods, namely init() and forward(). While the init()
		   method initiates the model with the given input parameters, the forward()
		   method defines how the forward propagation needs to be calculated.
		   Since PyTorch automatically defines back propagation, there is no need
		   to define back propagation method.

		   Attributes:
			   hidden_dim (int): The number of nodes in each layer
			   layer_dim (str): The number of layers in the network
			   gru (nn.GRU): The GRU model constructed with the input parameters.
			   fc (nn.Linear): The fully connected layer to convert the final state
							   of GRUs to our desired output shape.

		"""
		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			"""The __init__ method that initiates a GRU instance.

			Args:
				input_dim (int): The number of nodes in the input layer
				hidden_dim (int): The number of nodes in each layer
				layer_dim (int): The number of layers in the network
				output_dim (int): The number of nodes in the output layer
				dropout_prob (float): The probability of nodes being dropped out

			"""
			super(Forecasting_Models.BiGRUModel, self).__init__()

			# Defining the number of layers and the nodes in each layer
			self.layer_dim = layer_dim
			self.hidden_dim = hidden_dim

			# GRU layers
			self.gru = nn.GRU(
				input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=dropout_prob
			)

			# Fully connected layer
			self.fc = nn.Linear(hidden_dim*2, output_dim) # 2 for bidirection

		def forward(self, x):
			"""The forward method takes input tensor x and does forward propagation

			Args:
				x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

			Returns:
				torch.Tensor: The output tensor of the shape (batch size, output_dim)

			"""
			# Initializing hidden state for first input with zeros
			h0 = torch.zeros(self.layer_dim*2, x.size(0), self.hidden_dim).requires_grad_() # 2 for bidirection

			# Forward propagation by passing in the input and hidden state into the model
			out, _ = self.gru(x, h0.detach())

			# Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
			# so that it can fit into the fully connected layer
			out = out[:, -1, :]

			# Convert the final state to our desired output shape (batch_size, output_dim)
			out = self.fc(out)

			return out

	class SNNModel(nn.Module):
		pass


	class CNNModel(nn.Module):

		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			"""The __init__ method that initiates an CNN instance.

			Args:
				input_dim (int): The number of nodes in the input layer
				hidden_dim (int): The number of nodes in each layer
				layer_dim (int): The number of layers in the network
				output_dim (int): The number of nodes in the output layer
				dropout_prob (float): The probability of nodes being dropped out

			"""
			super(Forecasting_Models.CNNModel, self).__init__()

			# Defining the number of layers and the nodes in each layer
			self.hidden_dim = hidden_dim
			self.layer_dim = layer_dim

			# CNN layers
			self.cnn = nn.CNN(
				input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
			)
			# Fully connected layer
			self.fc = nn.Linear(hidden_dim, output_dim)

		def forward(self, x):
			"""The forward method takes input tensor x and does forward propagation

			Args:
				x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

			Returns:
				torch.Tensor: The output tensor of the shape (batch size, output_dim)

			"""
			# Initializing hidden state for first input with zeros
			h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

			# Forward propagation by passing in the input and hidden state into the model
			out, h0 = self.cnn(x, h0.detach())

			# Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
			# so that it can fit into the fully connected layer
			out = out[:, -1, :]

			# Convert the final state to our desired output shape (batch_size, output_dim)
			out = self.fc(out)
			return out


	class GNNModel(nn.Module):

		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
			
		 	# self, in_features = 7, hidden_dim = 64, classes = 2, dropout = 0.5

			super(Forecasting_Models.GNNModel, self).__init__()

			# GNN layers
			self.conv1 = GCN_Layer(input_dim, hidden_dim) # in_features, hidden_dim
			self.conv2 = GCN_Layer(hidden_dim, hidden_dim)
			self.conv3 = GCN_Layer(hidden_dim, hidden_dim)
			self.fc = nn.Linear(hidden_dim, output_dim)
			self.dropout = dropout_prob

		def forward(self, x, A):
			x = self.conv1(x, A)
			x = F.relu(x)
			x = self.conv2(x, A)
			x = F.relu(x)
			x = self.conv3(x, A)
			x = F.dropout(x, p=self.dropout, training=self.training)
			# aggregate node embeddings
			x = x.mean(dim=1)
			# final layer
			return self.fc(x)

	# Deep CNN

	class DeepCNNModel(nn.Module):

		"""
		Model : Class for DeepCNN model
		"""
		def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob): # self, LOOKBACK_SIZE, DIMENSION, KERNEL_SIZE
			super(Forecasting_Models.DeepCNNModel, self).__init__()
			import torch
			self.conv1d_1_layer = torch.nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3) # , stride=2
			self.relu_1_layer = torch.nn.ReLU()
			self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=(3)-1)
			self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
			self.relu_2_layer = torch.nn.ReLU()
			self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=(3)-1)
			self.flatten_layer = torch.nn.Flatten()
			self.dense_1_layer = torch.nn.Linear(80, 40)
			self.relu_3_layer = torch.nn.ReLU()
			self.dropout_layer = torch.nn.Dropout(p=dropout_prob)
			self.dense_2_layer = torch.nn.Linear(40, output_dim)
			
		def forward(self, x):
			x = self.conv1d_1_layer(x)
			x = self.relu_1_layer(x)
			x = self.maxpooling_1_layer(x)
			x = self.conv1d_2_layer(x)
			x = self.relu_2_layer(x)
			x = self.maxpooling_2_layer(x)
			x = self.flatten_layer(x)
			x = self.dense_1_layer(x)
			x = self.relu_3_layer(x)
			x = self.dropout_layer(x)
			return self.dense_2_layer(x)

	# GCN

	'''
	import torch
	from torch import nn

	class GCN(nn.Module):

		def __init__(self, *sizes):
			super().__init__()
			self.layers = nn.ModuleList([
				nn.Linear(x, y) for x, y in zip(sizes[:-1], sizes[1:])
			])
		def forward(self, vertices, edges):
			# ----- Build the adjacency matrix -----
			# Start with self-connections
			adj = torch.eye(len(vertices))
			# edges contain connected vertices: [vertex_0, vertex_1] 
			adj[edges[:, 0], edges[:, 1]] = 1 
			adj[edges[:, 1], edges[:, 0]] = 1
			
			# ----- Forward data pass -----
			for layer in self.layers:
				vertices = torch.sigmoid(layer(adj @ vertices))
	 
			return vertices
	'''

	'''
	# FFNN

	class FFModel(torch.nn.Module):

		def __init__(self, DIMENSION):
			super(Anamoly.MPLR, self).__init__()
			import torch
			self.layer_1 = torch.nn.Linear(DIMENSION, 16)
			self.layer_2 = torch.nn.Linear(16, 32)
			self.layer_3 = torch.nn.Linear(32, 16)
			self.layer_out = torch.nn.Linear(16, 1)
			self.relu = torch.nn.ReLU()

		def forward(self, inputs):
			x = self.relu(self.layer_1(inputs))
			x = self.relu(self.layer_2(x))
			x = self.relu(self.layer_3(x))
			x = self.layer_out(x)
			return (x)

		def predict(self, test_inputs):
			x = self.relu(self.layer_1(test_inputs))
			x = self.relu(self.layer_2(x))
			x = self.relu(self.layer_3(x))
			x = self.layer_out(x)
			return (x)

	# Deep CNN

	class DeepCNNModel(torch.nn.Module):
		"""
		Model : Class for DeepCNN model
		"""
		def __init__(self, LOOKBACK_SIZE, DIMENSION, KERNEL_SIZE):
			super(Anamoly.DeepCNN, self).__init__()
			import torch
			self.conv1d_1_layer = torch.nn.Conv1d(in_channels=LOOKBACK_SIZE, out_channels=16, kernel_size=KERNEL_SIZE) # , stride=2
			self.relu_1_layer = torch.nn.ReLU()
			self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=(KERNEL_SIZE)-1)
			self.conv1d_2_layer = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=KERNEL_SIZE)
			self.relu_2_layer = torch.nn.ReLU()
			self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=(KERNEL_SIZE)-1)
			self.flatten_layer = torch.nn.Flatten()
			self.dense_1_layer = torch.nn.Linear(80, 40)
			self.relu_3_layer = torch.nn.ReLU()
			self.dropout_layer = torch.nn.Dropout(p=0.25)
			self.dense_2_layer = torch.nn.Linear(40, DIMENSION)
			
		def forward(self, x):
			x = self.conv1d_1_layer(x)
			x = self.relu_1_layer(x)
			x = self.maxpooling_1_layer(x)
			x = self.conv1d_2_layer(x)
			x = self.relu_2_layer(x)
			x = self.maxpooling_2_layer(x)
			x = self.flatten_layer(x)
			x = self.dense_1_layer(x)
			x = self.relu_3_layer(x)
			x = self.dropout_layer(x)
			return self.dense_2_layer(x)

	# LSTMAENN

	class LSTMAEModel(torch.nn.Module):
		"""
		Model : Class for LSTMAENN model
		"""
		def __init__(self, LOOKBACK_SIZE, DIMENSION):
			import torch
			super(Anamoly.LSTMAENN, self).__init__()
			self.lstm_1_layer = torch.nn.LSTM(DIMENSION, 128, 1)
			self.dropout_1_layer = torch.nn.Dropout(p=0.2)
			self.lstm_2_layer = torch.nn.LSTM(128, 64, 1)
			self.dropout_2_layer = torch.nn.Dropout(p=0.2)
			self.lstm_3_layer = torch.nn.LSTM(64, 64, 1)
			self.dropout_3_layer = torch.nn.Dropout(p=0.2)
			self.lstm_4_layer = torch.nn.LSTM(64, 128, 1)
			self.dropout_4_layer = torch.nn.Dropout(p=0.2)
			self.linear_layer = torch.nn.Linear(128, DIMENSION)
			
		def forward(self, x):
			x, (_,_) = self.lstm_1_layer(x)
			x = self.dropout_1_layer(x)
			x, (_,_) = self.lstm_2_layer(x)
			x = self.dropout_2_layer(x)
			x, (_,_) = self.lstm_3_layer(x)
			x = self.dropout_3_layer(x)
			x, (_,_) = self.lstm_4_layer(x)
			x = self.dropout_4_layer(x)
			return self.linear_layer(x)

	# GNN

	# GAN

	class GAN(torch.nn.Module):
		"""
		Model: Class for Generative Adversarial Networks
		"""
		pass

	# TRANSFORMER

	class TF(torch.nn.Module):
		"""
		Model: Class for Transformer Networks
		"""
		pass


	# SNN

	# https://github.com/ajayarunachalam/pynmsnn/blob/main/pyNM/spiking_regressor.py

	class SNN(torch.nn.Module):
		"""
		Model: Class for Spiking Neural Networks
		"""
		pass


	# Using the auto time-series functionality 
	class AutoTimeSeries:
		from auto_ts import auto_timeseries

		def __init__(self, df, ts_column, time_interval, seasonality, seasonal_period, target, FORECAST_PERIOD):
			super(Anamoly.AutoTimeSeries, self).__init__()

		
		# Train Model

		self.ts_column =  ts_column
		self.target = target
		self.FORECAST_PERIOD = FORECAST_PERIOD
		self.interval = interval
		sep = ','

		#FORECAST_PERIOD = 2

		model = auto_timeseries(score_type='rmse',
						time_interval='QS',
						non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
						model_type=['prophet'],  # model_type available = ['best', 'prophet', 'stats', 'ml', 'arima','ARIMA','Prophet','SARIMAX', 'VAR', 'ML']
						verbose=2)

		model.fit(
			traindata=dataset,
			# traindata=file_path,  # Alternately, you can specify the file directly
			ts_column=ts_column,
			target=target,
			cv=5,
			sep=sep)

		model.get_leaderboard()

		# Make sure all models have the same number of CV folds
		model.get_cv_scores()

		model.plot_cv_scores()

		# Using Machine Learning Model
		future_predictions = model.predict(
			testdata=FORECAST_PERIOD
		)  

		# Forecasted Future Predictions dataframe
		future_predictions

	'''


	