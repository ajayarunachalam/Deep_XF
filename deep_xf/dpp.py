#! /usr/bin/env python
"""
@author: Ajay Arunachalam
Created on: 04/10/2021
Goal: Data Preprocessing for Time-series data 
Version: 0.0.1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# set default figure dimensions
sns.set(rc={'figure.figsize':(11, 4)})
import pathlib
from scipy import stats
from ecgdetectors import Detectors
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
from plotly.offline import iplot

class Preprocessing:
	
	# function to find missing values
	def missing(x):
		return sum(x.isnull())

	# function to print missing in whole dataset row & column wise

	def print_missing(df):
		print("Missing information for whole dataset")
		print("Missing by row", df.apply(Preprocessing.missing, axis=0)) # col
		print("Missing by col", df.apply(Preprocessing.missing, axis=1)) # row


	# function to convert to datetime object, and extract date & time columns seperately
	def datetime_extraction(df, datetimecolindex):
		import pandas as pd
		import datetime
		# convert to datetime object

		df.iloc[:,datetimecolindex] = pd.to_datetime(df.iloc[:,datetimecolindex])

		# Extract date & time separately
		df['Date'] = df.iloc[:,datetimecolindex].dt.date
		df['Time'] = df.iloc[:,datetimecolindex].dt.time

		#Reorder the DATE, TIME column back to BEGINNING index
		cols = list(df)
		# move the column to head of list 
		cols.pop(cols.index('Date')), cols.pop(cols.index('Time'))
		df = df[['Date','Time']+cols[1:]]
		return df

	# function to join date & time columns together as single timestamp attribute

	def datetime_merge(df, datecol:str, timecol:str):
		import pandas as pd
		import datetime

		# concat date & time to form timestamp column

		if df[datecol].dtypes=='O' or df[timecol].dtypes=='O': 

			df['TS'] = df[datecol] + " " + df[timecol]
			df['TS'] = pd.to_datetime(df[datecol] + " " + df[timecol])
			# move newly create TS column back to start postion
			cols = list(df)
			cols.pop(cols.index('TS'))
			df = df[['TS'] + cols[2:]]

		else:
			df['TS'] = df.apply(lambda x : pd.datetime.combine(x['datecol'],x['timecol']),1)
			# move newly create TS column back to start postion
			cols = list(df)
			cols.pop(cols.index('TS'))
			df = df[['TS'] + cols[2:]]
		
		return df

	# impute missing values
	def impute(df, modes:int): #start_sensor_column_index, 
		import pandas as pd
		if modes==0:
			#df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(0, inplace=True)
			df.fillna(0, inplace=True)
		elif modes==1:
			#df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(df.mean(), inplace=True)
			df.fillna(df.mean(), inplace=True)
		elif modes==2:
			#df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(df.median(), inplace=True)
			df.fillna(df.median(), inplace=True)
		else:
			#df.iloc[:,start_sensor_column_index:len(df.columns)].fillna(method='bfill', inplace=True) # backfill / bfill: use next valid observation to fill gap.
			df.fillna(method='bfill', inplace=True)
		df.to_csv('./df_no_NA.csv')
		return df

	# drop particular column
	def drop_columns(df, column_index_to_drop:int):
		df.drop(df.columns[column_index_to_drop], inplace=True, axis=1)
		return df



class ExploratoryDataAnalysis:
	
	# profiling your dataset
	def profiling(df): 
		import pandas_profiling as pp       
		# To Generate a HTML report file
		html_report = pp.ProfileReport(df, title="Data Profiling Report")
		html_report.to_file("./profiling_report.html")
	
	# filtering outliers
	def drop_outliers(df, z_threshold):
		from scipy import stats
		import numpy as np
		constraints = df.select_dtypes(include=[np.number]) \
			.apply(lambda x: np.abs(stats.zscore(x)) < z_threshold) \
			.all(axis=1)
		df.drop(df.index[~constraints], inplace=True)
		return df

	# univariate/multivariate plots
	def plots(df, cols):
		# plotting multivariate
		df[cols].plot()

	# Usage: ExploratoryDataAnalysis.plots(df=processed_data, cols=['col1', 'col2', 'col3', 'col4', 'col5'])


	# plot with plotly
	def plot_dataset(df, title):
		import plotly.graph_objs as go
		from plotly.offline import iplot
		data = []
		value = go.Scatter(
			x=df.index,
			y=df.value,
			mode="lines",
			name="values",
			marker=dict(),
			text=df.index,
			line=dict(color="rgba(0,0,0, 0.3)"),
		)
		data.append(value)

		layout = dict(
			title=title,
			xaxis=dict(title="Date", ticklen=5, zeroline=False),
			yaxis=dict(title="Value", ticklen=5, zeroline=False),
		)

		fig = dict(data=data, layout=layout)
		iplot(fig) 

	# Usage: 
	'''
	df = df.set_index(['your_datatime_column'])
	df = df.rename(columns={'your_column_name': 'value'})

	df.index = pd.to_datetime(df.index)
	if not df.index.is_monotonic:
		df = df.sort_index()
	
	ExploratoryDataAnalysis.plot_dataset(df, title='Enter your title of the plot')
	'''
		
	def sumplots_rw(df, cols, window:int):
		# sumplots of rolling window
		processed_data[cols].rolling(window=window).mean().plot(subplots=True)

	# Usage: ExploratoryDataAnalysis.sumplots_rw(processed_data, ['col1', 'col2', 'col3', 'col4', 'col5'], 12)

	# detects r-signal-to-signal peak
	# includes the following detector types, 
	# i.e., det_type = [swt_detector, christov_detector, two_average_detector, engzee_detector, hamilton_detector, 
	#                  pan_tompkins_detector, matched_filter_detector]
	def r_peak_plots_arg(dataframe, det_type, col_index, fs:int):
		unfiltered_ecg_dat = dataframe 
		unfiltered_ecg = unfiltered_ecg_dat.iloc[:,col_index]
		fs = fs
		detectors = Detectors(fs)
		det_type = str(det_type.strip('\"'))
		#print(det_type)
		detector_call = getattr(detectors, det_type);
		r_peaks = detector_call(unfiltered_ecg)
		plt.figure()
		plt.plot(unfiltered_ecg)
		plt.plot(r_peaks, unfiltered_ecg[r_peaks], 'ro')
		plt.title('Detected R-peaks')
		plt.show()
		return det_type, col_index 

	# Usage: ExploratoryDataAnalysis.r_peak_plots_arg(processed_data,det_type='christov_detector', col_index=6, fs=1000)
	

class DescriptiveStatistics:

	# Stats for whole dataset
	def desc_all(df, start_sensor_column_index):
		print("Descriptive Statistics")
		return df.iloc[:,start_sensor_column_index:len(df.columns)].describe() # include='all'
	
	# Stats with relation to response variable and any feature (tobe used if your data is labeled)
	def desc_target_feature(df, target, feature):
		print("Descriptive Statistics by Target-Feature")
		return df.groupby(target)[feature].describe()


class StatisticalEvaluation:

	# Test your data for normality/if the data follows normal distribution
	def normality_test(df, start_sensor_column_index):
		from scipy import stats, norm, shapiro, jarque_bera
		print("Shapiro-Wilk Normality test (W test statistic, the p-value) =======> {}  ".format(shapiro(df.iloc[:,start_sensor_column_index:len(df.columns)])))
		print("Jarque-Bera Normality test (W test statistic, the p-value) =======> {}".format(jarque_bera(df.iloc[:,start_sensor_column_index:len(df.columns)])))


class Features:

	def generate_time_lags(df, n_lags:int, col_name:str):
		df_n = df.copy()
		for n in range(1, n_lags + 1):
			df_n[f"lag{n}"] = df_n[col_name].shift(n) 
		df_n = df_n.iloc[n_lags:]
		return df_n

	def generate_date_time_features_hour(df, cols:list()):

		df_features = (
				df
				.assign(hour = df.index.hour)
				.assign(day = df.index.day)
				.assign(month = df.index.month)
				.assign(day_of_week = df.index.dayofweek)
				.assign(week_of_year = df.index.week)
		)
				
		return df_features

	def generate_date_time_features_month(df, cols:list()):

		df_features = (
				df
				.assign(month = df.index.month)
				.assign(day_of_week = df.index.dayofweek)
				.assign(week_of_year = df.index.week)
		)
				
		return df_features

	def generate_date_time_features_ohe(df, cols:list()):

		df_features = (
				df
				.assign(hour = df.index.hour)
				.assign(day = df.index.day)
				.assign(month = df.index.month)
				.assign(day_of_week = df.index.dayofweek)
				.assign(week_of_year = df.index.week)
		)
		df_features = df_features

		def onehot_encode_pd(df_features=df_features, cols=cols):
			import pandas as pd
			df_with_dummies = pd.get_dummies(df_features, columns=cols)
			df_with_features_dummies = pd.concat([df_features, df_with_dummies.iloc[:,1:]], axis=1)
			return df_with_features_dummies

		df_full_features = onehot_encode_pd(df_features=df_features, cols=cols)
		
		return df_full_features


	def generate_cyclic_features(df, col_name, period, start_num=0):
		kwargs = {
			f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
			f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
			 }
		df_full_features = df.assign(**kwargs).drop(columns=[col_name])
		return df_full_features



	def generate_other_related_features(df):
		from datetime import date
		import holidays

		country_holidays = holidays.US() # change according to user specific country

		def is_holiday(date):
			date = date.replace(hour = 0)
			return 1 if (date in country_holidays) else 0

		def add_holiday_col(df, holidays):
			return df.assign(is_holiday = df.index.to_series().apply(is_holiday))

		df_full_features = add_holiday_col(df=df, holidays=country_holidays)
		
		return df_full_features




