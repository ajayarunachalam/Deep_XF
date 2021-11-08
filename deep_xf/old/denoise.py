#! /usr/bin/env python
"""
@author: Ajay Arunachalam
Created on: 11/10/2021
Goal: Denoising time-series signals
Version: 0.0.1
"""
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Denoise:

	fs = globals()
	order = globals()
	cutoff_high = globals()
	cutoff_low = globals()
	powerline = globals()
	nyq = globals()

	def set_parameters(**kwargs):
		for key, value in kwargs.items():
			print("{0} = {1}" .format(key, value))

		fs = list(kwargs.values())[0]
		order = list(kwargs.values())[1]
		cutoff_high = list(kwargs.values())[2]
		cutoff_low = list(kwargs.values())[3]
		powerline = list(kwargs.values())[4]

		nyq = 0.5 * fs

		return fs, order, cutoff_high, cutoff_low, powerline, nyq

	assert fs == fs
	assert order == order
	assert cutoff_high == cutoff_high
	assert cutoff_low == cutoff_low
	assert powerline == powerline

	assert nyq == nyq

	## Butter high pass filter allows frequencies higher than a cut-off value
	def bhpf(cutoff, order, nyq):
		#nyq = 0.5 * fs
		normal_cutoff = cutoff/nyq
		b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
		return b, a

	## Butter low pass filter allows frequencies lower than a cut-off value
	def blpf(cutoff, order, nyq):
		#nyq = 0.5 * fs
		normal_cutoff = cutoff/nyq
		b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
		return b, a

	## Notch filter 
	def nf(cutoff, q, nyq):
		#nyq = 0.5 * fs
		freq = cutoff/nyq
		b, a = iirnotch(freq, q)
		return b, a

	## Applying high pass filter 
	def hpfd(data, cutoff_high, order, nyq):
		b,a = bhpf(cutoff_high, order, nyq)
		x = lfilter(b,a,data)
		return x

	## Applying low pass filter 
	def lpfd(data, cutoff_low, order, nyq):
		b,a = blpf(cutoff_low, order, nyq)
		y = lfilter(b,a,data)
		return y

	## Applying notch filter
	def nd(data, powerline, q):
		b,a = nf(powerline,q)
		z = lfilter(b,a,data)
		return z

	## Ensembling of the filter
	def overlay_filter(data, cutoff_high, cutoff_low, nyq, powerline, fs, order):
		b, a = Denoise.bhpf(cutoff_high, order, nyq)
		x = lfilter(b, a, data)
		d, c = Denoise.blpf(cutoff_low, order, nyq)
		y = lfilter(d, c, x)
		f, e = Denoise.nf(powerline, 30, nyq)
		z = lfilter(f, e, y)     
		return z

	def plot_and_write(df, featurecol_index:int, fs=fs, cutoff_high=cutoff_high, cutoff_low=cutoff_low, nyq=nyq, powerline=powerline, order=order):
		#fs = 500; cutoff_high = 0.5; cutoff_low = 2; powerline = 60
		#order = 5 ## Order of five works well with ECG signals

		plt.figure(1)
		ax1 = plt.subplot(121)
		plt.plot(df.iloc[:,featurecol_index:])
		ax1.set_title("Raw TimeSeries ECG Signal")

		filter_ts_signal = Denoise.overlay_filter(df.iloc[:,featurecol_index:], cutoff_high, cutoff_low, nyq, powerline, fs, order)
		ax2 = plt.subplot(122)
		plt.plot(filter_ts_signal)
		ax2.set_title("Denoised TimeSeries ECG Signal")
		plt.show()
		#filtered_dataset = pd.concat([df.iloc[:,:featurecol_index], filter_ts_signal])
		cols = list(df.iloc[:,featurecol_index:])
		#filtered_dataset = pd.DataFrame({'Date':df.Date, 'Time': df.Time,'ecg1': filter_ts_signal[:, 0], 'ecg2': filter_ts_signal[:, 1],'ecg3': filter_ts_signal[:, 2],'ecg4': filter_ts_signal[:, 3],'ecg5': filter_ts_signal[:, 4]})
		filter_ts_signal_dataset = pd.DataFrame(np.array(filter_ts_signal), columns=cols) 
		filtered_dataset = pd.concat([df.iloc[:, :featurecol_index].reset_index(), filter_ts_signal_dataset.reset_index()], axis=1)
		filtered_dataset.drop(columns=['index'], inplace=True)
		filtered_dataset.to_csv('Denoise_TS_Signal.csv', index=None)
		return filtered_dataset

	def write_csv():
		pass
		


