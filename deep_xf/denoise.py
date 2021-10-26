#! /usr/bin/env python
"""
@author: Ajay Arunachalam
Created on: 11/10/2021
Goal: Denoising time-series ecg signals
Version: 0.0.3
"""
from scipy import signal
from scipy.signal import butter, iirnotch, lfilter
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Denoise:
	## Butter high pass filter allows frequencies higher than a cut-off value
	def bhpf(cutoff, fs, order=5):
		nyq = 0.5*fs
		normal_cutoff = cutoff/nyq
		b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
		return b, a

	## Butter low pass filter allows frequencies lower than a cut-off value
	def blpf(cutoff, fs, order=5):
		nyq = 0.5*fs
		normal_cutoff = cutoff/nyq
		b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
		return b, a

	## Notch filter 
	def nf(cutoff, q):
		nyq = 0.5*fs
		freq = cutoff/nyq
		b, a = iirnotch(freq, q)
		return b, a

	## Applying high pass filter 
	def hpfd(data, fs, order=5):
		b,a = bhpf(cutoff_high, fs, order=order)
		x = lfilter(b,a,data)
		return x

	## Applying low pass filter 
	def lpfd(data, fs, order =5):
		b,a = blpf(cutoff_low, fs, order=order)
		y = lfilter(b,a,data)
		return y

	## Applying notch filter
	def nd(data, powerline, q):
		b,a = nf(powerline,q)
		z = lfilter(b,a,data)
		return z

	## Ensembling of the filter
	def overlay_filter(data, fs, order=5):
		b, a = Denoise.bhpf(cutoff_high, fs, order=order)
		x = lfilter(b, a, data)
		d, c = Denoise.blpf(cutoff_low, fs, order = order)
		y = lfilter(d, c, x)
		f, e = Denoise.nf(powerline, 30)
		z = lfilter(f, e, y)     
		return z

	def plot_and_write(df, featurecol_index:int):
		fs = 500; cutoff_high = 0.5; cutoff_low = 2; powerline = 60
		order = 5 ## Order of five works well with ECG signals

		plt.figure(1)
		ax1 = plt.subplot(121)
		plt.plot(df.iloc[:,featurecol_index:])
		ax1.set_title("Raw TimeSeries Signal")

		filter_ts_signal = Denoise.overlay_filter(df.iloc[:,featurecol_index:], fs, order)
		ax2 = plt.subplot(122)
		plt.plot(filter_signal)
		ax2.set_title("Denoised TimeSeries Signal")
		plt.show()
		filtered_dataset = pd.concat([df.iloc[:,:featurecol_index], filter_ts_signal])
		#filtered_dataset = pd.DataFrame({'Date':df., 'Time': ecg_signal.Time,'ecg1': filter_signal[:, 0], 'ecg2': filter_signal[:, 1],'ecg3': filter_signal[:, 2],'ecg4': filter_signal[:, 3],'ecg5': filter_signal[:, 4]})
		filtered_dataset.to_csv('Denoise_TS.csv', index=None)
		return filtered_dataset

	def write_csv():
		pass
		


