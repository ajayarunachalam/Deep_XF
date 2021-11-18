import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

import denoise


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Process raw ECG signals'
)

parser.add_argument(
    '-d', '--data', type=str, required=True,
    help='Path to raw ECG signal CSV data'
)
parser.add_argument(
    '-o', '--output-path', type=str, dest='output',
    help='Filepath to output filtered ECG signal'
)
parser.add_argument(
    '-r', '--sampling-freq', type=int, default=500,
    dest='fs', help='Sampling frequency of raw ECG signal'
)
parser.add_argument(
    '-s', '--snr-algorithm', default='default',
    choices=('default', 'klt'), dest='snr_alg',
    help='Algorithm to calculate SNR of ECG signal'
)
parser.add_argument(
    '-f', '--filter-algorithm', default='default',
    choices=('default', 'dwt'), dest='filter_alg',
    help='Algorithm to denoise/filter the raw ECG signal'
)


if __name__ == '__main__':
    args = parser.parse_args()

    # Read raw ECG signal
    signal = pd.read_csv(args.data, parse_dates=['tstamp'])
    date = signal['tstamp'].dt.date
    time = signal['tstamp'].dt.time
    signal.set_index('tstamp', inplace=True)

    # Filter signal
    if args.filter_alg == 'default':
        filtered = denoise.overlay_filter(signal, args.fs)
    elif args.filter_alg == 'dwt':
        filtered = denoise.wavelet_denoising(signal.copy().T).T
    
    # Output filtered signal
    if args.output is not None:
        filtered_df = pd.DataFrame(
            data=filtered, index=signal.index,
            columns=signal.columns
        )
        filtered_df.insert(0, 'Date', date.values)
        filtered_df.insert(1, 'Time', time.values)
        filtered_df.to_csv(args.output, index=False)
        filtered_df.drop(columns=['Date', 'Time'], inplace=True)

        # Save plots
        plt.figure(figsize=(16, 8))

        ax1 = plt.subplot(211)
        plt.plot(signal)
        ax1.set_title('Raw ECG signal')

        ax2 = plt.subplot(212)
        plt.plot(filtered_df)
        ax2.set_title('Clean ECG signal')

        plt.savefig(os.path.join(os.path.dirname(args.output), 'ecg_plot.png'))
    
    # Calculate SNR
    if args.snr_alg == 'default':
        noise = signal - filtered
        snr = 20 * np.log10(np.square(np.divide(
            np.sqrt(np.square(signal).mean(axis=0)),
            np.sqrt(np.square(noise).mean(axis=0))
        )))
    elif args.snr_alg == 'klt':
        # Process array
        filtered_data = denoise.filter_outliers(signal)
        mean_filtered_data = denoise.mean_and_subtract(filtered_data)
        compressed_data, m, val = denoise.KLTransform(mean_filtered_data)

        # Calculate SNR
        snr = np.where(
            compressed_data.std(axis=0, ddof=0) == 0,
            0, compressed_data.mean(axis=0)
        )
    
    snr_db = 20 * np.log10(np.abs(snr))
    for i in range(snr.shape[0]):
        print(f'SNR for {signal.columns[i]} is {snr[i]} OR {snr_db[i]} dB')
