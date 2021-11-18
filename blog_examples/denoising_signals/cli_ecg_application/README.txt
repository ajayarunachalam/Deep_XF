python ecg.py -h

usage: ecg.py [-h] -d DATA [-o OUTPUT] [-r FS] [-s {default,klt}]
              [-f {default,dwt}]

Process raw ECG signals

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Path to raw ECG signal CSV data (default: None)
  -o OUTPUT, --output-path OUTPUT
                        Filepath to output filtered ECG signal (default: None)
  -r FS, --sampling-freq FS
                        Sampling frequency of raw ECG signal (default: 500)
  -s {default,klt}, --snr-algorithm {default,klt}
                        Algorithm to calculate SNR of ECG signal (default:
                        default)
  -f {default,dwt}, --filter-algorithm {default,dwt}
                        Algorithm to denoise/filter the raw ECG signal
                        (default: default)


Example:-
python ecg.py -d ./Sitting_data.csv -o ./filtered_Sitting_data.csv -r 500 -s default -f default
