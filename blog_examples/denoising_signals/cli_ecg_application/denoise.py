from scipy import stats
from scipy.signal import butter, iirnotch, lfilter
import pywt
import numpy as np


## Default method
# A high pass filter allows frequencies higher than a cut-off value
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a


# A low pass filter allows frequencies lower than a cut-off value
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def notch_filter(cutoff, fs, q):
    nyq = 0.5 * fs
    freq = cutoff/nyq
    b, a = iirnotch(freq, q)
    return b, a


def highpass(data, fs, order=5):
    b,a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b,a,data)
    return x


def lowpass(data, fs, order =5):
    b,a = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(b,a,data)
    return y


def notch(data, powerline, q):
    b,a = notch_filter(powerline,q)
    z = lfilter(b,a,data)
    return z


def overlay_filter(
    data, fs, order=5, cutoff_high=0.5, cutoff_low=2, powerline=60
):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(d, c, x)
    f, e = notch_filter(powerline, fs, 30)
    z = lfilter(f, e, y)     
    return z


## Direct Wavelet Transform (DWT)
def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


## KL Transform (KLT)
def filter_outliers(dataframe, z_threshold=0.7):
    constraints = dataframe.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_threshold) \
        .all(axis=1)
    return dataframe.drop(dataframe.index[~constraints]).reset_index(drop=True)


def mean_and_subtract(dataframe):
    return dataframe - dataframe.mean(axis=0, skipna=True)


def KLTransform(a):
    """
    Returns Karhunen Loeve Transform of the input and the transformation matrix and eigenval
    
    Usage:
    import numpy as np
    a  = np.array([[1,2,4],[2,3,10]])
    
    klt,tm,eigval = KLT(a)
    print(klt)
    print(tm)
    print(eigval)
    
    # to check, the following should return the original 'a'
    original_data  = np.dot(klt.T,tm).T
        
    """

    val, vec = np.linalg.eig(np.cov(a))
    klt = np.dot(vec, a)
    return klt, vec, val
