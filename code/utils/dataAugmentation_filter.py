import pickle

import numpy as np
from scipy.signal import butter, lfilter
from scipy.fftpack import fft

from dataAugmentation_additive import add_noise

import matplotlib.pyplot as plt


# Filters out 60 Hz noise


def butter_bandstop_filter(data, lowcut, highcut, fs, order=2):
    def butter_bandstop(lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a
        
    b, a = butter_bandstop(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y





if __name__ == "__main__":
    # Get some data to play with
    #with open("data/dev/002_001_0317.pkl", "rb") as f:
    with open("data/train/002_001_0100.pkl", "rb") as f:
        audio, emg = pickle.load(f)

    fs = 600.
    lowcut = 49.
    highcut = 51.

    signal = emg["emg1"]
    signal2 = add_noise(signal, show_plot=False)
    signalCleaned = butter_bandstop_filter(signal2, lowcut, highcut, fs, order=2)


    DATA_XRANGE = np.array(range(200))
    DATA_YRANGE = (-32768, 32767) # 2-byte signed integer, as per corpus spec

    plt.figure()
    plt.plot(DATA_XRANGE/fs, signal2[DATA_XRANGE], color="red", label="Original")
    plt.plot(DATA_XRANGE/fs, signalCleaned[DATA_XRANGE], label="Filtered")
    plt.ylim(DATA_YRANGE)
    plt.title("Filtered Signal")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.legend()
    plt.show()

