import pickle

import numpy as np
import matplotlib.pyplot as plt


# Implements & illustrates intermittent 60 Hz noise.

# "Mains hum, electric hum, or power line hum is a sound associated with 
# alternating current at the frequency of the mains electricity. The fundamental 
# frequency of this sound is usually 50 Hz or 60 Hz, depending on the local 
# power-line frequency. The sound often has heavy harmonic content above 50–60 Hz.
# https://en.wikipedia.org/wiki/Mains_hum
#
# "The harmonics are going to depend on the particular power generation and will 
# vary based on uncontrolled variations in both the grid and how the hum is actually 
# being introduced in to the signal."
# https://sound.stackexchange.com/questions/25453/a-mathematical-formula-describing-mains-hum



def add_noise(emg_signal, show_plot=True):
    # Sampling
    # 1 second of data requires 600 frames.  And 600 fps is 600 Hz, sampling rate of EMG.
    fs = 600 
    Ts = 1/fs

    # Time vector
    t = np.arange(0, len(emg_signal)/fs, Ts) # each unit of t is a second

    # Noise
    randAmplitudeScale = np.random.random()*0.1
    randOffset = t[np.random.randint(0, fs)]

    fNoise = 50;                                           # Frequency [Hz]
    aNoise = randAmplitudeScale*abs(np.max(emg_signal))    # Amplitude
    noise  = aNoise * np.sin(2 * np.pi * t * fNoise + randOffset)

    # Signal + Noise
    signalNoise = emg_signal + noise
    
    # Plot an example of augmentation
    if show_plot:
        DATA_XRANGE = list(range(200))
        DATA_YRANGE = (-32768, 32767) # 2-byte signed integer, as per corpus spec

        plt.subplots(nrows=3, ncols=1)
        plt.tight_layout()

        plt.subplot(3, 1, 1)
        plt.title("Data Augmentation: Additive Noise")
        plt.plot(t[DATA_XRANGE], signal[DATA_XRANGE]) 
        plt.ylim(DATA_YRANGE)
        plt.text(.5,.98,'EMG Signal',
            horizontalalignment='center',
            verticalalignment='top',
            transform=plt.gca().transAxes)
        plt.ylabel("Amplitude")
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.subplot(3, 1, 2)
        plt.plot(t[DATA_XRANGE], noise[DATA_XRANGE])
        plt.ylim(DATA_YRANGE)
        plt.text(.5,.98, "50 Hz Noise\n(Amplitude of %4.2fx Signal Range, Offset of %4.2f sec)" %
            (randAmplitudeScale, randOffset),
            horizontalalignment='center',
            verticalalignment='top',
            transform=plt.gca().transAxes)
        plt.ylabel("Amplitude")
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.subplot(3, 1, 3)
        plt.plot(t[DATA_XRANGE], signalNoise[DATA_XRANGE])
        plt.ylim(DATA_YRANGE)
        plt.text(.5,.98,'Noisy Signal',
            horizontalalignment='center',
            verticalalignment='top',
            transform=plt.gca().transAxes)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
        plt.show()

    return signalNoise



if __name__ == "__main__":
    # Get some data to play with
    with open("data/train/002_001_0100.pkl", "rb") as f:
        audio, emg = pickle.load(f)

    signal = emg["emg1"]
    signalNoise = add_noise(signal)
