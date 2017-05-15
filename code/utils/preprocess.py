"""
@author Pol Rosello
"""

import pickle
import glob
import os
import numpy as np
import pandas as pd
import scipy.signal

EMG_F_SAMPLE = 600.0
# AUDIO_F_SAMPLE = 16e3

EMG_FRAME_LEN = int(27e-3 * EMG_F_SAMPLE) # 27ms frame length
EMG_SHIFT_LEN = int(10e-3 * EMG_F_SAMPLE) # 10ms frame shift
# AUDIO_FRAME_LEN = 160

EMG_SIGNALS = ["emg1", "emg2", "emg3", "emg4", "emg6"] # skip emg5

def stack_context(features, k=10):
    """
    Represents timestep t as the features at steps t-k, t-k+1, ..., t-1, t, t+1, ..., t+k-1, t+k
    concatenated together.
    Inputs:
        features: a 2D tensor of shape (n_feats, n_frames)
        k: the context length to consider
    Returns:
        stacked_features: a 2D tensor of shape (n_augmented_feats, n_context_frames) where
            n_augmented_feats = n_feats * (2*k + 1)
            n_context_frames = n_frames - (2*k + 1)
    """
    n_feats, n_frames = features.shape[0], features.shape[1]
    stacked_features = np.array([features[:,frame-k:frame+k+1] for frame in range(k,n_frames-k-1)])
    stacked_features = np.reshape(stacked_features, [-1, (2*k+1)*n_feats]).T
    return stacked_features


def spectrogram_features(data, signals=EMG_SIGNALS, frame_len=EMG_FRAME_LEN,
                         frame_shift=EMG_SHIFT_LEN, flatten=True):
    """
    Computes spectrogram features given input EMG data.
    Inputs:
        data: a pandas DataFrame containing EMG data of an utterance.
        signals: the names of the signals from the DataFrame to use.
        frame_len: the number of samples per FFT frame.
        frame_shift: the number of samples between the starts of consecutive frames.
        flatten: if True, returns a 2D tensor, else a 3D tensor (see below).
    Returns:
        spectrogram:
            If flatten is False, a numpy.ndarray of shape (n_signals, n_freqs, n_frames), where
            n_signals = the number of EMG electrodes used
            n_freqs = frame_len / 2 + 1
            n_frames = n_timesteps / frame_len
            spectrogram[x,y,z] contains the power density of frequency freqs[y] measured
            by electrode x at the z-th frame (i.e. at absolute time t[z]).
            If flatten is True, then the n_signals and n_freqs are flattened into one
            dimension, such that spectrogram is instead a 2D numpy.ndarray of shape
            (n_feats, n_frames), where n_feats = (n_signals * n_freqs).
    """
    samples = np.array(data[signals].T) # samples is n_signals x n_timesteps
    noverlap = frame_len - frame_shift
    freqs, t, spectrogram = scipy.signal.spectrogram(samples, fs=EMG_F_SAMPLE,
                                                     nperseg=frame_len, noverlap=noverlap)
    
    if flatten:
        n_frames = spectrogram.shape[2]
        spectrogram = np.reshape(spectrogram, [-1, n_frames])
    
    return spectrogram

def wand_features(data, signals=EMG_SIGNALS, frame_len=EMG_FRAME_LEN,
                  frame_shift=EMG_SHIFT_LEN, k=10):
    """
    Computes features from Michael Wand's dissertation, Advancing Electromyographic
    Continuous Speech Recognition: Signal Preprocessing and Modeling (2014), as
    described in Section 4.1.2.
    Inputs:
        data: a pandas DataFrame containing EMG data of an utterance.
        signals: the names of the signals from the DataFrame to use.
        frame_len: the number of samples per frame.
        frame_shift: the number of samples between the starts of consecutive frames.
        k: the context length to consider.
    Returns:
        wand: a 2D numpy.dnarray of shape (n_feats, n_frames) of features from Wand et al.
    """
    # samples is n_signals x n_timesteps
    samples = np.array(data[signals].T)

    n_signals, n_timesteps = samples.shape[0], samples.shape[1]

    # Create the 17-point weighted moving average filter shown in Figure 4.2.
    ramp_filter = np.linspace(0,0.1,num=9)
    ma_filter = np.concatenate((ramp_filter[:-1], ramp_filter[::-1]))
    assert len(ma_filter) == 17
    
    n_frames = int(n_timesteps / frame_shift)
    n_feats = 5
    features = np.zeros((n_signals, n_feats, n_frames))

    for i in range(n_signals):
        # Mean normalize
        x = samples[i] - np.mean(samples[i])

        # Apply moving average filter to compute low frequency signal w
        w = np.convolve(x, ma_filter, mode="same")

        # Compute high frequency signal p
        p = x - w

        # Compute rectified signal r
        r = abs(p)

        # Ignore any frames that are incomplete
        # (aka if n_timesteps is 2500 but 
        # n_frames is 416 and frame_shift is 6, 
        # count up to  416*6 = 2496 rather than 2500 timesteps,
        # so we don't end up with a unit in the features that
        # is made up of an incomplete set of samples)
        for frame_id, t in enumerate(range(0, n_frames*frame_shift, frame_shift)):
            w_frame = w[t:t+frame_len]
            p_frame = p[t:t+frame_len]
            r_frame = r[t:t+frame_len]
            M_w = np.mean(w_frame)           # Frame-based mean of w
            P_w = np.mean(w_frame * w_frame) # Frame-based power of w
            P_r = np.mean(r_frame * r_frame) # Frame-based power of r
            M_r = np.mean(r_frame)

            # Zero-crossing rate of p
            z_p = len(np.where(np.diff(np.signbit(p_frame)))[0]) / len(p_frame)

            features[i, :, frame_id] = np.array([M_w, P_w, P_r, z_p, M_r])

    features = np.reshape(features, [-1, n_frames])

    features = stack_context(features, k=k)

    return features

def extract_features(pkl_filename, feature_type):
    with open(pkl_filename, "rb") as f:
        audio, emg = pickle.load(f)

    emg["triplePhones"]  = someMath(emg["pno e"])
        
    if feature_type == "wand":
        return wand_features(emg)
    elif feature_type == "spectrogram":
        return spectrogram_features(emg)
    else:
        raise RuntimeError("Invalid feature type specified")
        
def extract_features_lda(directory, feature_type):
    # Be careful about LDA on test data -- point this 
    # function to a directory that is train-only, 
    # and save the LDA parameters so we can apply them
    # to test data
    
    # feats = extract_features_lda("sample-data", "wand")
    all_features = []
    for fullpath in glob.glob(os.path.join(directory, "*.pkl")):
        pathUnits = os.path.split(fullpath)
        fn = pathUnits[-1]
        if fn.startswith("utteranceInfo"):
            continue
        else:
            features = extract_features(fullpath, feature_type)
            all_features.append(features)
    return all_features
    


def extract_all_features(directory, feature_type):
    """
    Returns
        padded_samples: a numpy ndarray of shape (n_samples, n_features, max_timesteps).
            Samples of length < max_timesteps are padded with zeros.
        sample_lens: a numpy ndarray of shape (n_samples,) containing the
            length of sample padded_samples[i] in number of timesteps.
        transcripts: a list of strings of shape (n_samples,) containing the
            transcript of sample padded_samples[i].
    """
    samples = []
    transcripts = []

    with open(os.path.join(directory, "utteranceInfo.pkl"), "rb") as f:
        meta = pickle.load(f)
        
    for i, utterance in meta.iterrows():
        pkl_filename = os.path.join(directory, utterance["label"] + ".pkl")
        features = extract_features(pkl_filename, feature_type)
        transcript = utterance["transcript"]
        samples.append(features)
        transcripts.append(transcript)

    # samples is a list of 2D ndarrays of length (n_feats, n_timesteps)

    sample_lens = np.array([s.shape[1] for s in samples], dtype=np.int64)

    n_samples = len(samples)
    n_feats = samples[0].shape[0]

    assert all(s.shape[0] == n_feats for s in samples)

    padded_samples = np.zeros((n_samples, n_feats, max(sample_lens)))

    for i, sample in enumerate(samples):
        padded_samples[i,:,0:sample_lens[i]] = sample

    return padded_samples, sample_lens, np.array(transcripts)

if __name__ == "__main__":
    features = extract_features("sample-data/002_001_0311.pkl", feature_type="wand")
