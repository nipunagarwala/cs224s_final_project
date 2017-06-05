"""
@author Pol Rosello, Pamela Toman
"""

import pickle
import glob
import os
import scipy.signal

import numpy as np
import pandas as pd
from code.config import Config

from itertools import chain
from collections import Counter
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

EMG_F_SAMPLE = 600.0
# AUDIO_F_SAMPLE = 16e3

EMG_FRAME_LEN = int(27e-3 * EMG_F_SAMPLE) # 27ms frame length
EMG_SHIFT_LEN = int(10e-3 * EMG_F_SAMPLE) # 10ms frame shift
# AUDIO_FRAME_LEN = 160

EMG_SIGNALS = ["emg1", "emg2", "emg3", "emg4", "emg6"] # skip emg5

def stack_context(features, k=10, labels=None):
    """
    Represents timestep t as the features at steps t-k, t-k+1, ..., t-1, t, t+1, ..., t+k-1, t+k
    concatenated together.
    Inputs:
        features: a 2D tensor of shape (n_feats, n_frames)
        k: the context length to consider
        labels: a 1D tensor of shape (n_frames,) containing the label for each frame
    Returns:
        stacked_features: a 2D tensor of shape (n_augmented_feats, n_context_frames) where
            n_augmented_feats = n_feats * (2*k + 1)
            n_context_frames = n_frames - (2*k + 1)
        stacked_labels: a 1D tensor of shape (n_context_frames,) containing the label for each
            context frame
    """
    n_feats, n_frames = features.shape[0], features.shape[1]
    stacked_features = np.array([features[:,frame-k:frame+k+1] for frame in range(k,n_frames-k-1)])
    stacked_features = np.reshape(stacked_features, [-1, (2*k+1)*n_feats]).T

    if labels is not None:
        stacked_labels = labels[k:n_frames-k-1]
        return stacked_features, stacked_labels
    else:
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
        labels: None
    """
    samples = np.array(data[signals].T) # samples is n_signals x n_timesteps
    noverlap = frame_len - frame_shift
    freqs, t, spectrogram = scipy.signal.spectrogram(samples, fs=EMG_F_SAMPLE,
                                                     nperseg=frame_len, noverlap=noverlap)
    
    if flatten:
        n_frames = spectrogram.shape[2]
        spectrogram = np.reshape(spectrogram, [-1, n_frames])
    
    return spectrogram, None

def triphones(phone, n_phones):
    """
    Returns phone repeated n_phones times, evenly divided into 3 triphones.
    """
    if phone == "?" or phone == "SIL":
        return [phone] * n_phones

    tri_repeat = int(n_phones / 3)
    if n_phones % 3 == 0:
        tris = [phone + "_0"] * tri_repeat
        tris += [phone + "_1"] * tri_repeat
        tris += [phone + "_2"] * tri_repeat
    elif n_phones % 3 == 1:
        tris = [phone + "_0"] * tri_repeat
        tris += [phone + "_1"] * (tri_repeat + 1)
        tris += [phone + "_2"] * tri_repeat
    else:
        tris = [phone + "_0"] * (tri_repeat + 1)
        tris += [phone + "_1"] * tri_repeat
        tris += [phone + "_2"] * (tri_repeat + 1)

    return tris

def compute_subphones(phones):
    this_phone = phones[0]
    n_this_phone = 0
    subphones = []
    for p in phones:
        if p != this_phone:
            subphones += triphones(this_phone, n_this_phone)
            this_phone = p
            n_this_phone = 1
        else:
            n_this_phone += 1

    subphones += triphones(this_phone, n_this_phone)
    
    assert len(subphones) == len(phones)

    return subphones

def mode(lst):
    return Counter(lst).most_common(1)[0][0]

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
        features: a 2D numpy.ndarray of shape (n_feats, n_frames) of features from
            Wand et al.
        labels: a 1D list of length n_frames of subphone labels, one label per frame.
    """

    # samples is n_signals x n_timesteps
    samples = np.array(data[signals].T)
    phones = compute_subphones(data["phone"])

    n_signals, n_timesteps = samples.shape[0], samples.shape[1]

    # Create the 17-point weighted moving average filter shown in Figure 4.2.
    ramp_filter = np.linspace(0,0.1,num=9)
    ma_filter = np.concatenate((ramp_filter[:-1], ramp_filter[::-1]))
    assert len(ma_filter) == 17
    
    n_frames = int(n_timesteps / frame_shift)
    n_feats = 5
    features = np.zeros((n_signals, n_feats, n_frames))
    frame_phones = []

    for i in range(n_signals):
        # Mean normalize
        x = samples[i] - np.mean(samples[i])

        # Apply moving average filter to compute low frequency signal w
        w = np.convolve(x, ma_filter, mode="same")

        # Compute high frequency signal p
        p = x - w

        # Compute rectified signal r
        r = abs(p)

        # Ignore any frames that are incomplete (i.e. if n_timesteps is 2500 but 
        # n_frames is 416 and frame_shift is 6, count up to 416*6 = 2496 rather
        # than 2500 timesteps, so we don't end up with a unit in the features that
        # is made up of an incomplete set of samples)
        for frame_id, t in enumerate(range(0, n_frames*frame_shift, frame_shift)):
            w_frame = w[t:t+frame_len]
            p_frame = p[t:t+frame_len]
            r_frame = r[t:t+frame_len]
            M_w = np.mean(w_frame)           # Frame-based mean of w
            P_w = np.mean(w_frame * w_frame) # Frame-based power of w
            P_r = np.mean(r_frame * r_frame) # Frame-based power of r
            M_r = np.mean(r_frame)           # Frame-based mean of r

            # Zero-crossing rate of p
            z_p = len(np.where(np.diff(np.signbit(p_frame)))[0]) / len(p_frame)

            features[i, :, frame_id] = np.array([M_w, P_w, P_r, z_p, M_r])
            mode_phone = mode(phones[t:t+frame_len])
            frame_phones.append(mode_phone)

    features = np.reshape(features, [-1, n_frames])

    features, labels = stack_context(features, k=k, labels=frame_phones)

    return features, labels

def labels_to_int_lookup(phone_labels):
    next_value = 0
    lookup = {}

    for sample in phone_labels:
        for phoneme in sample:
            if phoneme not in lookup:
                lookup[phoneme] = next_value
                next_value += 1

    return lookup

def transform(samples, lda):
    """
    Transforms samples according to the specified LDA transformation. Careful:
    performs the transformation in place!

    Inputs:
        samples: a list of n_sample np.ndarrays, each of shape
            (n_feats, n_timesteps), where n_timesteps may vary.
        lda: a LinearDiscriminantAnalysis object, already fitted to some data.

    Returns:
        samples: a list of n_sample np.ndarrays, each of shape
            (n_components, n_timesteps), where n_timesteps may vary and
            n_components is determined by the dimensionality of the LDA
            transformation.
    """
    for i in range(len(samples)):
        samples[i] = lda.transform(samples[i].T).T

    return samples

def wand_lda(samples, phone_labels, n_components=12, subset_to_use=None):
    """
    Fits the n_components most discriminant features in the samples with respect 
    to the triphone labels, and transforms the samples accordingly. samples
    may be modified.

    Inputs:
        samples: a list of n_sample tensors, each of shape (n_feats, n_timesteps)
        phone_labels: a list of n_sample lists, each of length n_timesteps
        n_components: number of output components
        subset_to_use: a list of n_sample booleans, each one indicating whether
            to use the corresponding sample and phone_labels in fitting the LDA,
            or None if all should be used

        NOTE: n_timesteps can be different per sample!

    Returns:
        samples: a list of n_sample np.ndarrays, each of shape
            (n_components, n_timesteps)
    """
    if subset_to_use is None:
        tr_samples = samples
        tr_phone_labels = phone_labels
    else:
        tr_samples, tr_phone_labels = [], []
        for i in range(len(samples)):
            if subset_to_use[i]:
                tr_samples.append(samples[i])
                tr_phone_labels.append(phone_labels[i])
    
    if len(tr_samples) == 0:
        raise ValueError("Cannot perform LDA on no input data!")
    
    n_feats = tr_samples[0].shape[0]
    total_timesteps = sum(s.shape[1] for s in tr_samples)

    phone_lookup = labels_to_int_lookup(tr_phone_labels)

    X = np.zeros((total_timesteps, n_feats))
    y = np.zeros(total_timesteps, dtype=np.int32)

    cur_timestep = 0
    for s, (feats, labels) in enumerate(zip(tr_samples, tr_phone_labels)):
        for t in range(len(labels)):
            X[cur_timestep, :] = feats[:,t]
            y[cur_timestep] = phone_lookup[labels[t]]
            cur_timestep += 1

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X,y)
    samples = transform(samples, lda)

    return samples

def extract_features(pkl_filename, feature_type):
    with open(pkl_filename, "rb") as f:
        audio, emg = pickle.load(f)

    if feature_type == "wand" or feature_type == "wand_lda" or feature_type == "wand_ldaa":
        return wand_features(emg)
    elif feature_type == "spectrogram":
        return spectrogram_features(emg)
    else:
        raise RuntimeError("Invalid feature type specified")

def extract_all_features(directory, feature_type, session_type=None, 
    le=None, dummies=None, dummy_train=None, 
    use_scaler=True, scaler=None):
    """
    Extracts features from all files in a given directory according to the 
    `feature_type` and `session_type` requested

    Inputs:
        directory: a directory containing utteranceInfo.pkl and the pkl files
            for all utterances specified in utteranceInfo.pkl.
        feature_type: either "wand", "wand_lda", "wand_ldaa", or "spectrogram".
        session_type: None, "audible", "whispered", or "silent". if None, 
            extracts features for all sessions.
        le: an existing label encoder; can be None if a new label_encoder
            should be created
        dummies: a list of column names from utteranceInfo for which to create
            dummies to feed in over time, or None if no dummies should be created
            (e.g., dummies=["speakerId", "speakerSess", "gender", "mode"] )
        dummy_train: a pd.DataFrame containing the dummies from the training data,
            or None; the dataframe allows us to match new data to existing data 
        use_scaler: boolean indicating whether to use the scaler
        scaler: sklearn.preprocessing.StandardScaler object to use for transform,
            or None if a new transformer should be learned, or "ignore" if ignored

    Returns:
        padded_samples: a numpy ndarray of shape (n_samples, max_timesteps, n_features).
            Samples of length < max_timesteps are padded with zeros.
        sample_lens: a numpy ndarray of shape (n_samples,) containing the
            length of sample padded_samples[i] in number of timesteps.
        transcripts: a list of strings of shape (n_samples,) containing the
            transcript of sample padded_samples[i].
        label_encoder: a sklearn.preprocessing.LabelEncoder for the transcripts
        dummy_train: a pd.DataFrame containing the dummies from the training data;
            matches the incoming dummy_train unless incoming dummy_train was None
        scaler: a sklearn.preprocessing.StandardScaler object used for transform
    """
    samples = []
    original_transcripts = []
    phone_labels = []
    is_audible_sample = []
    modes = []
    sessions = []

    meta_info_path = os.path.join(directory, "utteranceInfo.pkl")
    try:
        with open(meta_info_path, "rb") as f:
            meta = pickle.load(f)
            meta["speakerSess"] =  meta["speakerId"] + "_" + meta["sessionId"]
            
            # If we want dummies, create/store the dummies as needed
            if dummies is not None:
                # Create them
                meta_dummies = pd.get_dummies(meta[dummies])
                if dummy_train is None:
                    # Store it for future
                    dummy_train = meta_dummies
                else:
                    # Reindex to match the training data, filling blanks with zeros
                    meta_dummies = meta_dummies.reindex(columns = dummy_train.columns, fill_value=0)
                meta_dummies = meta_dummies.as_matrix()
    except FileNotFoundError:
        print("Cannot open file %s -- check that directory to see if it needs to be renamed to the hardcoded path" % os.path.join(directory, "utteranceInfo.pkl"))

    for i, utterance in meta.iterrows():
        if session_type is not None and utterance["mode"] != session_type:
            continue
        pkl_filename = os.path.join(directory, utterance["label"] + ".pkl")
        features, phones = extract_features(pkl_filename, feature_type)
        samples.append(features)
        modes.append(utterance["mode"])
        sessions.append(utterance["speakerSess"])
        original_transcripts.append(utterance["transcript"])
        phone_labels.append(phones)
        is_audible_sample.append(utterance["mode"] == "audible")
        
    if len(samples) == 0:
        raise ValueError("Dataset %s has no entries when filtered for '%s' " % 
                         (meta_info_path, session_type if session_type is not None else "(none)"))
        
    if feature_type == "wand_lda":
        samples = wand_lda(samples, phone_labels)
    elif feature_type == "wand_ldaa":
        samples = wand_lda(samples, phone_labels, subset_to_use=is_audible_sample)

    # Build the encodings
    if le is None:
        le = preprocessing.LabelEncoder()
        le.fit(list(chain.from_iterable(list(x) for x in original_transcripts)))
    transcripts = []
    for text in original_transcripts:
        transcripts.append(le.transform([c for c in list(text) if c in le.classes_]))
        
    # Get lengths
    sample_lens = []
    for i, s in enumerate(samples):
        sample_lens.append(s.shape[1])
        if dummies is not None:
            dummies_through_time = np.ones((meta_dummies[i].shape[0], s.shape[1]))
            dummies_through_time *= meta_dummies[i][:,np.newaxis]
            s = np.vstack([s, dummies_through_time])
    sample_lens = np.array(sample_lens, dtype=np.int64)

    n_samples = len(samples)
    n_feats = samples[0].shape[0]

    assert all(s.shape[0] == n_feats for s in samples)

    padded_samples = np.zeros((n_samples, n_feats, max(sample_lens)))

    for i, sample in enumerate(samples):
        padded_samples[i,:,0:sample_lens[i]] = sample

    # Ensure samples are shaped (n_samples, max_timesteps, n_features)
    padded_samples = np.transpose(padded_samples, (0, 2, 1))
    n_signals, max_timesteps, n_feats = padded_samples.shape
    
    if use_scaler:
        if scaler is None:
            scaler = preprocessing.StandardScaler()
            padded_samples = np.reshape(padded_samples, (-1, n_feats))
            scaler.fit(padded_samples)
            padded_samples = np.reshape(padded_samples, (n_samples, max_timesteps, n_feats))
        padded_samples = np.reshape(padded_samples, (-1, n_feats))
        padded_samples = scaler.transform(padded_samples)
        padded_samples = np.reshape(padded_samples, (n_samples, max_timesteps, n_feats))
        
    return (padded_samples, sample_lens, np.array(transcripts), le, 
            dummy_train, np.array(modes), np.array(sessions), scaler)

def prep_data(args, path_to_data, feature_type, mode, label_encoder=None, 
                dummies=None, dummy_train=None, 
                use_scaler=True, scaler=None):
    print("Extracting features")
    # Extract features
    feat_info = extract_all_features(path_to_data, feature_type, mode, label_encoder, dummies, dummy_train, use_scaler, scaler)
    if label_encoder is None:
        if dummy_train is not None:
            raise ValueError("When label encoder is None, that means we're training -- so dummy_train should be None too. But it isn't.")
        if use_scaler and scaler is not None:
            raise ValueError("When label encoder is None, that means we're training -- so scaler should be None too. But it isn't.")
        samples, sample_lens, transcripts, label_encoder, dummy_train, modes, sessions, scaler = feat_info
        # Store label_encoder to disk
        label_fn = os.path.join(Config.checkpoint_dir, "labels.pkl")
        with open(label_fn, "wb") as f:
            pickle.dump(label_encoder, f)
        # Store dummy_train to disk
        dummy_fn = os.path.join(Config.checkpoint_dir, "dummy_train.pkl")
        with open(dummy_fn, "wb") as f:
            pickle.dump(dummy_train, f)
        print("Labels (label_encoder and dummy_train) stored")
        
        if use_scaler:
            # Store scaler
            scaler_fn = os.path.join(Config.checkpoint_dir, "scaler.pkl")
            with open(scaler_fn, "wb") as f:
                pickle.dump(scaler, f)
            print("Scaler stored")
    else:
        samples, sample_lens, transcripts, _, _, modes, sessions, scaler = feat_info
    
    # Verify to user load succeeded
    print("------")
    print("Features successfully extracted. Verification:")
    print("Input 0 shape (max_timesteps, n_features):")
    print(samples[0].shape)
    print("Input 0 active timesteps")
    print(sample_lens[0])
    print("Target 0")
    print(transcripts[0])
    print(label_encoder.inverse_transform(transcripts[0]))
    
    return samples, sample_lens, transcripts, label_encoder, dummy_train, modes, sessions, scaler



def get_separate_mode_features(args):
    
    samples_tr_comp, sample_lens_tr_comp, transcripts_tr_comp, label_encoder_comp,\
            dummy_train, mode_list_train, _, scaler = prep_data(args, 
        Config.train_path, Config.feature_type, None, None, Config.dummies, None, Config.use_scaler, None)
    # Get the dev data using the same label_encoder
    data_de_comp, lens_de_comp, transcripts_de_comp, _, _ , mode_list_dev, _, _ = prep_data(args, 
            Config.dev_path, Config.feature_type, None, label_encoder_comp, Config.dummies, dummy_train,
            Config.use_scaler, scaler)

    samples_tr_audible, sample_lens_tr_audible, transcripts_tr_audible, \
      samples_tr_whisp, sample_lens_tr_whisp, transcripts_tr_whisp, \
       samples_tr_silent, sample_lens_tr_silent, transcripts_tr_silent, \
       data_de_audible, lens_de_audible, transcripts_de_audible,\
       data_de_whisp, lens_de_whisp, transcripts_de_whisp,\
       data_de_silent, lens_de_silent, transcripts_de_silent = ([] for i in range(18))


    for i in range(len(mode_list_train)):
        if mode_list_train[i] == 'audible':
            samples_tr_audible.append(samples_tr_comp[i,:,:])
            sample_lens_tr_audible.append(sample_lens_tr_comp[i])
            transcripts_tr_audible.append(transcripts_tr_comp[i])
        elif mode_list_train[i] == 'whispered':
            samples_tr_whisp.append(samples_tr_comp[i,:,:])
            sample_lens_tr_whisp.append(sample_lens_tr_comp[i])
            transcripts_tr_whisp.append(transcripts_tr_comp[i])
        elif mode_list_train[i] == 'silent':
            samples_tr_silent.append(samples_tr_comp[i,:,:])
            sample_lens_tr_silent.append(sample_lens_tr_comp[i])
            transcripts_tr_silent.append(transcripts_tr_comp[i])
        else:
            raise RuntimeError("Cannot find mode %s" % str(mode_list_train[i]))

    for i in range(len(mode_list_dev)):
        if mode_list_dev[i] == 'audible':
            data_de_audible.append(data_de_comp[i,:,:])
            lens_de_audible.append(lens_de_comp[i])
            transcripts_de_audible.append(transcripts_de_comp[i])
        elif mode_list_dev[i] == 'whispered':
            data_de_whisp.append(data_de_comp[i,:,:])
            lens_de_whisp.append(lens_de_comp[i])
            transcripts_de_whisp.append(transcripts_de_comp[i])
        elif mode_list_dev[i] == 'silent':
            data_de_silent.append(data_de_comp[i,:,:])
            lens_de_silent.append(lens_de_comp[i])
            transcripts_de_silent.append(transcripts_de_comp[i])
        else:
            raise RuntimeError("Cannot find mode %s" % str(mode_list_dev[i]))

    print("Length of transcriptions: {0}".format(len(transcripts_tr_comp)))
    print("Dimensions per item: {0}".format(transcripts_tr_comp[0]))

    samples_tr_audible = np.concatenate(np.expand_dims(samples_tr_audible,axis=0), axis=0)
    sample_lens_tr_audible = np.array(sample_lens_tr_audible)
    transcripts_tr_audible = np.array(transcripts_tr_audible)
    samples_tr_whisp = np.concatenate(np.expand_dims(samples_tr_whisp,axis=0), axis=0)
    sample_lens_tr_whisp = np.array(sample_lens_tr_whisp)
    transcripts_tr_whisp = np.array(transcripts_tr_whisp)
    samples_tr_silent = np.concatenate(np.expand_dims(samples_tr_silent,axis=0), axis=0)
    sample_lens_tr_silent = np.array(sample_lens_tr_silent)
    transcripts_tr_silent = np.array(transcripts_tr_silent)

    data_de_audible = np.concatenate(np.expand_dims(data_de_audible,axis=0), axis=0)
    lens_de_audible = np.array(lens_de_audible)
    transcripts_de_audible = np.array(transcripts_de_audible)
    data_de_whisp = np.concatenate(np.expand_dims(data_de_whisp,axis=0), axis=0)
    lens_de_whisp = np.array(lens_de_whisp)
    transcripts_de_whisp = np.array(transcripts_de_whisp)
    data_de_silent = np.concatenate(np.expand_dims(data_de_silent,axis=0), axis=0)
    lens_de_silent = np.array(lens_de_silent)
    transcripts_de_silent = np.array(transcripts_de_silent)

    return  samples_tr_audible, sample_lens_tr_audible, transcripts_tr_audible, \
      samples_tr_whisp, sample_lens_tr_whisp, transcripts_tr_whisp, \
       samples_tr_silent, sample_lens_tr_silent, transcripts_tr_silent, \
       data_de_audible, lens_de_audible, transcripts_de_audible,\
       data_de_whisp, lens_de_whisp, transcripts_de_whisp,\
       data_de_silent, lens_de_silent, transcripts_de_silent, label_encoder_comp



if __name__ == "__main__":
    """
    To test, run from root directory: python3 code/utils/preprocess.py
    """
    samples, lens, transcripts, _, _ = extract_all_features("sample-data/train/", "wand")
    samples, lens, transcripts, _, _ = extract_all_features("sample-data/train/", "spectrogram")
    samples, lens, transcripts, _, _ = extract_all_features("sample-data/train/", "wand_lda")
