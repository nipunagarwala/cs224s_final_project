import numpy as np 
import sys
import os
import glob
import editdistance

def compute_wer(true_transcripts, decoded_transcripts):
    """
    Inputs:
        true_transcripts: a list of utterance strings
        decoded_transcripts: a second list of utterance strings

    Returns:
        Average WER per utterance, where tokens are split by whitespace.
    """
    wers = []
    for target, decoded in zip(true_transcripts, decoded_transcripts):
        target_words = target.split()
        decoded_words = decoded.split()
        distance = editdistance.eval(target_words, decoded_words)
        wer = distance / len(target_words)
        wers.append(wer)
    return wers

def compute_cer(true_transcripts, decoded_transcripts):
    cers = []
    for target, decoded in zip(true_transcripts, decoded_transcripts):
        distance = editdistance.eval(target, decoded)
        cer = distance / len(target)
        cers.append(cer)
    return cers

def make_batch(array, n_batches, batch_size):
    return np.stack([array[i*batch_size:i*batch_size+batch_size] for i in range(n_batches)])

def make_batches(samples, sample_lens, transcripts, batch_size, modes=None, sessions=None, shuffle=True):
    """
    Shuffles the input data into batches.

    General usage:
        samples, sample_lens, transcripts = preprocess.extract_all_features("../../data/", "spectrogram")
        batched_samples, batched_sample_lens, batched_transcripts = utils.make_batches(samples, sample_lens, transcripts, 32)

    Returns
        batched_samples: a numpy ndarray of shape (n_batches, batch_size, n_features, max_timesteps).
            Samples of length < max_timesteps are padded with zeros.
        batched_transcripts: a list of strings of shape (n_batches, batch_size) containing the
            transcript of sample batches_sample[x,y].
        batched_sample_lens: a numpy ndarray of shape (n_batches, batch_size) containing the
            length of sample batches_samples[x,y] in number of timesteps
    """

    n_batches = int(len(samples) / batch_size)
    if n_batches < 1:
        raise ValueError("Must have at least one batch of size %d to train model, but there are only %d datapoints available " % (batch_size, len(samples)))
    
    if shuffle:
        # Randomly permute the batches
        p = np.random.permutation(len(samples))
    else:
        p = range(len(samples))
    
    samples = samples[p]
    sample_lens = sample_lens[p]
    transcripts = transcripts[p]
    if modes is not None:
        modes = modes[p]
        batched_modes = []
    if sessions is not None:
        sessions = sessions[p]
        batched_sessions = []
    
    batched_samples = []
    batched_sample_lens = []
    batched_transcripts = []

    for i in range(n_batches):
        batched_samples.append(samples[i*batch_size: (i+1)*batch_size])
        batched_sample_lens.append(sample_lens[i*batch_size: (i+1)*batch_size])
        # Transcripts must be sparse because of tensorflow CTC requirements
        batched_transcripts.append(sparse_tuple_from(transcripts[i*batch_size: (i+1)*batch_size]))
        if modes is not None:
            batched_modes.append(modes[i*batch_size:(i+1)*batch_size])
        if sessions is not None:
            batched_sessions.append(sessions[i*batch_size:(i+1)*batch_size])

    return_args = [batched_samples, batched_transcripts, batched_sample_lens]

    if modes is not None:
        return_args.append(batched_modes)
    if sessions is not None:
        return_args.append(batched_sessions)

    return return_args


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape
