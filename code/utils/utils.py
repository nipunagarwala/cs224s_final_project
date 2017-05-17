import numpy as np 
import sys
import os
import argparse
import glob

def parse_commandline():
    """
    Parses the command line arguments to the run method for training and testing purposes
    Inputs:
        None
    Returns:
        args: An object with the command line arguments stored in the correct values.
            phase : Train or Test
            train_path : Path for the training data
            val_path : Path for the testing data
            save_every : (Int) How often to save the model
            save_to_file : (string) Path to file to save the model too
            load_from_file : (string) Path to load the model from
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--train_path', nargs='?', default='./data/hw3_train.dat', type=str, help="Give path to training data")
    parser.add_argument('--val_path', nargs='?', default='./data/hw3_val.dat', type=str, help="Give path to val data")
    parser.add_argument('--save_every', nargs='?', default=2, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--save_to_file', nargs='?', default=os.getcwd()+ '/' + 'checkpoints/model_ckpt', type=str, help="Provide filename prefix for saving intermediate models")
    parser.add_argument('--load_from_file', nargs='?', default=None, type=str, help="Provide filename to load saved model")
    args = parser.parse_args()
    return args

def make_batch(array, n_batches, batch_size):
    return np.stack([array[i*batch_size:i*batch_size+batch_size] for i in range(n_batches)])

def make_batches(samples, sample_lens, transcripts, batch_size):
    """
    Shuffles the input data into batches.

    General usage:
        samples, sample_lens, transcripts = preprocess.extract_all_features("../../data/", "spectrogram")
        batched_samples, batched_sample_lens, batched_transcripts = utils.make_batches(samples, sample_lens, transcripts, 32)

    Returns
        batched_samples: a numpy ndarray of shape (n_batches, batch_size, n_features, max_timesteps).
            Samples of length < max_timesteps are padded with zeros.
        batched_sample_lens: a numpy ndarray of shape (n_batches, batch_size) containing the
            length of sample batches_samples[x,y] in number of timesteps
        batched_transcripts: a list of strings of shape (n_batches, batch_size) containing the
            transcript of sample batches_sample[x,y].
    """

    p = np.random.permutation(len(samples))

    n_batches = int(len(samples) / batch_size)

    batched_samples = []
    batched_sample_lens = []
    batched_transcripts = []

    for i in range(n_batches):
        batched_samples.append(samples[i*batch_size: (i+1)*batch_size])
        batched_sample_lens.append(sample_lens[i*batch_size: (i+1)*batch_size])
        batched_transcripts.append(sparse_tuple_from(transcripts[i*batch_size: (i+1)*batch_size]))

    return batched_samples, batched_sample_lens, batched_transcripts


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

def convert_to_encodings(target_data):

    char_set = set('A')
    for i in range(target_data.shape[0]):
        new_set = set(target_data[i])
        char_set |= new_set

    char_list = list(char_set)
    encodings = range(len(char_list))


    encoded_targets = []

    for t in range(target_data.shape[0]):
        cur_target = list(target_data[t])
        for i in range(len(char_list)):
            for s in range(len(cur_target)):
                if cur_target[s] == char_list[i]:
                    cur_target[s] = encodings[i]

        encoded_targets.append(list(cur_target))

    return encoded_targets, len(encodings)

