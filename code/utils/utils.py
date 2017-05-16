import numpy as np 
import sys
import os
import argparse
import glob
<<<<<<< HEAD
=======
#import preprocess
>>>>>>> 25c37507b9d80d4a915f229676ac67b95502ecab

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
    parser.add_argument('--save_every', nargs='?', default=None, type=int, help="Save model every x iterations. Default is not saving at all.")
    parser.add_argument('--save_to_file', nargs='?', default='saved_models/saved_model_epoch', type=str, help="Provide filename prefix for saving intermediate models")
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

    batched_samples = make_batch(samples[p], n_batches, batch_size)
    batched_sample_lens = make_batch(sample_lens[p], n_batches, batch_size)
    batched_transcripts = make_batch(transcripts[p], n_batches, batch_size)

    return batched_samples, batched_sample_lens, batched_transcripts


def convert_to_encodings(target_data):
    char_set = set()
    for i in xrange(target_data.shape[0]):
        new_set = set(target_data)
        char_set.union(new_set)

    char_list = list(char_set)
    encodings = xrange(len(char_list))
    encoded_targets = []

    for t in xrange(len(target_data)):
        for i in xrange(len(char_list)):
            if char_list[i] in target_data[t]:
                encoded_targets.append(map(int,list(target[t].replace(ch,encodings[i]))))

    return encoded_targets

