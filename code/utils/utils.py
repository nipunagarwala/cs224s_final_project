import numpy as np 
import sys
import os
import argparse
import glob
import preprocess

def parse_commandline():
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
    Returns
        batched_samples: a numpy ndarray of shape (n_batches, batch_size, n_features, max_timesteps).
            Samples of length < max_timesteps are padded with zeros.
        batched_sample_lens: a numpy ndarray of shape (n_batches, batch_size) containing the
            length of sample batches_samples[x,y] in number of timesteps
        batched_transcripts: a list of strings of shape (n_batches, batch_size) containing the
            transcript of sample batches_sample[x,y] in number of timesteps
    """

    p = np.random.permutation(len(samples))

    n_batches = int(len(samples) / batch_size)

    batched_samples = make_batch(samples[p], n_batches, batch_size)
    batched_sample_lens = make_batch(sample_lens[p], n_batches, batch_size)
    batched_transcripts = make_batch(transcripts[p], n_batches, batch_size)

    return batched_samples, batched_sample_lens, batched_transcripts
