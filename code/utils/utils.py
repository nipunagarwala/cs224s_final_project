import numpy as np 
import sys
import os
import argparse


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

