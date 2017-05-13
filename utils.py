import numpy as np 
import sys
import os
import argparse



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

def make_batches(dataset, batch_size):





