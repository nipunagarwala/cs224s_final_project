from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import sys
import os
import math
import random
import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import *
from time import gmtime, strftime




def create_simple_model():
	model = SimpleAcousticNN()
	model.build_model()
	model.add_loss_op()
	model.add_optimizer_op()
	model.add_decoder_and_wer_op()
	model.add_summary_op()
	model.add_feed_dict()

	return model

def train_model(model):
	logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
	train_dataset = load_dataset(args.train_path)
	val_dataset = load_dataset(args.val_path)
	train_data_batches, train_labels_batches, train_seq_batches = make_batches(args.train_data)


	with tf.Graph().as_default():
		init = tf.global_variables_initializer()
		saver = tf.train.Saver(tf.trainable_variables())

		with tf.Session() as session:
			session.run(init)
			if args.load_from_file is not None:
				new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
				new_saver.restore(session, args.load_from_file)
			
			train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
			global_start = time.time()
			step_ii = 0

			for curr_epoch in range(Config.num_epochs):
				total_train_cost = total_train_wer = 0
				start = time.time()





def main(args):
	model = create_simple_model()
	if args.phase == 'train':
		train_model(model)
	if args.phase == 'test':
		test_model(model)



if __name__ == '__main__':
	args = parse_commandline()
	main(args)