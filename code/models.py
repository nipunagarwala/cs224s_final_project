import numpy as np 
import sys
import os
import tensorflow as tf


class Config(Object):

	def __init__(self):
		batch_size = 64
		lr = 1e-3
		l2_lambda = 0.0000001
		hidden_size = 128
		num_epochs = 50
		num_layers = 3
		num_classes = 20
		num_features = 100 #TO FIX!!!!
		max_norm = 10



class SimpleAcousticNN(Object):
	"""
    Implements a recurrent neural network with multiple hidden layers and CTC loss.
    """

	def __init__(self, num_features, seq_len, cell_type):
		self.config = Config()
		self.input_placeholder = tf.placeholder(tf.float32, shape=(None, None, num_features))
		self.target_placeholder = tf.sparse_placeholder(tf.int32)
		self.seq_len_placeholder = tf.placeholder(tf.int32, shape=(None))
		if cell_type == 'rnn':
			self.cell = tf.contrib.rnn.RNNCell(num_units = self.config.num_hidden, 
							input_size=(None, self.config.num_features))
		elif cell_type == 'gru':
			self.cell = tf.contrib.rnn.GRUCell(num_units = self.config.num_hidden, 
							input_size=(None, self.config.num_features))
		elif cell_type == 'lstm':
			self.cell = tf.contrib.rnn.LSTMCell(num_units = self.config.num_hidden, 
							input_size=(None, self.config.num_features))
		else:
			raise ValueError('Input correct cell type')


	def build_model(self):
		W = tf.get_variable("Weights", shape=[self.config.hidden_size, self.config.num_classes],
		                    initializer=tf.contrib.layers.xavier_initializer())
		b = tf.get_variable("Bias", shape=[Config.num_classes])

		rnnNet = tf.contrib.rnn.MultiRNNCell(cells = [self.cell]*self.config.num_layers, state_is_tuple=True)
		(rnnNet_out, rnnNet_state) = tf.nn.dynamic_rnn(cell = rnnNet, inputs=self.inputs_placeholder,
		                sequence_length=self.seq_len_placeholder,dtype=tf.float32)

		cur_shape = tf.shape(rnnNet_out)
		rnnOut_2d = tf.reshape(rnnNet_out, [-1, cur_shape[2]])

		logits_2d = tf.matmul(rnnOut_2d, W) + b
		logits = tf.reshape(logits_2d,[cur_shape[0], cur_shape[1], Config.num_classes])

		self.logits = logits


	def add_loss_op(self):
		self.logits = tf.transpose(self.logits,perm=[1,0,2])
		ctc_loss = tf.nn.ctc_loss(labels=self.targets_placeholder, inputs=self.logits,
		            sequence_length=self.seq_lens_placeholder, preprocess_collapse_repeated=False,
		            ctc_merge_repeated=True)

		l2_cost = 0
		train_vars = tf.trainable_variables()
		for v in train_vars:
			l2_cost += tf.nn.l2_loss(v)

		loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
		self.num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
		cost = tf.reduce_mean(loss_without_invalid_paths) 

		self.loss = Config.l2_lambda * l2_cost + cost

	def add_optimizer_op(self):
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),self.config.max_norm)
		optimizer = tf.train.AdamOptimizer(self.config.lr)
		self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 

	def add_decoder_and_wer_op(self):
		(all_decoded_sequence,log_probabilities) = tf.nn.ctc_beam_search_decoder(inputs=self.logits, sequence_length=self.seq_lens_placeholder,
		                          merge_repeated=False)
		wer = tf.reduce_mean(tf.edit_distance(hypothesis=tf.cast(all_decoded_sequence[0],tf.int32), 
		                                    truth=self.targets_placeholder))

		decoded_sequence = all_decoded_sequence[0]

		tf.summary.scalar("loss", self.loss)
		tf.summary.scalar("wer", wer)

		self.decoded_sequence = decoded_sequence
		self.wer = wer

	def add_summary_op(self):
		self.merged_summary_op = tf.summary.merge_all()

	def add_feed_dict(self, input_batch, target_batch, seq_batch):
		self.feed_dict = {self.inputs_placeholder:input_batch, self.targets_placeholder:target_batch,
							self.seq_len_placeholder:seq_batch}

	def train_one_batch(self, session, train=True):
		_,batch_cost, wer, batch_num_valid_ex, summary = session.run([self.train_op, self.loss, self.wer, 
													self.num_valid_examples, self.merged_summary_op], self.feed_dict)

		if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
			return 0

		return batch_cost, wer, summary


	def test_one_batch(self, session):
		batch_cost, wer, batch_num_valid_ex, summary = session.run([self.loss, self.wer, 
													self.num_valid_examples, self.merged_summary_op], self.feed_dict)

		return batch_cost, wer, summary



