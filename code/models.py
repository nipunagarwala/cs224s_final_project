import sys
import os
import math

import numpy as np 
import tensorflow as tf

from code.utils.preprocess import get_num_features

class SimpleEmgNN(object):
    """
    Implements a recurrent neural network with multiple hidden layers and CTC loss.
    """
    def __init__(self, config, alphabet_size):
        self.config = config
        
        self.num_features = get_num_features(self.config.feature_type)
        self.alphabet_size = alphabet_size
               
        if self.config.cell_type == 'rnn':
            self.cell = tf.contrib.rnn.RNNCell
        elif self.config.cell_type == 'gru':
            self.cell = tf.contrib.rnn.GRUCell
        elif self.config.cell_type == 'lstm':
            self.cell = tf.contrib.rnn.LSTMCell
        else:
            raise ValueError('Input correct cell type')

        self.global_step = tf.contrib.framework.get_or_create_global_step() 
      
        self.add_placeholders()
        self.build_model()
        self.add_loss_op()
        self.add_optimizer_op()
        self.add_decoder_and_wer_op()
        self.add_summary_op()
        
        # Needs to be last line -- graph must be created before saver is created
        self.saver = tf.train.Saver(tf.global_variables(), 
                           keep_checkpoint_every_n_hours=self.config.freq_of_longterm_checkpoint)
                           
    def add_placeholders(self):
        # TODO FIXME clean up
        # Inputs are batch size, by max sequence length, by num features.
        # Inputs are (batch_size,  max_timesteps, n_features)
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, None, self.num_features))
        # Targets are 1-D
        self.targets_placeholder = tf.sparse_placeholder(tf.int32)
        # Sequence lengths are batch size-by-
        self.seq_lens_placeholder = tf.placeholder(tf.int32, shape=(None))
                           
    def build_model(self):
        rnnNet = tf.contrib.rnn.MultiRNNCell([self.cell(self.config.hidden_size) 
                                              for _ in range(self.config.num_layers)])
        output, _ = tf.nn.dynamic_rnn(rnnNet, self.inputs_placeholder,
                        sequence_length=self.seq_lens_placeholder,
                        dtype=tf.float32)

        logits = tf.contrib.layers.fully_connected(output, 
                    num_outputs=self.alphabet_size,
					activation_fn=None, 
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
					biases_initializer=tf.constant_initializer(0.0), 
                    trainable=True)
        
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

        self.loss = self.config.l2_lambda * l2_cost + cost

    def add_optimizer_op(self):
        # Clips by global norm
        # This can be slow -- if we run into speed trouble, check out 
        # the gradient clipping docs
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_norm)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        

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

    def get_feed_dict(self, input_batch, target_batch, seq_batch):
        feed_dict = {
            self.inputs_placeholder: input_batch, 
            self.targets_placeholder: target_batch,
            self.seq_lens_placeholder: seq_batch
        }
        return feed_dict

    def train_one_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        _, batch_cost, wer, batch_num_valid_ex, summary = session.run([self.train_op, 
                                            self.loss, self.wer, 
                                            self.num_valid_examples, 
                                            self.merged_summary_op], 
                                            feed_dict)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0

        return batch_cost, wer, summary


    def test_one_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        batch_cost, wer, batch_num_valid_ex, summary = session.run([self.loss, self.wer, 
                                                    self.num_valid_examples, 
                                                    self.merged_summary_op], 
                                                    feed_dict)
        return batch_cost, wer, summary

    def get_config(self):
        return self.config




