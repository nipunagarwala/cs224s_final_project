import sys
import os
import math

import numpy as np 
import tensorflow as tf

class SimpleEmgNN(object):
    """
    Implements a recurrent neural network with multiple hidden layers and CTC loss.
    """
    def __init__(self, config, num_features, alphabet_size):
        self.config = config
        
        self.num_features = num_features
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
        # Inputs are batch size, by max seq len/timesteps, by num features
        self.inputs_placeholder = tf.placeholder(tf.float32, 
                                    shape=(None, None, self.num_features))
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
        tf.summary.scalar("loss", self.loss)

    def add_optimizer_op(self):
        # Clips by global norm
        # This can be slow -- if we run into speed trouble, check out 
        # the gradient clipping docs
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_norm)
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        

    def add_decoder_and_wer_op(self):
        # TODO decide out if we want to set merge_repeated == True
        # With merge_repeated=False:
        #     (31.4) BOTH OUR CORPORATE AND FOUNDATION COONTRIBUTIONS ERE UPRP ATS WELLR IATO OUR INDIVIDUAL CONTRIBUTIONS
        # With merge_repeated=True:
        #     (31.4) BOTH OUR CORPORATE AND FOUNDATION CONTRIBUTIONS ERE UPRP ATS WELR IATO OUR INDIVIDUAL CONTRIBUTIONS
        # (see "contributions", "wellr")

        self.all_decoded_sequences, self.all_decoded_probs = tf.nn.ctc_beam_search_decoder(
                                    inputs=self.logits, 
                                    sequence_length=self.seq_lens_placeholder,
                                    beam_width=self.config.beam_size, 
                                    top_paths=self.config.beam_size,
                                    merge_repeated=False)
        decoded_sequence = tf.cast(self.all_decoded_sequences[0], tf.int32)
        
        # TODO: add autocorrection before wer, to retrieve both WER before and after autocorrect
        # (this is non-trivial while we have the decoded_sequence data format --
        # should be easier if we move to a WER approach that is like the one we had
        # discussed)
        self.wer = tf.reduce_mean(tf.edit_distance(decoded_sequence,
                                              self.targets_placeholder,
                                              normalize=True))
        tf.summary.scalar("wer", self.wer)


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
        _, batch_cost, wer, summary, beam_decoded, beam_probs, b_m, p_m = session.run([self.train_op, 
                                            self.loss, 
                                            self.wer, 
                                            self.merged_summary_op,
                                            self.all_decoded_sequences,
                                            self.all_decoded_probs], 
                                            feed_dict)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0

        return batch_cost, wer, summary, beam_decoded, beam_probs, b_m, p_m


    def test_one_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        batch_cost, wer, summary, beam_decoded, beam_probs = session.run([self.loss, 
                                                self.wer, 
                                                self.merged_summary_op,
                                                self.all_decoded_sequences,
                                                self.all_decoded_probs], 
                                                feed_dict)
        return batch_cost, wer, summary, beam_decoded, beam_probs

    def get_config(self):
        return self.config




