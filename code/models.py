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
        self.all_decoded_sequences, self.all_decoded_probs = tf.nn.ctc_beam_search_decoder(
                                    inputs=self.logits, 
                                    sequence_length=self.seq_lens_placeholder,
                                    beam_width=self.config.beam_width, 
                                    top_paths=self.config.top_paths,
                                    merge_repeated=True)
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

    def train_one_batch(self, session, input_batch, target_batch, seq_batch, mode=None):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        _, batch_cost, wer, summary, beam_decoded, beam_probs = session.run([self.train_op, 
                                            self.loss, 
                                            self.wer, 
                                            self.merged_summary_op,
                                            self.all_decoded_sequences,
                                            self.all_decoded_probs], 
                                            feed_dict)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0

        return batch_cost, wer, summary, beam_decoded, beam_probs


    def test_one_batch(self, session, input_batch, target_batch, seq_batch, mode=None):
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

    def get_global_step(self, mode=None):
        return self.global_step
        



class MultiSharedEmgNN(object):

    def __init__(self, config_audio, config_whisp, config_silent, config_shared, num_features, alphabet_size):
        self.config_audio = config_audio
        self.config_whisp = config_whisp
        self.config_silent = config_silent
        self.config_shared = config_shared
        self.audible_scope = 'audible'
        self.whisp_scope = 'whispered'
        self.silent_scope = 'silent'
        self.shared_scope = 'shared'
        
        self.num_features = num_features
        self.alphabet_size = alphabet_size
               
        if self.config_audio.cell_type == 'rnn':
            self.audible_cell = tf.contrib.rnn.RNNCell
        elif self.config_audio.cell_type == 'gru':
            self.audible_cell = tf.contrib.rnn.GRUCell
        elif self.config_audio.cell_type == 'lstm':
            self.audible_cell = tf.contrib.rnn.LSTMCell
        else:
            raise ValueError('Input correct cell type')

        if self.config_whisp.cell_type == 'rnn':
            self.whisp_cell = tf.contrib.rnn.RNNCell
        elif self.config_whisp.cell_type == 'gru':
            self.whisp_cell = tf.contrib.rnn.GRUCell
        elif self.config_whisp.cell_type == 'lstm':
            self.whisp_cell = tf.contrib.rnn.LSTMCell
        else:
            raise ValueError('Input correct cell type')

        if self.config_silent.cell_type == 'rnn':
            self.silent_cell = tf.contrib.rnn.RNNCell
        elif self.config_silent.cell_type == 'gru':
            self.silent_cell = tf.contrib.rnn.GRUCell
        elif self.config_silent.cell_type == 'lstm':
            self.silent_cell = tf.contrib.rnn.LSTMCell
        else:
            raise ValueError('Input correct cell type')

        if self.config_shared.cell_type == 'rnn':
            self.shared_cell = tf.contrib.rnn.RNNCell
        elif self.config_shared.cell_type == 'gru':
            self.shared_cell = tf.contrib.rnn.GRUCell
        elif self.config_shared.cell_type == 'lstm':
            self.shared_cell = tf.contrib.rnn.LSTMCell
        else:
            raise ValueError('Input correct cell type')

        self.global_step_audible = tf.contrib.framework.get_or_create_global_step() 
        self.global_step_whisp = tf.contrib.framework.get_or_create_global_step()
        self.global_step_silent = tf.contrib.framework.get_or_create_global_step()
      
        self.add_placeholders()
        self.build_model()
        self.add_loss_op()
        self.add_optimizer_op()
        self.add_decoder_and_wer_op()
        self.add_summary_op()
        
        # Needs to be last line -- graph must be created before saver is created
        self.saver = tf.train.Saver(tf.global_variables(), 
                           keep_checkpoint_every_n_hours=self.config_shared.freq_of_longterm_checkpoint)

    def add_placeholders(self):
        self.inputs_audio_placeholder = tf.placeholder(tf.float32, 
                                    shape=(None, None, self.num_features), name='input_audio')

        self.targets_audio_placeholder = tf.sparse_placeholder(tf.int32, name='targets_audio')

        self.seq_lens_audio_placeholder = tf.placeholder(tf.int32, shape=(None), name='seq_len_audio')


        self.inputs_whisp_placeholder = tf.placeholder(tf.float32, 
                                    shape=(None, None, self.num_features), name='input_whisp')

        self.targets_whisp_placeholder = tf.sparse_placeholder(tf.int32, name='targets_whisp')

        self.seq_lens_whisp_placeholder = tf.placeholder(tf.int32, shape=(None), name='seq_len_whisp')

        self.inputs_silent_placeholder = tf.placeholder(tf.float32, 
                                    shape=(None, None, self.num_features), name='input_silent')

        self.targets_silent_placeholder = tf.sparse_placeholder(tf.int32, name='targets_silent')

        self.seq_lens_silent_placeholder = tf.placeholder(tf.int32, shape=(None), name='seq_len_audio')

    
    def build_rnn_model(self, inputs, seq_len, config, model_cell, shared_cell, model_scope, shared_scope, reuse):
        output_model = None
        state_output = None
        with tf.variable_scope(model_scope):
            rnnNet_model = tf.contrib.rnn.MultiRNNCell([model_cell(config.hidden_size) 
                                              for _ in range(config.num_layers)])
            output_model, state_output = tf.nn.dynamic_rnn(rnnNet_model, inputs,
                        sequence_length=seq_len,
                        dtype=tf.float32)


        logits = None        
        with tf.variable_scope(shared_scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            rnnNet_shared = tf.contrib.rnn.MultiRNNCell([shared_cell(config.hidden_size,
                        reuse=reuse) for _ in range(config.num_layers)])
            output_shared, state_shared = tf.nn.dynamic_rnn(rnnNet_shared, output_model,
                        sequence_length=seq_len,
                        dtype=tf.float32)
            logits = tf.contrib.layers.fully_connected(output_shared, 
                    num_outputs=self.alphabet_size,
                    activation_fn=None, 
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=tf.constant_initializer(0.0), reuse = reuse, scope='fc',
                    trainable=True)

        return logits


    def build_model(self):
        self.audible_logits = self.build_rnn_model(self.inputs_audio_placeholder, self.seq_lens_audio_placeholder, 
                                self.config_audio,self.audible_cell, self.shared_cell,  self.audible_scope,
                                self.shared_scope, False)
        self.whisp_logits = self.build_rnn_model(self.inputs_whisp_placeholder, self.seq_lens_whisp_placeholder, 
                                self.config_whisp,self.whisp_cell, self.shared_cell, self.whisp_scope,
                                self.shared_scope, True)
        self.silent_logits = self.build_rnn_model(self.inputs_silent_placeholder, self.seq_lens_silent_placeholder, 
                                self.config_silent,self.silent_cell, self.shared_cell, self.silent_scope, 
                                self.shared_scope, True)

    def add_model_loss_op(self, logits, targets, seq_len, config, summary_name, model_scope, shared_scope):
        logits = tf.transpose(logits,perm=[1,0,2])
        ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits,
                    sequence_length=seq_len, preprocess_collapse_repeated=False,
                    ctc_merge_repeated=True)

        l2_cost = 0
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope)
        for v in train_vars:
            l2_cost += tf.nn.l2_loss(v)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=shared_scope)
        for v in train_vars:
            l2_cost += tf.nn.l2_loss(v)

        loss_without_invalid_paths = tf.boolean_mask(ctc_loss, tf.less(ctc_loss, tf.constant(10000.)))
        num_valid_examples = tf.cast(tf.shape(loss_without_invalid_paths)[0], tf.int32)
        cost = tf.reduce_mean(loss_without_invalid_paths) 

        loss = config.l2_lambda * l2_cost + cost
        tf.summary.scalar(summary_name, loss)

        return num_valid_examples, loss


    def add_loss_op(self):
        self.audible_valid_examples, self.audible_loss = self.add_model_loss_op(self.audible_logits,
                 self.targets_audio_placeholder, self.seq_lens_audio_placeholder, self.config_audio,
                "audible_loss", self.audible_scope, self.shared_scope)

        self.whisp_valid_examples, self.whisp_loss = self.add_model_loss_op(self.whisp_logits,
                 self.targets_whisp_placeholder, self.seq_lens_whisp_placeholder, self.config_whisp,
                "whisp_loss", self.whisp_scope, self.shared_scope)

        self.silent_valid_examples, self.silent_loss = self.add_model_loss_op(self.silent_logits,
                 self.targets_silent_placeholder, self.seq_lens_silent_placeholder, self.config_silent,
                "silent_loss", self.silent_scope, self.shared_scope)


    def add_model_optimizer_op(self, loss, config, model_scope, shared_scope, global_step):
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope)
        tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=shared_scope)

        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.max_norm)
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

        return train_op


    def add_optimizer_op(self):

        self.audible_train_op = self.add_model_optimizer_op(self.audible_loss, self.config_audio, 
                                    self.audible_scope , self.shared_scope, self.global_step_audible)
        self.whisp_train_op = self.add_model_optimizer_op(self.whisp_loss, self.config_whisp, 
                                    self.whisp_scope , self.shared_scope, self.global_step_whisp)
        self.silent_train_op = self.add_model_optimizer_op(self.silent_loss, self.config_silent, 
                                    self.silent_scope , self.shared_scope, self.global_step_silent)

    def add_model_results_op(self,logits, seq_len, config,targets, summary_name):
        logits = tf.transpose(logits,perm=[1,0,2])
        all_decoded_sequences, all_decoded_probs = tf.nn.ctc_beam_search_decoder(
                                    inputs=logits, 
                                    sequence_length=seq_len,
                                    beam_width=config.beam_width, 
                                    top_paths=config.top_paths,
                                    merge_repeated=True)
        decoded_sequence = tf.cast(all_decoded_sequences[0], tf.int32)

        wer = tf.reduce_mean(tf.edit_distance(decoded_sequence,
                                              targets, normalize=True))
        tf.summary.scalar(summary_name, wer)

        return all_decoded_sequences, all_decoded_probs, wer


    def add_decoder_and_wer_op(self):
        self.audible_decoded_seq, self.audible_decoded_probs, self.audible_wer = \
            self.add_model_results_op(self.audible_logits, self.seq_lens_audio_placeholder, 
                                      self.config_audio, self.targets_audio_placeholder, 'audible_wer')

        self.whisp_decoded_seq, self.whisp_decoded_probs, self.whisp_wer = \
            self.add_model_results_op(self.whisp_logits, self.seq_lens_whisp_placeholder, 
                                      self.config_whisp, self.targets_whisp_placeholder, 'whisp_wer')

        self.silent_decoded_seq, self.silent_decoded_probs, self.silent_wer = \
            self.add_model_results_op(self.silent_logits, self.seq_lens_silent_placeholder, 
                                      self.config_silent, self.targets_silent_placeholder, 'silent_wer')


    def add_summary_op(self):
        self.merged_summary_op = tf.summary.merge_all()

    def get_audible_feed_dict(self, input_batch, target_batch, seq_batch):
        feed_dict = {
            self.inputs_audio_placeholder: input_batch, 
            self.targets_audio_placeholder: target_batch,
            self.seq_lens_audio_placeholder: seq_batch
        }
        return feed_dict

    def get_whisp_feed_dict(self, input_batch, target_batch, seq_batch):
        feed_dict = {
            self.inputs_whisp_placeholder: input_batch, 
            self.targets_whisp_placeholder: target_batch,
            self.seq_lens_whisp_placeholder: seq_batch
        }
        return feed_dict

    def get_silent_feed_dict(self, input_batch, target_batch, seq_batch):
        feed_dict = {
            self.inputs_silent_placeholder: input_batch, 
            self.targets_silent_placeholder: target_batch,
            self.seq_lens_silent_placeholder: seq_batch
        }
        return feed_dict


    def train_one_audible_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_audible_feed_dict(input_batch, target_batch, seq_batch)
        _, batch_cost, wer, beam_decoded, beam_probs = session.run([self.audible_train_op, 
                                            self.audible_loss, 
                                            self.audible_wer, 
                                            # self.merged_summary_op,
                                            self.audible_decoded_seq,
                                            self.audible_decoded_probs], 
                                            feed_dict)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0

        return batch_cost, wer, None, beam_decoded, beam_probs

    def train_one_whisp_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_whisp_feed_dict(input_batch, target_batch, seq_batch)
        _, batch_cost, wer, beam_decoded, beam_probs = session.run([self.whisp_train_op, 
                                            self.whisp_loss, 
                                            self.whisp_wer, 
                                            # self.merged_summary_op,
                                            self.whisp_decoded_seq,
                                            self.whisp_decoded_probs], 
                                            feed_dict)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0

        return batch_cost, wer, None, beam_decoded, beam_probs

    def train_one_silent_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_silent_feed_dict(input_batch, target_batch, seq_batch)
        _, batch_cost, wer, beam_decoded, beam_probs = session.run([self.silent_train_op, 
                                            self.silent_loss, 
                                            self.silent_wer, 
                                            # self.merged_summary_op,
                                            self.silent_decoded_seq,
                                            self.silent_decoded_probs], 
                                            feed_dict)

        if math.isnan(batch_cost): # basically all examples in this batch have been skipped 
            return 0

        return batch_cost, wer, None, beam_decoded, beam_probs

    def train_one_batch(self, session, input_batch, target_batch, seq_batch, mode):
        if mode == 'audible':
            return self.train_one_audible_batch(session, input_batch, target_batch, seq_batch)
        elif mode == 'whisp':
            return self.train_one_whisp_batch(session, input_batch, target_batch, seq_batch)
        elif mode == 'silent':
            return self.train_one_silent_batch(session, input_batch, target_batch, seq_batch)

    def test_one_audible_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        batch_cost, wer, summary, beam_decoded, beam_probs = session.run([self.audible_loss, 
                                                self.audible_wer, 
                                                self.merged_summary_op,
                                                self.audible_decoded_seq,
                                                self.audible_decoded_probs], 
                                                feed_dict)
        return batch_cost, wer, summary, beam_decoded, beam_probs

    def test_one_whisp_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        batch_cost, wer, summary, beam_decoded, beam_probs = session.run([self.whisp_loss, 
                                                self.whisp_wer, 
                                                self.merged_summary_op,
                                                self.whisp_decoded_seq,
                                                self.whisp_decoded_probs], 
                                                feed_dict)
        return batch_cost, wer, summary, beam_decoded, beam_probs

    def test_one_silent_batch(self, session, input_batch, target_batch, seq_batch):
        feed_dict = self.get_feed_dict(input_batch, target_batch, seq_batch)
        batch_cost, wer, summary, beam_decoded, beam_probs = session.run([self.silent_loss, 
                                                self.silent_wer, 
                                                self.merged_summary_op,
                                                self.silent_decoded_seq,
                                                self.silent_decoded_probs], 
                                                feed_dict)
        return batch_cost, wer, summary, beam_decoded, beam_probs

    def test_one_batch(self, session, input_batch, target_batch, seq_batch, mode):
        if mode == 'audible':
            return self.test_one_audible_batch(session, input_batch, target_batch, seq_batch)
        elif mode == 'whisp':
            return self.test_one_whisp_batch(session, input_batch, target_batch, seq_batch)
        elif mode == 'silent':
            return self.test_one_silent_batch(session, input_batch, target_batch, seq_batch)

    def get_global_step(self, mode):
        if mode == 'audible':
            return self.global_step_audible
        elif mode == 'whisp':
            return self.global_step_whisp
        elif mode == 'silent':
            return self.global_step_silent





