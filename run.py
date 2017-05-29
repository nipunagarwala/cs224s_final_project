from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import random
import time
import argparse
import pickle
import shutil
import autocorrect
import scipy

import numpy as np 
import tensorflow as tf
from time import gmtime, strftime

from code.config import Config
from code.models import SimpleEmgNN
from code.utils.preprocess import extract_all_features
from code.utils.utils import make_batches, compute_wer


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Usage:
#
# Train a new model on the train directory in Config:
#   python run.py [--phase train]
#
# Restore and keep training a new model on train directory in Config:
#   python run.py --restore true [--phase train]
#
# Restore and test against the test dataset from Config
#   python run.py --phase test [--restore true]
#
# See config.py for additional configuration parameters.

# A copy of config.py as well as labels.pkl is included in 
# the Config.checkpoint_dir for posterity 

def generate_all_str(sparse_matrix, label_encoder):
    """
    Given a sparse matrix in Tensorflow's format representing a decoded
    batch from beam search, return the string representation of all
    decodings in the batch.
    """
    indices, values, shape = sparse_matrix
    results = ["" for _ in range(shape[0])]

    characters = label_encoder.inverse_transform(values)

    # Assumes values are ordered in row-major order.
    for (example, timestep), character in zip(indices, characters):
        results[example] += character

    return results

def generate_str_example(sparse_matrix, example_to_print, label_encoder):
    """
    Given a sparse matrix in Tensorflow's format, and 
    an integer indicating the example_to_print aka row,
    iterate over the matrix and return the string representation
    of the desired elements.
    """
    # TODO Speed this function up with some 
    # "where row == example_to_print" clauses.
    indices, values, shape = sparse_matrix
    result_str = ""

    for (example, timestep), val in zip(indices, values):
        if example == example_to_print:
            result_str += label_encoder.inverse_transform(val)
        if example > example_to_print:
            # Break out early if we're past our point
            break
    return result_str
    
def print_details_on_example(example_to_print, split,
                             samples, lens, transcripts, 
                             beam_decoded, beam_probs,
                             label_encoder,
                             show_autocorrect=False,
                             limit_beam_to=None):
    """
    Prints details of `example_to_print`: its input shape, 
    active timesteps, target text, and the beam results and their
    probabilities as well as their auto-corrected versions.
    
    Inputs:
        example_to_print: integer indicating which example from batch
            to drill down on
        split: a string indicating which split the data is from
            (e.g., "train", "dev", "test")
        samples: a np.ndarray of shape (batch_size, max_timesteps, 
            num_features)
        lens: a np.ndarray of shape (batch_size,) in which each 
            element reflects the number of active timesteps in the
            corresponding example in `samples`
        transcripts: a sparse 3-tuple; the first element is an array
            of indices, the second element is an array of values,
            and the third element is an array containing the size;
            in dense form, the matrix is batch_size-by-max-length-of-
            any-truth-text-in-batch 
            (see utils.sparse_tuple_from for format)
        beam_decoded: first output of tf.nn.ctc_beam_search_decoder
        beam_probs: second output of tf.nn.ctc_beam_search_decoder
        label_encoder: sklearn.preprocessing.LabelEncoder instance
        show_autocorrect: boolean for whether to perform autocorrection;
            when True, performance can be very slow because it requires
            searching a dictionary for similar words
        limit_beam_to: integer or None; None prints entire beam
    """
    # TODO: include information about the mode of the sample (silent/audible/etc.)
    print("\nSample %d from a %s batch:" % (example_to_print, split))
    
    print("  Input shape (max_timesteps, n_features): ", end="")
    print(samples[example_to_print].shape)
    
    print("  Input active timesteps: %d" % lens[example_to_print])
    
    ex_truth = generate_str_example(transcripts, example_to_print, label_encoder)
    print("  Target:  %s" % ex_truth)
    
    print("  Decoded (top %s, %s autocorrect): " % 
                ("all" if limit_beam_to is None else str(limit_beam_to), 
                 "paired with" if show_autocorrect else "without" ))
    for path_id, beam_result in enumerate(beam_decoded):
        if limit_beam_to and path_id >= limit_beam_to:
            break
        ex_prob = beam_probs[example_to_print][path_id]
        ex_str = generate_str_example(beam_result, example_to_print, label_encoder)
        print("    (%4.1f) %s" % (ex_prob, ex_str))
        if show_autocorrect:
            ex_str_corr = " ".join([autocorrect.spell(word) for word in ex_str.split()])
            print("           %s" % (ex_str_corr))
    print()
    
def create_model(session, restore, num_features, alphabet_size):
    """
    Returns a model, which has been initialized in `session`.
    Re-opens saved model if so instructed; otherwise creates 
    a new model from scratch.
    """
    print("Creating model")
    model = SimpleEmgNN(Config, num_features, alphabet_size)
    
    ckpt = tf.train.get_checkpoint_state(Config.checkpoint_dir)
    if restore:
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            model.saver.restore(session, ckpt.model_checkpoint_path)
            print("Model restored.")
        else:
            raise RuntimeError("Cannot restore from nonexistent checkpoint at %s" % ckpt.model_checkpoint_path)        
    else:
        session.run(tf.global_variables_initializer())
        try:
            session.run(tf.assert_variables_initialized())
        except tf.errors.FailedPreconditionError:
            raise RuntimeError("Not all variables initialized!")
        print("Created model with fresh parameters.")
            
    return model

def train_model(args, samples_tr, sample_lens_tr, transcripts_tr, label_encoder,
                      samples_de, sample_lens_de, transcripts_de):  
    with tf.Graph().as_default():
        with tf.Session() as session:
            # Create or restore model
            model = create_model(session, args.restore, 
                        samples_tr.shape[-1], len(label_encoder.classes_)+1)
            
            # Create a tensorboard writer for this session
            start_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())
            logs_path_train = os.path.join(Config.tensorboard_dir, start_time , "train")
            logs_path_dev = os.path.join(Config.tensorboard_dir, start_time , "dev")
            train_writer = tf.summary.FileWriter(logs_path_train, session.graph)
            dev_writer = tf.summary.FileWriter(logs_path_dev, session.graph)
            
            # Perform the training
            dev_iter = 0
            batched_samples_dev = []
            for cur_epoch in range(Config.num_epochs):
                batched_samples, batched_transcripts, batched_sample_lens = make_batches(
                                                                    samples_tr, 
                                                                    sample_lens_tr, 
                                                                    transcripts_tr, 
                                                                    Config.batch_size)

                epoch_start = time.time()
                epoch_losses = []
                epoch_weres = []
                for iter, cur_batch_iter in enumerate(range(len(batched_samples))):
                    # Do training step
                    batch_start = time.time()
                    batch_cost, batch_wer, train_summary, beam_decoded, beam_probs = model.train_one_batch(
                                                    session, 
                                                    batched_samples[cur_batch_iter], 
                                                    batched_transcripts[cur_batch_iter], 
                                                    batched_sample_lens[cur_batch_iter])

                    global_step = model.global_step.eval()

                    should_train_report = (global_step % Config.steps_per_train_report == 0)
                    should_dev_report = (global_step % Config.steps_per_dev_report == 0)
                    should_checkpoint = (global_step % Config.steps_per_checkpoint == 0)
                    
                    # Monitor training -- training performance
                    if should_train_report:
                        # Print training information
                        batch_end = time.time()
                        batch_time = batch_end - batch_start
                        epoch_losses.append(batch_cost)
                        epoch_weres.append(batch_wer)
                        log = "Epoch {}/{}, overall step {}, batch {}/{} in epoch"
                        print(log.format(cur_epoch+1, 
                                         Config.num_epochs,
                                         global_step,
                                         iter+1,
                                         len(batched_samples)))
                        log = "    batch cost = {:.3f}, wer = {:.3f}, time = {:.3f}"
                        print(log.format(batch_cost,
                                         batch_wer,
                                         batch_time))
                        log = "    overall cost = {:.3f}+/-{:.3f}, wer = {:.3f}+/-{:.3f}, est epoch time = {:.3f} hrs (95%CI)"
                        print(log.format(np.mean(epoch_losses),
                                         2*scipy.stats.sem(epoch_losses),
                                         np.mean(epoch_weres),
                                         2*scipy.stats.sem(epoch_weres),
                                         ((Config.num_epochs / len(batched_samples)) * batch_time) / 3600.
                                         ))
                        # Watch performance
                        # TODO print this performance information to disk every n batches 
                        # (e.g., 100) so we have a qualitative trace for performance over time
                        print_details_on_example(0, "train", batched_samples[cur_batch_iter],
                                                    batched_sample_lens[cur_batch_iter],
                                                    batched_transcripts[cur_batch_iter],
                                                    beam_decoded,
                                                    beam_probs,
                                                    label_encoder)

                        # Tensorboard -- training
                        train_writer.add_summary(train_summary, global_step)   
                        train_writer.flush()
          
          
                    # Monitor training -- dev performance
                    if should_dev_report:
                        print("--------")
                        if dev_iter >= len(batched_samples_dev):
                            dev_iter = 0
                            batched_samples_dev, batched_transcripts_dev, batched_sample_lens_dev = make_batches(
                                                                    samples_de, 
                                                                    sample_lens_de, 
                                                                    transcripts_de, 
                                                                    Config.batch_size)
                        
                        dev_cost, dev_wer, dev_summary, dev_beam_decoded, dev_beam_probs = model.test_one_batch(
                                                session, 
                                                batched_samples_dev[dev_iter], 
                                                batched_transcripts_dev[dev_iter], 
                                                batched_sample_lens_dev[dev_iter])
                        log = "DEV: batch cost = {:.3f}, wer = {:.3f}"
                        print(log.format(dev_cost, dev_wer))
                        
                        # Watch performance
                        print_details_on_example(0, "dev", batched_samples_dev[dev_iter],
                                                    batched_sample_lens_dev[dev_iter],
                                                    batched_transcripts_dev[dev_iter],
                                                    dev_beam_decoded,
                                                    dev_beam_probs,
                                                    label_encoder)
                                             
                        print("Computing WER for batch")
                        true_transcripts = generate_all_str(batched_transcripts_dev[dev_iter], label_encoder)
                        decoded_transcripts = generate_all_str(dev_beam_decoded[0], label_encoder)
                        dev_wer = compute_wer(true_transcripts, decoded_transcripts)
                        print("Word WER of batch =", dev_wer)
                        dev_wer_summary = tf.Summary(value=[tf.Summary.Value(tag="word_wer", simple_value=dev_wer)])
                        dev_writer.add_summary(dev_wer_summary, global_step)

                        # Tensorboard -- dev results
                        dev_writer.add_summary(dev_summary, global_step)
                        dev_writer.flush()
                        
                        # Increment dev_iter
                        dev_iter += 1
                        print("--------")
                        
                    
                    # Save checkpoints as per configuration
                    if should_checkpoint:
                        # Checkpoints
                        checkpoint_path = os.path.join(Config.checkpoint_dir, "checkpoint.ckpt")
                        model.saver.save(session, checkpoint_path, 
                                                global_step=model.global_step)


def test_model(args, samples, sample_lens, transcripts, label_encoder):
    with tf.Graph().as_default():
        with tf.Session() as session:
            # Create or restore model
            model = create_model(session, args.restore, 
                        samples.shape[-1], len(label_encoder.classes_)+1)

            # Create a tensorboard writer for this session
            logs_path = os.path.join(Config.tensorboard_dir, 
                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()), "val")
            test_writer = tf.summary.FileWriter(logs_path, session.graph)

            # Create dataset
            batched_samples, batched_transcripts, batched_sample_lens = make_batches(samples, sample_lens, transcripts, Config.batch_size)
            
            test_start = time.time()
            test_loss_avg = 0
            test_wer_avg = 0
            for cur_batch_iter in range(len(batched_samples)):
                # Do test step
                batch_cost, wer, summary, beam_decoded, beam_probs = model.test_one_batch(
                                                session, 
                                                batched_samples[cur_batch_iter], 
                                                batched_transcripts[cur_batch_iter], 
                                                batched_sample_lens[cur_batch_iter])
                global_step = model.global_step.eval()

                # Show information to user
                test_loss_avg += (batch_cost - test_loss_avg)/(cur_batch_iter+1)
                test_wer_avg += (wer - test_wer_avg)/(cur_batch_iter+1)
                # Watch performance
                num_examples_in_batch = beam_probs.shape[0]
                for example_id in range(num_examples_in_batch):
                    # TODO dump the test results to a file rather than to screen
                    print_details_on_example(example_id, "test", batched_samples[cur_batch_iter],
                                                        batched_sample_lens[cur_batch_iter],
                                                        batched_transcripts[cur_batch_iter],
                                                        beam_decoded,
                                                        beam_probs,
                                                        label_encoder,
                                                        limit_beam_to=1)
                # Write to Tensorboard
                test_writer.add_summary(summary, global_step)
                test_writer.flush()
                
            log = "Test_cost = {:.3f}, test_wer = {:.3f}, time = {:.3f}"
            print(log.format(test_loss_avg, test_wer_avg, 
                             time.time() - test_start))


def parse_commandline():
    """
    Parses the command line arguments to the run method for training and testing purposes
    Inputs:
        None
    Returns:
        args: An object with the command line arguments stored in the correct values.
            phase : Train or Test
            restore: True or False
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--restore', nargs='?', default=False, type=bool, help="Whether to restore from checkpoint directory specified in Config (default is false; overriden to be True whenever phase is test)")
    args = parser.parse_args()
    return args

def prep_data(args, path_to_data, feature_type, mode, label_encoder=None):
    print("Extracting features")
    # Extract features
    feat_info = extract_all_features(path_to_data, feature_type, mode, label_encoder)
    if label_encoder is None:
        samples, sample_lens, transcripts, label_encoder = feat_info
        # Store label_encoder to disk
        label_fn = os.path.join(Config.checkpoint_dir, "labels.pkl")
        with open(label_fn, "wb") as f:
            pickle.dump(label_encoder, f)
        print("Labels stored")
    else:
        samples, sample_lens, transcripts, _ = feat_info
    
    # Verify to user load succeeded
    print("------")
    print("Features successfully extracted. Verification:")
    print("Input 0 shape (max_timesteps, n_features):")
    print(samples[0].shape)
    print("Input 0 active timesteps")
    print(sample_lens[0])
    print("Target 0")
    print(transcripts[0])
    print(label_encoder.inverse_transform(transcripts[0]))
    
    return samples, sample_lens, transcripts, label_encoder
  
def main(args):
    # TODO make the storing of config files with the more natural -- 
    # perhaps in json rather than in code 
    if not os.path.exists(Config.checkpoint_dir):
        os.makedirs(Config.checkpoint_dir)
    if os.path.isfile(os.path.join(Config.checkpoint_dir, "config.py")):
        raise RuntimeError("There is already a configuration file in directory '%s' -- please restore it by hand to code/config.py or delete it" % Config.checkpoint_dir)
    shutil.copy("code/config.py", Config.checkpoint_dir)
    
    if args.phase == 'train':
        # Get the training data
        data_tr, lens_tr, transcripts_tr, le = prep_data(args, 
                    Config.train_path, Config.feature_type, Config.mode, None)
        # Get the dev data using the same label_encoder
        data_de, lens_de, transcripts_de, _ = prep_data(args, 
                    Config.dev_path, Config.feature_type, Config.mode, le)
        # Run model training         
        train_model(args, data_tr, lens_tr, transcripts_tr, le,
                          data_de, lens_de, transcripts_de)
    
    elif args.phase == 'test':
        args.restore = True
        
        # Retrieve labels
        label_fn = os.path.join(Config.checkpoint_dir, "labels.pkl")
        if label_fn and os.path.isfile(label_fn):
            with open(label_fn, "rb") as f:
                label_encoder = pickle.load(f)
            print("Labels restored")
        else:
            raise RuntimeError("Cannot restore label_encoder from %s" % label_fn)
            
        # Prep data
        data, lens, transcripts, _ = prep_data(args, 
                    Config.train_path, Config.feature_type, Config.mode, label_encoder)
        # Run the model test           
        test_model(args, data, lens, transcripts, label_encoder)
    
    else:
        raise RuntimeError("Phase '%s' is unknown" % args.phase)

if __name__ == '__main__':
    args = parse_commandline()
    main(args)
