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

import numpy as np 
import tensorflow as tf
from time import gmtime, strftime

from code.config import Config
from code.models import SimpleEmgNN
from code.utils.preprocess import extract_all_features
from code.utils.utils import make_batches


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

def print_example(sparse_matrix, example_to_print, label_encoder):
    """
    Given a sparse matrix in Tensorflow's format, and 
    an integer indicating the example_to_print aka row,
    iterate over the matrix and print out the desired elements.
    
    Note: This is very slow if example_to_print isn't at
    the front.  Could be improved with some 
    "where row == example_to_print" clauses.
    """
    indices, values, shape = sparse_matrix
    for (example, timestep), val in zip(indices, values):
        if example == example_to_print:
            print(label_encoder.inverse_transform(val), end="" )
    print()
    
def print_details_on_example(example_to_print, 
                             samples, lens, transcripts, 
                             beam_decoded, beam_probs,
                             label_encoder,
                             limit_beam_to=None):
    """
    Prints details of `example_to_print`: its input shape, 
    active timesteps, target text, and the beam results and their
    probabilities.
    
    Inputs:
        example_to_print: integer indicating which example from batch
            to drill down on
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
        limit_beam_to: integer or None; None prints entire beam
    """
    # TODO: include information about the mode of the sample (silent/audible/etc.)
    print("\nSample %d from batch:" % example_to_print)
    
    print("  Input shape (max_timesteps, n_features): ", end="")
    print(samples[example_to_print].shape)
    
    print("  Input active timesteps: ", end="")
    print(lens[example_to_print])
    
    print("  Target:  ", end="")
    print_example(transcripts, example_to_print, label_encoder)
    
    print("  Decoded (top %s): " % ("all" if limit_beam_to is None else str(limit_beam_to)))
    for path_id, beam_result in enumerate(beam_decoded):
        if limit_beam_to and path_id >= limit_beam_to:
            break
        print("    (%4.1f) " % beam_probs[example_to_print][path_id], end="")
        print_example(beam_result, example_to_print, label_encoder)
    
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

def train_model(args, samples, sample_lens, transcripts, label_encoder):  
    with tf.Graph().as_default():
        with tf.Session() as session:
            # Create or restore model
            model = create_model(session, args.restore, 
                        samples.shape[-1], len(label_encoder.classes_)+1)
            
            # Create a tensorboard writer for this session
            logs_path = os.path.join(Config.tensorboard_dir, 
                             strftime("%Y_%m_%d_%H_%M_%S", gmtime()), "train")
            train_writer = tf.summary.FileWriter(logs_path, session.graph)
            
            # Perform the training
            for cur_epoch in range(Config.num_epochs):
                batched_samples, batched_transcripts, batched_sample_lens = make_batches(samples, sample_lens, transcripts, Config.batch_size)

                epoch_start = time.time()
                epoch_loss_avg = 0
                epoch_wer_avg = 0
                for cur_batch_iter in range(len(batched_samples)):
                    # Do training step
                    batch_cost, wer, summary, beam_decoded, beam_probs = model.train_one_batch(
                                                    session, 
                                                    batched_samples[cur_batch_iter], 
                                                    batched_transcripts[cur_batch_iter], 
                                                    batched_sample_lens[cur_batch_iter])

                    global_step = model.global_step.eval()

                    # Show information to user
                    log = "Epoch {}/{}, overall step {}, train_cost = {:.3f}, train_wer = {:.3f}, time = {:.3f}"
                    epoch_loss_avg += (batch_cost - epoch_loss_avg)/(cur_batch_iter+1)
                    epoch_wer_avg += (wer - epoch_wer_avg)/(cur_batch_iter+1)
                    print(log.format(cur_epoch+1, 
                                     Config.num_epochs, 
                                     global_step, 
                                     epoch_loss_avg, epoch_wer_avg, 
                                     time.time() - epoch_start))
                    # Watch performance
                    print_details_on_example(0, batched_samples[cur_batch_iter],
                                                batched_sample_lens[cur_batch_iter],
                                                batched_transcripts[cur_batch_iter],
                                                beam_decoded,
                                                beam_probs,
                                                label_encoder)
           
                    # Save checkpoints as per configuration
                    if global_step % Config.steps_per_checkpoint == 0:
                        # Checkpoints
                        checkpoint_path = os.path.join(Config.checkpoint_dir, "checkpoint.ckpt")
                        model.saver.save(session, checkpoint_path, 
                                                global_step=model.global_step)
                        # Tensorboard
                        train_writer.add_summary(summary, global_step)


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
                    print_details_on_example(example_id, batched_samples[cur_batch_iter],
                                                        batched_sample_lens[cur_batch_iter],
                                                        batched_transcripts[cur_batch_iter],
                                                        beam_decoded,
                                                        beam_probs,
                                                        label_encoder,
                                                        limit_beam_to=1)
                # Write to Tensorboard
                test_writer.add_summary(summary, global_step)
                
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

def prep_data(args, path_to_data):
    print("Extracting features")
    # Extract features
    feat_info = extract_all_features(path_to_data, Config.feature_type)
    samples, sample_lens, transcripts, label_encoder = feat_info
    
    # Restore labels from disk instead if restoring
    label_fn = os.path.join(Config.checkpoint_dir, "labels.pkl")
    if args.restore:
        if label_fn and os.path.isfile(label_fn):
            with open(label_fn, "rb") as f:
                label_encoder = pickle.load(f)
        else:
            raise RuntimeError("Cannot restore label_encoder from %s" % label_fn)
        print("Labels restored")
    else:
        with open(label_fn, "wb") as f:
            pickle.dump(label_encoder, f)
        print("Labels stored")
    
    # Verify to user load succeeded
    print("Features successfully extracted. Verification:")
    print("------")
    print("Input 0 shape (max_timesteps, n_features):")
    print(samples[0].shape)
    print("Input 0 active timesteps")
    print(sample_lens[0])
    print("Target 0")
    print(transcripts[0])
    print(label_encoder.inverse_transform(transcripts[0]))
    
    return samples, sample_lens, transcripts, label_encoder
  
def main(args):
    # TODO: add the ability to run a test on the training data
    # to check for overfitting -- aka add a validation set
    # and the setup for it
    if args.phase == 'train':
        data, lens, transcripts, le = prep_data(args, Config.train_path)
        train_model(args, data, lens, transcripts, le)
    elif args.phase == 'test':
        args.restore = True
        data, lens, transcripts, le = prep_data(args, Config.test_path)
        test_model(args, data, lens, transcripts, le)
    else:
        raise RuntimeError("Phase '%s' is unknown" % args.phase)

if __name__ == '__main__':
    args = parse_commandline()
    main(args)
