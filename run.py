from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import random
import time
import argparse

import numpy as np 
import tensorflow as tf
from time import gmtime, strftime

from code.config import Config
from code.models import SimpleEmgNN
from code.utils.preprocess import extract_all_features
from code.utils.utils import make_batches
from code.utils.utils import convert_to_encodings


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


def create_model(session, restore):
    """
    Returns a model, which has been initialized in `session`.
    Re-opens saved model if so instructed; otherwise creates 
    a new model from scratch.
    """
    print("Creating model")
    model = SimpleEmgNN(Config)
    
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

def train_model(args):
    print("Extracting features")
    feat_info = extract_all_features(Config.train_path, Config.feature_type)
    samples, sample_lens, transcripts = feat_info
    print("Finished reading files and extracting features ...")

    samples = np.transpose(samples, (0, 2, 1))
    transcripts, _ = convert_to_encodings(transcripts)
    print("Finished converting targets into encodings ...")

    with tf.Graph().as_default():
        with tf.Session() as session:
            # Create or restore model
            model = create_model(session, args.restore)
            
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
                    batch_cost, wer, summary = model.train_one_batch(session, 
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

                    # Save checkpoints as per configuration
                    if global_step % Config.steps_per_checkpoint == 0:
                        # Checkpoints
                        checkpoint_path = os.path.join(Config.checkpoint_dir, "checkpoint.ckpt")
                        model.saver.save(session, checkpoint_path, 
                                                global_step=model.global_step)
                        # Tensorboard
                        train_writer.add_summary(summary, global_step)


def test_model(args):
    # TODO: factor out the repeated dataset creation code from train_model 
    feat_info = extract_all_features(Config.test_path, Config.feature_type)
    samples, sample_lens, transcripts = feat_info
    samples = np.transpose(samples, (0, 2, 1))
    transcripts, _ = convert_to_encodings(transcripts)
    
    with tf.Graph().as_default():
        with tf.Session() as session:
            # Create or restore model
            model = create_model(session, True) # always restore

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
                batch_cost, wer, summary = model.test_one_batch(session, 
                                                batched_samples[cur_batch_iter], 
                                                batched_transcripts[cur_batch_iter], 
                                                batched_sample_lens[cur_batch_iter])
                global_step = model.global_step.eval()

                # Show information to user
                test_loss_avg += (batch_cost - test_loss_avg)/(cur_batch_iter+1)
                test_wer_avg += (wer - test_wer_avg)/(cur_batch_iter+1)
                log = "Batch {}; So far: test_cost = {:.3f}, test_wer = {:.3f}, time = {:.3f}"
                print(log.format(cur_batch_iter,
                                 test_loss_avg, test_wer_avg, 
                                 time.time() - test_start))
                
                # Write to Tensorboard
                test_writer.add_summary(summary, global_step)


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

  
def main(args):
    # TODO: add the ability to run a test on the training data
    # to check for overfitting
    if args.phase == 'train':
        train_model(args)
    if args.phase == 'test':
        test_model(args)


if __name__ == '__main__':
    args = parse_commandline()
    main(args)
