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


# python run.py --train sample-data/train




def create_simple_model(num_features, num_encodings, cell_type):
    model = SimpleEmgNN(Config, num_features, num_encodings, cell_type)

    return model

def train_model(args):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    print("Extracting features")
    samples, sample_lens, transcripts = extract_all_features(os.getcwd()+ '/' + args.train_path, "wand")
    print("Finished reading files and extracting features ...")

    samples = np.transpose(samples, (0, 2, 1))
    transcripts, num_encodings = convert_to_encodings(transcripts)
    print("Finished converting targets into encodings ...")

    with tf.Graph().as_default():
        print("Creating model")
        model = create_simple_model(samples.shape[2], num_encodings, 'lstm')
        print("Finished creating the model ...")
        init = tf.global_variables_initializer()
        
        with tf.Session() as session:
            session.run(init)
            if args.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
                print("Finished importing the saved model ...")

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
            global_start = time.time()
            step_ii = 0

            n_batches = int(len(samples) / Config.batch_size)
            if n_batches < 1:
                raise ValueError("Must have at least one batch of size %d to train model, but there are only %d datapoints available " % (Config.batch_size, len(samples)))
                
            for cur_epoch in range(Config.num_epochs):
                print("epoch", cur_epoch)
                batched_samples, batched_sample_lens, batched_transcripts = make_batches(samples, sample_lens, transcripts, Config.batch_size)

                total_train_cost = total_train_wer = 0
                start = time.time()

                epoch_loss_avg = 0
                epoch_wer_avg = 0
                cur_batch_iter = 0
                for cur_batch_iter in range(n_batches):
                    #print(batched_transcripts[cur_batch_iter])
                    batch_cost, wer, summary = model.train_one_batch(session, 
                                                    batched_samples[cur_batch_iter], 
                                                    batched_transcripts[cur_batch_iter], 
                                                    batched_sample_lens[cur_batch_iter])
                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1 
                    epoch_loss_avg += (batch_cost - epoch_loss_avg)/(cur_batch_iter+1)
                    epoch_wer_avg += (wer - epoch_wer_avg)/(cur_batch_iter+1)

                    # Show information to user
                    log = "Epoch {}/{}, train_cost = {:.3f}, train_wer = {:.3f}, time = {:.3f}"
                    print(log.format(cur_epoch+1, Config.num_epochs, epoch_loss_avg, epoch_wer_avg, time.time() - start))

                    global_step = model.global_step.eval()

                    # Save checkpoints
                    if global_step % Config.steps_per_checkpoint == 0:
                        print("saving", global_step)
                        model.saver.save(session, Config.checkpoint_dir, 
                                                global_step=model.global_step)


def test_model(model, args):

    samples, sample_lens, transcripts = extract_all_features(os.getcwd()+ '/' + args.train_path, "wand")
    samples = np.transpose(samples, (0, 2, 1))
    transcripts, num_encodings = convert_to_encodings(transcripts)
    train_data_batches, train_labels_batches, train_seq_batches = make_batches(args.train_path, Config.batch_size)

    with tf.Session() as session:
            session.run(init)
            if args.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
            else:
                raise ValueError('No pre-trained model found!')

            saver = tf.train.Saver()

            global_start = time.time()
            step_ii = 0
            n_batches = int(len(samples) / Config.batch_size)
            for cur_batch_iter in range(n_batches):
                # print(batched_transcripts[cur_batch_iter])
                batch_cost, wer, summary = model.test_one_batch(session, batched_samples[cur_batch_iter], 
                                            batched_transcripts[cur_batch_iter], batched_sample_lens[cur_batch_iter])
                train_writer.add_summary(summary, step_ii)
                step_ii += 1 
                epoch_loss_avg += (batch_cost - epoch_loss_avg)/(cur_batch_iter+1)
                epoch_wer_avg += (wer - epoch_wer_avg)/(cur_batch_iter+1)

                log = "Test_cost = {:.3f}, Test_wer = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, Config.num_epochs, epoch_loss_avg, epoch_wer_avg, time.time() - start))


def parse_commandline():
    """
    Parses the command line arguments to the run method for training and testing purposes
    Inputs:
        None
    Returns:
        args: An object with the command line arguments stored in the correct values.
            phase : Train or Test
            train_path : Path for the training data
            val_path : Path for the testing data
            save_to_file : (string) Path to file to save the model too
            load_from_file : (string) Path to load the model from
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'test'])
    parser.add_argument('--train_path', nargs='?', default='sample-data/train', type=str, help="Give path to training data")
    parser.add_argument('--val_path', nargs='?', default='sample-data/test', type=str, help="Give path to val data")
    parser.add_argument('--load_from_file', nargs='?', default=None, type=str, help="Provide filename to load saved model")
    args = parser.parse_args()
    return args

  
def main(args):
    if args.phase == 'train':
        train_model(args)
    if args.phase == 'test':
        test_model(args)



if __name__ == '__main__':
    args = parse_commandline()
    main(args)
