from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import sys
import os
import math
import random
import tensorflow as tf
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from code.models import *
from time import gmtime, strftime

from code.utils.utils import parse_commandline
from code.utils.preprocess import extract_all_features
from code.utils.utils import make_batches
from code.utils.utils import convert_to_encodings

BATCH_SIZE = 32

def create_simple_model(num_features, num_encodings, cell_type):
    model = SimpleAcousticNN(num_features,num_encodings, cell_type)
    model.build_model()
    model.add_loss_op()
    model.add_optimizer_op()
    model.add_decoder_and_wer_op()
    model.add_summary_op()

    return model

def train_model(args):
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    samples, sample_lens, transcripts = extract_all_features(os.getcwd()+ '/' + args.train_path, "wand")
    print("Finished reading files and extracting features ...")

    samples = np.transpose(samples, (0, 2, 1))
    transcripts, num_encodings = convert_to_encodings(transcripts)
    print("Finished converting targets into encodings ...")



    with tf.Graph().as_default():
        model = create_simple_model(samples.shape[2], num_encodings, 'lstm')
        print("Finished creating the model ...")
        model_config = model.get_config()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            session.run(init)
            if args.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
                print("Finished importing the saved model ...")

            saver = tf.train.Saver()

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
            global_start = time.time()
            step_ii = 0

            n_batches = int(len(samples) / BATCH_SIZE)
            for curr_epoch in range(model_config.num_epochs):
                batched_samples, batched_sample_lens, batched_transcripts = make_batches(samples, sample_lens, transcripts, BATCH_SIZE)

                total_train_cost = total_train_wer = 0
                start = time.time()

                epoch_loss_avg = 0
                epoch_wer_avg = 0
                cur_batch_iter = 0
                for cur_batch_iter in range(n_batches):
                    # print(batched_transcripts[cur_batch_iter])
                    batch_cost, wer, summary = model.train_one_batch(session, batched_samples[cur_batch_iter], 
                                                batched_transcripts[cur_batch_iter], batched_sample_lens[cur_batch_iter])
                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1 
                    epoch_loss_avg += (batch_cost - epoch_loss_avg)/(cur_batch_iter+1)
                    epoch_wer_avg += (wer - epoch_wer_avg)/(cur_batch_iter+1)

                    log = "Epoch {}/{}, train_cost = {:.3f}, train_wer = {:.3f}, time = {:.3f}"
                    print(log.format(curr_epoch+1, model_config.num_epochs, epoch_loss_avg, epoch_wer_avg, time.time() - start))

                if args.save_every is not None and args.save_to_file is not None and (curr_epoch + 1) % args.save_every == 0:
                    saver.save(session, args.save_to_file, global_step=curr_epoch + 1)


def test_model(model, args):

    samples, sample_lens, transcripts = extract_all_features(os.getcwd()+ '/' + args.train_path, "wand")
    samples = np.transpose(samples, (0, 2, 1))
    transcripts, num_encodings = convert_to_encodings(transcripts)
    train_data_batches, train_labels_batches, train_seq_batches = make_batches(args.train_path, BATCH_SIZE)

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
            n_batches = int(len(samples) / BATCH_SIZE)
            for cur_batch_iter in range(n_batches):
                # print(batched_transcripts[cur_batch_iter])
                batch_cost, wer, summary = model.test_one_batch(session, batched_samples[cur_batch_iter], 
                                            batched_transcripts[cur_batch_iter], batched_sample_lens[cur_batch_iter])
                train_writer.add_summary(summary, step_ii)
                step_ii += 1 
                epoch_loss_avg += (batch_cost - epoch_loss_avg)/(cur_batch_iter+1)
                epoch_wer_avg += (wer - epoch_wer_avg)/(cur_batch_iter+1)

                log = "Test_cost = {:.3f}, Test_wer = {:.3f}, time = {:.3f}"
                print(log.format(curr_epoch+1, model_config.num_epochs, epoch_loss_avg, epoch_wer_avg, time.time() - start))



  
def main(args):
    if args.phase == 'train':
        train_model(args)
    if args.phase == 'test':
        test_model(args)



if __name__ == '__main__':
    args = parse_commandline()
    main(args)
