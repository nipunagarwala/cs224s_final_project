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
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from code.models import *
from time import gmtime, strftime

from code.utils.utils import parse_commandline
from code.utils.utils import make_batches

def create_simple_model():
    model = SimpleAcousticNN()
    model.build_model()
    model.add_loss_op()
    model.add_optimizer_op()
    model.add_decoder_and_wer_op()
    model.add_summary_op()

    return model

def train_model(model, args):
    model_config = model.get_config()
    
    logs_path = "tensorboard/" + strftime("%Y_%m_%d_%H_%M_%S", gmtime())
    train_data_batches, train_labels_batches, train_seq_batches = make_batches(args.train_path)


    with tf.Graph().as_default():
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            if args.load_from_file is not None:
                new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
                new_saver.restore(session, args.load_from_file)
            saver = tf.train.Saver(tf.trainable_variables())

            train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
            global_start = time.time()
            step_ii = 0
            for curr_epoch in range(model_config.num_epochs):
                total_train_cost = total_train_wer = 0
                start = time.time()

                epoch_loss_avg = 0
                cur_batch_iter = 0
                for cur_batch in random.sample(range(num_batches_per_epoch),num_batches_per_epoch):
                    batch_cost, wer, summary = model.train_one_batch(self, session, input_batch, target_batch, seq_batch)
                    train_writer.add_summary(summary, step_ii)
                    step_ii += 1 
                    epoch_loss_avg += (batch_cost - epoch_loss_avg)/(cur_batch_iter+1)

                    log = "Epoch {}/{}, train_cost = {:.3f}, train_wer = {:.3f}, time = {:.3f}"
                    print(log.format(curr_epoch+1, model_config.num_epochs, train_cost, train_wer, time.time() - start))

                if args.save_every is not None and args.save_to_file is not None and (curr_epoch + 1) % args.save_every == 0:
                    saver.save(session, args.save_to_file, global_step=curr_epoch + 1)


def test_model(model, args):
    if args.load_from_file is not None:
        new_saver = tf.train.import_meta_graph('%s.meta'%args.load_from_file, clear_devices=True)
        new_saver.restore(session, args.load_from_file)
    else:
        raise ValueError('No pre-trained model found!')

    train_data_batches, train_labels_batches, train_seq_batches = make_batches(args.train_path, BATCH_SIZE)




  
def main(args):
    model = create_simple_model()
    print(args)
    if args.phase == 'train':
        train_model(model, args)
    if args.phase == 'test':
        test_model(model, args)



if __name__ == '__main__':
    args = parse_commandline()
    main(args)
