# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from models.basic_model import LSTMLM
from config import *
from utilts import *


flags = tf.flags
flags.DEFINE_string("data_path", "ptb_data", "Where the training/test data is stored.")
FLAGS = flags.FLAGS

def main(_):
    reader = Reader(FLAGS.data_path)
    config = SmallConfig()
  
    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)
        with tf.variable_scope('Model',reuse=None,initializer=initializer):
            trainm = LSTMLM(config,mode="Train")
        with tf.variable_scope('Model',reuse=True,initializer=initializer):
            validm = LSTMLM(config,mode="Valid")
        with tf.variable_scope('Model',reuse=True,initializer=initializer):
            testm  = LSTMLM(config,mode="Test")

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(config.epoch_num):
            # trainm.update_lr(session, lr_updater.get_lr())
            lr_decay = config.decay ** max(epoch + 1 - config.max_epoch, 0.0)
            trainm.update_lr(session,config.learning_rate*lr_decay)
   
            cost,ppl = trainm.run(session,reader)
            print "Epoch: %d Train Perplexity: %.3f" % (epoch + 1,ppl)

            cost,ppl = validm.run(session,reader,False)
            print "Epoch: %d Valid Perplexity: %.3f" % (epoch + 1,ppl)
            # lr_updater.update(ppl)
            cost,ppl = testm.run(session,reader,False)
            print "Epoch: %d Test  Perplexity: %.3f" % (epoch + 1,ppl)

            print "Epoch: %d Learing rate:%.3f" % (epoch+1,session.run(trainm.lr))

if __name__ == '__main__':
    tf.app.run()
