import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 48, "The size of image to use (will be center cropped). [48]")
flags.DEFINE_integer("input_width", 48, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 48, "The size of the output images to produce [48]")
flags.DEFINE_integer("output_width", 48, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("train_encoder", False, "Train encoder [False]")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(
              sess,
              input_width=FLAGS.input_width,
              input_height=FLAGS.input_height,
              output_width=FLAGS.output_width,
              output_height=FLAGS.output_height,
              batch_size=FLAGS.batch_size,
              sample_num=FLAGS.batch_size,
              y_dim=7,
              crop=FLAGS.crop,
              checkpoint_dir=FLAGS.checkpoint_dir,
              sample_dir=FLAGS.sample_dir)
        show_all_variables()
        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        # Below is codes for visualization
        OPTION = 1
        visualize(sess, dcgan, FLAGS)

if __name__ == '__main__':
  tf.app.run()
