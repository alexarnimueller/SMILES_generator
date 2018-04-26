#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import SMILESmodel
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("model_path", "checkpoint/chembl", "model path within checkpoint directory")
flags.DEFINE_string("output_file", "generated/sampled.csv", "output file for molecules")
flags.DEFINE_integer("epoch_to_load", 5, "epoch_to_load")
flags.DEFINE_integer("num_sample", 1000, "number of points to sample from trained model")
flags.DEFINE_float("temp", 1.0, "temperature to sample at")
flags.DEFINE_string("frag", "G", "Fragment to grow SMILES from")
FLAGS = flags.FLAGS


def main(_):
    model = SMILESmodel()
    model.load_model_from_file(FLAGS.model_path, FLAGS.epoch_to_load)
    if FLAGS.frag[0] != 'G':
        frag = 'G' + FLAGS.frag
    else:
        frag = FLAGS.frag
    valid_mols = model.sample_points(FLAGS.num_sample, FLAGS.temp, frag)
    mol_file = open(FLAGS.output_file, 'w')
    mol_file.write("\n".join(set(valid_mols)))
    print("Valid:{}/{}".format(len(valid_mols), FLAGS.num_sample))
    print("Unique:{}".format(len(set(valid_mols))))


if __name__ == '__main__':
    tf.app.run()
