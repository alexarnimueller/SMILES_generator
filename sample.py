#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model import SMILESmodel
import tensorflow as tf

flags = tf.compat.v1.app.flags
flags.DEFINE_string("model", "checkpoint/combined_data_a5_adLR/", "model path within checkpoint directory")
flags.DEFINE_string("out", "generated/combined_data_a5_adLR_10k_sampled.csv", "output file for molecules")
flags.DEFINE_integer("epoch", 14, "epoch_to_load")
flags.DEFINE_integer("num", 10000, "number of points to sample from trained model")
flags.DEFINE_float("temp", 1.0, "temperature to sample at")
flags.DEFINE_string("frag", "^", "Fragment to grow SMILES from. default: start character '^'")
flags.DEFINE_float("seed", 42, "random seed to use")
FLAGS = flags.FLAGS


def main(_):
    print("\n----- Running SMILES LSTM model -----\n")
    model = SMILESmodel(dataset=FLAGS.model, seed=FLAGS.seed)
    model.load_model_from_file(FLAGS.model, FLAGS.epoch)
    if FLAGS.frag[0] != '^':
        frag = '^' + FLAGS.frag
    else:
        frag = FLAGS.frag
    print("Starting character(s): %s" % frag)
    valid_mols = model.sample_points(n_sample=FLAGS.num, temp=FLAGS.temp, prime_text=frag)
    mol_file = open(FLAGS.out, 'w')
    mol_file.write("\n".join(set(valid_mols)))
    mol_file.close()
    print("Valid:{}/{}".format(len(valid_mols), FLAGS.num))
    print("Unique:{}".format(len(set(valid_mols))))


if __name__ == '__main__':
    tf.compat.v1.app.run()
