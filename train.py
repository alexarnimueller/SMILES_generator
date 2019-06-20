#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from model import SMILESmodel

flags = tf.app.flags
flags.DEFINE_string("dataset", "data/combined_data.csv", "dataset file containing smiles strings")
flags.DEFINE_string("name", "combined_data_a5_adLR", "run name for log and checkpoint files")
flags.DEFINE_float("lr", 0.005, "learning rate")
flags.DEFINE_integer("batch", 256, "batch size")
flags.DEFINE_integer("after", 2, "sample after how many epochs")
flags.DEFINE_integer("sample", 50, "number of molecules to sample per sampling round")
flags.DEFINE_integer("train", 20, "epochs to train")
flags.DEFINE_integer("augment", 5, "whether different SMILES strings should generated for the same molecule, [1-n]")
flags.DEFINE_boolean("preprocess", True, "whether to pre-process stereo chemistry/salts etc.")
flags.DEFINE_integer("stereo", 1, "whether stereo chemistry information should be included [0, 1]")
flags.DEFINE_boolean("reinforce", False, "whether to add most similar but novel generated mols back to training")
flags.DEFINE_string("ref", None, "a molecule to compare the generated ones to and pick similar ones to reinforce")
flags.DEFINE_float("val", 0.1, "fraction of the data to use as a validation set")
flags.DEFINE_float("seed", 42, "random seed to use")
FLAGS = flags.FLAGS


def main(_):
    print("\n----- Running SMILES LSTM model -----\n")
    model = SMILESmodel(batch_size=FLAGS.batch, dataset=FLAGS.dataset, num_epochs=FLAGS.train,
                        lr=FLAGS.lr, run_name=FLAGS.name, sample_after=FLAGS.after,
                        reinforce=FLAGS.reinforce, validation=FLAGS.val, reference=FLAGS.ref, seed=FLAGS.seed)
    model.load_data(preprocess=FLAGS.preprocess, stereochem=FLAGS.stereo, augment=FLAGS.augment)
    model.build_model()
    model.train_model(n_sample=FLAGS.sample)


if __name__ == '__main__':
    tf.app.run()
