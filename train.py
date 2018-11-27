#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from model import SMILESmodel

flags = tf.app.flags
flags.DEFINE_string("dataset", "data/training.csv", "dataset file containing smiles strings")
flags.DEFINE_string("run_name", "combined_data", "run name for log and checkpoint files")
flags.DEFINE_string("tokenizer", "default", "Tokenizer for one-hot encoding: 'default': 71 tokens;"
                                            "'generate': generate new tokenizer from input data")
flags.DEFINE_float("learning_rate", 0.002, "learning rate")
flags.DEFINE_integer("batch_size", 512, "batch size")
flags.DEFINE_integer("sample_after", 1, "sample after how many epochs")
flags.DEFINE_integer("n_sample", 100, "number of molecules to sample per sampling round")
flags.DEFINE_integer("epochs", 15, "epochs to train")
flags.DEFINE_integer("augment", 5, "whether different SMILES strings should generated for the same molecule, [1-n]")
flags.DEFINE_boolean("preprocess", True, "whether to pre-process stereo chemistry/salts etc.")
flags.DEFINE_integer("stereochemistry", 1, "whether stereo chemistry information should be included [0, 1]")
flags.DEFINE_boolean("reinforce", True, "whether to add most similar but novel generated mols back to training")
flags.DEFINE_float("validation", 0.2, "fraction of the data to use as a validation set")

FLAGS = flags.FLAGS


def main(_):
    print("\n----- Running SMILES LSTM model -----\n")
    model = SMILESmodel(batch_size=FLAGS.batch_size, dataset=FLAGS.dataset, num_epochs=FLAGS.epochs,
                        lr=FLAGS.learning_rate, run_name=FLAGS.run_name, sample_after=FLAGS.sample_after,
                        reinforce=FLAGS.reinforce, validation=FLAGS.validation)
    model.load_data(preprocess=FLAGS.preprocess, stereochem=FLAGS.stereochemistry, augment=FLAGS.augment)
    model.build_tokenizer(tokenize=FLAGS.tokenizer)
    model.build_model()
    model.train_model(n_sample=FLAGS.n_sample)


if __name__ == '__main__':
    tf.app.run()
