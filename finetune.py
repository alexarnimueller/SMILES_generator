#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import tensorflow as tf

from model import SMILESmodel

flags = tf.app.flags
flags.DEFINE_string("model_path", "checkpoint/chembl24_augment5/", "path (folder) of the pretrained model")
flags.DEFINE_string("dataset", "", "[REQUIRED] dataset for fine tuning")
flags.DEFINE_string("runname", "", "run_name for output files")
flags.DEFINE_integer("epoch_to_load", 24, "epoch_to_load")
flags.DEFINE_integer("epochs_to_train", 10, "number of epochs to fine tune")
flags.DEFINE_integer("num_sample", 100, "number of points to sample from trained model")
flags.DEFINE_float("temp", 1.0, "temperature to sample at")
flags.DEFINE_integer("sample_after", 2, "sample after how many epochs (if 0, no sampling)")
flags.DEFINE_integer("augment", 10, "whether different SMILES strings should generated for the same molecule, [1-n]")
flags.DEFINE_integer("batch_size", 16, "batchsize used for finetuning")
flags.DEFINE_boolean("preprocess", False, "whether to preprocess stereochemistry/salts etc.")
flags.DEFINE_integer("stereochemistry", 1, "whether stereochemistry information should be included [0, 1]")
flags.DEFINE_float("validation", 0., "Fraction of the data to use as a validation set")

FLAGS = flags.FLAGS


def main(_):
    print("\n----- Running SMILES LSTM model -----\n")
    if len(FLAGS.dataset) == 0:
        raise ValueError("Please specify the dataset for fine tuning!")

    if len(FLAGS.runname) == 0:
        run = FLAGS.dataset.split("/")[-1].split('.')[0]
    else:
        run = FLAGS.runname

    model = SMILESmodel(dataset=FLAGS.dataset, num_epochs=FLAGS.epochs_to_train, run_name=run,
                        batch_size=FLAGS.batch_size, validation=FLAGS.validation, sample_after=FLAGS.sample_after)
    model.load_data(preprocess=FLAGS.preprocess, stereochem=FLAGS.stereochemistry, augment=FLAGS.augment)
    model.load_model_from_file(FLAGS.model_path, FLAGS.epoch_to_load)
    print("Pre-trained model loaded, finetuning...")

    model.train_model()

    valid_mols = model.sample_points(FLAGS.num_sample, FLAGS.temp)
    mol_file = open('./generated/' + run + '_finetuned.csv', 'a')
    mol_file.write("\n".join(valid_mols))
    print("Valid:\t{}/{}".format(len(valid_mols), FLAGS.num_sample))

    os.system("cp %s*.json ./checkpoint/%s/" % (FLAGS.model_path, run))  # copy tokenizer files to fine-tuned folder
    # json.dump(FLAGS.__flags.items(), open('./checkpoint/%s/flags.json' % run, 'w'))  # save used flags


if __name__ == '__main__':
    tf.app.run()
