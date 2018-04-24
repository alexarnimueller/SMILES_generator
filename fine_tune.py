#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import tensorflow as tf

from model import SMILESmodel

flags = tf.app.flags
flags.DEFINE_string("model_path", "checkpoint/chembl/", "path (folder) of the pretrained model")
flags.DEFINE_string("dataset", "", "[REQUIRED] dataset for fine tuning")
flags.DEFINE_integer("epoch_to_load", 5, "epoch_to_load")
flags.DEFINE_integer("epochs_to_train", 5, "number of epochs to fine tune")
flags.DEFINE_integer("num_sample", 100, "number of points to sample from trained model")
flags.DEFINE_float("temp", 0.75, "temperature to sample at")
flags.DEFINE_string("run_name", "", "run_name for output files")
flags.DEFINE_boolean("preprocess", True, "whether to preprocess stereochemistry/salts etc.")
flags.DEFINE_integer("stereochemistry", 1, "whether stereochemistry information should be included [0, 1]")
flags.DEFINE_float("percent_length", 0.8, "percent of length to take into account")
flags.DEFINE_float("validation", 0.2, "Fraction of the data to use as a validation set")

FLAGS = flags.FLAGS


def main(_):
    if len(FLAGS.dataset) == 0:
        print("ERROR: Please specify the dataset for fine tuning!")

    if len(FLAGS.run_name) == 0:
        run = FLAGS.dataset.split("/")[-1].split('.')[0]
    else:
        run = FLAGS.run_name

    model = SMILESmodel(dataset=FLAGS.dataset, num_epochs=FLAGS.epochs_to_train, run_name=run,
                        validation=FLAGS.validation)
    model.load_data(preprocess=FLAGS.preprocess, stereochem=FLAGS.stereochemistry, percent_length=FLAGS.percent_length)
    model.load_model_from_file(FLAGS.model_path, FLAGS.epoch_to_load)
    print("Pre-trained model loaded...")

    model.train_model()

    valid_mols = model.sample_points(FLAGS.num_sample, FLAGS.temp)
    mol_file = open('./generated/' + run + '_finetuned.txt', 'a')
    mol_file.write("\n".join(valid_mols))
    print("Valid:\t{}/{}".format(len(valid_mols), FLAGS.num_sample))

    os.system("cp %s*.json ./checkpoint/%s/" % (FLAGS.model_path, run))  # copy tokenizer files to fine-tuned folder
    json.dump(FLAGS.__dict__, open('./checkpoint/%s/flags.json' % run, 'w'))  # save used flags


if __name__ == '__main__':
    tf.app.run()
