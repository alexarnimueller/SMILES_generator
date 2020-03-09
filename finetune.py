#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from model import SMILESmodel

flags = tf.compat.v1.app.flags
flags.DEFINE_string("model", "checkpoint/combined_data_a5_adLR/", "path (folder) of the pretrained model")
flags.DEFINE_string("dataset", "data/actives.csv", "[REQUIRED] dataset for fine tuning")
flags.DEFINE_string("name", "combined_a5LR_hits_ft_actives", "run_name for output files")
flags.DEFINE_float("lr", 0.005, "learning rate")
flags.DEFINE_integer("epoch", 19, "epoch_to_load")
flags.DEFINE_integer("train", 20, "number of epochs to fine tune")
flags.DEFINE_integer("sample", 100, "number of points to sample during and after training")
flags.DEFINE_float("temp", 1.0, "temperature to sample at")
flags.DEFINE_integer("after", 1, "sample after how many epochs (if 0, no sampling)")
flags.DEFINE_integer("augment", 10, "whether different SMILES strings should generated for the same molecule, [1-n]")
flags.DEFINE_integer("batch", 16, "batchsize used for finetuning")
flags.DEFINE_boolean("preprocess", False, "whether to preprocess stereochemistry/salts etc.")
flags.DEFINE_integer("stereo", 1, "whether stereochemistry information should be included [0, 1]")
flags.DEFINE_boolean("reinforce", False, "whether to add most similar but novel generated mols back to training")
flags.DEFINE_string("mw_filter", "250,400", "allowed thresholds for reinforcing molecules")
flags.DEFINE_string("reference", "", "reference molecule to compare to for reinforcement")
flags.DEFINE_float("val", 0., "Fraction of the data to use as a validation set")
flags.DEFINE_float("seed", 42, "random seed to use")
flags.DEFINE_integer("workers", 1, "number of threads to use for the data generator")
FLAGS = flags.FLAGS


def main(_):
    print("\n----- Running SMILES LSTM model -----\n")
    if len(FLAGS.dataset) == 0:
        raise ValueError("Please specify the dataset for fine tuning!")

    if len(FLAGS.name) == 0:
        run = FLAGS.dataset.split("/")[-1].split('.')[0]
    else:
        run = FLAGS.name

    model = SMILESmodel(dataset=FLAGS.dataset, num_epochs=FLAGS.train, run_name=run,
                        reinforce=bool(FLAGS.reinforce), batch_size=FLAGS.batch, validation=FLAGS.val,
                        mw_filter=FLAGS.mw_filter.split(','), sample_after=FLAGS.after, lr=FLAGS.lr,
                        reference=FLAGS.reference, workers=FLAGS.workers, seed=FLAGS.seed)
    model.load_data(preprocess=FLAGS.preprocess, stereochem=FLAGS.stereo, augment=FLAGS.augment)
    model.load_model_from_file(checkpoint_dir=FLAGS.model, epoch=FLAGS.epoch)
    model.model.layers[1].trainable = False  # freeze first LSTM layer
    model.model.compile(loss='categorical_crossentropy', optimizer=model.model.optimizer, metrics=['accuracy'])

    print("Pre-trained model loaded, finetuning...")

    model.train_model(n_sample=FLAGS.sample)

    valid_mols = model.sample_points(FLAGS.sample, FLAGS.temp)
    mol_file = open('./generated/' + run + '_finetuned.csv', 'a')
    mol_file.write("\n".join(valid_mols))
    print("Valid:\t{}/{}".format(len(valid_mols), FLAGS.sample))


if __name__ == '__main__':
    tf.compat.v1.app.run()
