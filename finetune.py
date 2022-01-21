#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from argparse import ArgumentParser
from model import SMILESmodel


def main(flags):
    print("\n----- Running SMILES LSTM model -----\n")
    if len(flags.dataset) == 0:
        raise ValueError("Please specify the dataset for fine tuning!")

    if len(flags.name) == 0:
        run = flags.dataset.split("/")[-1].split('.')[0]
    else:
        run = flags.name

    model = SMILESmodel(dataset=flags.dataset, num_epochs=flags.train, run_name=run,
                        reinforce=bool(flags.reinforce), batch_size=flags.batch, validation=flags.val,
                        mw_filter=flags.mw_filter.split(','), sample_after=flags.after, lr=flags.lr,
                        reference=flags.reference, num_reinforce=flags.num_reinforce, workers=flags.workers, seed=flags.seed)
    model.load_data(preprocess=flags.preprocess, stereochem=flags.stereo, augment=flags.augment)
    model.load_model_from_file(checkpoint_dir=flags.model, epoch=flags.epoch)
    model.model.layers[1].trainable = False  # freeze first LSTM layer
    model.model.compile(loss='categorical_crossentropy', optimizer=model.model.optimizer, metrics=['accuracy'])

    print("Pre-trained model loaded, finetuning...")

    model.train_model(n_sample=flags.sample)

    valid_mols = model.sample_points(flags.sample, flags.temp)
    mol_file = open('./generated/' + run + '_finetuned.csv', 'a')
    mol_file.write("\n".join(valid_mols))
    print("Valid:\t{}/{}".format(len(valid_mols), flags.sample))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="path (folder) of the pretrained model")
    parser.add_argument("--dataset", required=True, type=str, help="dataset for fine tuning")
    parser.add_argument("--name", type=str, default="transferlearn", help="run_name for output files")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epoch", type=int, default=19, help="epoch_to_load")
    parser.add_argument("--train", type=int, default=5, help="number of epochs to fine tune")
    parser.add_argument("--sample", type=int, default=25, help="number of points to sample during and after training")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature to sample at")
    parser.add_argument("--after", type=int, default=1, help="sample after how many epochs (if 0, no sampling)")
    parser.add_argument("--augment", type=int, default=5,
                        help="whether different SMILES strings should generated for the same molecule, [1-n]")
    parser.add_argument("--batch", type=int, default=64, help="batchsize used for finetuning")
    parser.add_argument('--preprocess', dest='preprocess', action='store_true', default=True,
                        help="pre-process stereo chemistry/salts etc.")
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false',
                        help="don't pre-process stereo chemistry/salts etc.")
    parser.add_argument("--stereo", type=int, default=0,
                        help="whether stereochemistry information should be included [0, 1]")
    parser.add_argument('--reinforce', dest='reinforce', action='store_true', default=True,
                        help="add most similar but novel generated mols back to training")
    parser.add_argument('--no-reinforce', dest='reinforce', action='store_false',
                        help="add most similar but novel generated mols back to training")
    parser.add_argument('--num_reinforce', type=int, help="number of generated compounds to add back to training set; "
                                                          "only active if reinforce = True")
    parser.add_argument("--mw_filter", type=str, default="250,400", help="allowed thresholds for reinforcing molecules")
    parser.add_argument("--reference", type=str, default="", help="reference molecule to compare to for reinforcement")
    parser.add_argument("--val", type=float, default=0., help="Fraction of the data to use as a validation set")
    parser.add_argument("--seed", type=float, default=42, help="random seed to use")
    parser.add_argument("--workers", type=int, default=-1, help="number of threads to use for the data generator")
    args = parser.parse_args()

    with tf.device('/GPU:0'):
        main(args)
