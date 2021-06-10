#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from argparse import ArgumentParser
from model import SMILESmodel


def main(flags):
    print("\n----- Running SMILES LSTM model -----\n")
    model = SMILESmodel(batch_size=flags.batch, dataset=flags.dataset, num_epochs=flags.train,
                        lr=flags.lr, run_name=flags.name, sample_after=flags.after,
                        reinforce=flags.reinforce, validation=flags.val, reference=flags.ref, seed=flags.seed)
    model.load_data(preprocess=flags.preprocess, stereochem=flags.stereo, augment=flags.augment)
    model.build_model()

    model.train_model(n_sample=flags.sample)


if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/chembl24_10uM_20-100.csv",
                        help="dataset file containing smiles strings")
    parser.add_argument("--name", type=str, default="chembl24_20-100_a5", help="run name for log and checkpoint files")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--after", type=int, default=2, help="sample after how many epochs")
    parser.add_argument("--sample", type=int, default=25, help="number of molecules to sample per sampling round")
    parser.add_argument("--train", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--augment", type=int, default=5,
                        help="whether different SMILES strings should generated for the same molecule, [1-n]")
    parser.add_argument('--preprocess', dest='preprocess', action='store_true',
                        help="pre-process stereo chemistry/salts etc.")
    parser.add_argument('--no-preprocess', dest='preprocess', action='store_false',
                        help="don't pre-process stereo chemistry/salts etc.")
    parser.set_defaults(preprocess=True)
    parser.add_argument("--stereo", type=int, default=1,
                        help="whether stereo chemistry information should be included [0, 1]")
    parser.add_argument("--reinforce", type=bool, default=False,
                        help="whether to add most similar but novel generated mols back to training")
    parser.add_argument("--ref", type=str, default=None,
                        help="a molecule to compare the generated ones to and pick similar ones to reinforce")
    parser.add_argument("--val", type=float, default=0.1, help="fraction of the data to use as a validation set")
    parser.add_argument("--seed", type=float, default=42, help="random seed to use")
    args = parser.parse_args()

    # run
    with tf.device('/GPU:0'):
        main(args)
