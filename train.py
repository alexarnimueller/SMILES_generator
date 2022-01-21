#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from argparse import ArgumentParser
from model import SMILESmodel


def main(flags):
    print("\n----- Running SMILES LSTM model -----\n")
    print("Initializing...")
    model = SMILESmodel(batch_size=flags.batch, dataset=flags.dataset, num_epochs=flags.train,
                        lr=flags.lr, run_name=flags.name, sample_after=flags.after,
                        reinforce=flags.reinforce, validation=flags.val, reference=flags.ref, seed=flags.seed)
    print("Loading data...")
    model.load_data(preprocess=flags.preprocess, stereochem=flags.stereo, augment=flags.augment)
    print("Building model...")
    model.build_model()
    print("Training...")
    model.train_model(n_sample=flags.sample)


if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/chembl24_10uM_20-100.csv",
                        help="dataset file containing smiles strings")
    parser.add_argument("--name", type=str, default="chembl24", help="run name for log and checkpoint files")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--after", type=int, default=2, help="sample after how many epochs")
    parser.add_argument("--sample", type=int, default=25, help="number of molecules to sample per sampling round")
    parser.add_argument("--train", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--augment", type=int, default=5,
                        help="number different SMILES strings to be generated for the same molecule, [1-n]")
    parser.add_argument('--preprocess', dest='preprocess', action='store_true',
                        help="pre-process stereo chemistry/salts etc.")
    parser.add_argument('--no_preprocess', dest='preprocess', action='store_false',
                        help="don't pre-process stereo chemistry/salts etc.")
    parser.set_defaults(preprocess=False)
    parser.add_argument("--stereo", type=int, default=0,
                        help="whether stereo chemistry information should be included [0, 1]")
    parser.add_argument("--reinforce", type=bool, default=False,
                        help="whether to add most similar but novel generated mols back to training")
    parser.add_argument("--ref", type=str, default=None,
                        help="a molecule to compare the generated ones to and pick similar ones to reinforce")
    parser.add_argument("--val", type=float, default=0.1, help="fraction of the data to use as a validation set")
    parser.add_argument("--seed", type=float, default=42, help="random seed to use")
    args = parser.parse_args()

    # run with custom multi-GPU configuration
    # gpus = tf.config.list_physical_devices('GPU')
    # _ = [tf.config.experimental.set_memory_growth(g, enable=True) for g in gpus]
    # tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=7000)])
    # tf.config.set_logical_device_configuration(gpus[1], [tf.config.LogicalDeviceConfiguration(memory_limit=11000)])
    strategy = tf.distribute.MirroredStrategy(tf.config.list_logical_devices('GPU'))
    # with strategy.scope():
    with tf.device("/gpu:0"):
        main(args)
