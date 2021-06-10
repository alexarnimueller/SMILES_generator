#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from argparse import ArgumentParser
from model import SMILESmodel


def main(flags):
    print("\n----- Running SMILES LSTM model -----\n")
    model = SMILESmodel(dataset=flags.model, seed=flags.seed)
    model.load_model_from_file(flags.model, flags.epoch)
    if flags.frag[0] != '^':
        frag = '^' + flags.frag
    else:
        frag = flags.frag
    print("Starting character(s): %s" % frag)
    valid_mols = model.sample_points(n_sample=flags.num, temp=flags.temp, prime_text=frag)
    mol_file = open(flags.out, 'w')
    mol_file.write("\n".join(set(valid_mols)))
    mol_file.close()
    print("Valid:{}/{}".format(len(valid_mols), flags.num))
    print("Unique:{}".format(len(set(valid_mols))))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoint/combined_data_a5_adLR/",
                        help="model path within checkpoint directory")
    parser.add_argument("--out", type=str, default="generated/combined_data_a5_adLR_10k_sampled.csv",
                        help="output file for molecules")
    parser.add_argument("--epoch", type=int, default=14, help="epoch to load")
    parser.add_argument("--num", type=int, default=10000, help="number of points to sample from trained model")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature to sample at")
    parser.add_argument("--frag", type=str, default="^",
                        help="Fragment to grow SMILES from. default: start character '^'")
    parser.add_argument("--seed", type=float, default=42, help="random seed to use")
    args = parser.parse_args()
    with tf.device('/GPU:0'):
        main(args)
