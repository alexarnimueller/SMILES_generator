#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from argparse import ArgumentParser
from cats import cats_descriptor
from descriptorcalculation import numpy_maccs, numpy_fps, parallel_pairwise_similarities
from rdkit.Chem import MolFromSmiles

from plotting import sim_hist, pca_plot, plot_top_n
from utils import compare_mollists

from FCD import FCD_to_ref


def main(flags):
    generated = [line.strip() for line in open(flags.generated, 'r')]
    reference = [line.strip() for line in open(flags.reference, 'r')]
    novels = compare_mollists(generated, reference, False)
    print("\n%i\tgenerated molecules read" % len(generated))
    print("%i\treference molecules read" % len(reference))
    print("%i\tof the generated molecules are not in the reference set" % len(novels))
    print("\nCalculating Fréchet ChEMBLNET Distance...")
    fcd = FCD_to_ref(generated, reference, n=min([len(generated), len(reference)]))
    print("\nFréchet ChEMBLNET Distance to reference set:  %.4f" % fcd)

    print("\nCalculating %s similarities..." % flags.fingerprint)
    if flags.fingerprint == 'ECFP4':
        similarity = 'tanimoto'
        generated_fp = numpy_fps([MolFromSmiles(s) for s in generated], r=2, features=False, bits=1024)
        reference_fp = numpy_fps([MolFromSmiles(s) for s in reference], r=2, features=False, bits=1024)
    elif flags.fingerprint == 'MACCS':
        similarity = 'tanimoto'
        generated_fp = numpy_maccs([MolFromSmiles(s) for s in generated])
        reference_fp = numpy_maccs([MolFromSmiles(s) for s in reference])
    elif flags.fingerprint == 'CATS':
        similarity = 'euclidean'
        generated_fp = cats_descriptor([MolFromSmiles(s) for s in generated])
        reference_fp = cats_descriptor([MolFromSmiles(s) for s in reference])
    else:
        raise NotImplementedError('Only "MACCS", "CATS" or "ECFP4" are available as fingerprints!')
    sims = parallel_pairwise_similarities(generated_fp, reference_fp, similarity)
    sim_hist(sims.reshape(-1, 1), filename='./plots/%s_sim_hist.pdf' % flags.name)
    pca_plot(data=generated_fp, reference=reference_fp, filename='./plots/%s_pca.png' % flags.name)
    plot_top_n(smiles=novels, ref_smiles=reference, n=flags.num, fp=flags.fingerprint, sim=similarity,
               filename='./plots/%s_similar_mols.png' % flags.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generated", type=str, default="generated/combined_data_10k_sampled.csv",
                        help="the sampled molecules after finetuning")
    parser.add_argument("--reference", type=str, default="data/combined_data.csv",
                        help="the molecules used as input for finetuning")
    parser.add_argument("--name", type=str, default="analyze_",
                        help="name that will be prepended to the output filenames")
    parser.add_argument("--num", type=int, default=3,
                        help="number of most similar molecules to return per reference molecule")
    parser.add_argument("--fingerprint", type=str, default="ECFP4",
                        help="fingerprint to use for searching similar molecules; available: MACCS: MACCS keys, "
                             "ECFP4: radial fingerprint with diameter 4; CATS: pharmacophore atompair FP")
    args = parser.parse_args()

    with tf.device('/GPU:0'):
        main(args)
