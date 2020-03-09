#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from cats import cats_descriptor
from descriptorcalculation import numpy_maccs, numpy_fps, parallel_pairwise_similarities
from rdkit.Chem import MolFromSmiles

from plotting import sim_hist, pca_plot, plot_top_n
from utils import compare_mollists

from FCD import FCD_to_ref

flags = tf.app.flags
flags.DEFINE_string("generated", "generated/combined_data_10k_sampled.csv", "the sampled molecules after finetuning")
flags.DEFINE_string("reference", "data/combined_data.csv", "the molecules used as input for finetuning")
flags.DEFINE_string("name", "ftENGA", "name that will be prepended to the output filenames")
flags.DEFINE_integer("n", 3, "number of most similar molecules to return per reference molecule")
flags.DEFINE_string("fingerprint", "ECFP4", "fingerprint to use for searching similar molecules; available:"
                    "MACCS: MACCS keys, ECFP4: radial fingerprint with diameter 4; CATS: pharmacophore atompair FP")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    generated = [line.strip() for line in open(FLAGS.generated, 'r')]
    reference = [line.strip() for line in open(FLAGS.reference, 'r')]
    novels = compare_mollists(generated, reference, False)
    print("\n%i\tgenerated molecules read" % len(generated))
    print("%i\treference molecules read" % len(reference))
    print("%i\tof the generated molecules are not in the reference set" % len(novels))
    print("\nCalculating Fréchet ChEMBLNET Distance...")
    fcd = FCD_to_ref(generated, reference, n=min([len(generated), len(reference)]))
    print("\nFréchet ChEMBLNET Distance to reference set:  %.4f" % fcd)

    print("\nCalculating %s similarities..." % FLAGS.fingerprint)
    if FLAGS.fingerprint == 'ECFP4':
        similarity = 'tanimoto'
        generated_fp = numpy_fps([MolFromSmiles(s) for s in generated], r=2, features=False, bits=1024)
        reference_fp = numpy_fps([MolFromSmiles(s) for s in reference], r=2, features=False, bits=1024)
    elif FLAGS.fingerprint == 'MACCS':
        similarity = 'tanimoto'
        generated_fp = numpy_maccs([MolFromSmiles(s) for s in generated])
        reference_fp = numpy_maccs([MolFromSmiles(s) for s in reference])
    elif FLAGS.fingerprint == 'CATS':
        similarity = 'euclidean'
        generated_fp = cats_descriptor([MolFromSmiles(s) for s in generated])
        reference_fp = cats_descriptor([MolFromSmiles(s) for s in reference])
    else:
        raise NotImplementedError('Only "MACCS", "CATS" or "ECFP4" are available as fingerprints!')
    sims = parallel_pairwise_similarities(generated_fp, reference_fp, similarity)
    sim_hist(sims.reshape(-1, 1), filename='./plots/%s_sim_hist.pdf' % FLAGS.name)
    pca_plot(data=generated_fp, reference=reference_fp, filename='./plots/%s_pca.png' % FLAGS.name)
    plot_top_n(smiles=novels, ref_smiles=reference, n=FLAGS.n, fp=FLAGS.fingerprint, sim=similarity,
               filename='./plots/%s_similar_mols.png' % FLAGS.name)
