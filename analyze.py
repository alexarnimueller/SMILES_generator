#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import maccs_keys, tanimoto, compare_mollists
from plotting import sim_hist, pca_plot, plot_top_n

flags = tf.app.flags
flags.DEFINE_string("generated", "generated/sampled.csv", "the sampled molecules after finetuning")
flags.DEFINE_string("reference", "data/chemregmed.csv", "the molecules used as input for finetuning")
flags.DEFINE_string("name", "", "name that will be prepended to the output filenames")
flags.DEFINE_integer("n", 3, "number of most similar molecules to return per reference molecule")
FLAGS = flags.FLAGS


if __name__ == "__main__":
    generated = [line.strip() for line in open(FLAGS.generated, 'r')]
    reference = [line.strip() for line in open(FLAGS.reference, 'r')]
    novels = compare_mollists(generated, reference)
    print("\n%i\tgenerated molecules read" % len(generated))
    print("%i\treference molecules read" % len(reference))
    print("%i\tof the generated molecules are not in the reference set" % len(novels))
    print("\ncalculating MACCS key similarities...")
    generated_maccs = maccs_keys(novels)
    reference_maccs = maccs_keys(reference)
    sims = list()
    for g in generated_maccs:
        for r in reference_maccs:
            sims.append(tanimoto(g, r))
    sim_hist(sims, filename='./plots/%s_sim_hist.pdf' % FLAGS.name)
    # pca_plot(data=generated_maccs, reference=reference_maccs, filename='./plots/pca.png')
    plot_top_n(smiles=novels, ref_smiles=reference, n=FLAGS.n, filename='./plots/%s_similar_mols.png' % FLAGS.name)
