#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from utils import maccs_keys, tanimoto, compare_mollists
from plotting import sim_hist, pca_plot

flags = tf.app.flags
flags.DEFINE_string("generated", "generated/sampled.csv", "the sampled molecules after finetuning")
flags.DEFINE_string("reference", "data/chemregmed.csv", "the molecules used as input for finetuning")
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
    sim_hist(sims, filename='./plots/sim_hist.pdf')
    # pca_plot(data=generated_maccs, reference=reference_maccs, filename='./plots/pca.png')
