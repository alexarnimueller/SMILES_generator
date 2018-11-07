#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from descriptorcalculation import maccs_keys, tanimoto, compare_mollists, numpy_fps
from plotting import sim_hist, pca_plot, plot_top_n

flags = tf.app.flags
flags.DEFINE_string("generated", "generated/sampled.csv", "the sampled molecules after finetuning")
flags.DEFINE_string("reference", "data/chemregmed.csv", "the molecules used as input for finetuning")
flags.DEFINE_string("name", "", "name that will be prepended to the output filenames")
flags.DEFINE_integer("n", 3, "number of most similar molecules to return per reference molecule")
flags.DEFINE_string("fingerprint", "FCFP4", "fingerprint to use for searching similar molecules; available:"
                                            "MACCS: MACCS keys"
                                            "FCFP4: radial fingerprint with features and diameter 4")

FLAGS = flags.FLAGS


if __name__ == "__main__":
    generated = [line.strip() for line in open(FLAGS.generated, 'r')]
    reference = [line.strip() for line in open(FLAGS.reference, 'r')]
    novels = compare_mollists(generated, reference)
    print("\n%i\tgenerated molecules read" % len(generated))
    print("%i\treference molecules read" % len(reference))
    print("%i\tof the generated molecules are not in the reference set" % len(novels))
    print("\ncalculating %s similarities..." % FLAGS.fingerprint)
    if FLAGS.fingerprint == 'FCFP4':
        generated_fp = numpy_fps(novels, r=2, features=True, bits=1024)
        reference_fp = numpy_fps(reference, r=2, features=True, bits=1024)
    elif FLAGS.fingerprint == 'MACCS':
        generated_fp = maccs_keys(novels)
        reference_fp = maccs_keys(reference)
    else:
        raise NotImplementedError('Only "MACCS" or "FCFP4" are available as fingerprints!')
    sims = list()
    for g in generated_fp:
        for r in reference_fp:
            sims.append(tanimoto(g, r))
    sim_hist(sims, filename='./plots/%s_sim_hist.pdf' % FLAGS.name)
    # pca_plot(data=generated_maccs, reference=reference_maccs, filename='./plots/pca.png')
    plot_top_n(smiles=novels, ref_smiles=reference, n=FLAGS.n, fp=FLAGS.fingerprint,
               filename='./plots/%s_similar_mols.png' % FLAGS.name)
