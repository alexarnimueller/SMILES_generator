#! /usr/bin/env python
# -*- coding: utf-8 -*-

import re
from multiprocessing import Process, Queue, cpu_count
from time import time

import numpy as np
import pandas as pd
from progressbar import ProgressBar
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, ChemicalFeatures, Descriptors3D, AddHs
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
from rdkit.Chem.Pharm2D import Generate, SigFactory
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.DataStructs import ConvertToNumpyArray, cDataStructs
from rdkit.DataStructs import FingerprintSimilarity, TanimotoSimilarity
from rdkit.SimDivFilters import MaxMinPicker
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def _rdk2numpy(fps):
    """ private function to transform RDKit fingerprints into numpy arrays

    :param fps: {list} list of RDKit fingerprints
    :return: {numpy.ndarray} fingerprints in array
    """
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    return np.array(np_fps).reshape((len(fps), len(np_fps[0])))


def tanimoto(vector1, vector2):
    """ function to calculate the taniomoto similarity of two binary vectors of the same length. only on-bits are
    considered. The formula used is:

    .. math::

            S = c / (a + b - c)

            a = on-bits in vector1
            b = on-bits in vector2
            c = on-bits in both vectors

    :param vector1: {numpy.ndarray or list} first binary vector
    :param vector2: {numpy.ndarray or list} second binary vector
    :return: tanimoto similarity
    """
    a = np.where(vector1 == 1)[0]
    b = np.where(vector2 == 1)[0]
    return len(np.intersect1d(a, b)) / (float(len(a) + len(b)) - len(np.intersect1d(a, b)))


def cosine_dist(vector1, vector2):
    """Calculate the cosine distance of two given vectors

    :param vector1: {numpy.ndarray or list} first vector
    :param vector2: {numpy.ndarray or list} second vector
    :return: cosine similarity
    """
    return cosine_distances(vector1.reshape(1, -1), vector2.reshape(1, -1)).flatten()


def euclidean_dist(vector1, vector2):
    """Calculate the euclidean distance of two given vectors

    :param vector1: {numpy.ndarray or list} first vector
    :param vector2: {numpy.ndarray or list} second vector
    :return: cosine similarity
    """
    return euclidean_distances(vector1.reshape(1, -1), vector2.reshape(1, -1)).flatten()


def numpy_fps(mols, r, features=True, bits=1024):
    """ Calculate RDKit morgan fingerprints and output them as a numpy array

    :param mols: {list} list of molecules (RDKit mols)
    :param r: {int} radius to consider when calculating the fingerprints
    :param features: {bool} whether to use features like in FCFP
    :param bits: {int} size of the fingerprint (e.g. 1024, 2048)
    :return: numpy array containing row-wise fingerprints for every molecule
    """
    return _rdk2numpy(
        [AllChem.GetMorganFingerprintAsBitVect(m, r, useFeatures=features, nBits=bits) for m in mols if m])


def numpy_rdk_fps(mols):
    """ Calculate RDKit daylight style fingerprints and output them as a numpy array

    :param mols: {list} list of molecules (RDKit mols)
    :return: numpy array containing row-wise fingerprints for every molecule
    """
    return _rdk2numpy([FingerprintMol(m) for m in mols if m])


def numpy_pp_fps(mols):
    """ Calculate Gobbi and Poppinger pharmacophore fingerprints and return them as numpy.ndarrays

    :param mols: {list} list of molecules (RDKit mols)
    :return: numpy array containing row-wise fingerprints for every molecule
    """
    feat_fact = ChemicalFeatures.BuildFeatureFactory()
    sig_fact = SigFactory(feat_fact, useCounts=False, minPointCount=2, maxPointCount=3)
    sig_fact.SetBins([(0, 2), (2, 4), (4, 6), (6, 8), (8, 100)])
    sig_fact.Init()
    return _rdk2numpy([Generate.Gen2DFingerprint(m, sig_fact) for m in mols if m])


def numpy_maccs(mols):
    """ Calculate MACCS keys and output them as a numpy array

    :param mols: {list} list of molecules (RDKit mols)
    :return: numpy array containing row-wise MACCS keys for every molecule
    """
    return _rdk2numpy([MACCSkeys.GenMACCSKeys(m) for m in mols if m])


def numpy_atompair(mols):
    """ Calculate atom pair fingerprints and output them as a numpy array

    :param mols: {list} list of molecules (RDKit mols)
    :return: numpy array containing row-wise fingerprints for every molecule
    """
    return _rdk2numpy([MACCSkeys.GenMACCSKeys(m) for m in mols if m])


def rdkit_descirptors(mols, regex="(MolWt)|(MolLogP)|(TPSA)|(.*Count)|(Num.*)|(FractionCSP3)|(.*VSA.*)|(Topliss.*)"
                                  "|(Chi.*)|(.*Density.*)|(MQNs)|(Autocorr2D)"):
    """ calculates a set of RDKit descriptors for given molecules (RDKit) ``mols``

    :param mols: {str} RDKit molecules
    :param regex: {str} regular expression to match RDKit functions
    :return: {pandas DataFrame} descriptor names and values
    """
    # create results dictionary with descriptors as keys and append list of values for all mols
    rslt = dict()
    desc_regex = re.compile(regex)
    for descriptor, func in Descriptors.descList:
        if desc_regex.match(descriptor):
            rslt[descriptor] = list()
            for mol in mols:
                rslt[descriptor].append(func(mol))
    return pd.DataFrame(rslt)


def rdkit_3d_descirptors(mols, regex="(NPR1)|(NPR2)|(PMI1)|(PMI2)|(PMI3)|(SpherocityIndex)|(InertialShapeFactor)|"
                                     "(Eccentricity)|(Asphericity)"):
    """ embeds molecules in 3D and calculates a set of RDKit descriptors for given molecules (RDKit) ``mols``

    :param mols: {str} RDKit molecules
    :param regex: {str} regular expression to match RDKit functions
    :return: {pandas DataFrame} descriptor names and values
    """
    # embed molecules in 3D
    mols = [AddHs(m) for m in mols]
    for i, m in enumerate(mols):
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
        # AllChem.MMFFOptimizeMolecule(m)

    # create results dictionary with descriptors as keys and append list of values for all mols
    rslt = dict()
    desc_regex = re.compile(regex)
    for descriptor in Descriptors3D.__dict__.keys():
        if desc_regex.match(descriptor):
            print("\t%s..." % descriptor)
            func = getattr(Descriptors3D, descriptor)
            pbar = ProgressBar()
            rslt[descriptor] = list()
            for mol in pbar(mols):
                rslt[descriptor].append(func(mol))
    return pd.DataFrame(rslt)


def fp_similarity(fp1, fp2, metric='tanimoto'):
    """ Calculate the Tanimoto similarity between two fingerprints

    :param fp1: {numpy array / RDKit fingerprint} Fingerprint 1
    :param fp2: {numpy array / RDKit fingerprint} Fingerprint 2
    :param metric: {str} which similarity metric to use, default: tanimoto; available for numpy fingerprints:
        tanimoto, cosine, euclidean
    :return: Tanimoto similarity
    """
    if isinstance(fp1, cDataStructs.ExplicitBitVect):
        return FingerprintSimilarity(fp1, fp2, metric=TanimotoSimilarity)
    elif isinstance(fp1, np.ndarray):
        if metric.lower() == 'tanimoto':
            return tanimoto(fp1, fp2)
        elif metric.lower() == 'cosine':
            return cosine_dist(fp1, fp2)
        elif metric.lower() == 'euclidean':
            return euclidean_dist(fp1, fp2)
        else:
            raise NotImplementedError('Only the following distance metrics are available: tanimoto, cosine, euclidean')
    else:
        raise TypeError("Fingerprints must be of type numpy.ndarray or rdkit.DataStructs.cDataStructs.ExplicitBitVect")


def list2batches(lst, n):
    """Divide a list into n batches

    :param lst: {list}
    :param n: {int}
    :return: list of n lists
    """
    p = len(lst) // n
    if len(lst) - p > 0:
        return [lst[:p]] + list2batches(lst[p:], n - 1)
    else:
        return [lst]


def _batch_vs_all(batch, fps, q, mtrc):
    """Function to calculate pairwise similarities from a batch of fingerprints to all fingerprints in a set of molecules

    :param batch: {list} list of RDKit fingerprints, a subset of ``fps``
    :param fps: {list} list of RDKit fingerprints of all molecules
    :param q: {multiprocessing queue} queue for multiprocessing
    :param mtrc: {str} metric to use, available: tanimoto, cosine, euclidean
    :return: {list} pairwise similarities batch-to-fps as the Tanimoto distance
    """
    q.put(np.asarray([[fp_similarity(fp1, fp2, mtrc) for fp2 in fps] for fp1 in batch]))


def parallel_pairwise_similarities(fps, fps2=None, metric='tanimoto'):
    """Function for parallel pairwise similarity calculation of RDKit-type fingerprints

    :param fps: {list} list of fingerprints (or numpy array) to calculate pairwise similarities for
    :param fps2: {list} list of fingerprints to calculate pairwise similarities to fps; if None, only pairwise
        similarities of all fingerprints in fps are calculated.
    :param metric: {str} available for RDKit fingerprints: tanimoto, available for numpy: tanimoto, cosine, euclidean
    :return: {numpy.ndarray} array of Tanimoto similarities
    """
    if not (isinstance(fps2, np.ndarray) or isinstance(fps2, list)):
        fps2 = fps
    if len(fps.shape) == 1:
        fps = fps.reshape(1, -1)
    if len(fps) < int(
            10 * cpu_count()):  # if only small array, don't parallelize and calculate all internal similarities
        rslt = np.array([list(map(lambda x: fp_similarity(fp, x, metric), fps)) for fp in fps2]).reshape(
            (len(fps), len(fps2), 1))
    else:
        queue = Queue()
        rslt = []
        for batch in list2batches(fps, cpu_count()):
            p = Process(target=_batch_vs_all, args=(batch, fps2, queue, metric,))
            p.start()
        for _ in range(cpu_count()):
            rslt.extend(queue.get())
    return np.array(rslt).astype('float')


def get_n_neighbors(fps, fps2, n, metric='tanimoto'):
    """ Function to get "N" nearest neighbors for a given set of molecules (´fps´, represented as fingerprint /
    descriptor) compared to a set of different molecules (´fps2´, same representation).

    :param fps: {list} list of RDKit fingerprints to calculate pairwise similarities for
    :param fps2: {list} list of RDKit fingerprints to calculate pairwise similarities to fps; if None, only pairwise
        similarities of all fingerprints in fps are calculated.
    :param n: {int} number of neighbors to return
    :param metric: {str} available for RDKit fingerprints: tanimoto, available for numpy: tanimoto, cosine, euclidean
    :return: {numpy.ndarray} n indices for every member of fps, corresponding to the indices of the molecules in fps2
    """
    sims = parallel_pairwise_similarities(fps, fps2, metric)
    return np.argsort(sims)[:, -n:][:, ::-1]  # indices of n most similar members of fps2 for every member of fps


def minmax(m, num=10, metric='tanimoto', seed=42):
    """ MinMax selection algorithm

    :param m: {array} Input matrix with vectors to make MinMax selection from
    :param num: {int} Number of selections to do
    :param metric: {str} Metric to use for distance / similarity calculation (tanimoto, cosine, euclidean)
    :param seed: {int} random seed to use for initialization
    :returns: indices of the vectors that were selected
    """
    np.random.seed(seed)
    if num > m.shape[0]:
        raise ValueError("Number of selections can't be larger than number of instances in M.")

    start = time()  # tic
    pool = m  # Store pool from which the selections get removed

    # Randomly selecting first molecule into the sele
    idx = int(np.random.randint(0, m.shape[0], 1))
    sele = pool[idx:idx + 1, :]
    minmaxidx = np.where(np.all(m == pool[idx:idx + 1, :], axis=1))[0].tolist()  # store original indices to return

    # Deleting selected molecule in selection from pool
    pool = np.delete(pool, idx, axis=0)

    pbar = ProgressBar()
    for _ in pbar(range(num - 1)):
        # Calculating distance from selected instances to the rest of the pool
        dist = parallel_pairwise_similarities(pool, sele, metric)
        if metric.lower() == 'tanimoto':
            dist = 1 - dist

        # Choosing maximal distances for every selected instance
        maxidx = np.argmax(dist, axis=0)  # index of most distant instance to each of compounds that were selected
        maxcols = np.max(dist, axis=0)  # value of most distant instance to each of compounds that were selected

        # Choosing minimal distance among the maximal distances
        minmax = np.argmin(maxcols)  # index of lowest distance value within maxcols
        idx = int(maxidx[minmax])  # index of the least most distant instance to each of the selected ones

        # Adding it to selection minmax indices and removing from pool
        sele = np.vstack((sele, pool[idx:idx + 1, :]))
        minmaxidx.extend(np.where(np.all(m == pool[idx:idx + 1, :], axis=1))[0])
        pool = np.delete(pool, idx, axis=0)

    print("MinMax selection took %.1f" % (time() - start))  # toc
    return minmaxidx


def minmax_rdkit(mols, num=10):
    """ RDKit implementation of the MinMax picker for fingerprints

    :param mols: RDKit molecules
    :param num:
    :return: picked molecules
    """
    start = time()
    mmp = MaxMinPicker()
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols if m]
    picks = mmp.LazyBitVectorPick(fps, len(fps), num)
    print("MinMax selection took %.1f" % (time() - start))  # toc
    return [i for i in picks]


def get_cats_factory(features='cats', names=False):
    """ Get the feature combinations paired to all possible distances

    :param features: {str} which pharmacophore features to consider; available: ["cats", "rdkit"]
    :param names: {bool} whether to return an array describing the bits with names of features and distances
    :return: RDKit signature factory to be used for 2D pharmacophore fingerprint calculation
    """
    if features == 'cats':
        fdef = fdef_cats
    else:
        fdef = fdef_rdkit
    factory = ChemicalFeatures.BuildFeatureFactoryFromString(fdef)
    sigfactory = SigFactory.SigFactory(factory, useCounts=True, minPointCount=2, maxPointCount=2)
    sigfactory.SetBins([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)])
    sigfactory.Init()
    if names:
        descs = [sigfactory.GetBitDescription(i) for i in range(sigfactory.GetSigSize())]
        return sigfactory, descs
    else:
        return sigfactory


def _cats_corr(mols, q):
    """ private cats descriptor function to be used in multiprocessing

    :param mols: {list/array} molecules (RDKit mol) to calculate the descriptor for
    :param q: {queue} multiprocessing queue instance
    :return: {numpy.ndarray} calculated descriptor vectors
    """
    factory = get_cats_factory()
    fps = []
    for mol in mols:
        arr = np.zeros((1,))
        ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
        scale = np.array([10 * [sum(arr[i:i + 10])] for i in range(0, 210, 10)]).flatten()
        fps.append(np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0))
    q.put(np.array(fps).reshape((len(mols), 210)).astype('float32'))


def _one_cats(mol):
    """ Function to calculate the CATS pharmacophore descriptor for one molecule.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mol: {RDKit molecule} molecule to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vector
    """
    factory = get_cats_factory()
    arr = np.zeros((1,))
    ConvertToNumpyArray(Generate.Gen2DFingerprint(mol, factory), arr)
    scale = np.array([10 * [sum(arr[i:i + 10])] for i in range(0, 210, 10)]).flatten()
    return np.divide(arr, scale, out=np.zeros_like(arr), where=scale != 0).astype('float32')


def cats_descriptor(mols):
    """ Function to calculate the CATS pharmacophore descriptor for a set of molecules.
    Descriptions of the individual features can be obtained from the function ``get_cats_sigfactory``.

    :param mols: {list/array} molecules (RDKit mol) to calculate the descriptor for
    :return: {numpy.ndarray} calculated descriptor vectors
    """
    queue = Queue()
    rslt = []
    if len(mols) < 4 * cpu_count():  # if only small array, don't parallelize
        for mol in mols:
            rslt.append(_one_cats(mol))
    else:
        for m in np.array_split(np.array(mols), cpu_count()):
            p = Process(target=_cats_corr, args=(m, queue,))
            p.start()
        for _ in range(cpu_count()):
            rslt.extend(queue.get(10))
    return np.array(rslt).reshape((len(mols), 210)).astype('float32')


fdef_cats = """
AtomType Hydroxylgroup [O;H1;+0]
AtomType OxygenAtom [#8]
AtomType PosCharge [+,++,+++,++++,++++]
AtomType NegCharge [-,--,---,----]
AtomType Carbon_AttachedOther [#6;$([#6]~[#7,#8,#9,#15,#16,#17,#35,#53,#14,#5,#34])]
AtomType CarbonLipophilic [#6;+0;!{Carbon_AttachedOther}]
AtomType ClBrI [#17,#35,#53]
AtomType SC2 [#16;X2]([#6])[#6]
AtomType NH_NH2_NH3 [#7;H1,H2,H3;+0]
AtomType NH0 [#7;H0;+0]
AtomType FlCl [#9,#17]
AtomType NH2 [#7;H2]
AtomType CSPOOH [C,S,P](=O)-[O;H1]
AtomType AromR4 [a]
AtomType AromR5 [a]
AtomType AromR6 [a]
AtomType AromR7 [a]
AtomType AromR8 [a]

DefineFeature SingleAtomDonor [{Hydroxylgroup},{NH_NH2_NH3}]
  Family Donor
  Weights 1
EndFeature

DefineFeature SingleAtomAcceptor [{OxygenAtom},{NH0},{FlCl}]
  Family Acceptor
  Weights 1
EndFeature

DefineFeature SingleAtomPositive [{PosCharge},{NH2}]
  Family PosIonizable
  Weights 1
EndFeature

DefineFeature SingleAtomNegative [{NegCharge},{CSPOOH}]
  Family NegIonizable
  Weights 1
EndFeature

DefineFeature SingleAtomLipophilic [!a;{CarbonLipophilic},{ClBrI},{SC2}]
  Family Hydrophobe
  Weights 1
EndFeature

DefineFeature Arom4 [{AromR4}]1[{AromR4}][{AromR4}][{AromR4}]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom5 [{AromR5}]1[{AromR5}][{AromR5}][{AromR5}][{AromR5}]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom6 [{AromR6}]1[{AromR6}][{AromR6}][{AromR6}][{AromR6}][{AromR6}]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom7 [{AromR7}]1[{AromR7}][{AromR7}][{AromR7}][{AromR7}][{AromR7}][{AromR7}]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

DefineFeature Arom8 [{AromR8}]1[{AromR8}][{AromR8}][{AromR8}][{AromR8}][{AromR8}][{AromR8}][{AromR8}]1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
"""

fdef_rdkit = """
# $Id$
#
# RDKit base fdef file.
# Created by Greg Landrum
#

AtomType NDonor [N&!H0&v3,N&!H0&+1&v4,n&H1&+0]
AtomType AmideN [$(N-C(=O))]
AtomType SulfonamideN [$([N;H0]S(=O)(=O))]
AtomType NDonor [$([Nv3](-C)(-C)-C)]

AtomType NDonor [$(n[n;H1]),$(nc[n;H1])]

AtomType ChalcDonor [O,S;H1;+0]
DefineFeature SingleAtomDonor [{NDonor},{ChalcDonor}]
  Family Donor
  Weights 1
EndFeature

# aromatic N, but not indole or pyrole or fusing two rings
AtomType NAcceptor [n;+0;!X3;!$([n;H1](cc)cc)]
AtomType NAcceptor [$([N;H0]#[C&v4])]
# tertiary nitrogen adjacent to aromatic carbon
AtomType NAcceptor [N&v3;H0;$(Nc)]

# removes thioether and nitro oxygen
AtomType ChalcAcceptor [O;H0;v2;!$(O=N-*)]
Atomtype ChalcAcceptor [O;-;!$(*-N=O)]

# Removed aromatic sulfur from ChalcAcceptor definition
Atomtype ChalcAcceptor [o;+0]

# Hydroxyls and acids
AtomType Hydroxyl [O;H1;v2]

# F is an acceptor so long as the C has no other halogen neighbors. This is maybe
# a bit too general, but the idea is to eliminate things like CF3
AtomType HalogenAcceptor [F;$(F-[#6]);!$(FC[F,Cl,Br,I])]

DefineFeature SingleAtomAcceptor [{Hydroxyl},{ChalcAcceptor},{NAcceptor},{HalogenAcceptor}]
  Family Acceptor
  Weights 1
EndFeature

# this one is delightfully easy:
DefineFeature AcidicGroup [C,S](=[O,S,P])-[O;H1,H0&-1]
  Family NegIonizable
  Weights 1.0,1.0,1.0
EndFeature

AtomType Carbon_NotDouble [C;!$(C=*)]
AtomType BasicNH2 [$([N;H2&+0][{Carbon_NotDouble}])]
AtomType BasicNH1 [$([N;H1&+0]([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType PosNH3 [$([N;H3&+1][{Carbon_NotDouble}])]
AtomType PosNH2 [$([N;H2&+1]([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType PosNH1 [$([N;H1&+1]([{Carbon_NotDouble}])([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType BasicNH0 [$([N;H0&+0]([{Carbon_NotDouble}])([{Carbon_NotDouble}])[{Carbon_NotDouble}])]
AtomType QuatN [$([N;H0&+1]([{Carbon_NotDouble}])([{Carbon_NotDouble}])([{Carbon_NotDouble}])[{Carbon_NotDouble}])]


DefineFeature BasicGroup [{BasicNH2},{BasicNH1},{BasicNH0};!$(N[a])]
  Family PosIonizable
  Weights 1.0
EndFeature

# 14.11.2007 (GL): add !$([N+]-[O-]) constraint so we don't match
# nitro (or similar) groups
DefineFeature PosN [#7;+;!$([N+]-[O-])]
 Family PosIonizable
 Weights 1.0
EndFeature

# imidazole group can be positively charged (too promiscuous?)
DefineFeature Imidazole c1ncnc1
  Family PosIonizable
  Weights 1.0,1.0,1.0,1.0,1.0
EndFeature
# guanidine group is positively charged (too promiscuous?)
DefineFeature Guanidine NC(=N)N
  Family PosIonizable
  Weights 1.0,1.0,1.0,1.0
EndFeature

# the LigZn binder features were adapted from combichem.fdl
DefineFeature ZnBinder1 [S;D1]-[#6]
  Family ZnBinder
  Weights 1,0
EndFeature
DefineFeature ZnBinder2 [#6]-C(=O)-C-[S;D1]
  Family ZnBinder
  Weights 0,0,1,0,1
EndFeature
DefineFeature ZnBinder3 [#6]-C(=O)-C-C-[S;D1]
  Family ZnBinder
  Weights 0,0,1,0,0,1
EndFeature

DefineFeature ZnBinder4 [#6]-C(=O)-N-[O;D1]
  Family ZnBinder
  Weights 0,0,1,0,1
EndFeature
DefineFeature ZnBinder5 [#6]-C(=O)-[O;D1]
  Family ZnBinder
  Weights 0,0,1,1
EndFeature
DefineFeature ZnBinder6 [#6]-P(=O)(-O)-[C,O,N]-[C,H]
  Family ZnBinder
  Weights 0,0,1,1,0,0
EndFeature


# aromatic rings of various sizes:
#
# Note that with the aromatics, it's important to include the ring-size queries along with
# the aromaticity query for two reasons:
#   1) Much of the current feature-location code assumes that the feature point is
#      equidistant from the atoms defining it. Larger definitions like: a1aaaaaaaa1 will actually
#      match things like 'o1c2cccc2ccc1', which have an aromatic unit spread across multiple simple
#      rings and so don't fit that requirement.
#   2) It's *way* faster.
#

#
# 21.1.2008 (GL): update ring membership tests to reflect corrected meaning of
# "r" in SMARTS parser
#
AtomType AromR4 [a;r4,!R1&r3]
DefineFeature Arom4 [{AromR4}]1:[{AromR4}]:[{AromR4}]:[{AromR4}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0
EndFeature
AtomType AromR5 [a;r5,!R1&r4,!R1&r3]
DefineFeature Arom5 [{AromR5}]1:[{AromR5}]:[{AromR5}]:[{AromR5}]:[{AromR5}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0
EndFeature
AtomType AromR6 [a;r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom6 [{AromR6}]1:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:[{AromR6}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
AtomType AromR7 [a;r7,!R1&r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom7 [{AromR7}]1:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:[{AromR7}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature
AtomType AromR8 [a;r8,!R1&r7,!R1&r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature Arom8 [{AromR8}]1:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:[{AromR8}]:1
 Family Aromatic
 Weights 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

# hydrophobic features
# any carbon that is not bonded to a polar atom is considered a hydrophobe
#
# 23.11.2007 (GL): match any bond (not just single bonds); add #6 at
#  beginning to make it more efficient
AtomType Carbon_Polar [#6;$([#6]~[#7,#8,#9])]
# 23.11.2007 (GL): don't match charged carbon
AtomType Carbon_NonPolar [#6;+0;!{Carbon_Polar}]

DefineFeature ThreeWayAttach [D3,D4;{Carbon_NonPolar}]
  Family Hydrophobe
  Weights 1.0
EndFeature

DefineFeature ChainTwoWayAttach [R0;D2;{Carbon_NonPolar}]
  Family Hydrophobe
  Weights 1.0
EndFeature

# hydrophobic atom
AtomType Hphobe [c,s,S&H0&v2,Br,I,{Carbon_NonPolar}]
AtomType RingHphobe [R;{Hphobe}]

# nitro groups in the RD code are always: *-[N+](=O)[O-]
DefineFeature Nitro2 [N;D3;+](=O)[O-]
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0
EndFeature

#
# 21.1.2008 (GL): update ring membership tests to reflect corrected meaning of
# "r" in SMARTS parser
#
AtomType Ring6 [r6,!R1&r5,!R1&r4,!R1&r3]
DefineFeature RH6_6 [{Ring6};{RingHphobe}]1[{Ring6};{RingHphobe}][{Ring6};{RingHphobe}][{Ring6};{RingHphobe}][{Ring6};{RingHphobe}][{Ring6};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0,1.0,1.0,1.0
EndFeature

AtomType Ring5 [r5,!R1&r4,!R1&r3]
DefineFeature RH5_5 [{Ring5};{RingHphobe}]1[{Ring5};{RingHphobe}][{Ring5};{RingHphobe}][{Ring5};{RingHphobe}][{Ring5};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0,1.0,1.0
EndFeature

AtomType Ring4 [r4,!R1&r3]
DefineFeature RH4_4 [{Ring4};{RingHphobe}]1[{Ring4};{RingHphobe}][{Ring4};{RingHphobe}][{Ring4};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0,1.0
EndFeature

AtomType Ring3 [r3]
DefineFeature RH3_3 [{Ring3};{RingHphobe}]1[{Ring3};{RingHphobe}][{Ring3};{RingHphobe}]1
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0
EndFeature

DefineFeature tButyl [C;!R](-[CH3])(-[CH3])-[CH3]
  Family LumpedHydrophobe
  Weights 1.0,0.0,0.0,0.0
EndFeature

DefineFeature iPropyl [CH;!R](-[CH3])-[CH3]
  Family LumpedHydrophobe
  Weights 1.0,1.0,1.0
EndFeature
"""
