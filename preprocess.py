#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from rdkit.Chem import CanonSmiles
from multiprocessing import Process, Queue, cpu_count


def keep_longest(smls):
    """ function to keep the longest fragment of a smiles string after fragmentation by splitting at '.'

    :param smls: {list} list of smiles strings
    :return: {list} list of longest fragments
    """
    out = list()
    for s in smls:
        if '.' in s:
            f = s.split('.')
            lengths = [len(m) for m in f]
            n = np.argmax(lengths)
            out.append(f[n])
        else:
            out.append(s)
    return out


def harmonize_sc(mols):
    """ harmonize the sidechains of given SMILES strings to a normalized format

    :param mols: {list} molecules as SMILES string
    :return: {list} harmonized molecules as SMILES string
    """
    out = list()
    for mol in mols:
        # TODO: add more problematic sidechain representation that occur
        pairs = [('[N](=O)[O-]', '[N+](=O)[O-]'),
                 ('[O-][N](=O)', '[O-][N+](=O)')]
        for b, a in pairs:
            mol = mol.replace(b, a)
        out.append(mol)
    return out


def preprocess_smiles(smiles, stereochem=1):
    """ desalt, canonicalize and harmonize side chains of molecules in a list of smiles strings

    :param smiles: {list} list of SMILES strings
    :param stereochem: {[0, 1]} whether stereochemistry should be considered (1) or not (0)
    :return: list of preprocessed SMILES strings
    """
    def process(s, q):
        smls = keep_longest(s)
        smls = harmonize_sc(smls)
        mols = list()
        for s in smls:
            try:
                mols.append(CanonSmiles(s, stereochem))
            except:
                print("Error! Can not process SMILES string %s" % s)
                mols.append(None)
        q.put(mols)

    print("Preprocessing...")
    queue = Queue()
    for m in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=process, args=(m, queue,))
        p.start()
    rslt = []
    for _ in range(cpu_count()):
        rslt.extend(queue.get(10))
    return np.random.choice(rslt, len(rslt), replace=False)  # return shuffled
