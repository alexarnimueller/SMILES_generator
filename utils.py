#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import progressbar
import re
import numpy as np

from rdkit.Chem import CanonSmiles, MolFromSmiles, MolToSmiles, Descriptors, MACCSkeys, ReplaceSidechains, ReplaceCore
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_token(text, position):
    return list(text)[position], position + 1


def read_smiles_file(dataset, pickled_name):
    """ read a file with molecules and pickle it after doing so, to be faster for the next time...

    :param dataset: {str} filename of file with SMILES strings, one per line
    :param pickled_name: {str} pickled version of dataset file, expected to be text of molecules, with each molecule
        separated by newline
    """
    text = ""
    num_lines = 0
    if not os.path.isfile(pickled_name):
        with open(dataset) as f:
            for line in f:
                if num_lines % 10000 == 0:
                    print("read %i lines" % num_lines)
                fields = line.split(',')
                smiles = fields[-1].strip()
                text += smiles + '\n'
                num_lines += 1
        pickle.dump(text, open(pickled_name, 'wb'))
    else:
        print("Reading pickled file %s" % pickled_name)
        text = pickle.load(open(pickled_name, 'rb'))
    print("Done reading all input...")
    return text


def pad_seqs(sequences, pad_char, given_len=0):
    """ pad SMILES strings all to the same length

    :param sequences: {list} list of SMILES strings
    :param pad_char: {str} character to use for padding
    :param given_len: {int} fixed length to pad to. If 0, given_len= len(the longest sequence)
    :return: padded sequences
    """
    if given_len == 0:
        length = max([len(seq) for seq in sequences])
    else:
        length = given_len
    padded_seqs = []
    for seq in sequences:
        padded_seq = seq + [pad_char] * (length - len(seq))
        padded_seqs += [padded_seq]
    return padded_seqs


def transform_temp(preds, temp):
    """ transform predicted probabilities with a temperature

    :param preds: {list} list of probabilities to transform
    :param temp: {float} temperature to use for transformation
    :return: transformed probabilities
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def one_hot_encode(token_lists, n_chars):
    """ produce a one-hot encoding scheme from integer token lists

    :param token_lists: {list} list of lists with integers to transform into one-hot encoding
    :param n_chars: {int} vocabulary size
    :return: one-hot encoding of input
    """
    output = np.zeros((len(token_lists), len(token_lists[0]), n_chars))
    for i, token_list in enumerate(token_lists):
        for j, token in enumerate(token_list):
            output[i, j, int(token)] = 1
    return output


def tokenize_molecules(smiles, token_indices):
    """ Tokenizes a list of SMILES strings into a list of token lists

    :param smiles: {list} list of molecules as SMILES strings
    :param token_indices: {dict} translation dictionary for token to indices
    :return: tokenized SMILES strings in a list of lists
    """
    print("One-hot encoding...")
    tokens = []
    pbar = progressbar.ProgressBar()
    for molecule in pbar(smiles):
        mol_tokens = []
        posit = 0
        while posit < len(molecule):
            t, p = get_token(molecule, posit)
            posit = p
            mol_tokens += [token_indices[t]]
        tokens.append(mol_tokens)
    return tokens


def generate_Xy(tokens, maxlen, step=1):
    """ generate the training and target token lists from a given set of tokens

    :param tokens: {list} list containing tokens to be split into X and y
    :param maxlen: {int} maximal observed length of a SMILES string
    :param step: {int} step size to take for producing y
    :return: X and y
    """
    inputs = []
    targets = []
    for token_list in tokens:
        for i in range(0, len(token_list) - maxlen, step):
            inputs.append(token_list[i:i + maxlen])
            targets.append(token_list[(i + 1):(i + maxlen + 1)])
    return np.array(inputs), np.array(targets)


def tokenize_smiles(text, mode='default'):
    """ function to generate all possible token of a text and put them into two translation dictionaries

    :param text: text to tokenize
    :param mode: whether to use the default complete dictionary of 71 tokens ("default"), the ChEMBL token set
        with 43 token ("chembl") or generate a new token set from the input text ("generate")
    """
    if mode == 'generate':
        chars = list(set(text))
        indices_token = {str(i): chars[i] for i in range(len(chars))}
        token_indices = {v: k for k, v in indices_token.items()}

    elif mode == 'default':
        indices_token = {"0": 'H', "1": '9', "2": 'D', "3": 'r', "4": 'T', "5": 'R', "6": 'V', "7": '4',
                              "8": 'c', "9": 'l', "10": 'b', "11": '.', "12": 'C', "13": 'Y', "14": 's', "15": 'B',
                              "16": 'k', "17": '+', "18": 'p', "19": '2', "20": '7', "21": '8', "22": 'O',
                              "23": '%', "24": 'o', "25": '6', "26": 'N', "27": 'A', "28": 't', "29": '$',
                              "30": '(', "31": 'u', "32": 'Z', "33": '#', "34": 'M', "35": 'P', "36": 'G',
                              "37": 'I', "38": '=', "39": '-', "40": 'X', "41": '@', "42": 'E', "43": '":',
                              "44": '\\', "45": ')', "46": 'i', "47": 'K', "48": '/', "49": '{', "50": 'h',
                              "51": 'L', "52": 'n', "53": 'U', "54": '[', "55": '0', "56": 'y', "57": 'e',
                              "58": '3', "59": 'g', "60": 'f', "61": '}', "62": '1', "63": 'd', "64": 'W',
                              "65": '5', "66": 'S', "67": 'F', "68": ']', "69": 'a', "70": 'm'}
        token_indices = {v: k for k, v in indices_token.items()}
    else:
        raise NotImplementedError
    return indices_token, token_indices


def preprocess_smiles(smiles, stereochem=1, keep_fraction=1.):
    """ desalt, uniquify, canonicalize and harmonize sidechains of molecules in a list of smiles strings

        :param smiles: {list} list of SMILES strings
        :param stereochem: {[0, 1]} whether stereochemistry should be considered (1) or not (0)
        :param keep_fraction: {float} central fraction of the data to keep, selected by length
        :return: list of preprocessed SMILES strings
        """
    smls = keep_longest(smiles)
    mols = list()
    for c, s in enumerate(list(set(smls))):
        if c % 1000 == 0:
            print("%i molecules preprocessed..." % c)
        s = harmonize_sc(s)
        try:
            mols.append(CanonSmiles(s, stereochem))
        except:
            print("[Error] Can not process SMILES string %s" % s)
            continue
    if len(smls) - len(mols):
        print("%i duplicates removed" % (len(smls) - len(mols)))
    mols.sort(key=len)
    lower = int((0.5 - (keep_fraction / 2.)) * len(mols))
    upper = int((0.5 + (keep_fraction / 2.)) * len(mols)) - 1
    print("{}% of string length considered".format(keep_fraction * 100))
    print("Keeping lengths from %i to %i" % (len(mols[lower]), len(mols[upper])))
    selected = [mols[i] for i in range(lower, upper)]
    return np.random.choice(selected, len(selected))  # return unordered


def is_valid_mol(smiles, return_smiles=False):
    """ function to check a generated SMILES string for validity

    :param smiles: {str} SMILES string to be checked
    :param return_smiles: {bool} whether the checked valid SMILES string should be returned
    :return: {bool} validity
    """
    if smiles[0] == 'G':
        smiles = smiles[1:]
    if 'E' in smiles:
        end_index = smiles.find('E')
        if end_index != -1:
            smiles = smiles[:end_index]
    try:
        m = CanonSmiles(smiles.strip(), 1)
    except:
        m = None
    if return_smiles:
        return m is not None, m
    else:
        return m is not None


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


def harmonize_sc(mol):
    """ harmonize the sidechains in a SMILES to a normalized format

    :param mol: {str} molecule as SMILES string
    :return: {str} harmonized molecule as SMILES string
    """
    # TODO: add more problematic sidechain representation that occur
    pairs = [('[N](=O)[O-]', '[N+](=O)[O-]'),
             ('[O-][N](=O)', '[O-][N+](=O)')]
    for b, a in pairs:
        mol = mol.replace(b, a)
    return mol


def extract_murcko_scaffolds(mol):
    """ Extract Bemis-Murcko scaffolds from a smile string.

    :param mol: {str} smiles string of a molecule.
    :return: smiles string of a scaffold.
    """

    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        scaf = MolToSmiles(core, isomericSmiles=True)
    except:
        return ''

    return scaf


def extract_murcko_scaffolds_marked(mol, mark='[*]'):
    """ Extract Bemis-Murcko scaffolds from a smile string.

    :param mol: {str} smiles string of a molecule.
    :param mark: character to mark attachment points.
    :return: smiles string of a scaffold, side chains replaced with [R].
    """
    pos = range(0, 20)
    set_pos = ['[' + str(x) + '*]' for x in pos]

    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        tmp = ReplaceSidechains(m1, core)
        smi = MolToSmiles(tmp, isomericSmiles=True)  # isomericSmiles adds a number to the dummy atoms.
    except:
        return ''

    for i in pos:
        smi = smi.replace(''.join(set_pos[i]), mark)
    return smi


def extract_side_chains(mol, remove_duplicates=False, mark='[*]'):
    """ Extract side chains from a smiles string. Core is handled as Murcko scaffold.

    :param mol: {str} smiles string of a molecule.
    :param remove_duplicates: {bool} Keep or remove duplicates.
    :param mark: character to mark attachment points.
    :return: smiles strings of side chains in a list, attachment points replaced by [R].
    """
    pos = range(0, 20)
    set_pos = ['[' + str(x) + '*]' for x in pos]

    m1 = MolFromSmiles(mol)
    try:
        core = MurckoScaffold.GetScaffoldForMol(m1)
        side_chain = ReplaceCore(m1, core)
        smi = MolToSmiles(side_chain, isomericSmiles=True)  # isomericSmiles adds a number to the dummy atoms.
    except:
        return list()

    for i in pos:
        smi = smi.replace(''.join(set_pos[i]), mark)

    if remove_duplicates:
        return list(set(smi.split('.')))
    else:
        return smi.split('.')


def decorate_scaffold(scaffold, sidechains, num=10):
    """ Decorate a given scaffold containing marked attachment points ([*]) randomly with the given side chains

    :param scaffold: {str} smiles string of a scaffold with attachment points marked as [*]
    :param sidechains: {str} point-separated side chains as smiles strings
    :param num: {int} number of unique molecules to generate
    :return: ``num``-molecules in a list
    """
    # check if side chains contain rings & adapt the ring number to not confuse them with the ones already in the scaff
    try:
        ring_scaff = int(max(list(filter(str.isdigit, scaffold))))  # get highest number of ring in scaffold
        ring_sc = list(filter(str.isdigit, scaffold))  # get number of rings in side chains
        for r in ring_sc:
            sidechains = sidechains.replace(r, str(ring_scaff + int(r)))  # replace the ring number with the adapted one
    except ValueError:
        pass

    # do the decoration
    mols = list()
    tmp = scaffold.replace('[*]', '*')
    schns = sidechains.split('.')
    invalcntr = 0
    while len(mols) < num and invalcntr < 50:
        scaff = tmp
        while '*' in scaff:
            scafflist = list(scaff)
            scafflist[scafflist.index('*')] = np.random.choice(schns, replace=False)
            scaff = ''.join(scafflist)
        if is_valid_mol(scaff) and (scaff not in mols):
            scaff = CanonSmiles(scaff)
            print(sidechains + "." + scaffold + ">>" + scaff)
            mols.append(sidechains + "." + scaffold + ">>" + scaff)
        else:
            invalcntr += 1
    return mols


def get_rdkit_desc_functions():
    desc_regex = re.compile("(.*MolWt)|(BertzCT)|(.*Count)|(Num.*)|(MolLogP)")
    functions = []
    descriptors = []
    for descriptor, func in Descriptors.descList:
        if desc_regex.match(descriptor):
            descriptors = descriptors + [descriptor]
            functions = functions + [func]
    return functions, descriptors


def rdkit_desc(all_mols, functions):
    """
    Calculate all Molecular Weight, atom count, num atoms, molLogP descriptors from a list of molecules.
    Return calculated descriptors, as well as list of which descriptors were calculated.
    """
    total_mols = len(all_mols)
    desc = np.zeros([total_mols, len(functions)])
    for i in range(total_mols):
        if i % 1000 == 0 and i > 0:
            print("Processed ", str(i), "molecules")
        smiles = all_mols[i]
        if smiles[0] == 'G':
            smiles = smiles[1:]
        end_index = smiles.find('E')
        if end_index != -1:
            smiles = smiles[:end_index]
        if is_valid_mol(smiles):
            mol = MolFromSmiles(smiles.strip())
        else:
            continue
        j = 0
        for func in functions:
            desc[i, j] = func(mol)
            j += 1
    return desc


def maccs_keys(smls):
    """ Calculate MACCS keys and output them as a numpy array

    :param smls: {list} list of molecules (RDKit mols)
    :return: numpy array containing row-wise MACCS keys for every molecule
    """
    mols = [MolFromSmiles(s) for s in smls if MolFromSmiles(s)]
    fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    np_fps = []
    for fp in fps:
        arr = np.zeros((1,))
        ConvertToNumpyArray(fp, arr)
        np_fps.append(arr)
    return np.array(np_fps).reshape((len(mols), len(np_fps[-1])))


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
    c = len(np.intersect1d(a, b))
    return float(c) / float(len(a) + len(b) - c)


def compare_mollists(smiles, reference):
    """ get the molecules from ``smiles`` that are not in ``reference``

    :param smiles: {list} list of SMILES strings to check for known reference in ``reference``
    :param reference: {list} reference molecules as SMILES strings to compare to ``smiles``
    :return: {list} unique molecules from ``smiles`` as SMILES strings
    """
    mols = set([MolFromSmiles(s) for s in smiles if MolFromSmiles(s)])
    refs = set([MolFromSmiles(s) for s in reference if MolFromSmiles(s)])
    return [MolToSmiles(m, True) for m in mols if not m in refs]
