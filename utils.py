#!/usr/bin/env python
# -*- coding: utf-8 -*-

import progressbar
import numpy as np

from multiprocessing import cpu_count, Process, Queue

from rdkit.Chem import CanonSmiles, MolFromSmiles, MolToSmiles, RenumberAtoms, ReplaceSidechains, ReplaceCore
from rdkit.Chem.Scaffolds import MurckoScaffold


def _get_token(text, position):
    return list(text)[position], position + 1


def is_valid_mol(smiles, return_smiles=False):
    """ function to check a generated SMILES string for validity

    :param smiles: {str} SMILES string to be checked
    :param return_smiles: {bool} whether the checked valid SMILES string should be returned
    :return: {bool} validity
    """
    try:
        m = CanonSmiles(smiles.replace('G', '').replace('E', '').strip(), 1)
    except:
        m = None
    if return_smiles:
        return m is not None, m
    else:
        return m is not None


def read_smiles_file(dataset):
    """ read a file with molecules and pickle it after doing so, to be faster for the next time...

    :param dataset: {str} filename of file with SMILES strings, containing one per line
    :return: list of read SMILES strings
    """
    smls = list()
    print("Reading %s..." % dataset)
    pbar = progressbar.ProgressBar()
    with open(dataset) as f:
        for line in pbar(f):
            smls.append(line.strip())
    return smls


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
    return np.array(padded_seqs)


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
    tokens = []
    pbar = progressbar.ProgressBar()
    for molecule in pbar(smiles):
        mol_tokens = []
        posit = 0
        while posit < len(molecule):
            t, posit = _get_token(molecule, posit)
            mol_tokens += [token_indices[t]]
        tokens.append(mol_tokens)
    return np.array(tokens)


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
                         "37": 'I', "38": '=', "39": '-', "40": 'X', "41": '@', "42": 'E', "43": ':',
                         "44": '\\', "45": ')', "46": 'i', "47": 'K', "48": '/', "49": '{', "50": 'h',
                         "51": 'L', "52": 'n', "53": 'U', "54": '[', "55": '0', "56": 'y', "57": 'e',
                         "58": '3', "59": 'g', "60": 'f', "61": '}', "62": '1', "63": 'd', "64": 'W',
                         "65": '5', "66": 'S', "67": 'F', "68": ']', "69": 'a', "70": 'm'}
        token_indices = {v: k for k, v in indices_token.items()}
    else:
        raise NotImplementedError
    return indices_token, token_indices


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


def compare_mollists(smiles, reference):
    """ get the molecules from ``smiles`` that are not in ``reference``

    :param smiles: {list} list of SMILES strings to check for known reference in ``reference``
    :param reference: {list} reference molecules as SMILES strings to compare to ``smiles``
    :return: {list} unique molecules from ``smiles`` as SMILES strings
    """
    smiles = [s.replace('G', '').replace('E', '').replace('A', '') for s in smiles]
    reference = [s.replace('G', '').replace('E', '').replace('A', '') for s in reference]
    mols = set([CanonSmiles(s, 1) for s in smiles if MolFromSmiles(s)])
    refs = set([CanonSmiles(s, 1) for s in reference if MolFromSmiles(s)])
    return [m for m in mols if m not in refs]


def randomize_smiles(smiles, num=10, isomeric=True):
    """ Generate different SMILES representations for the same molecule

    :param smiles: {str} SMILES string
    :param num: {int} number of different SMILES strings to generate
    :param isomeric: {bool} whether to consider stereo centers
    :return: different SMILES representation for same molecule
    """
    m = MolFromSmiles(smiles)
    res = list()
    while len(set(res)) < num:
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = RenumberAtoms(m, ans)
        res.append(MolToSmiles(nm, canonical=False, isomericSmiles=isomeric))
    return res


def randomize_smileslist(smiles, num=10, isomeric=True):
    """ Generate different SMILES representations for the same molecule from a list of smiles strings

    :param smiles: {str} list of SMILES strings
    :param num: {int} number of different SMILES strings to generate
    :param isomeric: {bool} whether to consider stereo centers
    :return: different SMILES representation for all molecule in the input list
    """
    def _one_random(smls, n, iso, q):
        res = list()
        for s in smls:
            r = list()
            m = MolFromSmiles(s)
            if m:
                while len(set(r)) < n:
                    ans = list(range(m.GetNumAtoms()))
                    np.random.shuffle(ans)
                    nm = RenumberAtoms(m, ans)
                    r.append(MolToSmiles(nm, canonical=False, isomericSmiles=iso))
                res.extend(r)
        q.put(res)

    queue = Queue()
    rslt = []
    for l in np.array_split(np.array(smiles), cpu_count()):
        p = Process(target=_one_random, args=(l, num, isomeric, queue))
        p.start()
    for _ in range(cpu_count()):
        rslt.extend(queue.get(10))
    return list(set(rslt))
