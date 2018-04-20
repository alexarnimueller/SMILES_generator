import os
import numpy as np
import pickle
import progressbar


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
                    print("read ", str(num_lines), " lines")
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
