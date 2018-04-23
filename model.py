import os
import pickle
import json
import random
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, BatchNormalization, Activation, TimeDistributed, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import read_smiles_file, tokenize_molecules, pad_seqs, generate_Xy, one_hot_encode, transform_temp
from utils_chem import tokenize_smiles, is_valid_mol, preprocess_smiles

np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
sess = tf.Session()
K.set_session(sess)


class SMILESmodel(object):
    def __init__(self, batch_size=128, dataset='data/default', num_epochs=25, lr=0.001,
                 sample_after=0, run_name="default", n_mols=0, validation=0.2):
        self.lr = lr
        self.dataset = dataset
        self.n_mols = n_mols
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sample_after = sample_after
        if self.sample_after == 0:
            self.sample_after = self.num_epochs + 1
            print("Sampling after %i epochs" % self.sample_after)
        self.run_name = run_name
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.n_chars = None
        self.molecules = None
        self.token_indices = None
        self.indices_token = None
        self.model = None
        self.maxlen = None
        self.validation = validation

    def load_data(self, preprocess=False, stereochem=1., percent_length=0.8):
        pickled_name = self.dataset.split(".")[0] + ".p"
        text = read_smiles_file(self.dataset, pickled_name)
        all_mols = text.split('\n')
        if preprocess:
            all_mols = preprocess_smiles(all_mols, stereochem, percent_length)
        if self.n_mols == 0:
            self.n_mols = len(all_mols)
        kept_ind = random.sample(range(0, len(all_mols)), self.n_mols)
        self.molecules = ["G" + all_mols[i].strip() + "E" for i in kept_ind]
        del kept_ind, all_mols, text, pickled_name
        print("Molecules loaded...")

    def build_tokenizer(self, tokenize='default', pad_char="A"):
        text = "".join(self.molecules) + pad_char
        self.indices_token, self.token_indices = tokenize_smiles(text, mode=tokenize)
        self.n_chars = len(self.indices_token.keys())
        json.dump(self.indices_token, open(self.checkpoint_dir + "indices_token.json", 'w'))
        json.dump(self.token_indices, open(self.checkpoint_dir + "token_indices.json", 'w'))
        del text
        print("Molecules tokenized, token saved...")

    def build_model(self, layers=2, neurons=256, dropoutfrac=0.2):
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))
        for l in range(layers):
            self.model.add(LSTM(neurons, unit_forget_bias=True, dropout=dropoutfrac * (l+1), return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(self.n_chars, name="dense1")))
        self.model.add(Activation('softmax'))
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        print("Model built...")

    def train_model(self):
        val_split = int(self.validation * len(self.molecules))
        shuffled_mol = random.sample(self.molecules, len(self.molecules))
        print("Molecules shuffeled...")
        tokens = tokenize_molecules(shuffled_mol, self.token_indices)
        del shuffled_mol
        print("SMILES tokenized...")
        tokens = pad_seqs(tokens, pad_char=self.token_indices["A"])
        print("SMILES padded...")
        self.maxlen = len(tokens[0])
        mols_val = tokens[:val_split]
        mols_train = tokens[(val_split + 1):]
        data, targs = generate_Xy(mols_val, self.maxlen - 1)
        X_val = one_hot_encode(data, self.n_chars)
        y_val = one_hot_encode(targs, self.n_chars)

        writer = tf.summary.FileWriter('./logs/' + self.run_name, graph=sess.graph)
        mol_file = open("./generated/" + self.run_name + "_generated.csv", 'a')

        print("Training model...")
        i = 0
        while i < self.num_epochs:
            print("\n------ ITERATION %i ------" % i)
            mols_train = random.sample(mols_train, len(mols_train))
            train_tokens, train_next_tokens = generate_Xy(mols_train, self.maxlen - 1)
            X_train = one_hot_encode(train_tokens, self.n_chars)
            y_train = one_hot_encode(train_next_tokens, self.n_chars)
            chkpntr = ModelCheckpoint(filepath=self.checkpoint_dir + 'model_epoch_{:02d}.hdf5'.format(i), verbose=1)
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size,
                                     epochs=1, validation_data=(X_val, y_val), shuffle=False, callbacks=[chkpntr])
            val_loss_sum = tf.Summary(
                value=[tf.Summary.Value(tag="val_loss", simple_value=history.history['val_loss'][-1])])
            writer.add_summary(val_loss_sum, i)

            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=history.history['loss'][-1])])
            writer.add_summary(loss_sum, i)

            if (i + 1) % self.sample_after == 0:
                n_sample = 100
                for temp in [0.75, 1.0, 1.2]:
                    valid_mols = self.sample_points(n_sample, temp)
                    mol_file.write("\n".join(valid_mols))
                    n_valid = len(valid_mols)
                    valid_sum = tf.Summary(value=[
                        tf.Summary.Value(tag="molecules_" + str(temp), simple_value=(float(n_valid) / n_sample))])
                    writer.add_summary(valid_sum, i)
                    print("\nValid:\t{}/{}".format(n_valid, n_sample))
                    print("Unique:\t{}\n".format(len(set(valid_mols))))
                    del valid_mols
            i += 1

    def sample_points(self, n_sample, temp, prime_text="G"):
        valid_mols = []
        print("\n SAMPLING POINTS \n")
        print("----- temp:", temp)
        for x in range(n_sample):
            smiles = self.sample(temp, prime_text)
            print(smiles[1:])
            if is_valid_mol(smiles):
                valid_mols += [smiles[1:]]
            if x % 100 == 0:
                print("Sampled {} points".format(x + 1))
        return valid_mols

    def sample(self, temp, prime_text="G", maxlen=100):
        generated = str()
        seed_token = []
        for i in range(len(prime_text)):
            t = list(prime_text)[i]
            generated += t
            seed_token += [self.token_indices[t]]
        while generated[-1] != 'E' and len(generated) < maxlen:
            x_seed = one_hot_encode([seed_token], self.n_chars)
            preds = self.model.predict(x_seed, verbose=0)[0]
            next_char_ind = transform_temp(preds[-1, :], temp)
            next_char = self.indices_token[str(next_char_ind)]
            generated += next_char
            seed_token += [next_char_ind]
        return generated[:-1]

    def load_model_from_file(self, checkpoint_dir, epoch):
        model_file = checkpoint_dir + 'model_epoch_{:02d}.hdf5'.format(epoch)
        print("Loading model from file: " + model_file)
        self.model = load_model(model_file)
        self.load_token(checkpoint_dir)

    def load_token(self, dirname):
        print("Loading token sets from %s ..." % dirname)
        try:
            self.indices_token = json.load(open(os.path.join(dirname, "indices_token.json"), 'r'))
            self.token_indices = json.load(open(os.path.join(dirname, "token_indices.json"), 'r'))
        except:
            self.indices_token = pickle.load(open(os.path.join(dirname, "indices_token.p"), 'rb'))
            self.token_indices = pickle.load(open(os.path.join(dirname, "token_indices.p"), 'rb'))
        self.n_chars = len(self.indices_token.keys())
