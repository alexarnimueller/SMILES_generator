import json
import os
import random

import numpy as np
import tensorflow as tf
from keras import backend as kb
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, GaussianDropout, BatchNormalization, TimeDistributed, \
    RepeatVector
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

from preprocess import preprocess_smiles
from utils import read_smiles_file, tokenize_molecules, pad_seqs, generate_Xy, one_hot_encode, transform_temp, \
    tokenize_smiles, is_valid_mol, randomize_smileslist, compare_mollists

np.random.seed(42)
random.seed(42)
tf.set_random_seed(42)
sess = tf.Session()
kb.set_session(sess)


class SMILESmodel(object):
    def __init__(self, batch_size=128, dataset='data/default', num_epochs=25, lr=0.001,
                 sample_after=0, run_name="default", validation=0.2):
        self.lr = lr
        self.dataset = dataset
        self.n_mols = 0
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

    def load_data(self, preprocess=False, stereochem=1., augment=1):
        all_mols = read_smiles_file(self.dataset)
        if preprocess:
            all_mols = preprocess_smiles(all_mols, stereochem)
        self.molecules = all_mols
        self.maxlen = max([len(m) for m in self.molecules]) + 2
        del all_mols
        print("%i molecules loaded from %s..." % (len(self.molecules), self.dataset))
        if augment > 1:
            print("augmenting SMILES %i-fold..." % augment)
            augmented_mols = randomize_smileslist(self.molecules, num=augment)
            print("%i SMILES strings generated for %i molecules" % (len(augmented_mols), len(self.molecules)))
            self.molecules = augmented_mols
            del augmented_mols
        self.molecules = pad_seqs(["G%sE" % m for m in self.molecules], 'A', given_len=self.maxlen)
        self.n_mols = len(self.molecules)

    def build_tokenizer(self, tokenize='default'):
        self.indices_token, self.token_indices = tokenize_smiles(mode=tokenize)
        self.n_chars = len(self.indices_token.keys())
        json.dump(self.indices_token, open(self.checkpoint_dir + "indices_token.json", 'w'))
        json.dump(self.token_indices, open(self.checkpoint_dir + "token_indices.json", 'w'))

    def build_model(self, layers=2, neurons=256, dropoutfrac=0.2):
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=(None, self.n_chars)))
        for l in range(layers):
            self.model.add(LSTM(neurons, unit_forget_bias=True, return_sequences=True, name='LSTM%i' % l))
            self.model.add(GaussianDropout(dropoutfrac * (l + 1)))
        self.model.add(BatchNormalization())
        self.model.add(TimeDistributed(Dense(self.n_chars, activation='softmax', name="dense")))
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def generator(self, train_or_val):
        print("Generating data...")
        for batch in np.array_split(self.molecules, len(self.molecules) / self.batch_size):
            token = tokenize_molecules(np.random.choice(batch, len(batch), replace=False), self.token_indices)
            mols_val, mols_train = np.split(token, [int(self.validation * len(batch))])
            if train_or_val == 'val':
                val_tokens, val_next_tokens = generate_Xy(mols_val, self.maxlen - 2)
                x_val = one_hot_encode(val_tokens, self.n_chars)
                y_val = one_hot_encode(val_next_tokens, self.n_chars)
                yield x_val, y_val
            elif train_or_val == 'train':
                train_tokens, train_next_tokens = generate_Xy(mols_train, self.maxlen - 2)
                x_train = one_hot_encode(train_tokens, self.n_chars)
                y_train = one_hot_encode(train_next_tokens, self.n_chars)
                yield x_train, y_train

    def train_model(self, n_sample=10):
        print("Training model...")
        writer = tf.summary.FileWriter('./logs/' + self.run_name, graph=sess.graph)
        mol_file = open("./generated/" + self.run_name + "_generated.csv", 'a')
        i = 0
        while i < self.num_epochs:
            print("\n------ ITERATION %i ------" % i)
            chkpntr = ModelCheckpoint(filepath=self.checkpoint_dir + 'model_epoch_{:02d}.hdf5'.format(i), verbose=1)

            if self.validation:
                steps = len(np.array_split(self.molecules, len(self.molecules) / self.batch_size))
                history = self.model.fit_generator(generator=self.generator('train'), steps_per_epoch=steps,
                                                   epochs=1, validation_data=self.generator('val'),
                                                   validation_steps=steps, callbacks=[chkpntr])
                val_loss_sum = tf.Summary(
                    value=[tf.Summary.Value(tag="val_loss", simple_value=history.history['val_loss'][-1])])
                writer.add_summary(val_loss_sum, i)

            else:
                history = self.model.fit_generator(generator=self.generator('train'), steps_per_epoch=self.n_mols,
                                                   epochs=1, callbacks=[chkpntr])

            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=history.history['loss'][-1])])
            writer.add_summary(loss_sum, i)

            if (i + 1) % self.sample_after == 0:
                temp = 1.
                valid_mols = self.sample_points(n_sample, temp)
                n_valid = len(valid_mols)
                novel = compare_mollists(valid_mols, np.array(self.molecules))
                mol_file.write("\n".join(set(valid_mols)))

                valid_sum = tf.Summary(value=[
                    tf.Summary.Value(tag="valid_molecules_" + str(temp), simple_value=(float(n_valid) / n_sample))])
                novel_sum = tf.Summary(value=[tf.Summary.Value(tag="new_molecules_" + str(temp),
                                                               simple_value=(float(len(set(novel))) / n_sample))])
                writer.add_summary(valid_sum, i)
                writer.add_summary(novel_sum, i)
                print("\nValid:\t{}/{}".format(n_valid, n_sample))
                print("Unique:\t{}".format(len(set(valid_mols))))
                print("Novel:\t{}\n".format(len(novel)))
                del valid_mols, valid_sum, novel_sum
            i += 1

    def sample_points(self, n_sample, temp, prime_text="G"):
        valid_mols = []
        print("\n SAMPLING POINTS \n")
        print("----- temp: %.2f -----" % temp)
        for x in range(n_sample):
            smiles = self.sample(temp, prime_text)
            val, s = is_valid_mol(smiles, True)
            print(s)
            if val:
                valid_mols.append(s)
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
        return generated

    def load_model_from_file(self, checkpoint_dir, epoch):
        model_file = checkpoint_dir + 'model_epoch_{:02d}.hdf5'.format(epoch)
        print("Loading model from file: " + model_file)
        self.model = load_model(model_file)
        self.load_token(checkpoint_dir)

    def load_token(self, dirname):
        print("Loading token sets from %s ..." % dirname)
        self.indices_token = json.load(open(os.path.join(dirname, "indices_token.json"), 'r'))
        self.token_indices = json.load(open(os.path.join(dirname, "token_indices.json"), 'r'))
        self.n_chars = len(self.indices_token.keys())


class SMILESautoencoder(SMILESmodel):
    def build_model(self, layers=2, neurons=32, dropoutfrac=0.3):
        # encoder
        print(len(self.molecules), self.maxlen, self.n_chars)
        inputs = Input(shape=(self.maxlen, self.n_chars))
        encoded = LSTM(self.n_chars, unit_forget_bias=True, return_sequences=True)(inputs)
        encoded = GaussianDropout(dropoutfrac)(encoded)
        encoded = LSTM(neurons)(encoded)
        self.encoder = Model(inputs, encoded)

        # decoder
        decoded = RepeatVector(self.maxlen)(encoded)
        decoded = LSTM(neurons, unit_forget_bias=True, return_sequences=True)(decoded)
        decoded = GaussianDropout(dropoutfrac)(decoded)
        decoded = LSTM(self.n_chars, return_sequences=True)(decoded)
        decoded = Dense(self.n_chars, activation='softmax')(decoded)

        # autoencoder
        self.model = Model(inputs, decoded)
        self.model.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy')
        print("Model built...")

# TODO: make data for AE right
