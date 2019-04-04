import os
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
from cats import cats_descriptor
from descriptorcalculation import parallel_pairwise_similarities
from keras import backend as kb
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import BatchNormalization, Dense, GaussianDropout, Input, LSTM
from keras.models import Model, load_model
from keras.optimizers import Adam
from rdkit.Chem import MolFromSmiles

from generator import DataGenerator
from preprocess import preprocess_smiles
from utils import read_smiles_file, pad_seqs, one_hot_encode, transform_temp, tokenizer, is_valid_mol, \
    randomize_smileslist, compare_mollists

np.random.seed(42)
tf.set_random_seed(42)
sess = tf.Session()
kb.set_session(sess)


class SMILESmodel(object):
    def __init__(self, batch_size=128, dataset='data/default', num_epochs=25, lr=0.001, sample_after=1, temp=1.,
                 run_name="default", reference=None, step=1, reinforce=True, validation=0.2):
        self.lr = lr
        self.dataset = dataset
        self.n_mols = 0
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sample_after = sample_after
        if self.sample_after == 0:
            self.sample_after = self.num_epochs + 1
        self.run_name = run_name
        self.checkpoint_dir = './checkpoint/' + run_name + "/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.temp = temp
        self.reference = reference
        self.step = step
        self.reinforce = reinforce
        self.validation = validation
        self.n_chars = None
        self.molecules = None
        self.val_mols = None
        self.train_mols = None
        self.token_indices = None
        self.indices_token = None
        self.padded = None
        self.smiles = None
        self.model = None
        self.maxlen = None

    def load_data(self, preprocess=False, stereochem=1., augment=1):
        all_mols = read_smiles_file(self.dataset)
        if preprocess:
            all_mols = preprocess_smiles(all_mols, stereochem)
        self.molecules = all_mols
        self.smiles = all_mols
        del all_mols
        print("%i molecules loaded from %s..." % (len(self.molecules), self.dataset))
        self.maxlen = max([len(m) for m in self.molecules]) + 2
        print("Maximal sequence length: %i" % (self.maxlen - 2))
        if augment > 1:
            print("augmenting SMILES %i-fold..." % augment)
            augmented_mols = randomize_smileslist(self.molecules, num=augment)
            print("%i SMILES strings generated for %i molecules" % (len(augmented_mols), len(self.molecules)))
            self.smiles = self.molecules
            self.molecules = augmented_mols
            del augmented_mols
        self.padded = pad_seqs(["^%s$" % m for m in self.molecules], ' ', given_len=self.maxlen)
        self.n_mols = len(self.molecules)
        self.val_mols, self.train_mols = np.split(np.random.choice(range(self.n_mols), self.n_mols, replace=False),
                                                  [int(self.validation * self.n_mols)])
        print("Using %i examples for training and %i for valdiation" % (len(self.train_mols), len(self.val_mols)))
        self.build_tokenizer()

    def build_tokenizer(self, tokenize='default'):
        self.indices_token, self.token_indices = tokenizer(mode=tokenize)
        self.n_chars = len(self.indices_token.keys())

    def build_model(self):
        l_in = Input(shape=(None, self.n_chars), name='Input')
        l_out = LSTM(512, unit_forget_bias=True, return_sequences=True, name='LSTM_1')(l_in)
        l_out = GaussianDropout(0.4, name='Dropout_1')(l_out)
        l_out = LSTM(256, unit_forget_bias=True, return_sequences=True, name='LSTM_2')(l_out)
        l_out = GaussianDropout(0.2, name='Dropout_2')(l_out)
        l_out = BatchNormalization(name='BatchNorm')(l_out)
        l_out = Dense(self.n_chars, activation='softmax', name="Dense")(l_out)
        self.model = Model(l_in, l_out)
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr), metrics=['accuracy'])

    def train_model(self, n_sample=100):
        print("Training model...")
        lr_scheduler = LearningRateScheduler(self._step_decay)
        writer = tf.summary.FileWriter('./logs/' + self.run_name, graph=sess.graph)
        mol_file = open("./generated/" + self.run_name + "_generated.csv", 'a')
        i = 0
        while i < self.num_epochs:
            print("\n------ ITERATION %i ------" % i)
            chkpntr = ModelCheckpoint(filepath=self.checkpoint_dir + 'model_epoch_{:02d}.hdf5'.format(i), verbose=1)
            if self.validation:
                generator_train = DataGenerator(self.padded, self.train_mols, self.maxlen - 1, self.token_indices,
                                                self.step, self.batch_size)
                generator_val = DataGenerator(self.padded, self.val_mols, self.maxlen - 1, self.token_indices,
                                              self.step, self.batch_size)
                history = self.model.fit_generator(generator=generator_train, epochs=1, validation_data=generator_val,
                                                   use_multiprocessing=True, workers=cpu_count() - 1,
                                                   callbacks=[chkpntr, lr_scheduler])
                val_loss_sum = tf.Summary(
                    value=[tf.Summary.Value(tag="val_loss", simple_value=history.history['val_loss'][-1])])
                writer.add_summary(val_loss_sum, i)

            else:
                generator = DataGenerator(self.padded, range(self.n_mols), self.maxlen - 1, self.token_indices,
                                          self.step, self.batch_size)
                history = self.model.fit_generator(generator=generator, epochs=1, use_multiprocessing=True,
                                                   workers=cpu_count() - 1, callbacks=[chkpntr, lr_scheduler])

            loss_sum = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=history.history['loss'][-1])])
            writer.add_summary(loss_sum, i)
            lr_sum = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=kb.get_value(self.model.optimizer.lr))])
            writer.add_summary(lr_sum, i)
            # todo: learning rate scheduler not working?

            if (i + 1) % self.sample_after == 0:
                valid_mols = self.sample_points(n_sample, self.temp)
                n_valid = len(valid_mols)
                if n_valid:
                    print("Comparing novelty...")
                    novel = np.array(compare_mollists(valid_mols, np.array(self.smiles)))
                    mol_file.write("\n----- epoch %i -----\n" % i)
                    mol_file.write("\n".join(set(valid_mols)))
                else:
                    novel = []

                valid_sum = tf.Summary(value=[
                    tf.Summary.Value(tag="valid", simple_value=(float(n_valid) / n_sample))])
                novel_sum = tf.Summary(value=[tf.Summary.Value(tag="novel",
                                                               simple_value=(float(len(set(novel))) / n_sample))])
                writer.add_summary(valid_sum, i)
                writer.add_summary(novel_sum, i)
                print("\nValid:\t{}/{}".format(n_valid, n_sample))
                print("Unique:\t{}".format(len(set(valid_mols))))
                print("Novel:\t{}\n".format(len(novel)))

                if self.reinforce:
                    if len(novel) > (n_sample / 5):
                        print("Calculating similarities of novel generated molecules to SMILES pool...")
                        fp_novel = cats_descriptor([MolFromSmiles(s) for s in novel])
                        if self.reference:  # if a reference mol(s) is given, calculate distance to that one
                            fp_train = cats_descriptor([MolFromSmiles(self.reference)])
                        else:  # else calculate the distance to all training mols
                            fp_train = cats_descriptor([MolFromSmiles(s) for s in self.smiles])
                        sims = parallel_pairwise_similarities(fp_novel, fp_train, metric='euclidean')
                        top = sims[range(len(novel)), np.argsort(sims, axis=1)[:, 0, 0]].flatten()
                        # take most similar third of the novel mols and add it to self.padded
                        print("Adding %i most similar but novel molecules to SMILES pool" % int(len(top) / 3))
                        add = novel[np.argsort(top)[:int(len(top) / 3)]]
                        padd_add = pad_seqs(["^%s$" % m for m in add], ' ', given_len=self.maxlen)
                        for q, r in enumerate(np.random.choice(range(len(self.padded)), len(add), False)):
                            self.padded[r] = padd_add[q]
            i += 1  # next epoch

    def sample_points(self, n_sample=100, temp=1.0, prime_text="^", maxlen=100):
        valid_mols = []
        print("\n SAMPLING POINTS \n")
        print("----- temp: %.2f -----" % temp)
        for x in range(n_sample):
            smiles = self.sample(temp, prime_text, maxlen)
            val, s = is_valid_mol(smiles, True)
            if val:
                print(s)
                valid_mols.append(s)
        return valid_mols

    def sample(self, temp=1.0, prime_text="^", maxlen=100):
        generated = str()
        seed_token = []
        for t in list(prime_text):
            generated += t
            seed_token += [self.token_indices[t]]
        while generated[-1] != '$' and len(generated) < maxlen:
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
        self.build_tokenizer()

    def _step_decay(self, epoch):
        return self.lr * np.power(0.5, np.floor((1 + epoch) / int(self.num_epochs / 4)))

# todo: track learning rate in tensor board summary
