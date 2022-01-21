import os
from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf
from datetime import datetime
from descriptors import parallel_pairwise_similarities, cats_descriptor
from rdkit.Chem import MolFromSmiles, Descriptors

from generator import DataGenerator
from preprocess import preprocess_smiles
from utils import read_smiles_file, pad_seqs, one_hot_encode, transform_temp, tokenizer, is_valid_mol, \
    randomize_smileslist, compare_inchikeys, inchikey_from_smileslist


class SMILESmodel(object):
    def __init__(self, batch_size=128, dataset='data/default', num_epochs=25, lr=0.005, sample_after=1, temp=1., step=1,
                 run_name="default", reference=None, reinforce=False, num_reinforce=3, mw_filter=None, workers=1,
                 validation=0.2, seed=42):
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))
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
        self.num_reinforce = num_reinforce
        self.validation = validation
        self.mw_filter = mw_filter
        self.n_chars = None
        self.molecules = None
        self.val_mols = None
        self.train_mols = None
        self.token_indices = None
        self.indices_token = None
        self.padded = None
        self.smiles = None
        self.inchi = None
        self.model = None
        self.maxlen = None
        if workers == -1:
            self.workers = cpu_count()
            self.multi = True
        elif workers == 1:
            self.workers = 1
            self.multi = False
        else:
            self.workers = workers
            self.multi = True

    def load_data(self, preprocess=False, stereochem=1., augment=1):
        all_mols = read_smiles_file(self.dataset)
        if preprocess:
            all_mols = preprocess_smiles(all_mols, stereochem)
        self.molecules = all_mols
        self.smiles = all_mols
        print("%i molecules loaded from %s..." % (len(self.molecules), self.dataset))
        self.maxlen = max([len(m) for m in self.molecules]) + 2
        print("Maximal sequence length: %i" % (self.maxlen - 2))
        print("Creating InChI keys...")
        self.inchi = inchikey_from_smileslist(all_mols)
        del all_mols
        if augment > 1:
            print("Augmenting SMILES %i-fold..." % augment)
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
        l_in = tf.keras.layers.Input(shape=(None, self.n_chars), name='Input')
        l_out = tf.keras.layers.LSTM(512, unit_forget_bias=True, return_sequences=True, name='LSTM_1')(l_in)
        l_out = tf.keras.layers.GaussianDropout(0.25, name='Dropout_1')(l_out)
        l_out = tf.keras.layers.LSTM(256, unit_forget_bias=True, return_sequences=True, name='LSTM_2')(l_out)
        l_out = tf.keras.layers.GaussianDropout(0.25, name='Dropout_2')(l_out)
        l_out = tf.keras.layers.BatchNormalization(name='BatchNorm')(l_out)
        l_out = tf.keras.layers.Dense(self.n_chars, activation='softmax', name="Dense")(l_out)
        self.model = tf.keras.models.Model(l_in, l_out)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics=['accuracy'])

    def train_model(self, n_sample=100):
        print("Training model...")
        log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = tf.summary.create_file_writer(log_dir)
        mol_file = open("./generated/" + self.run_name + "_generated.csv", 'a')
        i = 0
        while i < self.num_epochs:
            print("\n------ ITERATION %i ------" % i)
            self.set_lr(i)
            print("\nCurrent learning rate: %.5f" % tf.keras.backend.get_value(self.model.optimizer.lr))
            chkpntr = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir + 'model_epoch_{:02d}.hdf5'.format(i), verbose=1)
            if self.validation:
                generator_train = DataGenerator(self.padded, self.train_mols, self.maxlen - 1, self.token_indices,
                                                self.step, self.batch_size)
                generator_val = DataGenerator(self.padded, self.val_mols, self.maxlen - 1, self.token_indices,
                                              self.step, self.batch_size)
                history = self.model.fit(generator_train, epochs=1, validation_data=generator_val,
                                         use_multiprocessing=self.multi, workers=self.workers,
                                         callbacks=[chkpntr])
                with writer.as_default():
                    tf.summary.scalar('val_loss', history.history['val_loss'][-1], step=i)

            else:
                generator = DataGenerator(self.padded, range(self.n_mols), self.maxlen - 1, self.token_indices,
                                          self.step, self.batch_size)
                history = self.model.fit(generator, epochs=1, use_multiprocessing=self.multi,
                                         workers=self.workers, callbacks=[chkpntr])

            # write losses to tensorboard log
            with writer.as_default():
                tf.summary.scalar('loss', history.history['loss'][-1], step=i)
                tf.summary.scalar('lr', tf.keras.backend.get_value(self.model.optimizer.lr), step=i)

            if (i + 1) % self.sample_after == 0:
                valid_mols = self.sample_points(n_sample, self.temp)
                n_valid = len(valid_mols)
                if n_valid:
                    print("Comparing novelty...")
                    inchi_valid = inchikey_from_smileslist(valid_mols)
                    inchi_novel, idx_novel = compare_inchikeys(inchi_valid, self.inchi)
                    novel = np.array(valid_mols)[idx_novel]
                    n_novel = float(len(set(inchi_novel))) / n_valid
                    mol_file.write("\n----- epoch %i -----\n" % i)
                    mol_file.write("\n".join(set(valid_mols)))
                else:
                    novel = []
                    n_novel = 0
                # write generated compound summary to tensorboard log
                with writer.as_default():
                    tf.summary.scalar('valid', (float(n_valid) / n_sample), step=i)
                    tf.summary.scalar('novel', n_novel, step=i)
                    tf.summary.scalar('unique_valid', len(set(valid_mols)), step=i)
                print("\nValid:\t{}/{}".format(n_valid, n_sample))
                print("Unique:\t{}".format(len(set(valid_mols))))
                print("Novel:\t{}\n".format(len(novel)))

                if self.reinforce:  # reinforce = add most similar generated compounds to training pool
                    if len(novel) > (n_sample / 5):
                        if self.mw_filter:
                            # only consider molecules in given MW range
                            mw = np.array([Descriptors.MolWt(MolFromSmiles(s)) if MolFromSmiles(s) else 0 for s in novel])
                            mw_idx = np.where((int(self.mw_filter[0]) < mw) & (mw < int(self.mw_filter[1])))[0]
                            novel = np.array(novel)[mw_idx]

                        print("Calculating CATS similarities of novel generated molecules to SMILES pool...")
                        fp_novel = cats_descriptor([MolFromSmiles(s) for s in novel])
                        if self.reference:  # if a reference mol(s) is given, calculate distance to that one
                            fp_train = cats_descriptor([MolFromSmiles(self.reference)])
                        else:  # else calculate the distance to all training mols
                            fp_train = cats_descriptor([MolFromSmiles(s) for s in self.smiles])
                        sims = parallel_pairwise_similarities(fp_novel, fp_train, metric='euclidean')
                        top = sims[range(len(novel)), np.argsort(sims, axis=1)[:, 0, 0]].flatten()
                        # take most similar part of the novel mols and add it to self.padded
                        print(f"Adding top {self.num_reinforce} most similar but novel molecules to SMILES pool")
                        add = randomize_smileslist(novel[np.argsort(top)[:self.num_reinforce]], num=self.num_reinforce)
                        padd_add = pad_seqs([f"^{m}$" for m in add], ' ', given_len=self.maxlen)
                        self.padded = np.hstack((self.padded, padd_add))
                        self.padded = np.random.choice(self.padded, len(self.padded), False)  # shuffle

            i += 1  # next epoch

    def sample_points(self, n_sample=100, temp=1.0, prime_text="^", maxlen=100):
        valid_mols = []
        print("\n\n----- SAMPLING POINTS AT TEMP %.2f -----" % temp)
        for x in range(n_sample):
            smiles = str()  # final smiles string will be stored in "generated"
            seed_token = []
            for t in list(prime_text):  # prepare seed token
                smiles += t
                seed_token += [self.token_indices[t]]
            while smiles[-1] != '$' and len(smiles) < maxlen:  # start sampling chars until maxlen or $ is reached
                x_seed = one_hot_encode([seed_token], self.n_chars)
                preds = self.model.predict(x_seed, verbose=0)[0]
                next_char_ind = transform_temp(preds[-1, :], temp)
                next_char = self.indices_token[str(next_char_ind)]
                smiles += next_char
                seed_token += [next_char_ind]
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
        self.model = tf.keras.models.load_model(model_file)
        self.build_tokenizer()

    def set_lr(self, epoch):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr * np.power(0.5, np.floor((epoch+1) / 5)))
