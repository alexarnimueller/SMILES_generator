# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from utils import tokenize_molecules, one_hot_encode


class DataGenerator(tf.keras.utils.Sequence):
    """Data generator to generate training and validation examples for the model's fit_generator method"""
    def __init__(self, smiles, ids, window, t2i, step, batch_size, shuffle=True):
        """Initialization"""
        self.smiles = smiles
        self.ids = ids
        self.window = window
        self.t2i = t2i
        self.step = step
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Returns the number of batches per epoch"""
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """One batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.generate_xy([self.ids[k] for k in indexes])

    def __call__(self, *args, **kwargs):
        return self

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def generate_xy(self, indexes):
        """generate sequence input and sequence output for one batch of SMILES"""
        x, y = [], []
        for idx in indexes:
            s = self.smiles[idx]
            inputs, targets = [], []
            # split up into windows
            for i in range(0, len(s) - self.window, self.step):
                inputs.append(s[i:i + self.window])
                targets.append(s[(i + 1):(i + self.window + 1)])

            # tokenize windows
            input_token = tokenize_molecules(inputs, self.t2i)
            target_token = tokenize_molecules(targets, self.t2i)

            # one-hot encode tokenized windows
            x.extend(one_hot_encode(input_token, len(self.t2i)).tolist())
            y.extend(one_hot_encode(target_token, len(self.t2i)).tolist())
        return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)
