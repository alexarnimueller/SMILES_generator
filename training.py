import json
import tensorflow as tf
from model import SMILESmodel

flags = tf.app.flags
flags.DEFINE_string("dataset", "data/chembl_smiles_preprocessed.csv", "dataset file (expecting csv or pickled file)")
flags.DEFINE_string("run_name", "test", "run name for log and checkpoint files")
flags.DEFINE_string("tokenizer", "default", "Tokenizer for one-hot encoding: 'default': 71 tokens;"
                    "'generate': generate new tokenizer from input data")
flags.DEFINE_integer("layers", 2, "number of LSTM layers in the network")
flags.DEFINE_float("dropout", 0.2, "fraction of dropout to apply; layer 1 gets 1*dropout, layer 2 2*dropout etc.")
flags.DEFINE_integer("neurons", 256, "number of neurons per layer")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("batch_size", 128, "Batch size")
flags.DEFINE_integer("step", 2, "Step size")
flags.DEFINE_integer("sample_after", 5, "Sample after how many epochs")
flags.DEFINE_integer("epochs", 10, "Epochs to train")
flags.DEFINE_integer("n_mols", 0, "Number of molecules to keep from the dataset")
flags.DEFINE_boolean("preprocess", True, "Whether to preprocess stereochemistry/salts etc.")
flags.DEFINE_integer("stereochemistry", 1, "whether stereochemistry information should be included [0, 1]")
flags.DEFINE_float("percent_length", 0.8, "percent of length to take into account ")
flags.DEFINE_float("validation", 0.2, "Fraction of the data to use as a validation set")

FLAGS = flags.FLAGS


def main(_):
    print("Running SMILES LSTM model...")
    model = SMILESmodel(batch_size=FLAGS.batch_size, dataset=FLAGS.dataset,
                        num_epochs=FLAGS.epochs, lr=FLAGS.learning_rate, n_mols=FLAGS.n_mols,
                        run_name=FLAGS.run_name, sample_after=FLAGS.sample_after, validation=FLAGS.validation)
    model.load_data(preprocess=FLAGS.preprocess, stereochem=FLAGS.stereochemistry, percent_length=FLAGS.percent_length)
    model.build_tokenizer(tokenize=FLAGS.tokenizer, pad_char='A')
    model.build_model(layers=FLAGS.layers, neurons=FLAGS.neurons, dropoutfrac=FLAGS.dropout)
    model.train_model()
    json.dump(FLAGS.__dict__, open('./checkpoint/%s/flags.json' % FLAGS.run_name, 'w'))  # save used flags


if __name__ == '__main__':
    tf.app.run()
