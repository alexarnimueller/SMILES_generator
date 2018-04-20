import os
import tensorflow as tf

from model import SMILESmodel

flags = tf.app.flags
flags.DEFINE_string("model_path", "final_model", "model path within checkpoint directory")
flags.DEFINE_string("dataset", "", "[REQUIRED] dataset for fine tuning")
flags.DEFINE_integer("epoch_to_load", 5, "epoch_to_load")
flags.DEFINE_integer("epochs_to_train", 5, "number of epochs to fine tune")
flags.DEFINE_integer("num_sample", 0, "[REQUIRED] number of points to sample from trained model")
flags.DEFINE_float("temp", 0.75, "temperature to sample at")
flags.DEFINE_string("run_name", "", "run_name for output files")
flags.DEFINE_boolean("preprocess", True, "Whether to preprocess stereochemistry/salts etc.")
flags.DEFINE_boolean("stereochemistry", False, "whether stereochemistry information should be included")
flags.DEFINE_float("percent_length", 0.8, "percent of length to take into account")
flags.DEFINE_bool("ask", True, "whether the run stops after the training and shows distances to sampled molecules")

FLAGS = flags.FLAGS


def main(_):
    if FLAGS.num_sample == 0 or len(FLAGS.dataset) == 0:
        print("ERROR: Please specify required inputs!")

    if len(FLAGS.run_name) == 0:
        run = FLAGS.dataset.split(".")[0]
    else:
        run = FLAGS.run_name
    checkpoint_dir = './checkpoint/' + FLAGS.model_path + "/"

    model = SMILESmodel(dataset=FLAGS.dataset, num_epochs=FLAGS.epochs_to_train, run_name=run, validation=False)
    print("Model initialized...")
    model.load_data(preprocess=FLAGS.preprocess, percent_length=FLAGS.preprocess_lengths,
                    stereochem=FLAGS.stereochemistry)
    model.load_model_from_file(checkpoint_dir, FLAGS.epoch_to_load)
    print("Model loaded...")
    model.train_model()

    valid_mols = model.sample_points(FLAGS.num_sample, FLAGS.temp)
    mol_file = open('./generated/' + run + '_finetune_generated.txt', 'a')
    mol_file.write("\n".join(valid_mols))
    print("Valid:{:02d}/{:02d}".format(len(valid_mols), FLAGS.num_sample))

    os.system("cp %s*.p ./checkpoint/%s/" % (checkpoint_dir, run))  # copy tokenizer to fine-tuned model folder


if __name__ == '__main__':
    tf.app.run()
