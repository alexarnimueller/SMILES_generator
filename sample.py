from model import SMILESmodel
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("model_path", "final_model", "model path within checkpoint directory")
flags.DEFINE_string("output_file", "", "output file for molecules")
flags.DEFINE_integer("epoch_to_load", 5, "epoch_to_load")
flags.DEFINE_integer("num_sample", 1000, "number of points to sample from trained model")
flags.DEFINE_float("temp", 0.75, "temperature to sample at")
flags.DEFINE_string("frag", "G", "Fragment to grow")
FLAGS = flags.FLAGS


def main(_):
    checkpoint_dir = './checkpoint/' + FLAGS.model_path + "/"
    model = SMILESmodel()
    model.load_model_from_file(checkpoint_dir, FLAGS.epoch_to_load)
    if FLAGS.frag[0] != 'G':
        frag = 'G' + FLAGS.frag
    else:
        frag = FLAGS.frag
    valid_mols = model.sample_points(FLAGS.num_sample, FLAGS.temp, frag)
    mol_file = open('./generated/' + FLAGS.output_file, 'a')
    mol_file.write("\n".join(valid_mols))
    print("Valid:{:02d}/{:02d}".format(len(valid_mols), FLAGS.num_sample))


if __name__ == '__main__':
    tf.app.run()
