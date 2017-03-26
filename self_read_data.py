import tensorflow as tf

import readers
import utils
from train import get_input_data_tensors


from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'video', 'video or frame level model')

# TODO, change according to running environment. Set as '' to be passed in python running command.
flags.DEFINE_string(
    "train_data_pattern", "/Users/Sophie/Documents/youtube-8m-data/train/train*.tfrecord",
    "File glob for the training dataset. If the files refer to Frame Level "
    "features (i.e. tensorflow.SequenceExample), then set --reader_type "
    "format. The (Sequence)Examples are expected to have 'rgb' byte array "
    "sequence feature as well as a 'labels' int64 context feature.")

flags.DEFINE_string('feature_names', 'mean_rgb', 'features to be used, separated by ,.')

flags.DEFINE_string('feature_sizes', '1024', 'dimensions of features to be used, separated by ,.')

flags.DEFINE_integer('batch_size', 1024, 'size of batch gradient descent')


def get_reader():
    """
    similar to train.get_reader()
    :return:
    """
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(FLAGS.feature_names, FLAGS.feature_sizes)

    if FLAGS.model_type == 'video':
        return readers.YT8MAggregatedFeatureReader(feature_sizes=feature_sizes, feature_names=feature_names)
    elif FLAGS.model_type == 'frame':
        return readers.YT8MFrameFeatureReader(feature_sizes=feature_sizes, feature_names=feature_names)
    else:
        raise NotImplementedError('Not supported model type. Supported ones are video and frame.')

if __name__ == '__main__':
    reader = get_reader()

    batch_data = get_input_data_tensors(reader, FLAGS.train_data_pattern)

    command = raw_input("Help:")
    if command == ':q':
        exit(0)