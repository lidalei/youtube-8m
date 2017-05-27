"""
Multi-layer Perceptron (MLP).

Note:
    1. Normalizing features will lead to much faster convergence but worse performance.
    2. Instead, standard scaling features will help achieve better performance. 
"""
import tensorflow as tf
import numpy as np

from readers import get_reader
from utils import DataPipeline, random_sample, load_features_mean_var
from tensorflow import flags, logging, app
from utils import gap_fn
from linear_model import LogisticRegression

from os.path import join as path_join

FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640


def multi_layer_transform(data, mean=None, variance=None, **kwargs):
    """
    Multi-layer Perceptron transform data, incl. standard scale data using given mean and var.

    Args:
        data: The second dimension represents the features. A 2D tensorflow tensor.
        mean: features mean. A 1D numpy array.
        variance: features variance. A 1D numpy array.
        kwargs: For accepting other useless arguments. Here, there are reshape and size.
    Returns:
        transformed data. A tensorflow tensor.
    """
    with tf.name_scope('standard_scale'):
        features_mean = tf.Variable(initial_value=mean, trainable=False, name='features_mean')
        features_var = tf.Variable(initial_value=variance, trainable=False, name='features_var')
        standardized_data = tf.nn.batch_normalization(data,
                                                      mean=features_mean, variance=features_var,
                                                      offset=None, scale=None, variance_epsilon=1e-12,
                                                      name='standardized')
        # Add layers.



        return standardized_data


def train(init_learning_rate, decay_steps, decay_rate=0.95, l2_reg_rate=0.01, epochs=None):
    """
    Training.

    Args:
        init_learning_rate: Initial learning rate.
        decay_steps: How many training steps to decay learning rate once.
        decay_rate: How much to decay learning rate.
        l2_reg_rate: l2 regularization rate.
        epochs: The maximal epochs to pass all training data.

    Returns:

    """
    output_dir = FLAGS.output_dir
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    validate_data_pattern = FLAGS.validate_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    is_bootstrap = FLAGS.is_bootstrap

    # Increase num_readers.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=2 * num_readers)

    # Sample validate set for line search in linear classifier or logistic regression early stopping.
    _, validate_data, validate_labels, _ = random_sample(0.05, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline)

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    # Load train data mean and std.
    train_features_mean, train_features_var = load_features_mean_var(reader)

    tr_data_fn = standard_scale
    tr_data_paras = {'mean': train_features_mean, 'variance': train_features_var,
                     'reshape': False, 'size': None}

    # Run logistic regression.
    log_reg = LogisticRegression(logdir=path_join(output_dir, 'log_reg'))
    log_reg.fit(train_data_pipeline,
                tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                validate_set=(validate_data, validate_labels), validate_fn=gap_fn, bootstrap=is_bootstrap,
                init_learning_rate=init_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
                epochs=epochs, l2_reg_rate=l2_reg_rate, pos_weights=None,
                initial_weights=None, initial_biases=None)


def main(unused_argv):
    logging.set_verbosity(logging.INFO)

    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate
    l2_reg_rate = FLAGS.l2_reg_rate
    train_epochs = FLAGS.train_epochs

    train(init_learning_rate, decay_steps, decay_rate=decay_rate, l2_reg_rate=l2_reg_rate, epochs=train_epochs)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train/traina*.tfrecord',
                        'File glob for the training data set.')

    flags.DEFINE_string('validate_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/validate/validateq*.tfrecord',
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '128', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_float('init_learning_rate', 0.01, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_float('l2_reg_rate', 0.01, 'l2 regularization rate.')

    flags.DEFINE_integer('train_epochs', 200, 'Training epochs, one epoch means passing all training data once.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/video_level',
                        'The directory where intermediate and model checkpoints should be written.')

    app.run()
