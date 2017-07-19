"""
Multi-layer Perceptron (MLP).

Note:
    1. Normalizing features will lead to much faster convergence but worse performance.
    2. Instead, standard scaling features will help achieve better performance.
"""
import tensorflow as tf
import numpy as np

from readers import get_reader
from utils import DataPipeline, random_sample, load_features_mean_var, load_sum_labels
from tensorflow import flags, logging, app
from utils import gap_fn
from linear_model import LogisticRegression

from os.path import join as path_join

FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640


def create_hidden_layer(data, name, pre_size, size, pos_activation, pos_transform=None, pos_transform_paras=None):
    """
    Create a hidden fully connected layer.

    Args:
        data: Data output from previous layer. A tf tensor.
        name: Layer name.
        pre_size: Previous layer size, namely number of neurons.
        size: Number of neurons.
        pos_activation: Activation function.
        pos_transform: Other transform, such as dropout and batch normalization.
        pos_transform_paras: pos transform parameters.
    Returns:
        Transformed data, i.e., data after passing this layer. A tensorflow tensor.
    """
    with tf.name_scope(name):
        # Initialize weights based on fan-in.
        weights = tf.Variable(initial_value=tf.truncated_normal(
            [pre_size, size], stddev=1.0 / np.sqrt(pre_size)), name='weights')
        biases = tf.Variable(initial_value=tf.zeros([size]), name='biases')

        inner_product = tf.matmul(data, weights) + biases
        activation = pos_activation(inner_product)

        # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)

        # Add to summary.
        tf.summary.histogram('model/weights', weights)
        tf.summary.histogram('model/biases', biases)
        tf.summary.histogram('model/activation', activation)

        if pos_transform is not None:
            if pos_transform_paras is None:
                pos_transform_paras = dict()
            transformed_data = pos_transform(activation, **pos_transform_paras)
            return transformed_data
        else:
            return activation


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
    with tf.name_scope('mlp_transform'):
        with tf.name_scope('standard_scale'):
            features_mean = tf.Variable(initial_value=mean, trainable=False, name='features_mean')
            features_var = tf.Variable(initial_value=variance, trainable=False, name='features_var')
            standardized = tf.nn.batch_normalization(data, mean=features_mean, variance=features_var,
                                                     offset=None, scale=None, variance_epsilon=1e-12,
                                                     name='standardized')

        current_size = mean.shape[-1]
        current_data = standardized

        # First Hidden layers---#
        layer_idx = 1
        layer_name = 'hidden_{}'.format(layer_idx)
        layer_size = 600
        # Try relu and tanh, no sigmoid.
        hidden_activation = create_hidden_layer(current_data, layer_name, current_size, layer_size, tf.nn.tanh)
        # ---First Hidden layers#
        current_size = layer_size
        current_data = hidden_activation

        """
        # Second Hidden layers---#
        layer_idx = 2
        layer_name = 'hidden_{}'.format(layer_idx)
        layer_size = 200
        hidden_activation = create_hidden_layer(current_data, layer_name, current_size, layer_size, tf.tanh)
        # ---Second Hidden layers#
        current_size = layer_size
        current_data = hidden_activation
        """

        return current_data


def main(unused_argv):
    """
        init_learning_rate: Initial learning rate.
        decay_steps: How many training steps to decay learning rate once.
        decay_rate: How much to decay learning rate.
        l2_reg_rate: l2 regularization rate.
        epochs: The maximal epochs to pass all training data.
    """
    logging.set_verbosity(logging.INFO)

    start_new_model = FLAGS.start_new_model
    output_dir = FLAGS.output_dir

    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate
    l2_reg_rate = FLAGS.l2_reg_rate
    train_epochs = FLAGS.train_epochs

    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    validate_data_pattern = FLAGS.validate_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    # Increase num_readers.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=num_readers)

    # Sample validate set for line search in linear classifier or logistic regression early stopping.
    _, validate_data, validate_labels, _ = random_sample(0.05, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline)

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    # If start a new model or output dir does not exist, truly start a new model.
    start_new_model = start_new_model or (not tf.gfile.Exists(output_dir))

    if start_new_model:
        # Load train data mean and std.
        train_features_mean, train_features_var = load_features_mean_var(reader)

        tr_data_fn = multi_layer_transform
        tr_data_paras = {'mean': train_features_mean, 'variance': train_features_var,
                         'reshape': True, 'size': 600}

        # Set pos_weights for extremely imbalanced situation in one-vs-all classifiers.
        try:
            # Load sum_labels in training set, numpy float format to compute pos_weights.
            train_sum_labels = load_sum_labels()
            # num_neg / num_pos, assuming neg_weights === 1.0.
            pos_weights = np.sqrt(float(NUM_TRAIN_EXAMPLES) / train_sum_labels - 1.0)
            logging.info('Computing pos_weights based on sum_labels in train set successfully.')
        except:
            logging.error('Cannot load train sum_labels. Use default value.')
            pos_weights = None
        finally:
            pos_weights = None
    else:
        tr_data_fn = None
        tr_data_paras = dict()
        pos_weights = None

    # Run logistic regression.
    log_reg = LogisticRegression(logdir=output_dir)
    log_reg.fit(train_data_pipeline, start_new_model=start_new_model,
                tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                validate_set=(validate_data, validate_labels), validate_fn=gap_fn, bootstrap=False,
                init_learning_rate=init_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
                epochs=train_epochs, l1_reg_rate=None, l2_reg_rate=l2_reg_rate, pos_weights=pos_weights,
                initial_weights=None, initial_biases=None)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train_validate/train*.tfrecord',
                        'File glob for the training data set.')

    flags.DEFINE_string('validate_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train_validate/validate*.tfrecord',
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_rgb,mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '1024,128', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_bool('start_new_model', True, 'To start a new model or restore from output dir.')

    flags.DEFINE_float('init_learning_rate', 0.001, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_float('l2_reg_rate', None, 'l2 regularization rate.')

    flags.DEFINE_integer('train_epochs', 20, 'Training epochs, one epoch means passing all training data once.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/video_level/mlp',
                        'The directory where intermediate and model checkpoints should be written.')

    app.run()
