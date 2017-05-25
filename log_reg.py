"""
One-vs-all logistic regression.

Note:
    1. Normalizing features will lead to much faster convergence but worse performance.
    2. Instead, standard scaling features will help achieve better performance.
    3. Initializing with linear regression will help get even better result.
    4. Bagging is implemented as training separately but combining inferences from multiple models. 
TODO:
    Add layers to form a neural network.
"""
import tensorflow as tf
import numpy as np

from readers import get_reader
from utils import DataPipeline, random_sample, load_sum_labels, load_features_mean_var
from tensorflow import flags, logging, app
from eval_util import calculate_gap
from linear_model import LinearClassifier, LogisticRegression

from os.path import join as path_join


FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640


def gap_fn(predictions=None, labels=None):
    """
    Make predictions and labels to be specified explicitly.
    :param predictions: Model output.
    :param labels: Targets or ground truth.
    :return: GAP - global average precision.
    """
    return calculate_gap(predictions, labels)


def standard_scale(data, mean=None, variance=None, **kwargs):
    """
    Standard scale data using given mean and var.
    
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
                                                      name='standardized_video_batch')
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
    init_with_linear_clf = FLAGS.init_with_linear_clf
    is_bootstrap = FLAGS.is_bootstrap

    # Increase num_readers.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=2 * num_readers)

    # Sample validate set for line search in linear classifier or logistic regression early stopping.
    _, validate_data, validate_labels, _ = random_sample(0.1, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline)

    # Set pos_weights for extremely imbalanced situation in one-vs-all classifiers.
    try:
        # Load sum_labels in training set, numpy float format to compute pos_weights.
        train_sum_labels = load_sum_labels()
        # num_neg / num_pos, assuming neg_weights === 1.0.
        pos_weights = np.sqrt((float(NUM_TRAIN_EXAMPLES) - train_sum_labels) / train_sum_labels)
        logging.info('Computing pos_weights based on sum_labels in train set successfully.')
    except:
        logging.error('Cannot load train sum_labels. Use default value.')
        pos_weights = None
    finally:
        # Set it as None to disable pos_weights.
        pos_weights = None

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    if init_with_linear_clf:
        # ...Start linear classifier...
        # Compute weights and biases of linear classifier using normal equation.
        # Linear search helps little.
        linear_clf = LinearClassifier(logdir=path_join(output_dir, 'linear_classifier'))
        linear_clf.fit(data_pipeline=train_data_pipeline, l2_regs=[0.01],
                       validate_set=(validate_data, validate_labels), line_search=False)
        linear_clf_weights, linear_clf_biases = linear_clf.weights, linear_clf.biases

        logging.info('linear classifier weights and biases with shape {}, {}'.format(
            linear_clf_weights.shape, linear_clf_biases.shape))
        logging.debug('linear classifier weights and {} biases: {}.'.format(
            linear_clf_weights, linear_clf_biases))
        # ...Exit linear classifier...
    else:
        linear_clf_weights, linear_clf_biases = None, None

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
                epochs=epochs, l2_reg_rate=l2_reg_rate, pos_weights=pos_weights,
                initial_weights=linear_clf_weights, initial_biases=linear_clf_biases)


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

    flags.DEFINE_string('test_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/test/test4*.tfrecord',
                        'Test data pattern, to be specified when making predictions.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '128', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_bool('is_bootstrap', False, 'Boolean variable indicating using bootstrap or not.')

    flags.DEFINE_boolean('init_with_linear_clf', True,
                         'Boolean variable indicating whether to init logistic regression with linear classifier.')

    flags.DEFINE_float('init_learning_rate', 0.01, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_float('l2_reg_rate', 0.01, 'l2 regularization rate.')

    flags.DEFINE_integer('train_epochs', 200, 'Training epochs, one epoch means passing all training data once.')

    flags.DEFINE_boolean('is_tuning_hyper_para', False,
                         'Boolean variable indicating whether to perform hyper-parameter tuning.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/video_level',
                        'The directory where intermediate and model checkpoints should be written.')

    app.run()
