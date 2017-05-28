"""
The referenced paper is
Schwenker F, Kestler H A, Palm G. Three learning phases for radial-basis-function networks[J].
Neural networks, 2001, 14(4): 439-458.
and
Zhang M L. ML-RBF: RBF neural networks for multi-label learning[J]. Neural Processing Letters, 2009, 29(2): 61-74.

In this implementation, training process is split as three phases as described in Schwenker's. Moreover, different
 second-phases are compared and a third phased is added or not.

More specifically, three different frameworks are implemented. 1, for all L labels, finding a number of centers
 and train L binary logistic regression models on these centers. 2, for each label, finding a number of centers
 and train a logistic regression model on these centers. In total, there are L groups of centers and L logistic
 regression models. 3, for each label, finding a certain number of centers and train L logistic regression models on
 all these centers as a whole group. The first framework is described in Schwenker's as multi-class classification.
 The second one works as one-vs-all. And the third is described in Zhang's.

First layer, feature transform using radius bias kernel functions.
Second layer, logistic regression to learn classifiers, including tuning feature transformers and weights and biases.
"""
import tensorflow as tf
from tensorflow import flags, logging, app

from kmeans import KMeans
from linear_model import LinearClassifier, LogisticRegression
from readers import get_reader
from utils import DataPipeline, random_sample, gap_fn, load_sum_labels, compute_data_mean_var

from os.path import join as path_join
import numpy as np
import scipy.spatial.distance as sci_distance


FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640

MAX_TRAIN_STEPS = 1000000


def initialize(num_centers_ratio, data_pipeline, method=None, metric='cosine',
               max_iter=100, tol=0.005, scaling_method=1, alpha=1.0, p=3):
    """
    This functions initializes representative prototypes (RBF centers) c and scaling factors sigma.

    This function will generate one group of centers for all labels as a whole. Be cautious with initialize_per_label.
    Args:
        num_centers_ratio: The number of centers to be decided / total number of examples that belong to label l,
            for l = 0, ..., num_classes - 1.
        data_pipeline: A namedtuple consisting of the following elements.
            reader, video-level features reader or frame-level features reader.
            data_pattern, File Glob of data set.
            batch_size, How many examples to handle per time.
            num_readers, How many IO threads to prefetch examples.
        method: The method to decide the centers. Possible choices are random selection, kmeans and online (kmeans).
         Default is None, which represents randomly selecting a certain number of examples as centers.
        metric: Distance metric, euclidean distance or cosine distance.
        max_iter: The maximal number of iterations clustering to be done.
        tol: The minimal reduction of objective function of clustering to be reached to stop iteration.
        scaling_method: There are four choices. 1, all of them use the same sigma, the p smallest pairs of distances.
         2, average of p nearest centers.
         3, distance to the nearest center that has a different label (Not supported!).
         4, mean distance between this center and all of its points.
        alpha: The alpha parameter that should be set heuristically. It works like a learning rate. (mu in Zhang)
        p: When scaling_method is 1 or 2, p is needed.
    Returns:
        centers (prototypes) and scaling factors (sigmas).
    Raises:
        ValueError if num_centers_ratio is not between 0.0 (open) and 1.0 (closed).
        ValueError if metric is not euclidean or cosine.
        ValueError if method is not one of None, kmeans or online.
        NotImplementedError if scaling_method is 3 or 5.
        ValueError if scaling method is not 1 - 5.
    """
    logging.info('Generate a group of centers for all labels. See Schwenker.')
    # Argument checking.
    if (num_centers_ratio <= 0.0) or (num_centers_ratio > 1.0):
        raise ValueError('num_centers_ratio must be larger than 0.0 and no greater than 1.0.')
    logging.info('num_centers_ratio is {}.'.format(num_centers_ratio))

    if ('euclidean' == metric) or ('cosine' == metric):
        logging.info('Using {} distance. The larger, the less similar.'.format(metric))
    else:
        raise ValueError('Only euclidean and cosine distance are supported, {} passed.'.format(metric))

    # Sample features only.
    _, centers, _, _ = random_sample(num_centers_ratio, mask=(False, True, False, False),
                                     data_pipeline=data_pipeline, name_scope='sample_centers')
    logging.info('Sampled {} centers totally.'.format(centers.shape[0]))
    logging.debug('Randomly selected centers: {}'.format(centers))

    # Used in scaling method 4. Average distance of each point with its cluster center.
    per_clu_mean_dist = None
    # Perform k-means or online k-means.
    if method is None:
        logging.info('Using randomly selected centers as model prototypes (centers).')
    elif 'online' == method:
        raise NotImplementedError('Only None (randomly select examples), online, kmeans are supported.')
    elif 'kmeans' == method:
        logging.info('Using k-means clustering result as model prototypes (centers).')

        return_mean_clu_dist = (scaling_method == 4)
        kmeans = KMeans(centers, data_pipeline=data_pipeline, metric=metric,
                        return_mean_clu_dist=return_mean_clu_dist)
        kmeans.fit(max_iter=max_iter, tol=tol)
        # Get current centers and update centers.
        centers = kmeans.current_centers
        per_clu_mean_dist = kmeans.per_clu_mean_dist

    else:
        raise ValueError('Only None (randomly select examples), online, kmeans are supported.')

    # Compute scaling factors based on these centers.
    num_centers = centers.shape[0]
    sigmas = None
    if scaling_method == 1:
        # Equation 27.
        pairwise_distances = sci_distance.pdist(centers, metric=metric)
        p = min(p, len(pairwise_distances))
        logging.info('Using {} minimal pairwise distances.'.format(p))
        # np.partition second argument begins with 0.
        sigmas = np.array([alpha * np.mean(np.partition(pairwise_distances, p - 1)[:p])] * num_centers,
                          dtype=np.float32)
    elif scaling_method == 2:
        # Equation 28.
        p = min(p, num_centers - 1)
        logging.info('Using {} minimal distances per center.'.format(p))

        if 'euclidean' == metric:
            dis_fn = sci_distance.euclidean
        else:
            dis_fn = sci_distance.cosine
        sigmas = []
        for c in centers:
            distances = [dis_fn(c, _c) for _c in centers]
            # The distance between c and itself is zero and is in the left partition.
            sigmas.append(alpha * np.sum(np.partition(distances, p)[:p + 1]) / float(p))

        sigmas = np.array(sigmas, dtype=np.float32)
    elif scaling_method == 3:
        # Equation 29.
        raise NotImplementedError('Not supported when all labels use the same centers.')
    elif scaling_method == 4:
        # Equation 30.
        if per_clu_mean_dist is None:
            kmeans = KMeans(centers, data_pipeline=data_pipeline, metric=metric, return_mean_clu_dist=True)
            kmeans.fit(max_iter=1, tol=tol)

            centers = kmeans.current_centers
            per_clu_mean_dist = kmeans.per_clu_mean_dist

            logging.info('Compute mean distance per cluster using kmeans or online kmeans.')
        else:
            logging.info('Reuse mean distance per cluster computed in kmeans or online kmeans.')

        sigmas = alpha * per_clu_mean_dist
    elif scaling_method == 5:
        # Equation 31.
        raise NotImplementedError('Only three methods are supported. Please read the documentation.')
    else:
        raise ValueError('Only three methods are supported. Please read the documentation.')

    logging.debug('Scaling factor sigmas: {}'.format(sigmas))

    return centers, sigmas


def initialize_per_label():
    """
    This functions implements the following two phases:
    1. To initialize representative prototypes (RBF centers) c and scaling factors sigma.
    2. And to fit output weights.

    It is different from the function initialize because it will generate L groups of centers (one per each label)
    instead of one group of centers for all labels as a whole.

    :return:
    """
    # Must consider the labels are super imbalanced!
    # The counts are stored in 'sum_labels.pickle' with Python3 protocol.
    # logging.error
    raise NotImplementedError('It is a little troubling, will be implemented later! Be patient.')


def rbf_transform(data, centers=None, sigmas=None, metric='cosine', **kwargs):
    """
    Transform data using given rbf centers and sigmas.
    
    Args:
        data: A 2D tensorflow tensor. The second dimension represents the features.
        centers: rbf centers. A numpy array. The second dimension equals data.
        sigmas: rbf scaling factors. A 1D numpy array. One sigma for each center.
        metric: distance metric. A string. cosine or euclidean.
        kwargs: For accepting other useless arguments. Here, there are reshape and size.
    Returns:
        transformed data. A tensorflow tensor.
    Raises:
        ValueError if metric is not cosine or euclidean.
    """
    if ('cosine' == metric) or ('euclidean' == metric):
        logging.info('rbf transform using {} distance.'.format(metric))
    else:
        raise ValueError('Only supported cosine and euclidean. Passed {}.'.format(metric))

    transform_name = 'rbf_transform_{}'.format(metric)
    with tf.name_scope(transform_name):
        if ('reuse' in kwargs) and (kwargs['reuse'] is True):
            # Get from collection.
            prototypes = tf.get_collection('prototypes')[0]
            basis_f_mul = tf.get_collection('basis_f_mul')[0]
        else:
            if 'cosine' == metric:
                normalized_centers = centers / np.clip(
                    np.linalg.norm(centers, axis=-1, keepdims=True), 1e-6, np.PINF)
                # prototypes are trainable.
                prototypes = tf.Variable(initial_value=normalized_centers, trainable=False,
                                         dtype=tf.float32, name='prototypes')
            else:
                # prototypes are trainable.
                prototypes = tf.Variable(initial_value=centers, trainable=False,
                                         dtype=tf.float32, name='prototypes')

            neg_twice_sq_sigmas = np.multiply(-2.0, np.square(sigmas))
            inv_neg_twice_sq_sigmas = np.divide(1.0, neg_twice_sq_sigmas)
            expanded_inv_neg_twice_sq_sigmas = np.expand_dims(inv_neg_twice_sq_sigmas, axis=0)
            # [-2.0 * sigmas ** 2], basis function denominators.
            basis_f_mul = tf.Variable(initial_value=expanded_inv_neg_twice_sq_sigmas, trainable=False,
                                      dtype=tf.float32, name='basis_f_mul')
            # Add to collection for future use, e.g., validate and train share the same variables.
            tf.add_to_collection('prototypes', prototypes)
            tf.add_to_collection('basis_f_mul', basis_f_mul)

            # For debug.
            tf.summary.histogram('{}/prototypes'.format(transform_name), prototypes)
            tf.summary.histogram('{}/basis_f_mul'.format(transform_name), basis_f_mul)
        # Do transform.
        if 'cosine' == metric:
            normalized_data = tf.nn.l2_normalize(data, -1)
            cosine_sim = tf.matmul(normalized_data, prototypes, transpose_b=True)
            squared_dist = tf.square(tf.subtract(1.0, cosine_sim), name='cosine_square_dist')
        else:
            # Make use of broadcasting feature.
            expanded_centers = tf.expand_dims(prototypes, axis=0)
            expanded_video_batch = tf.expand_dims(data, axis=1)

            sub = tf.subtract(expanded_video_batch, expanded_centers)
            # element-wise square.
            squared_sub = tf.square(sub)
            # Compute distances with centers video-wisely.
            # Shape [batch_size, num_initial_centers]. negative === -.
            squared_dist = tf.reduce_sum(squared_sub, axis=-1, name='euclidean_square_dist')

        rbf_fs = tf.exp(tf.multiply(squared_dist, basis_f_mul), name='basis_function')

        if 'mean' in kwargs and 'variance' in kwargs:
            logging.info('Standard scale is performed after rbf transform.')
            mean = kwargs['mean']
            variance = kwargs['variance']
            with tf.name_scope('standard_scale'):
                features_mean = tf.Variable(initial_value=mean, trainable=False, name='features_mean')
                features_var = tf.Variable(initial_value=variance, trainable=False, name='features_var')
                standardized_data = tf.nn.batch_normalization(rbf_fs,
                                                              mean=features_mean, variance=features_var,
                                                              offset=None, scale=None, variance_epsilon=1e-12,
                                                              name='standardized')
                return standardized_data
        else:
            return rbf_fs


def main(unused_argv):
    """
    Train the rbf network.
    """
    logging.set_verbosity(logging.INFO)

    start_new_model = FLAGS.start_new_model
    output_dir = FLAGS.output_dir

    # The ratio of examples to sample as centers (prototypes).
    num_centers_ratio = FLAGS.num_centers_ratio
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    validate_data_pattern = FLAGS.validate_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    # distance metric, cosine or euclidean.
    dist_metric = FLAGS.dist_metric
    init_with_linear_clf = FLAGS.init_with_linear_clf

    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate
    train_epochs = FLAGS.train_epochs
    l1_reg_rate = FLAGS.l1_reg_rate
    l2_reg_rate = FLAGS.l2_reg_rate

    # ....Start rbf network...
    logging.info('Entering rbf network...')
    # Validate set is not stored in graph or meta data. Re-create it any way.
    # Sample validate set for logistic regression early stopping.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=num_readers)

    _, validate_data, validate_labels, _ = random_sample(0.05, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline,
                                                         name_scope='sample_validate')

    # DataPipeline consists of reader, batch size, no. of readers and data pattern.
    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    # If start a new model or output dir does not exist, truly start a new model.
    start_new_model = start_new_model or (not tf.gfile.Exists(output_dir))

    if start_new_model:
        # PHASE ONE - selecting prototypes c, computing scaling factors sigma.
        # num_centers = FLAGS.num_centers
        # num_centers_ratio = float(num_centers) / NUM_TRAIN_EXAMPLES

        # metric is euclidean or cosine. If cosine, alpha=1.0, otherwise can be less than 1.0.
        if 'cosine' == dist_metric:
            # 200 will lead to decreasing drastically and increasing slowly.
            alpha = 1.0
        else:
            alpha = 1.0
        centers, sigmas = initialize(num_centers_ratio, data_pipeline=train_data_pipeline,
                                     method='kmeans', metric=dist_metric,
                                     scaling_method=4, alpha=alpha)

        # PHASE TWO - computing linear regression weights and biases.
        num_centers = centers.shape[0]
        # Compute mean and variance after data transform.
        tr_data_fn = rbf_transform
        tr_data_paras = {'centers': centers, 'sigmas': sigmas, 'metric': dist_metric,
                         'reshape': True, 'size': num_centers}
        # Not necessary to perform standard scale.
        if init_with_linear_clf:
            # Call linear classification to get a good initial values of weights and biases.
            linear_clf = LinearClassifier(logdir=path_join(output_dir, 'linear_classifier'))
            linear_clf.fit(data_pipeline=train_data_pipeline,
                           tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                           l2_regs=0.01, validate_set=(validate_data, validate_labels), line_search=False)
            linear_clf_weights, linear_clf_biases = linear_clf.weights, linear_clf.biases
        else:
            linear_clf_weights, linear_clf_biases = None, None

        # Set pos_weights for extremely imbalanced situation in one-vs-all classifiers.
        try:
            # Load sum_labels in training set, numpy float format to compute pos_weights.
            train_sum_labels = load_sum_labels()
            # num_neg / num_pos, assuming neg_weights === 1.0.
            pos_weights = np.sqrt(float(NUM_TRAIN_EXAMPLES) / train_sum_labels - 1.0)
            logging.info('Computing pos_weights based on sum_labels in train set successfully.')
        except IOError:
            logging.error('Cannot load train sum_labels. Use default value.')
            pos_weights = None
        finally:
            pos_weights = None

        # Include standard scale to rbf transform.
        tr_data_mean, tr_data_var = compute_data_mean_var(train_data_pipeline,
                                                          tr_data_fn=tr_data_fn,
                                                          tr_data_paras=tr_data_paras)
        logging.debug('tr_data_mean: {}\ntr_data_var: {}'.format(tr_data_mean, tr_data_var))
        tr_data_paras.update({'mean': tr_data_mean, 'variance': tr_data_var})

    else:
        linear_clf_weights, linear_clf_biases = None, None
        tr_data_fn, tr_data_paras = None, None
        pos_weights = None

    # PHASE THREE - fine tuning prototypes c, scaling factors sigma and weights and biases.
    log_reg_clf = LogisticRegression(logdir=path_join(output_dir, 'log_reg'))
    log_reg_clf.fit(train_data_pipeline=train_data_pipeline, start_new_model=start_new_model,
                    tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                    validate_set=(validate_data, validate_labels), validate_fn=gap_fn,
                    init_learning_rate=init_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
                    epochs=train_epochs, l1_reg_rate=l1_reg_rate, l2_reg_rate=l2_reg_rate,
                    pos_weights=pos_weights,
                    initial_weights=linear_clf_weights, initial_biases=linear_clf_biases)

    # ....Exit rbf network...
    logging.info('Exit rbf network.')


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    flags.DEFINE_string('yt8m_home', '/Users/Sophie/Documents/youtube-8m-data',
                        'YT8M dataset home.')
    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        path_join(FLAGS.yt8m_home, 'train_validate/traina*.tfrecord'),
                        'File glob for the training data set.')

    flags.DEFINE_string('validate_data_pattern',
                        path_join(FLAGS.yt8m_home, 'train_validate/validateq*.tfrecord'),
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_rgb,mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '1024,128', 'Dimensions of features to be used, separated by ,.')

    # Set by the memory limit.
    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 1, 'Number of readers to form a batch.')

    flags.DEFINE_bool('start_new_model', True, 'To start a new model or restore from output dir.')

    flags.DEFINE_float('num_centers_ratio', 0.001, 'The number of centers in RBF network.')

    flags.DEFINE_string('dist_metric', 'cosine', 'Distance metric, cosine or euclidean.')

    flags.DEFINE_boolean('init_with_linear_clf', True,
                         'Boolean variable indicating whether to init logistic regression with linear classifier.')

    flags.DEFINE_float('init_learning_rate', 0.01, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')
    # Regularization rates.
    flags.DEFINE_float('l1_reg_rate', 0.01, 'l1 regularization rate.')
    flags.DEFINE_float('l2_reg_rate', 0.01, 'l2 regularization rate.')

    flags.DEFINE_integer('train_epochs', 20, 'Training epochs, one epoch means passing all training data once.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/video_level/rbf_network',
                        'The directory where intermediate and model checkpoints should be written.')

    app.run()
