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
import time

from linear_model import LinearClassifier, LogisticRegression
from readers import get_reader
from utils import get_input_data_tensors, DataPipeline, random_sample, gap_fn
from tensorflow import flags, logging, app

from os.path import join as path_join
import numpy as np
import scipy.spatial.distance as sci_distance


FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640

MAX_TRAIN_STEPS = 1000000


class KMeans(object):
    def __init__(self, initial_centers, data_pipeline=None, metric='cosine', return_mean_clu_dist=False):
        """
        Args:
            initial_centers: A list of centers (as a numpy array).
            data_pipeline: A namedtuple consisting the following elements.
                reader, Video-level features reader or frame-level features reader.
                data_pattern, tf data Glob. Supports *, ? and [] wildcards.
                batch_size, How many examples to read per batch.
                num_readers, How many IO threads to read examples.
            metric: Distance metric, support euclidean and cosine.
            return_mean_clu_dist: boolean. If True, compute mean distance per cluster. Else, return None.
        Returns:
            Optimized centers and corresponding average cluster-center distance and mean distance per cluster.
        Raises:
            ValueError if distance is not euclidean or cosine.
        """
        self.current_centers = np.copy(initial_centers)
        self.data_pipeline = data_pipeline
        self.metric = metric
        self.return_mean_clu_dist = return_mean_clu_dist

        if (self.metric == 'euclidean') or (self.metric == 'cosine'):
            logging.info('Perform k-means clustering using {} distance.'.format(metric))
        else:
            raise ValueError('Only euclidean and cosine distance metrics are supported.')

        # Normalize current centers if distance metric is cosine.
        if self.metric == 'cosine':
            normalized_centers = self.current_centers / np.clip(
                np.linalg.norm(self.current_centers, axis=-1, keepdims=True), 1e-6, np.PINF)
            self.current_centers = normalized_centers

        self.graph = None
        # Attributes of the graph, tensor type.
        self.current_centers_initializer = None
        self.current_centers_init_op = None
        self.per_clu_sum_initializer = None
        self.per_clu_sum_init_op = None
        self.total_dist = None
        self.per_clu_sum = None
        self.per_clu_count = None
        self.per_clu_total_dist = None
        self.update_non_op = None
        self.init_op = None

        # Build iteration graph - initialize all attributes related to the graph.
        self.build_iter_graph()

        initialize_success = self.check_graph_initialized()
        if initialize_success:
            logging.info('Succeeded initializing a Tensorflow graph to perform k-means.')
        else:
            raise ValueError('Failed to initialize a Tensorflow Graph to perform k-means.')

        # clustering objective function.
        self.mean_dist = np.PINF
        self.per_clu_mean_dist = None

    def build_iter_graph(self):
        num_centers = self.current_centers.shape[0]

        # Create the graph to traverse all training data once.
        graph = tf.Graph()
        with graph.as_default():
            # Define current centers as a variable in graph and use placeholder to hold large number of centers.
            current_centers_initializer = tf.placeholder(tf.float32, shape=self.current_centers.shape,
                                                         name='centers_initializer')
            # Setting collections=[] keeps the variable out of the GraphKeys.GLOBAL_VARIABLES collection
            # used for saving and restoring checkpoints.
            current_centers = tf.Variable(initial_value=current_centers_initializer,
                                          trainable=False, collections=[], name='current_centers')

            # Objective function. POSSIBLE ISSUE, overflow in initial iteration.
            total_dist = tf.Variable(initial_value=0.0, dtype=tf.float32, name='total_distance')
            # Define sum per clu as Variable and use placeholder to hold large number of centers.
            per_clu_sum_initializer = tf.placeholder(tf.float32, shape=self.current_centers.shape)
            per_clu_sum = tf.Variable(initial_value=per_clu_sum_initializer,
                                      trainable=False, collections=[], name='per_cluster_sum')
            per_clu_count = tf.Variable(initial_value=tf.zeros([num_centers]), dtype=tf.float32)
            if self.return_mean_clu_dist:
                per_clu_total_dist = tf.Variable(initial_value=tf.zeros([num_centers]), name='per_clu_total_dist')
            else:
                per_clu_total_dist = tf.Variable(initial_value=0.0, dtype=tf.float32, name='per_clu_total_dist')

            # Construct data read pipeline.
            video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
                get_input_data_tensors(self.data_pipeline, num_epochs=1, name_scope='k_means_reader'))

            # Assign video batch to current centers (clusters).
            if self.metric == 'euclidean':
                # Make use of broadcasting feature.
                expanded_current_centers = tf.expand_dims(current_centers, axis=0)
                expanded_video_batch = tf.expand_dims(video_batch, axis=1)

                sub = tf.subtract(expanded_video_batch, expanded_current_centers)
                # element-wise square.
                squared_sub = tf.square(sub)
                # Compute distances with centers video-wisely. Shape [batch_size, num_initial_centers]. negative === -.
                neg_dist = tf.negative(tf.sqrt(tf.reduce_sum(squared_sub, axis=-1)))
                # Compute assignments and the distance with nearest centers video-wisely.
                neg_topk_nearest_dist, topk_assignments = tf.nn.top_k(neg_dist, k=1)
                nearest_topk_dist = tf.negative(neg_topk_nearest_dist)
                # Remove the last dimension due to k.
                nearest_dist = tf.squeeze(nearest_topk_dist, axis=[-1])
                assignments = tf.squeeze(topk_assignments, axis=[-1])

                # Compute new centers sum and number of videos that belong to each cluster within this video batch.
                batch_per_clu_sum = tf.unsorted_segment_sum(video_batch, assignments, num_centers)

            else:
                normalized_video_batch = tf.nn.l2_normalize(video_batch, -1)
                cosine_sim = tf.matmul(normalized_video_batch, current_centers, transpose_b=True)
                nearest_topk_cosine_sim, topk_assignments = tf.nn.top_k(cosine_sim, k=1)
                nearest_topk_dist = tf.subtract(1.0, nearest_topk_cosine_sim)
                # Remove the last dimension due to k.
                nearest_dist = tf.squeeze(nearest_topk_dist, axis=[-1])
                assignments = tf.squeeze(topk_assignments, axis=[-1])

                # Compute new centers sum and number of videos that belong to each cluster with this video batch.
                batch_per_clu_sum = tf.unsorted_segment_sum(normalized_video_batch, assignments, num_centers)

            batch_per_clu_count = tf.unsorted_segment_sum(tf.ones_like(video_id_batch, dtype=tf.float32),
                                                          assignments, num_centers)
            # Update total distance, namely objective function.
            if self.return_mean_clu_dist:
                batch_per_clu_total_dist = tf.unsorted_segment_sum(nearest_dist, assignments, num_centers)
                update_per_clu_total_dist = tf.assign_add(per_clu_total_dist, batch_per_clu_total_dist)

                total_batch_dist = tf.reduce_sum(batch_per_clu_total_dist)
            else:
                update_per_clu_total_dist = tf.no_op()
                total_batch_dist = tf.reduce_sum(nearest_dist)

            update_total_dist = tf.assign_add(total_dist, total_batch_dist)
            update_per_clu_sum = tf.assign_add(per_clu_sum, batch_per_clu_sum)
            update_per_clu_count = tf.assign_add(per_clu_count, batch_per_clu_count)

            # Avoid unnecessary fetches.
            with tf.control_dependencies(
                    [update_total_dist, update_per_clu_sum, update_per_clu_count, update_per_clu_total_dist]):
                update_non_op = tf.no_op()

            # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        graph.finalize()

        # Update the corresponding attributes of the class.
        self.graph = graph
        self.current_centers_initializer = current_centers_initializer
        self.current_centers_init_op = current_centers.initializer
        self.per_clu_sum_initializer = per_clu_sum_initializer
        self.per_clu_sum_init_op = per_clu_sum.initializer
        self.total_dist = total_dist
        self.per_clu_sum = per_clu_sum
        self.per_clu_count = per_clu_count
        self.per_clu_total_dist = per_clu_total_dist
        self.update_non_op = update_non_op
        self.init_op = init_op

    def check_graph_initialized(self):
        """
        To check if all graph operations and the graph itself are initialized successfully.

        Return:
            True if graph and all graph ops are not None, otherwise False.
        """
        graph_ops = [self.current_centers_initializer, self.current_centers_init_op, self.per_clu_sum_initializer,
                     self.per_clu_sum_init_op, self.total_dist, self.per_clu_sum, self.per_clu_count,
                     self.per_clu_total_dist, self.update_non_op, self.init_op]

        return (self.graph is not None) and (graph_ops.count(None) == 0)

    def kmeans_iter(self):
        logging.info('Entering k-means iter ...')
        # Create a new session due to closed queue cannot be reopened.
        sess = tf.Session(graph=self.graph)
        sess.run(self.init_op)

        # initialize current centers variable in tf graph.
        sess.run(self.current_centers_init_op,
                 feed_dict={self.current_centers_initializer: self.current_centers})

        # initializer per_clu_sum in tf graph.
        sess.run(self.per_clu_sum_init_op,
                 feed_dict={self.per_clu_sum_initializer: np.zeros_like(self.current_centers)})

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                _ = sess.run(self.update_non_op)

        except tf.errors.OutOfRangeError:
            logging.info('One k-means iteration done. One epoch limit reached.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        # Get final values.
        final_total_dist, final_per_clu_sum, final_per_clu_count, final_per_clu_total_dist = sess.run(
            [self.total_dist, self.per_clu_sum, self.per_clu_count, self.per_clu_total_dist])
        logging.info('Exiting k-means iter ...')
        sess.close()

        # Deal with empty cluster situation.
        nonzero_indices = np.nonzero(final_per_clu_count)
        per_nonempty_clu_count = final_per_clu_count[nonzero_indices]
        # Expand to each feature to make use of broadcasting.
        per_nonempty_clu_feat_count = np.expand_dims(per_nonempty_clu_count, axis=1)
        total_num_points = np.sum(per_nonempty_clu_count)

        per_nonempty_clu_sum = final_per_clu_sum[nonzero_indices]
        per_nonempty_clu_total_dist = final_per_clu_total_dist[nonzero_indices]

        if self.return_mean_clu_dist:
            # Numpy array divide element-wisely.
            per_nonempty_clu_mean_dist = per_nonempty_clu_total_dist / per_nonempty_clu_count
        else:
            per_nonempty_clu_mean_dist = None

        # Numpy array divide element-wisely.
        return ((per_nonempty_clu_sum / per_nonempty_clu_feat_count), final_total_dist / total_num_points,
                per_nonempty_clu_mean_dist)

    def fit(self, max_iter=100, tol=0.01):
        """
        This function works as sk-learn estimator fit.
        :param max_iter: 
        :param tol: Percentage not improved one iteration, stop iteration.
        :return: Update current centers and current objective function value (member variables).
        """
        for iter_count in xrange(max_iter):
            start_time = time.time()
            new_centers, new_mean_dist, new_per_clu_mean_dist = self.kmeans_iter()
            print('The {}-th iteration took {} s.'.format(iter_count + 1, time.time() - start_time))

            # There are empty centers (clusters) being removed.
            need_rebuild_graph = new_centers.shape[0] != self.current_centers.shape[0]

            # Update current centers and mean distance per cluster.
            # Normalize current centers if distance metric is cosine.
            if self.metric == 'cosine':
                self.current_centers = new_centers / np.clip(
                    np.linalg.norm(new_centers, axis=-1, keepdims=True), 1e-6, np.PINF)
            else:
                self.current_centers = new_centers

            self.per_clu_mean_dist = new_per_clu_mean_dist

            # Converged, break!
            if not np.isinf(self.mean_dist) and (self.mean_dist - new_mean_dist) / self.mean_dist < tol:
                # Update current objective function value.
                self.mean_dist = new_mean_dist
                logging.info('Done k-means clustering. Final centers have shape {}. Final mean dist is {}.'.format(
                    self.current_centers.shape, self.mean_dist))
                break
            else:
                # Update current objective function value.
                self.mean_dist = new_mean_dist

            if need_rebuild_graph:
                # Re-build graph using updated current centers.
                self.build_iter_graph()
                initialize_success = self.check_graph_initialized()
                if initialize_success:
                    logging.info('Succeeded re-initializing a Tensorflow graph to perform k-means.')
                else:
                    raise ValueError('Failed to re-initialize a Tensorflow Graph to perform k-means.')

            logging.debug('new_centers: {}'.format(self.current_centers))
            logging.info('new_centers shape: {}'.format(self.current_centers.shape))
            logging.info('New mean point-center distance: {}'.format(self.mean_dist))


def mini_batch_kmeans():
    raise NotImplementedError('Not implemented. Batch kmeans works fast enough now.')


def initialize(num_centers_ratio, data_pipeline, method=None, metric='cosine',
               max_iter=100, tol=0.01, scaling_method=1, alpha=1.0, p=3):
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
                                     data_pipeline=data_pipeline)
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


def rbf_transform(data, centers, sigmas, metric='cosine', **kwargs):
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

    with tf.name_scope('rbf_transform_{}'.format(metric)):
        if ('reuse' in kwargs) and (kwargs['reuse'] is True):
            # Get from collection.
            prototypes = tf.get_collection('prototypes')[0]
            basis_f_mul = tf.get_collection('basis_f_mul')[0]
        else:
            if metric == 'cosine':
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
            tf.summary.histogram('prototypes', prototypes)
            tf.summary.histogram('basis_f_mul', basis_f_mul)
        # Do transform.
        if metric == 'cosine':
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

    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate
    train_epochs = FLAGS.train_epochs
    l2_reg_rate = FLAGS.l2_reg_rate

    # ....Start rbf network...
    logging.info('Entering rbf network...')

    # If start a new model or output dir does not exist, truly start a new model.
    start_new_model = start_new_model or (not tf.gfile.Exists(output_dir))

    if start_new_model:
        # PHASE ONE - selecting prototypes c, computing scaling factors sigma.
        # num_centers = FLAGS.num_centers
        # num_centers_ratio = float(num_centers) / NUM_TRAIN_EXAMPLES
        # DataPipeline consists of reader, batch size, no. of readers and data pattern.
        _train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                            batch_size=8192, num_readers=4)

        # metric is euclidean or cosine. If cosine, alpha=1.0, otherwise can be less than 1.0.
        if 'cosine' == dist_metric:
            alpha = 1.0
        else:
            alpha = 1.0
        centers, sigmas = initialize(num_centers_ratio, data_pipeline=_train_data_pipeline,
                                     method='kmeans', metric=dist_metric, scaling_method=4, alpha=alpha)

        # PHASE TWO - computing linear regression weights and biases.
        num_centers = centers.shape[0]
        tr_data_fn = rbf_transform
        tr_data_paras = {'centers': centers, 'sigmas': sigmas, 'metric': dist_metric,
                         'reshape': True, 'size': num_centers}

        # Call linear classification to get a good initial values of weights and biases.
        linear_clf = LinearClassifier(logdir=path_join(output_dir, 'linear_classifier'))
        linear_clf.fit(data_pipeline=_train_data_pipeline,
                       tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                       l2_regs=0.01, line_search=False)

        linear_clf_weights, linear_clf_biases = linear_clf.weights, linear_clf.biases
    else:
        linear_clf_weights, linear_clf_biases = None, None
        tr_data_fn, tr_data_paras = None, None

    # PHASE THREE - fine tuning prototypes c, scaling factors sigma and weights and biases.
    # DataPipeline consists of reader, batch size, no. of readers and data pattern.
    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    # Validate set is not stored in graph or meta data. Re-create it any way.
    # Sample validate set for logistic regression early stopping.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=8192, num_readers=4)

    _, validate_data, validate_labels, _ = random_sample(0.05, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline)

    log_reg_clf = LogisticRegression(logdir=path_join(output_dir, 'log_reg'))
    log_reg_clf.fit(train_data_pipeline=train_data_pipeline, start_new_model=start_new_model,
                    tr_data_fn=tr_data_fn, tr_data_paras=tr_data_paras,
                    validate_set=(validate_data, validate_labels), validate_fn=gap_fn,
                    init_learning_rate=init_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
                    epochs=train_epochs, l2_reg_rate=l2_reg_rate,
                    initial_weights=linear_clf_weights, initial_biases=linear_clf_biases)

    # ....Exit rbf network...
    logging.info('Exit rbf network.')


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train/train*.tfrecord',
                        'File glob for the training dataset.')

    flags.DEFINE_string('validate_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/validate/validate*.tfrecord',
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '128', 'Dimensions of features to be used, separated by ,.')

    # Set by the memory limit (52GB).
    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_bool('start_new_model', True, 'To start a new model or restore from output dir.')

    flags.DEFINE_float('num_centers_ratio', 0.001, 'The number of centers in RBF network.')

    flags.DEFINE_string('dist_metric', 'euclidean', 'Distance metric, cosine or euclidean.')

    flags.DEFINE_float('init_learning_rate', 0.01, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_float('l2_reg_rate', 0.01, 'l2 regularization rate.')

    flags.DEFINE_integer('train_epochs', 200, 'Training epochs, one epoch means passing all training data once.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/video_level/rbf_network',
                        'The directory where intermediate and model checkpoints should be written.')

    app.run()
