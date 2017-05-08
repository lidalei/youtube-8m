"""
The referenced paper is
Schwenker F, Kestler H A, Palm G. Three learning phases for radial-basis-function networks[J].
Neural networks, 2001, 14(4): 439-458.
and
Zhang M L. ML-RBF: RBF neural networks for multi-label learning[J]. Neural Processing Letters, 2009, 29(2): 61-74.

In this implementation, training process is split as three phases as described in Schwenker's. Moreover, different
 second-phases are compared and a third phased is added or not.

More specifically, three different frameworks are implemented. 1, for all L labels, finding a certain number of centers
 and train L binary logistic regression models on these centers. 2, for each label, finding a certain number of centers
 and train a logistic regression model on these centers. In total, there are L groups of centers and L logistic
 regression models. 3, for each label, finding a certain number of centers and train L logistic regression models on
 all these centers as a whole group. The first framework is described in Schwenker's as multi-class classification.
 The second one works as one-vs-all. And the third is described in Zhang's.
"""
import tensorflow as tf
import time

from readers import get_reader
from utils import partial_data_features_mean
from tensorflow import flags, gfile, logging, app

from os.path import join as path_join
import pickle
import numpy as np
import scipy.spatial.distance as sci_distance

from inference import format_lines

FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1, num_epochs=1, name_scope='input'):
    """Creates the section of the graph which reads the input data.

    Similar to the same-name function in train.py.
    Args:
        reader: A class which parses the input data.
        data_pattern: A 'glob' style path to the data files.
        batch_size: How many examples to process at a time.
        num_readers: How many I/O threads to use.
        num_epochs: How many passed to go through the data files.
        name_scope: An identifier of this code.

    Returns:
        A tuple containing the features tensor, labels tensor, and optionally a
        tensor containing the number of frames per video. The exact dimensions
        depend on the reader being used.

    Raises:
        IOError: If no files matching the given pattern were found.
    """
    # Adapted from namesake function in inference.py.
    with tf.name_scope(name_scope):
        # Glob() can be replace with tf.train.match_filenames_once(), which is an operation.
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find input files. data_pattern='{}'".format(data_pattern))
        logging.info("Number of input files: {}".format(len(files)))
        # Pass test data once. Thus, num_epochs is set as 1.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=False, capacity=128)
        examples_and_labels = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

        # In shuffle_batch_join,
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        capacity = num_readers * batch_size + 1024
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            tf.train.batch_join(examples_and_labels,
                                batch_size=batch_size,
                                capacity=capacity,
                                allow_smaller_final_batch=True,
                                enqueue_many=True))
        return video_id_batch, video_batch, video_labels_batch, num_frames_batch


def random_sample(sample_ratio, reader, data_pattern, batch_size, num_readers):
    """
    Randomly sample sample_ratio examples from data that specified reader by and data_pattern.

    :param sample_ratio: The ratio of examples to be sampled.
    :param reader: See readers.py.
    :param data_pattern: File Glob of data.
    :param batch_size: The size of a batch. The last a few batches might have less examples.
    :param num_readers: How many IO threads to enqueue example queue.
    :return: Might not the exact ratio of examples be returned.
    """
    # Create the graph to traverse all data once.
    with tf.Graph().as_default() as graph:
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(reader=reader, data_pattern=data_pattern, batch_size=batch_size,
                                   num_readers=num_readers, num_epochs=1, name_scope='rnd_sample'))

        num_batch_videos = tf.shape(video_batch)[0]
        rnd_nums = tf.random_uniform([num_batch_videos])
        sample_mask = tf.less_equal(rnd_nums, sample_ratio)
        partial_sample = tf.boolean_mask(video_batch, sample_mask)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    sess = tf.Session(graph=graph)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Find num_centers_ratio of the total examples.
    sample = []
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Remove num_batch_videos_val and num_batch_videos.
            # num_batch_videos_val, partial_sample_val = sess.run([num_batch_videos, partial_sample])
            # print('length of video_batch: {}'.format(num_batch_videos_val))
            partial_sample_val = sess.run(partial_sample)

            # print('partial_sample_val: {}'.format(partial_sample_val))
            # print('magnitude of examples: {}'.format(np.linalg.norm(partial_sample_val, ord=2, axis=1)))

            if partial_sample_val.size > 0:
                sample.append(partial_sample_val)

    except tf.errors.OutOfRangeError:
        logging.info('Done sampling -- one epoch finished.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    a_sample = np.concatenate(sample, axis=0)
    logging.info('The sample has shape: {}.'.format(a_sample.shape))

    return a_sample


def kmeans_iter(centers, reader, data_pattern, batch_size, num_readers, metric='cosine', return_mean_clu_dist=False):
    """
    k-means clustering one iteration.
    Args:
        centers: A list of centers (as a numpy array).
        reader: Video-level features reader or frame-level features reader.
        data_pattern: tf data Glob.
        batch_size: How many examples to read per batch.
        num_readers: How many IO threads to read examples.
        metric: Distance metric, support euclidean and cosine.
        return_mean_clu_dist: boolean. If True, compute mean distance per cluster. Else, return None.
    Returns:
        Optimized centers and corresponding average cluster-center distance and mean distance per cluster.
    Raises:
        NotImplementedError if distance is not euclidean or cosine.
    """
    num_centers = len(centers)

    if (metric == 'euclidean') or (metric == 'cosine'):
        logging.info('Perform k-means clustering using {} distance.'.format(metric))
    else:
        raise NotImplementedError('Only euclidean and cosine distance metrics are supported.')

    # The dir where intermediate results and model checkpoints should be written.
    # output_dir = FLAGS.output_dir

    # Create the graph to traverse all training data once.
    with tf.Graph().as_default() as graph:
        # Define current centers as a variable in graph and use placeholder to hold large number of centers.
        centers_initializer = tf.placeholder(tf.float32, shape=centers.shape, name='centers_initializer')
        # Setting collections=[] keeps the variable out of the GraphKeys.GLOBAL_VARIABLES collection
        # used for saving and restoring checkpoints.
        current_centers = tf.Variable(initial_value=centers_initializer, trainable=False, collections=[],
                                      name='current_centers')

        # Objective function. TODO, avoid overflow in initial iteration.
        total_dist = tf.Variable(initial_value=0, dtype=tf.float32, name='total_distance')
        # Define new centers as Variable and use placeholder to hold large number of centers.
        per_clu_sum_initializer = tf.placeholder(tf.float32, shape=centers.shape)
        per_clu_sum = tf.Variable(initial_value=per_clu_sum_initializer, trainable=False, collections=[],
                                  name='per_cluster_sum')
        per_clu_count = tf.Variable(initial_value=tf.zeros([num_centers]), dtype=tf.float32)
        if return_mean_clu_dist:
            per_clu_total_dist = tf.Variable(initial_value=tf.zeros([num_centers]))
        else:
            per_clu_total_dist = tf.Variable(initial_value=0.0, dtype=tf.float32)

        # Construct data read pipeline.
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(reader=reader, data_pattern=data_pattern, batch_size=batch_size,
                                   num_readers=num_readers, num_epochs=1, name_scope='k_means_reader'))

        # Assign video batch to current centers (clusters).
        if metric == 'euclidean':
            # Make use of broadcasting feature.
            expanded_current_centers = tf.expand_dims(current_centers, axis=0)
            expanded_video_batch = tf.expand_dims(video_batch, axis=1)

            sub = tf.subtract(expanded_video_batch, expanded_current_centers)
            # element-wise square.
            squared_sub = tf.square(sub)
            # Compute distances with centers video-wisely. Shape [batch_size, num_initial_centers]. negative === -.
            minus_dist = tf.negative(tf.sqrt(tf.reduce_sum(squared_sub, axis=-1)))
            # Compute assignments and the distance with nearest centers video-wisely.
            minus_topk_nearest_dist, topk_assignments = tf.nn.top_k(minus_dist, k=1)
            nearest_topk_dist = tf.negative(minus_topk_nearest_dist)
            # Remove the last dimension due to k.
            nearest_dist = tf.squeeze(nearest_topk_dist, axis=[-1])
            assignments = tf.squeeze(topk_assignments, axis=[-1])

            # Compute new centers sum and number of videos that belong to each center (cluster) with this video batch.
            batch_per_clu_sum = tf.unsorted_segment_sum(video_batch, assignments, num_centers)

        else:
            normalized_video_batch = tf.nn.l2_normalize(video_batch, -1)
            cosine_sim = tf.matmul(normalized_video_batch, current_centers, transpose_b=True)
            nearest_topk_cosine_sim, topk_assignments = tf.nn.top_k(cosine_sim, k=1)
            nearest_topk_dist = tf.subtract(1.0, nearest_topk_cosine_sim)
            # Remove the last dimension due to k.
            nearest_dist = tf.squeeze(nearest_topk_dist, axis=[-1])
            assignments = tf.squeeze(topk_assignments, axis=[-1])

            # Compute new centers sum and number of videos that belong to each center (cluster) with this video batch.
            batch_per_clu_sum = tf.unsorted_segment_sum(normalized_video_batch, assignments, num_centers)

        batch_per_clu_count = tf.unsorted_segment_sum(tf.ones_like(video_id_batch, dtype=tf.float32),
                                                      assignments, num_centers)
        # Update total distance, namely objective function.
        if return_mean_clu_dist:
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

    sess = tf.Session(graph=graph)
    sess.run(init_op)

    # initialize centers variable in tf graph.
    if metric == 'euclidean':
        sess.run(current_centers.initializer, feed_dict={centers_initializer: centers})
    else:
        normalized_centers = centers / np.clip(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-6, np.PINF)
        sess.run(current_centers.initializer, feed_dict={centers_initializer: normalized_centers})

    # initializer per_clu_sum.
    sess.run(per_clu_sum.initializer, feed_dict={per_clu_sum_initializer: np.zeros_like(centers)})

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # TODO, deal with empty cluster situation.
    try:
        while not coord.should_stop():
            _ = sess.run(update_non_op)

    except tf.errors.OutOfRangeError:
        logging.info('One k-means iteration done. One epoch limit reached.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    final_total_dist, final_per_clu_sum, final_per_clu_count, final_per_clu_total_dist = sess.run(
        [total_dist, per_clu_sum, per_clu_count, per_clu_total_dist])

    sess.close()

    # Expand to each feature.
    accum_per_clu_count_per_feat = np.expand_dims(final_per_clu_count, axis=1)
    if return_mean_clu_dist:
        # Numpy array divide element-wisely.
        return ((final_per_clu_sum / accum_per_clu_count_per_feat), final_total_dist,
                final_per_clu_total_dist / final_per_clu_count)
    else:
        # Numpy array divide element-wisely.
        return (final_per_clu_sum / accum_per_clu_count_per_feat), final_total_dist, None


def mini_batch_kmeans():
    pass


def initialize(num_centers_ratio, reader, data_pattern, batch_size=1024, num_readers=1,
               method=None, metric='cosine', max_iter=20, tol=1.0, scaling_method=1, alpha=0.1, p=3):
    """
    This functions implements the following two phases:
    1. To initialize representative prototypes (RBF centers) c and scaling factors sigma.
    2. And to fit output weights.

    This function will generate one group of centers for all labels as a whole. Be cautious with initialize_per_label.
    Args:
        num_centers_ratio: The number of centers to be decided / total number of examples that belong to label l,
         for l = 0, ..., num_classes - 1.
        reader: video-level features reader or frame-level features reader.
        data_pattern: File Glob of data set.
        batch_size: How many examples to handle per time.
        num_readers: How many IO threads to prefetch examples.
        method: The method to decide the centers. Possible choices are random selection, kmeans and online(kmeans).
         Default is None, which represents randomly selecting a certain number of examples as centers.
        metric: Distance metric, euclidean distance or cosine distance.
        max_iter: The maximal number of iterations clustering to be done.
        tol: The minimal reduction of objective function of clustering to be reached to stop iteration.
        scaling_method: There are four choices. 1, all of them use the same sigma, the p smallest pairs of distances.
         2, average of p nearest centers. 3, distance to the nearest center that has a different label (Not supported!).
         4, mean distance between this center and all of its points.
        alpha: The alpha parameter that should be set heuristically. It works like a learning rate. (mu in Zhang's)
        p: When scaling_method is 1 or 2, p is needed.
    Returns:
        centers (prototypes) and scaling factors (sigmas).
    Raises:
        ValueError if num_centers_ratio is not between 0.0 (open) and 1.0 (closed).
        NotImplementedError if metric is not euclidean or cosine.
    """
    logging.info('Generate a group of centers for all labels. See Schwenker.')
    if num_centers_ratio <= 0.0 or num_centers_ratio > 1.0:
        raise ValueError('num_centers_ratio must be larger than 0.0 and no greater than 1.0.')
    logging.info('num_centers_ratio is {}.'.format(num_centers_ratio))

    if ('euclidean' == metric) or ('cosine' == metric):
        logging.info('Using {} distance. The larger, the less similar.'.format(metric))
    else:
        raise NotImplementedError('Only euclidean and cosine distance are supported, {} passed.'.format(metric))

    centers = random_sample(num_centers_ratio, reader, data_pattern, batch_size, num_readers)
    logging.info('Sampled {} centers totally.'.format(len(centers)))
    logging.debug('Randomly selected centers: {}'.format(centers))

    # Used in scaling method 4.
    per_clu_mean_dist = None
    # Perform kmeans or online kmeans.
    if method is None:
        logging.info('Using randomly selected centers as model prototypes (centers).')
    elif 'online' == method:
        # TODO.
        raise NotImplementedError('Only None (randomly select examples), online, kmeans are supported.')
    elif 'kmeans' == method:
        logging.info('Using k-means clustering result as model prototypes (centers).')
        iter_count = 0
        # clustering objective function.
        obj = np.PINF
        return_mean_clu_dist = (scaling_method == 4)

        while iter_count < max_iter:
            start_time = time.time()
            new_centers, new_obj, per_clu_mean_dist = kmeans_iter(centers, reader, data_pattern,
                                                                  batch_size, num_readers, metric=metric,
                                                                  return_mean_clu_dist=return_mean_clu_dist)
            iter_count += 1
            print('The {}-th iteration took {} s.'.format(iter_count, time.time() - start_time))
            logging.debug('new_centers: {}'.format(new_centers))
            print('new_obj: {}'.format(new_obj))
            centers = new_centers

            if not np.isinf(obj) and (obj - new_obj) < tol:
                logging.info('Done k-means clustering.')
                break

            obj = new_obj

    else:
        raise NotImplementedError('Only None (randomly select examples), online, kmeans are supported.')

    # Compute scaling factors based on these centers.
    num_centers = len(centers)
    sigmas = []
    if scaling_method == 1:
        # Equation 27.
        pairwise_distances = sci_distance.pdist(centers, metric=metric)
        p = min(p, len(pairwise_distances))
        logging.info('Using {} minimal pairwise distances.'.format(p))
        # np.partition begins with 1 instead of 0.
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
            _, _, per_clu_mean_dist = kmeans_iter(centers, reader, data_pattern,
                                                  batch_size, num_readers, metric=metric, return_mean_clu_dist=True)
            logging.info('Compute mean distance per cluster using kmeans or online kmeans.')
        else:
            logging.info('Reuse results from kmeans or online kmeans.')

        sigmas = alpha * per_clu_mean_dist
    elif scaling_method == 5:
        # Equation 31.
        raise NotImplementedError('Only three methods are supported. Please read the documentation.')
    else:
        raise NotImplementedError('Only three methods are supported. Please read the documentation.')

    print('Scaling factor sigmas: {}'.format(sigmas))

    return centers, sigmas


def _compute_data_mean_std(reader, data_pattern, batch_size=1024, num_readers=1, tr_data_fn=None):
    """
    Compute mean and standard deviations per feature (column) and mean of each label.

    Note:
        From Spark StandardScaler documentation.
        * The "unit std" is computed using the corrected sample standard deviation
        * (https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation),
        * which is computed as the square root of the unbiased sample variance.
    Args:
        reader: video-level or frame-level data reader.
        data_pattern: train set Glob pattern.
        batch_size: How many examples to process per time.
        num_readers: How many threads to (pre-)fetch examples.
        tr_data_fn: a function that transforms input data.

    Returns:
        Mean values of each feature column as a numpy array of rank 1.
        Standard deviations of each feature column as a numpy array of rank 1.
        Mean values of each label as a numpy array of rank 1.
    """
    feature_names = reader.feature_names
    feature_sizes = reader.feature_sizes
    # Total number of features.
    features_size = sum(feature_sizes)
    num_classes = reader.num_classes

    logging.info('Computing mean and std of {} features with sizes {} and mean of #{} labels.'.format(
        feature_names, feature_sizes, num_classes))

    # features_mean on partial data (600 + train files).
    # Note, can only be used locally, not in google cloud.
    try:
        par_features_mean = partial_data_features_mean()
    except IOError:
        logging.error('Cannot locate partial_data_features_mean data file.')
        par_features_mean = None

    if par_features_mean is None:
        approx_features_mean = np.zeros([features_size], dtype=np.float32)
    else:
        approx_features_mean = np.concatenate([par_features_mean[e] for e in feature_names])

    # numerical stability with
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data.
    # Create the graph to traverse all data once.
    with tf.Graph().as_default() as graph:
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(reader=reader, data_pattern=data_pattern, batch_size=batch_size,
                                   num_readers=num_readers, num_epochs=1, name_scope='features_mean_std'))

        video_count = tf.Variable(initial_value=0.0, name='video_count')
        features_sum = tf.Variable(initial_value=tf.zeros([features_size]), name='features_sum')
        features_squared_sum = tf.Variable(initial_value=tf.zeros([features_size]), name='features_squared_sum')
        labels_sum = tf.Variable(initial_value=tf.zeros([num_classes]), name='labels_sum')

        batch_video_count = tf.cast(tf.shape(video_batch)[0], tf.float32)
        # Compute shift features sum and squared sum.
        shift = tf.constant(approx_features_mean, dtype=tf.float32, name='shift')
        if par_features_mean is None:
            # Don't shift, though not good.
            shifted_video_batch = tf.identity(video_batch)
        else:
            shifted_video_batch = tf.subtract(video_batch, shift)

        batch_features_sum = tf.reduce_sum(shifted_video_batch, axis=0, name='batch_features_sum')
        batch_features_squared_sum = tf.reduce_sum(tf.square(shifted_video_batch), axis=0,
                                                   name='batch_features_squared_sum')
        batch_labels_sum = tf.reduce_sum(tf.cast(video_labels_batch, tf.float32), axis=0, name='batch_labels_sum')

        update_video_count = tf.assign_add(video_count, batch_video_count)
        update_features_sum = tf.assign_add(features_sum, batch_features_sum)
        update_features_squared_sum = tf.assign_add(features_squared_sum, batch_features_squared_sum)
        update_labels_sum = tf.assign_add(labels_sum, batch_labels_sum)

        with tf.control_dependencies(
                [update_video_count, update_features_sum, update_features_squared_sum, update_labels_sum]):
            update_accum_non_op = tf.no_op()

        # Define final results. To be run after all data have been handled.
        features_mean = tf.add(tf.divide(features_sum, video_count), shift, name='features_mean')
        # Corrected sample standard deviation.
        features_variance = tf.divide(
            tf.subtract(features_squared_sum, tf.scalar_mul(video_count, tf.square(features_mean))),
            tf.subtract(video_count, 1.0), name='features_var')
        features_std = tf.sqrt(features_variance, name='features_std')
        labels_mean = tf.divide(labels_sum, video_count)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    sess = tf.Session(graph=graph)

    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            _ = sess.run(update_accum_non_op)

    except tf.errors.OutOfRangeError:
        logging.info('Done features sum and squared sum and count computation -- one epoch finished.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    # After all data have been handled, fetch the statistics.
    features_mean_val, features_std_val, labels_mean_val = sess.run([features_mean, features_std, labels_mean])

    sess.close()

    return features_mean_val, features_std_val, labels_mean_val


def linear_classifier(reader, train_data_pattern, batch_size=1024, num_readers=1, tr_data_fn=None, l2_reg=0.001):
    """
    Compute weights and biases of linear classifier using normal equation.
    Args:
        reader: video-level or frame-level data reader.
        train_data_pattern: train set Glob pattern.
        batch_size: How many examples to process per time.
        num_readers: How many threads to (pre-)fetch examples.
        tr_data_fn: a function that transforms input data.
        l2_reg: How much the linear classifier weights should be penalized.
    Returns: Weights and biases fit on the given data set, where biases are appended as the last row.

    """
    num_classes = reader.num_classes
    feature_names = reader.feature_names
    logging.info('Linear regression using {} features.'.format(feature_names))

    output_dir = FLAGS.output_dir

    # Method two - append an all-one column to X.
    feature_sizes = reader.feature_sizes
    feature_size = sum(feature_sizes)

    # Create the graph to traverse all data once.
    with tf.Graph().as_default() as graph:
        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
        global_step_inc_op = tf.assign_add(global_step, 1)

        # X.transpose * X
        norm_equ_1_initializer = tf.placeholder(tf.float32, shape=[feature_size, feature_size])
        norm_equ_1 = tf.Variable(initial_value=norm_equ_1_initializer, collections=[], name='X_Tr_X')

        # X.transpose * Y
        norm_equ_2_initializer = tf.placeholder(tf.float32, shape=[feature_size, num_classes])
        norm_equ_2 = tf.Variable(initial_value=norm_equ_2_initializer, collections=[], name='X_Tr_Y')

        video_count = tf.Variable(initial_value=0.0, name='video_count')
        features_sum = tf.Variable(initial_value=tf.zeros([feature_size]), name='features_sum')
        labels_sum = tf.Variable(initial_value=tf.zeros([num_classes]), name='labels_sum')

        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(reader=reader, data_pattern=train_data_pattern, batch_size=batch_size,
                                   num_readers=num_readers, num_epochs=1, name_scope='input'))
        if tr_data_fn is None:
            video_batch_transformed = tf.identity(video_batch)
        else:
            video_batch_transformed = tr_data_fn(video_batch)

        with tf.name_scope('batch_increment'):
            video_batch_transformed_tr = tf.matrix_transpose(video_batch_transformed, name='X_Tr')
            batch_norm_equ_1 = tf.matmul(video_batch_transformed_tr, video_batch_transformed)
            # batch_norm_equ_1 = tf.add_n(tf.map_fn(lambda x: tf.einsum('i,j->ij', x, x),
            #                                       video_batch_transformed), name='X_Tr_X')
            batch_norm_equ_2 = tf.matmul(video_batch_transformed_tr, tf.cast(video_labels_batch, tf.float32))
            batch_video_count = tf.cast(tf.shape(video_batch)[0], tf.float32)
            batch_features_sum = tf.reduce_sum(video_batch, axis=0, name='batch_features_sum')
            batch_labels_sum = tf.reduce_sum(tf.cast(video_labels_batch, tf.float32), axis=0, name='batch_labels_sum')

        with tf.name_scope('update_ops'):
            update_norm_equ_1_op = tf.assign_add(norm_equ_1, batch_norm_equ_1)
            update_norm_equ_2_op = tf.assign_add(norm_equ_2, batch_norm_equ_2)
            update_video_count = tf.assign_add(video_count, batch_video_count)
            update_features_sum = tf.assign_add(features_sum, batch_features_sum)
            update_labels_sum = tf.assign_add(labels_sum, batch_labels_sum)

        with tf.control_dependencies([update_norm_equ_1_op, update_norm_equ_2_op, update_video_count,
                                      update_features_sum, update_labels_sum, global_step_inc_op]):
            update_equ_non_op = tf.no_op(name='unified_update_op')

        with tf.name_scope('solution'):
            # After all data being handled, compute weights.
            l2_reg_term = tf.diag(tf.fill([feature_size], l2_reg), name='l2_reg')
            # X.transpose * X + lambda * Id, where d is the feature dimension.
            norm_equ_1_with_reg = tf.add(norm_equ_1, l2_reg_term)

            # Concat other blocks to form the final norm equation terms.
            final_norm_equ_1_top = tf.concat([norm_equ_1_with_reg, tf.expand_dims(features_sum, 1)], 1)
            final_norm_equ_1_bot = tf.concat([features_sum, tf.expand_dims(video_count, 0)], 0)
            final_norm_equ_1 = tf.concat([final_norm_equ_1_top, tf.expand_dims(final_norm_equ_1_bot, 0)], 0,
                                         name='norm_equ_1')
            final_norm_equ_2 = tf.concat([norm_equ_2, tf.expand_dims(labels_sum, 0)], 0,
                                         name='norm_equ_2')

            weights_biases = tf.matrix_solve(final_norm_equ_1, final_norm_equ_2, name='weights_biases')

        summary_op = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.Session(graph=graph)
    # Initialize variables.
    sess.run(init_op)
    sess.run([norm_equ_1.initializer, norm_equ_2.initializer], feed_dict={
        norm_equ_1_initializer: np.zeros([feature_size, feature_size], dtype=np.float32),
        norm_equ_2_initializer: np.zeros([feature_size, num_classes], dtype=np.float32)
    })

    summary_writer = tf.summary.FileWriter(path_join(output_dir, 'linear_classifier'), graph=sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            _, summary, global_step_val = sess.run([update_equ_non_op, summary_op, global_step])
            summary_writer.add_summary(summary, global_step=global_step_val)
    except tf.errors.OutOfRangeError:
        logging.info('Finished normal equation terms computation -- one epoch done.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    # Extract weights of num_classes linear classifiers. Each column corresponds to a classifier.
    weights_biases_val = sess.run(weights_biases)

    sess.close()

    return weights_biases_val


def initialize_per_label():
    """
    This functions implements the following two phases:
    1. To initialize representative prototypes (RBF centers) c and scaling factors sigma.
    2. And to fit output weights.

    It is different from the function initialize because it will generate L groups of centers (one per each label)
    instead of one group of centers for all labels as a whole.

    :return:
    """
    # Must consider the labels are super imbalanced! The counts are stored in 'sum_labels.pickle' with Python3 protocol.
    # logging.error
    raise NotImplementedError('It is a little troubling, will be implemented later! Be patient.')


def build_graph():
    """
    Build training and test graph.

    :return:
    """
    pass


def train(init_learning_rate, decay_steps, decay_rate=0.95, epochs=None, debug=False):
    """
    Training.

    Args:
        init_learning_rate: Initial learning rate.
        decay_steps: How many training steps to decay learning rate once.
        decay_rate: How much to decay learning rate.
        epochs: The maximal epochs to pass all training data.
        debug: boolean, True to print detailed debug information, False, silent.

    Returns:

    """
    num_centers_ratio = FLAGS.num_centers_ratio
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    # The dir where intermediate results and model checkpoints should be written.
    output_dir = FLAGS.output_dir

    # distance metric, cosine or euclidean.
    dist_metric = FLAGS.dist_metric

    # Compute weights and biases of linear classifier using normal equation.
    weights_biases = linear_classifier(reader, train_data_pattern, batch_size=batch_size, num_readers=2)
    logging.info('linear classifier weights_biases with shape {}'.format(weights_biases.shape))
    logging.info('linear classifier weights_biases: {}.'.format(weights_biases))
    """
    # num_centers = FLAGS.num_centers
    # num_centers_ratio = float(num_centers) / NUM_TRAIN_EXAMPLES
    # metric is euclidean or cosine.
    centers, sigmas = initialize(num_centers_ratio, reader, train_data_pattern, batch_size, num_readers,
                                 method='kmeans', metric=dist_metric, scaling_method=4)
    """
    """
    num_centers = centers.shape[0]

    num_classes = reader.num_classes

    # Build logistic regression graph and optimize it.
    with tf.Graph().as_default() as graph:
        if dist_metric == 'cosine':
            normalized_centers = centers / np.clip(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-6, np.PINF)
            prototypes = tf.Variable(initial_value=normalized_centers, dtype=tf.float32)
        else:
            prototypes = tf.Variable(initial_value=centers, dtype=tf.float32)

        neg_two_times_sq_sigmas = np.multiply(-2.0, np.square(sigmas))
        expanded_neg_two_times_sq_sigmas = np.expand_dims(neg_two_times_sq_sigmas, axis=0)
        # [-2.0 * sigmas ** 2], basis function denominators.
        neg_basis_f_deno = tf.Variable(initial_value=expanded_neg_two_times_sq_sigmas, dtype=tf.float32)

        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(reader=reader, data_pattern=train_data_pattern, batch_size=batch_size,
                                   num_readers=num_readers, num_epochs=epochs, name_scope='lr_weights'))

        if dist_metric == 'cosine':
            normalized_video_batch = tf.nn.l2_normalize(video_batch, -1)
            cosine_sim = tf.matmul(normalized_video_batch, prototypes, transpose_b=True)
            squared_dist = tf.square(tf.subtract(1.0, cosine_sim), name='cosine_square_dist')
        else:
            # Make use of broadcasting feature.
            expanded_centers = tf.expand_dims(prototypes, axis=0)
            expanded_video_batch = tf.expand_dims(video_batch, axis=1)

            sub = tf.subtract(expanded_video_batch, expanded_centers)
            # element-wise square.
            squared_sub = tf.square(sub)
            # Compute distances with centers video-wisely. Shape [batch_size, num_initial_centers]. negative === -.
            squared_dist = tf.reduce_sum(squared_sub, axis=-1, name='euclidean_square_dist')

        rbf_fs = tf.exp(tf.divide(squared_dist, neg_basis_f_deno), name='basis_function')

        # Define num_classes logistic regression models parameters. num_centers is new feature dimension.
        weights = tf.Variable(initial_value=tf.truncated_normal([num_centers, num_classes]),
                              dtype=tf.float32, name='weights')
        basis = tf.Variable(initial_value=tf.zeros([num_classes]))

        lr_output = tf.matmul(rbf_fs, weights) + basis
        lr_pred_prob = tf.nn.sigmoid(lr_output, name='lr_pred_probability')
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(video_labels_batch, tf.float32),
                                                       logits=lr_output, name='loss')

        # TODO, Add regularization.

        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
        rough_num_examples_processed = tf.multiply(global_step, batch_size)
        adap_learning_rate = tf.train.exponential_decay(init_learning_rate, rough_num_examples_processed,
                                                        decay_steps, decay_rate)
        optimizer = tf.train.GradientDescentOptimizer(adap_learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # TODO, save checkpoints.

    sess = tf.Session(graph=graph)
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            sess.run(train_op)

    except tf.errors.OutOfRangeError:
        logging.info('Done training -- {} epochs finished.'.format(epochs))
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    """


def inference(out_file_location, top_k, debug=False):
    pass


def main(unused_argv):
    is_train = FLAGS.is_train
    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate

    train_epochs = FLAGS.train_epochs
    is_tuning_hyper_para = FLAGS.is_tuning_hyper_para

    is_debug = FLAGS.is_debug

    output_file = FLAGS.output_file
    top_k = FLAGS.top_k

    logging.set_verbosity(logging.INFO)

    if is_train:
        if is_tuning_hyper_para:
            raise NotImplementedError('Implementation is under progress.')
        else:
            train(init_learning_rate, decay_steps, decay_rate, epochs=train_epochs, debug=is_debug)
    else:
        inference(output_file, top_k)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train/traina*.tfrecord',
                        'File glob for the training dataset.')

    flags.DEFINE_string('validate_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/validate/validateo*.tfrecord',
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    flags.DEFINE_string('test_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/test/test4*.tfrecord',
                        'Test data pattern, to be specified when making predictions.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '128', 'Dimensions of features to be used, separated by ,.')

    # Set by the memory limit (52GB).
    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_float('num_centers_ratio', 0.001, 'The number of centers in RBF network.')

    flags.DEFINE_string('dist_metric', 'cosine', 'Distance metric, cosine or euclidean.')

    flags.DEFINE_boolean('is_train', True, 'Boolean variable to indicate training or test.')

    flags.DEFINE_float('init_learning_rate', 0.01, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_integer('train_epochs', 20, 'Training epochs, one epoch means passing all training data once.')

    flags.DEFINE_boolean('is_tuning_hyper_para', False,
                         'Boolean variable indicating whether to perform hyper-parameter tuning.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/ml-knn-{}'.format(int(time.time())),
                        'The directory where intermediate and model checkpoints should be written.')

    # TODO, change it.
    flags.DEFINE_boolean('is_debug', True, 'Boolean variable to indicate debug or not.')

    flags.DEFINE_string('output_file', '/tmp/ml-knn/predictions.csv', 'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
