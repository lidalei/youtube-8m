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
import utils
from tensorflow import flags, gfile, logging, app
from inference import format_lines

import pickle
import numpy as np
import scipy.spatial.distance as sci_distance

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
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=False)
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


def kmeans_iter(centers, reader, data_pattern, batch_size, num_readers, metric='euclidean'):
    """
    k-means clustering one iteration.

    :param centers: A list of centers (as a numpy array).
    :param reader: Video-level features reader or frame-level features reader.
    :param data_pattern: tf data Glob.
    :param batch_size: How many examples to read per batch.
    :param num_readers: How many IO threads to read examples.
    :param metric: Distance metric, support euclidean and cosine.
    :return: Optimized centers and corresponding average cluster-center distance.
    """
    num_centers = len(centers)
    if num_centers >= 40000:
        logging.warn('Too many ({}) initial centers which can not be held in a tf graph.'.format(num_centers))

    if (metric == 'euclidean') or (metric == 'cosine'):
        logging.info('Using {} distance.'.format(metric))
    else:
        raise NotImplementedError('Only euclidean and cosine distance metrics are supported.')

    graph = tf.Graph()
    # Create the graph to traverse all training data once.
    with graph.as_default():
        # Keep current centers in graph.
        if metric == 'euclidean':
            current_centers = tf.constant(centers, dtype=tf.float32, name='current_centers')
        else:
            normalized_centers = centers / np.clip(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-6, np.PINF)
            current_centers = tf.constant(normalized_centers, dtype=tf.float32, name='current_centers')

        # Objective function. TODO, avoid overflow in initial iteration.
        total_distance = tf.Variable(initial_value=0, dtype=tf.float32, name='total_distance')
        # Define new centers as Variable.
        new_centers_sum = tf.Variable(initial_value=tf.zeros_like(current_centers), dtype=tf.float32)
        new_centers_count = tf.Variable(initial_value=tf.zeros_like(current_centers), dtype=tf.float32)
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
            # TODO, sqrt can be omitted.
            minus_dist = tf.negative(tf.sqrt(tf.reduce_sum(squared_sub, axis=-1)))
            # Compute assignments and the distance with nearest centers video-wisely.
            minus_nearest_dist, topk_assignments = tf.nn.top_k(minus_dist, k=1)
            nearest_dist = tf.negative(minus_nearest_dist)
            # Remove the last dimension due to k.
            assignments = tf.squeeze(topk_assignments, axis=[-1])

            # Compute new centers sum and number of videos that belong to each center (cluster) with this video batch.
            sum_per_clu = tf.unsorted_segment_sum(video_batch, assignments, num_centers)
            count_per_clu = tf.unsorted_segment_sum(tf.ones_like(video_batch, dtype=tf.float32),
                                                    assignments, num_centers)

        else:
            normalized_video_batch = tf.nn.l2_normalize(video_batch, -1)
            cosine_sim = tf.matmul(normalized_video_batch, current_centers, transpose_b=True)
            nearest_cosine_sim, topk_assignments = tf.nn.top_k(cosine_sim, k=1)
            nearest_dist = tf.subtract(1.0, nearest_cosine_sim)
            # Remove the last dimension due to k.
            assignments = tf.squeeze(topk_assignments, axis=[-1])

            # Compute new centers sum and number of videos that belong to each center (cluster) with this video batch.
            sum_per_clu = tf.unsorted_segment_sum(normalized_video_batch, assignments, num_centers)
            count_per_clu = tf.unsorted_segment_sum(tf.ones_like(normalized_video_batch), assignments, num_centers)

        # Update total distance, namely objective function.
        total_batch_dist = tf.reduce_sum(nearest_dist)
        update_total_distance = tf.assign_add(total_distance, total_batch_dist)

        update_new_centers_sum = tf.assign_add(new_centers_sum, sum_per_clu)
        update_new_centers_count = tf.assign_add(new_centers_count, count_per_clu)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    graph.finalize()

    sess = tf.Session(graph=graph)
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            accum_distance, accum_new_centers_sum, accum_new_centers_count = sess.run(
                [update_total_distance, update_new_centers_sum, update_new_centers_count])

    except tf.errors.OutOfRangeError:
        logging.info('One k-means iteration done. One epoch limit reached.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    # Numpy array divide element-wisely.
    return (accum_new_centers_sum / accum_new_centers_count), accum_distance


def mini_batch_kmeans():
    pass


def dbscan():
    pass


def initialize(num_centers_ratio, method=None, metric='euclidean', scaling_method=1, alpha=0.1, p=3, debug=False):
    """
    This functions implements the following two phases:
    1. To initialize representative prototypes (RBF centers) c and scaling factors sigma.
    2. And to fit output weights.

    This function will generate one group of centers for all labels as a whole. Be cautious with initialize_per_label.

    :param num_centers_ratio: The number of centers to be decided / total number of examples that belong to label l,
     for l = 0, ..., num_classes - 1.
    :param method: The method to decide the centers. Possible choices are kmeans, online(kmeans), and lvq(learning).
     Default is None, which represents randomly selecting a certain number of examples as centers.
    :param metric: Distance metric, euclidean distance or cosine distance.
    :param scaling_method: There are four choices. 1, all of them use the same sigma, the p smallest pairs of distances.
     2, average of p nearest centers. 3, the distance to the nearest center that has a different label (Not supported!).
     4, mean distance between this center and all of its points.
    :param alpha: The alpha parameter that should be set heuristically. It works like a learning rate. (mu in Zhang's)
    :param p: When scaling_method is 1 or 2, p is needed.
    :param debug: If True, prints detailed intermediate results.
    :return:
    """
    logging.info('Generate a group of centers for all labels. See Schwenker.')
    if num_centers_ratio <= 0.0 or num_centers_ratio >= 1.0:
        raise ValueError('num_centers_ratio must be larger than 0.0 and no greater than 1.0.')

    logging.info('num_centers_ratio is {}.'.format(num_centers_ratio))
    if ('euclidean' == metric) or ('cosine' == metric):
        logging.info('Using {} distance. The larger, the less similar.'.format(metric))
    else:
        raise NotImplementedError('Only euclidean distance and cosine distance are supported.')

    train_data_pattern = FLAGS.train_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes

    reader = get_reader(model_type, feature_names, feature_sizes)
    initial_centers = random_sample(num_centers_ratio, reader, train_data_pattern, batch_size, num_readers)
    print('initial_centers: {}'.format(initial_centers))
    num_initial_centers = len(initial_centers)
    print('Sampled {} centers totally.'.format(num_initial_centers))

    # Perform kmeans or online kmeans.
    if method is None:
        logging.info('Using randomly selected initial centers as model prototypes (centers).')
    elif 'online' == method:
        # TODO.
        raise NotImplementedError('Only None (randomly select examples), online, kmeans and lvq are supported.')
    elif 'kmeans' == method:
        start_time = time.time()
        logging.info('Using k-means clustering result as model prototypes (centers).')
        new_centers, new_dist = kmeans_iter(initial_centers, reader,
                                            train_data_pattern, batch_size, num_readers, metric='cosine')
        print('One iteration needs {} s.'.format(time.time() - start_time))
        print('new_centers: {}\nnew_dist: {}'.format(new_centers, new_dist))
    elif 'lvq' == method:
        raise NotImplementedError('Only None (randomly select examples), online, kmeans and lvq are supported.')
    else:
        raise NotImplementedError('Only None (randomly select examples), online, kmeans and lvq are supported.')

    # Compute scaling factors based on these centers.
    if scaling_method == 1:
        if ('euclidean' == metric) or ('cosine' == metric):
            pairwise_distances = sci_distance.pdist(initial_centers, metric=metric)
            p = min(p, len(pairwise_distances))
            logging.info('Using {} minimal pairwise distances.'.format(p))
            # np.partition begins with 1 instead of 0.
            sigmas = [alpha * np.mean(np.partition(pairwise_distances, p - 1)[:p])] * num_initial_centers
    elif scaling_method == 2:
        p = min(p, num_initial_centers - 1)
        logging.info('Using {} minimal pairwise distances.'.format(p))

        if 'euclidean' == metric:
            dis_fn = sci_distance.euclidean
        else:
            dis_fn = sci_distance.cosine

        sigmas = []
        for c in initial_centers:
            distances = [dis_fn(c, _c) for _c in initial_centers]
            # The distance between c and itself is zero and is in the left partition.
            sigmas.append(alpha * np.sum(np.partition(distances, p)[:p + 1]) / float(p))
    elif scaling_method == 3:
        raise NotImplementedError('Not supported when all labels use the same centers.')
    elif scaling_method == 4:
        logging.info('Reuse results from kmeans or online kmeans.')
        # TODO. Consider code repeatability.
    else:
        raise NotImplementedError('Only four methods are supported. Please read the documentation.')


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


def train(debug=False):
    num_centers_ratio = FLAGS.num_centers_ratio

    # num_centers = FLAGS.num_centers
    # num_centers_ratio = float(num_centers) / NUM_TRAIN_EXAMPLES
    initialize(num_centers_ratio, method='kmeans', debug=debug)


def inference(out_file_location, top_k, debug=False):
    pass


def main(unused_argv):
    is_train = FLAGS.is_train
    is_tuning_hyper_para = FLAGS.is_tuning_hyper_para
    is_debug = FLAGS.is_debug

    output_file = FLAGS.output_file
    top_k = FLAGS.top_k

    logging.set_verbosity(logging.INFO)

    if is_train:
        if is_tuning_hyper_para:
            raise NotImplementedError('Implementation is under progress.')
        else:
            train(debug=is_debug)
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

    flags.DEFINE_string('feature_names', 'mean_rgb', 'Features to be used, separated by ,.')

    flags.DEFINE_string('feature_sizes', '1024', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_float('num_centers_ratio', 0.01, 'The number of centers in RBF network.')

    # Set by the memory limit (52GB).
    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 1, 'Number of readers to form a batch.')

    flags.DEFINE_boolean('verbosity', False, 'Whether print intermediate results, default no.')

    flags.DEFINE_string('output_dir', '/tmp/ml-knn/',
                        'The directory to which prior and posterior probabilities should be written.')

    flags.DEFINE_boolean('is_train', True, 'Boolean variable to indicate training or test.')

    flags.DEFINE_boolean('is_tuning_hyper_para', False,
                         'Boolean variable indicating whether to perform hyper-parameter tuning.')

    # TODO, change it.
    flags.DEFINE_boolean('is_debug', True, 'Boolean variable to indicate debug ot not.')

    flags.DEFINE_string('output_file', '/tmp/ml-knn/predictions.csv', 'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
