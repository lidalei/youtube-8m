import tensorflow as tf
import readers
import utils
from tensorflow import flags, gfile, logging, app

import pickle
import numpy as np

FLAGS = flags.FLAGS


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
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find input files. data_pattern='{}'".format(data_pattern))
        logging.info("number of input files: {}".format(len(files)))
        # Pass test data once. Thus, num_epochs is set as 1.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=False)
        examples_and_labels = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            tf.train.batch_join(examples_and_labels,
                                batch_size=batch_size,
                                allow_smaller_final_batch=True,
                                enqueue_many=True))
        return video_id_batch, video_batch, video_labels_batch, num_frames_batch


def compute_prior_posterior_prob():
    pass


def compute_prior_prob(reader, data_pattern, smooth_para=1, verbosity=False):
    """
    Compute prior probabilities for future use in ml-knn.
    :param reader:
    :param data_pattern:
    :param smooth_para:
    :return: (total number of labels per label, total number of videos processed, prior probabilities)
    """
    # Generate example queue. Traverse the queue to traverse the dataset.
    video_id_batch, video_batch, video_labels_batch, num_frames_batch = get_input_data_tensors(
        reader=reader, data_pattern=data_pattern, batch_size=8192, num_readers=1, num_epochs=1)

    num_classes = reader.num_classes

    sum_labels_onehot = tf.Variable(tf.zeros([num_classes]))
    total_num_videos = tf.Variable(0, dtype=tf.float32)

    sum_labels_onehot_op = sum_labels_onehot.assign_add(tf.reduce_sum(tf.cast(video_labels_batch, tf.float32), axis=0))
    accum_num_videos_op = total_num_videos.assign_add(tf.cast(tf.shape(video_labels_batch)[0], tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # sum video labels
            sum_labels_val, accum_num_videos_val = sess.run([sum_labels_onehot_op, accum_num_videos_op])

    except tf.errors.OutOfRangeError:
        logging.info('Done the whole dataset.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    labels_prior_prob_val = (smooth_para + sum_labels_val) / (smooth_para * 2 + accum_num_videos_val)

    sess.close()

    if verbosity:
        print('sum_labels_val: {}\n accum_num_videos_val: {}'.format(sum_labels_val, accum_num_videos_val))
        print('compute_labels_prob: {}'.format(labels_prior_prob_val))

    return sum_labels_val, accum_num_videos_val, labels_prior_prob_val


def find_k_nearest_neighbors(video_id_batch, video_batch, reader, data_pattern, k=3, verbosity=True):
    """
    Return k-nearest neighbors. https://www.tensorflow.org/programmers_guide/reading_data.
    :param video_id_batch: Must be a value.
    :param video_batch: Must be a value.
    :param reader:
    :param data_pattern:
    :param k:
    :param verbosity:
    :return: k-nearest videos, representing by (video_ids, video_labels)
    """

    # Generate example queue. Traverse the queue to traverse the dataset.
    # Works as the inner loop of finding k-nearest neighbors.
    video_id_batch_inner, video_batch_inner, video_labels_batch_inner, num_frames_batch_inner = get_input_data_tensors(
        reader=reader, data_pattern=data_pattern, batch_size=8192, num_readers=1, num_epochs=1, name_scope='inner_loop')

    # batch_size = int(video_id_batch.shape[0])
    # Define variables representing k nearest video_ids, video_labels and video_similarities.
    topk_video_ids = None  # batch_size * [[]]
    # Variable-length labels.
    topk_video_labels = None  # batch_size * [[]]
    topk_video_sims = None  # batch_size * [[0.0] * k]

    # Initialization.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    # normalization
    feature_dim = len(video_batch.shape) - 1

    video_batch_normalized = tf.nn.l2_normalize(video_batch, feature_dim)
    video_batch_inner_normalized = tf.nn.l2_normalize(video_batch_inner, feature_dim)

    # compute cosine similarities
    similarities = tf.matmul(video_batch_normalized, video_batch_inner_normalized, transpose_b=True)
    # top k similar videos per video in video_batch_normalized.
    # values and indices are in shape [batch_size, k].
    # k = k + 1, to avoid the video itself.
    topk_sim_op = tf.nn.top_k(similarities, k=k+1)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run results are numpy arrays.
            video_id_batch_inner_val, video_labels_batch_inner_val, (batch_topk_sims, batch_topk_sim_indices) = sess.run(
                [video_id_batch_inner, video_labels_batch_inner, topk_sim_op])

            # stack them into numpy array with shape [batch_size, k] (id) or [batch_size, k, num_classes] (labels).
            # np.stack() can be np.array().
            batch_topk_video_ids = np.stack([video_id_batch_inner_val[ind] for ind in batch_topk_sim_indices])
            batch_topk_video_labels = np.stack([video_labels_batch_inner_val[ind] for ind in batch_topk_sim_indices])

            if verbosity:
                # Debug mode.
                print('video_id_batch: {}'.format(video_id_batch))
                print('batch_topk_sims: {}\nbatch_topk_sim_indices: {}'.format(batch_topk_sims,
                                                                               batch_topk_sim_indices))
                print('batch_topk_video_ids: {}\nbatch_topk_video_labels: {}'.format(batch_topk_video_ids,
                                                                                     batch_topk_video_labels))

                coord.request_stop()

            # Update top k similar videos.
            if (topk_video_ids is None) or (topk_video_labels is None) or (topk_video_sims is None):
                # The first batch.
                topk_video_ids = batch_topk_video_ids
                topk_video_labels = batch_topk_video_labels
                topk_video_sims = batch_topk_sims
            else:
                # Combine batch top k video ids and labels into current top k ids and labels.
                if verbosity:
                    print('topk_video_ids shape: {}, batch_topk_video_ids shape: {}'.format(topk_video_ids.shape,
                                                                                            batch_topk_video_ids.shape))
                top2k_video_ids = np.concatenate((topk_video_ids, batch_topk_video_ids), axis=1)
                if verbosity:
                    print('topk_video_labels shape: {}, batch_topk_video_labels shape: {}'.format(
                        topk_video_labels.shape, batch_topk_video_labels.shape))
                top_2k_video_labels = np.concatenate((topk_video_labels, batch_topk_video_labels), axis=1)
                top2k_video_sims = np.concatenate((topk_video_sims, batch_topk_sims), axis=1)
                # k=k+1, to exclude the example itself finally.
                topk_video_sims, topk_video_sims_indices = sess.run(tf.nn.top_k(top2k_video_sims, k=k+1))

                topk_video_ids = np.stack([top2k_video_ids[i, ind] for i, ind in enumerate(topk_video_sims_indices)])
                topk_video_labels = np.stack(
                    [top_2k_video_labels[i, ind] for i, ind in enumerate(topk_video_sims_indices)])

    except tf.errors.OutOfRangeError:
        logging.info('Done the whole dataset.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    # Exclude the example itself if it exists.
    clean_topk_video_ids, clean_topk_video_labels = [], []
    for video_id, topk_video_id, topk_video_label in zip(video_id_batch, topk_video_ids, topk_video_labels):
        if topk_video_id[0] == video_id:
            clean_topk_video_ids.append(topk_video_id[1:])
            clean_topk_video_labels.append(topk_video_label[1:])
        else:
            clean_topk_video_ids.append(topk_video_id[:k])
            clean_topk_video_labels.append(topk_video_label[:k])

    return np.stack(clean_topk_video_ids), np.stack(clean_topk_video_labels)


def main(unused_argv):
    k = FLAGS.k
    smooth_para = FLAGS.smooth_para
    train_data_pattern = FLAGS.train_data_pattern
    verbosity = FLAGS.verbosity
    reader = get_reader()

    # batch_data = get_input_data_tensors(reader, train_data_pattern)

    """
    # Compute prior probabilities and store the results.
    sum_labels, accum_num_videos, labels_prior_prob = compute_prior_prob(reader, train_data_pattern,
                                                                         smooth_para)

    with open('sum_labels.pickle', 'wb') as pickle_file:
        pickle.dump(sum_labels, pickle_file)
    pickle_file.close()

    with open('accum_num_videos.pickle', 'wb') as pickle_file:
        pickle.dump(accum_num_videos, pickle_file)
    pickle_file.close()

    with open('labels_prior_prob.pickle', 'wb') as pickle_file:
        pickle.dump(labels_prior_prob, pickle_file)
    pickle_file.close()
    """
    # Total number of classes.
    num_classes = reader.num_classes
    range_num_classes = range(num_classes)
    # For each possible class, define a count and counter_count to count.
    # Compute the posterior probability, namely, given a label l, counting the number of training examples that have
    # exactly j (0 <= j <= k) nearest neighbors that have label l and normalizing it.
    # Here, j is considered as a random variable.
    count = np.zeros([k + 1, num_classes], dtype=np.float32)
    counter_count = np.zeros([k + 1, num_classes], dtype=np.float32)

    # TODO, change to whole dataset. For test, use a single tfrecord file.
    video_id_batch, video_batch, video_labels_batch, num_frames_batch = get_input_data_tensors(
        reader, train_data_pattern, 8192)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    inner_reader = get_reader()

    # Be cautious to not be blocked by queue.
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:

        while not coord.should_stop():
            # Run training steps or whatever
            video_id_batch_val, video_batch_val, video_labels_batch_val = sess.run(
                [video_id_batch, video_batch, video_labels_batch])

            if verbosity:
                print('video_id_batch_val: {}\nvideo_batch_val: {}'.format(video_id_batch_val, video_batch_val))

            # Pass values instead of tensors.
            topk_video_ids, topk_video_labels = find_k_nearest_neighbors(video_id_batch_val,
                                                                         video_batch_val, inner_reader,
                                                                         data_pattern=train_data_pattern,
                                                                         k=k, verbosity=False)

            if verbosity:
                print('topk_video_ids: {}\ntopk_video_labels: {}'.format(topk_video_ids, topk_video_labels))
            # Update count and counter_count.
            # batch_size * delta.
            deltas = topk_video_labels.astype(np.int32).sum(axis=1)
            # Update count and counter_count for each example.
            for delta, video_labels_val in zip(deltas, video_labels_batch_val):
                inc = video_labels_val.astype(np.float32)
                count[delta, range_num_classes] += inc
                counter_count[delta, range_num_classes] += 1 - inc

            if verbosity:
                print('count: {}\ncounter_count: {}'.format(count, counter_count))

            # TODO, Debug mode.
            # coord.request_stop()

    except tf.errors.OutOfRangeError:
        logging.info('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    # Write to files for future use.
    with open('count_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(count, pickle_file)
    pickle_file.close()

    with open('counter_count_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(counter_count, pickle_file)
    pickle_file.close()

    # Compute posterior probabilities.
    pos_prob_positive = (smooth_para + count) / (smooth_para * (k + 1) + count.sum(axis=0))
    pos_prob_negative = (smooth_para + counter_count) / (smooth_para * (k + 1) + counter_count.sum(axis=0))

    # Write to files for future use.
    with open('pos_prob_positive_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(pos_prob_positive, pickle_file)
    pickle_file.close()

    with open('pos_prob_negative_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(pos_prob_negative, pickle_file)
    pickle_file.close()

    sess.close()

if __name__ == '__main__':
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

    flags.DEFINE_integer('batch_size', 8192, 'size of batch processing')

    flags.DEFINE_integer('k', 8, 'k-nearest neighbor')

    flags.DEFINE_float('smooth_para', 1.0, 'smooth parameter, default as 1.0')

    flags.DEFINE_boolean('verbosity', False, 'whether print intermediate results, default no.')

    app.run()
