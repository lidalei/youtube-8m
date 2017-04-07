import tensorflow as tf
import time

from readers import get_reader
import utils
from tensorflow import flags, gfile, logging, app
from inference import format_lines

import pickle
import numpy as np

FLAGS = flags.FLAGS


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


def compute_prior_prob(reader, data_pattern, smooth_para=1, verbosity=False):
    """
    Compute prior probabilities for future use in ml-knn.
    :param reader:
    :param data_pattern:
    :param smooth_para:
    :param verbosity:
    :return: (total number of labels per label, total number of videos processed, prior probabilities)
    """
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    # Generate example queue. Traverse the queue to traverse the dataset.
    video_id_batch, video_batch, video_labels_batch, num_frames_batch = get_input_data_tensors(
        reader=reader, data_pattern=data_pattern, batch_size=batch_size, num_readers=num_readers, num_epochs=1)

    num_classes = reader.num_classes

    sum_labels_onehot = tf.Variable(tf.zeros([num_classes]))
    total_num_videos = tf.Variable(0, dtype=tf.float32)

    sum_labels_onehot_op = sum_labels_onehot.assign_add(tf.reduce_sum(tf.cast(video_labels_batch, tf.float32), axis=0))
    accum_num_videos_op = total_num_videos.assign_add(tf.cast(tf.shape(video_labels_batch)[0], tf.float32))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
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
        sess.close()

    labels_prior_prob_val = (smooth_para + sum_labels_val) / (smooth_para * 2 + accum_num_videos_val)

    if verbosity:
        print('sum_labels_val: {}\n accum_num_videos_val: {}'.format(sum_labels_val, accum_num_videos_val))
        print('compute_labels_prob: {}'.format(labels_prior_prob_val))

    return sum_labels_val, accum_num_videos_val, labels_prior_prob_val


def find_k_nearest_neighbors(video_id_batch, video_batch, reader, data_pattern, batch_size=8192, k=3, verbosity=True):
    """
    Return k-nearest neighbors. https://www.tensorflow.org/programmers_guide/reading_data.
    :param video_id_batch: Must be a value.
    :param video_batch: Must be a value.
    :param reader:
    :param data_pattern:
    :param batch_size:
    :param k:
    :param verbosity:
    :return: k-nearest videos, representing by (video_ids, video_labels)
    """

    num_readers = FLAGS.num_readers
    # Create a new graph to compute k-nearest neighbors from video_batch_inner for each video of video_batch.
    with tf.Graph().as_default() as graph:
        # Generate example queue. Traverse the queue to traverse the dataset.
        # Works as the inner loop of finding k-nearest neighbors.
        video_id_batch_inner, video_batch_inner, video_labels_batch_inner, num_frames_batch_inner = (
            get_input_data_tensors(
                reader=reader, data_pattern=data_pattern, batch_size=batch_size,
                num_readers=num_readers, num_epochs=1, name_scope='inner_loop'))

        # normalization
        feature_dim = len(video_batch_inner.get_shape()) - 1

        video_batch_normalized = tf.nn.l2_normalize(video_batch, feature_dim)
        video_batch_inner_normalized = tf.nn.l2_normalize(video_batch_inner, feature_dim)

        # compute cosine similarities
        similarities = tf.matmul(video_batch_normalized, video_batch_inner_normalized, transpose_b=True)
        # top k similar videos per video in video_batch_normalized.
        # values and indices are in shape [batch_size, k].
        # k = k + 1, to avoid the video itself.
        topk_sim_op = tf.nn.top_k(similarities, k=k + 1)

        # Initialization.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # batch_size = int(video_id_batch.shape[0])
    # Define variables representing k nearest video_ids, video_labels and video_similarities.
    topk_video_ids = None  # batch_size * [[]]
    # Variable-length labels.
    topk_video_labels = None  # batch_size * [[]]
    topk_video_sims = None  # batch_size * [[0.0] * k]

    # A new graph needs a new session. Thus, create one.
    with tf.Session(graph=graph) as sess:
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # Run results are numpy arrays.
                video_id_batch_inner_val, video_labels_batch_inner_val, (
                    batch_topk_sims, batch_topk_sim_indices) = sess.run(
                    [video_id_batch_inner, video_labels_batch_inner, topk_sim_op])

                # stack them into numpy array with shape [batch_size, k] (id) or [batch_size, k, num_classes] (labels).
                # np.stack() can be np.array().
                batch_topk_video_ids = np.stack([video_id_batch_inner_val[ind] for ind in batch_topk_sim_indices])
                batch_topk_video_labels = np.stack(
                    [video_labels_batch_inner_val[ind] for ind in batch_topk_sim_indices])

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
                    topk_video_sims, topk_video_sims_indices = sess.run(tf.nn.top_k(top2k_video_sims, k=k + 1))

                    topk_video_ids = np.stack(
                        [top2k_video_ids[i, ind] for i, ind in enumerate(topk_video_sims_indices)])
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


def store_prior_prob(sum_labels, accum_num_videos, labels_prior_prob, folder=''):
    with open(folder + 'sum_labels.pickle', 'wb') as pickle_file:
        pickle.dump(sum_labels, pickle_file)

    with open(folder + 'accum_num_videos.pickle', 'wb') as pickle_file:
        pickle.dump(accum_num_videos, pickle_file)

    with open(folder + 'labels_prior_prob.pickle', 'wb') as pickle_file:
        pickle.dump(labels_prior_prob, pickle_file)


def recover_prior_prob(folder=''):
    with open(folder + 'sum_labels.pickle', 'rb') as pickle_file:
        sum_labels = pickle.load(pickle_file)

    with open(folder + 'accum_num_videos.pickle', 'rb') as pickle_file:
        accum_num_videos = pickle.load(pickle_file)

    with open(folder + 'labels_prior_prob.pickle', 'rb') as pickle_file:
        labels_prior_prob = pickle.load(pickle_file)

    return sum_labels, accum_num_videos, labels_prior_prob


def store_posterior_prob(count, counter_count, pos_prob_positive, pos_prob_negative, k, folder=''):
    with open(folder + 'count_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(count, pickle_file)

    with open(folder + 'counter_count_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(counter_count, pickle_file)

    with open(folder + 'pos_prob_positive_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(pos_prob_positive, pickle_file)

    with open(folder + 'pos_prob_negative_{}.pickle'.format(k), 'wb') as pickle_file:
        pickle.dump(pos_prob_negative, pickle_file)


def recover_posterior_prob(k, folder=''):
    with open(folder + 'count_{}.pickle'.format(k), 'rb') as pickle_file:
        count = pickle.load(pickle_file)

    with open(folder + 'counter_count_{}.pickle'.format(k), 'rb') as pickle_file:
        counter_count = pickle.load(pickle_file)

    with open(folder + 'pos_prob_positive_{}.pickle'.format(k), 'rb') as pickle_file:
        pos_prob_positive = pickle.load(pickle_file)

    with open(folder + 'pos_prob_negative_{}.pickle'.format(k), 'rb') as pickle_file:
        pos_prob_negative = pickle.load(pickle_file)

    return count, counter_count, pos_prob_positive, pos_prob_negative


def compute_prior_posterior_prob(k=8, smooth_para=1.0, debug=False):
    train_data_pattern = '/Users/Sophie/Documents/youtube-8m-data/train/trainaW.tfrecord' \
        if debug else FLAGS.train_data_pattern

    batch_size = 1024 if debug else FLAGS.batch_size
    num_readers = 1 if debug else FLAGS.num_readers
    verbosity = FLAGS.verbosity
    model_dir = FLAGS.model_dir
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes

    reader = get_reader(model_type, feature_names, feature_sizes)
    """
    # Step 1. Compute prior probabilities and store the results.
    sum_labels, accum_num_videos, labels_prior_prob = compute_prior_prob(reader, train_data_pattern, smooth_para)
    store_prior_prob(sum_labels, accum_num_videos, labels_prior_prob, model_dir)
    """

    # Step 2. Compute posterior probabilities, actually likelihood function or sampling distribution.

    # Total number of classes.
    num_classes = reader.num_classes
    range_num_classes = range(num_classes)
    # For each possible class, define a count and counter_count to count.
    # Compute the posterior probability, namely, given a label l, counting the number of training examples that have
    # exactly j (0 <= j <= k) nearest neighbors that have label l and normalizing it.
    # Here, j is considered as a random variable.
    count = np.zeros([k + 1, num_classes], dtype=np.float32)
    counter_count = np.zeros([k + 1, num_classes], dtype=np.float32)

    global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
    global_step_inc_op = global_step.assign_add(1)

    # For debug, use a single tfrecord file (debug mode).
    video_id_batch, video_batch, video_labels_batch, num_frames_batch = (get_input_data_tensors(
        reader, train_data_pattern, batch_size, num_readers=num_readers))

    tf.summary.scalar('global_step', global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.Session()

    writer = tf.summary.FileWriter(model_dir, sess.graph)
    summary_op = tf.summary.merge_all()

    sess.run(init_op)

    inner_reader = get_reader(model_type, feature_names, feature_sizes)

    # TODO, add a train.Saver.

    # Be cautious to not be blocked by queue.
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    processing_count, num_examples_processed = 0, 0

    try:

        while not coord.should_stop():
            # Run training steps or whatever.
            start_time = time.time()
            video_id_batch_val, video_batch_val, video_labels_batch_val = sess.run(
                [video_id_batch, video_batch, video_labels_batch])

            if debug or verbosity:
                print('video_id_batch_val: {}\nvideo_batch_val: {}'.format(video_id_batch_val, video_batch_val))

            # Pass values instead of tensors.
            topk_video_ids, topk_video_labels = find_k_nearest_neighbors(video_id_batch_val,
                                                                         video_batch_val, inner_reader,
                                                                         data_pattern=train_data_pattern,
                                                                         batch_size=batch_size,
                                                                         k=k, verbosity=False)

            if debug or verbosity:
                print('topk_video_ids: {}\ntopk_video_labels: {}'.format(topk_video_ids, topk_video_labels))
            # Update count and counter_count.
            # batch_size * delta.
            deltas = topk_video_labels.astype(np.int32).sum(axis=1)
            # Update count and counter_count for each example.
            for delta, video_labels_val in zip(deltas, video_labels_batch_val):
                inc = video_labels_val.astype(np.float32)
                count[delta, range_num_classes] += inc
                counter_count[delta, range_num_classes] += 1 - inc

            if debug or verbosity:
                print('count: {}\ncounter_count: {}'.format(count, counter_count))

            if debug:
                coord.request_stop()

            global_step_val, summary = sess.run([global_step_inc_op, summary_op])
            now = time.time()
            num_examples_processed += video_id_batch_val.shape[0]
            print('Batch processing step: {}, elapsed seconds: {}, total number of examples processed: {}'.format(
                global_step_val, now - start_time, num_examples_processed))

            writer.add_summary(summary, global_step=global_step_val)

    except tf.errors.OutOfRangeError:
        logging.info('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    # Compute posterior probabilities.
    pos_prob_positive = (smooth_para + count) / (smooth_para * (k + 1) + count.sum(axis=0))
    pos_prob_negative = (smooth_para + counter_count) / (smooth_para * (k + 1) + counter_count.sum(axis=0))

    # Write to files for future use.
    store_posterior_prob(count, counter_count, pos_prob_positive, pos_prob_negative, k, model_dir)


def make_predictions(out_file_location, top_k, k=8, debug=False):
    """

    :param out_file_location: The file to which predictions should be written to. Supports gcloud file.
    :param top_k: See FLAGS.top_k.
    :param k: The k in ml-knn.
    :param debug: If True, make predictions on a single file and find k nearest neighbors from a single file.
    :return:
    """
    train_data_pattern = '/Users/Sophie/Documents/youtube-8m-data/train/trainaW.tfrecord' \
        if debug else FLAGS.train_data_pattern

    test_data_pattern = '/Users/Sophie/Documents/youtube-8m-data/test/test4a.tfrecord' \
        if debug else FLAGS.test_data_pattern

    verbosity = FLAGS.verbosity
    model_dir = FLAGS.model_dir
    batch_size = 1024 if debug else FLAGS.batch_size
    num_readers = 1 if debug else FLAGS.num_readers
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes

    # Load prior and posterior probabilities.
    sum_labels, accum_num_videos, labels_prior_prob = recover_prior_prob(folder=model_dir)
    count, counter_count, pos_prob_positive, pos_prob_negative = recover_posterior_prob(k, folder=model_dir)

    # Make batch predictions.
    reader = get_reader(model_type, feature_names, feature_sizes)
    inner_reader = get_reader(model_type, feature_names, feature_sizes)

    # Total number of classes.
    num_classes = reader.num_classes
    range_num_classes = range(num_classes)

    with tf.Session() as sess, gfile.Open(out_file_location, "w+") as out_file:

        # For debug, use a single tfrecord file (debug mode).
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = get_input_data_tensors(
            reader, test_data_pattern, batch_size, num_readers=num_readers)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_op)

        # Be cautious to not be blocked by queue.
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        processing_count, num_examples_processed = 0, 0
        out_file.write("VideoId,LabelConfidencePairs\n")

        try:

            while not coord.should_stop():
                # Run training steps or whatever.
                start_time = time.time()
                video_id_batch_val, video_batch_val, video_labels_batch_val = sess.run(
                    [video_id_batch, video_batch, video_labels_batch])

                if debug or verbosity:
                    print('video_id_batch_val: {}\nvideo_batch_val: {}'.format(video_id_batch_val, video_batch_val))

                # Pass values instead of tensors.
                topk_video_ids, topk_video_labels = find_k_nearest_neighbors(video_id_batch_val,
                                                                             video_batch_val, inner_reader,
                                                                             data_pattern=train_data_pattern,
                                                                             batch_size=batch_size,
                                                                             k=k, verbosity=False)

                if debug or verbosity:
                    print('topk_video_ids: {}\ntopk_video_labels: {}'.format(topk_video_ids, topk_video_labels))
                # batch_size * delta.
                deltas = topk_video_labels.astype(np.int32).sum(axis=1)

                batch_predictions_prob = []
                for delta in deltas:
                    positive_prob_numerator = labels_prior_prob * pos_prob_positive[delta, range_num_classes]
                    negative_prob_numerator = (1.0 - labels_prior_prob) * pos_prob_negative[delta, range_num_classes]
                    # predictions = positive_prob_numerator > negative_prob_numerator

                    batch_predictions_prob.append(
                        positive_prob_numerator / (positive_prob_numerator + negative_prob_numerator))

                # Write batch predictions to files.
                for line in format_lines(video_id_batch_val, batch_predictions_prob, top_k):
                    out_file.write(line)
                out_file.flush()

                if debug:
                    coord.request_stop()

                now = time.time()
                processing_count += 1
                num_examples_processed += video_id_batch_val.shape[0]
                print('Batch processing step: {}, elapsed seconds: {}, total number of examples processed: {}'.format(
                    processing_count, now - start_time, num_examples_processed))

        except tf.errors.OutOfRangeError:
            logging.info('Done with inference. The predictions were written to {}'.format(out_file_location))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        sess.close()
        out_file.close()


def main(unused_argv):
    is_train = FLAGS.is_train
    is_tuning_hyper_para = FLAGS.is_tuning_hyper_para
    is_debug = FLAGS.is_debug
    output_file = FLAGS.output_file
    top_k = FLAGS.top_k

    if is_train:
        if is_tuning_hyper_para:
            # TODO, implement.
            raise NotImplementedError('Implementation is under progress.')
        else:
            compute_prior_posterior_prob(debug=is_debug)
    else:
        make_predictions(output_file, top_k, debug=is_debug)


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

    # Set by the memory limit (52GB).
    flags.DEFINE_integer('batch_size', 24576, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 4, 'Number of readers to form a batch.')

    flags.DEFINE_boolean('verbosity', False, 'Whether print intermediate results, default no.')

    flags.DEFINE_string('model_dir', '/tmp/ml-knn/',
                        'The directory to which prior and posterior probabilities should be written.')

    flags.DEFINE_boolean('is_train', True, 'Boolean variable to indicate training or test.')

    flags.DEFINE_boolean('is_tuning_hyper_para', False,
                         'Boolean variable indicating whether to perform hyper-parameter tuning.')

    # TODO, change it.
    flags.DEFINE_boolean('is_debug', False, 'Boolean variable to indicate debug ot not.')

    flags.DEFINE_string('output_file', '/tmp/ml-knn/predictions.csv', 'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
