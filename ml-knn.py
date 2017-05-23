"""
The referenced paper is
Zhang M L, Zhou Z H. ML-KNN: A lazy learning approach to multi-label learning[J].
Pattern recognition, 2007, 40(7): 2038-2048.
"""
import tensorflow as tf
import numpy as np

from readers import get_reader
from tensorflow import flags, gfile, logging, app
from inference import format_lines
from utils import DataPipeline, get_input_data_tensors, random_sample
from utils import save_prior_prob, save_posterior_prob, restore_prior_prob, restore_posterior_prob
from eval_util import calculate_gap

import time

FLAGS = flags.FLAGS


def compute_prior_prob(data_pipeline, smooth_para=1.0):
    """
    Compute prior probabilities for future use in ml-knn.
    :param data_pipeline:
    :param smooth_para:
    :return: (total number of labels per label, total number of videos processed, prior probabilities)
    """
    reader = data_pipeline.reader
    num_classes = reader.num_classes

    with tf.Graph().as_default() as g:
        sum_labels_onehot = tf.Variable(tf.zeros([num_classes]))
        total_num_videos = tf.Variable(0, dtype=tf.float32)

        # Generate example queue. Traverse the queue to traverse the data set.
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = get_input_data_tensors(
            data_pipeline, num_epochs=1, name_scope='prior_prob_input')

        sum_labels_onehot_op = sum_labels_onehot.assign_add(
            tf.reduce_sum(tf.cast(video_labels_batch, tf.float32), axis=0))
        accum_num_videos_op = total_num_videos.assign_add(tf.cast(tf.shape(video_labels_batch)[0], tf.float32))

        with tf.control_dependencies([sum_labels_onehot_op, accum_num_videos_op]):
            accum_non_op = tf.no_op()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session(graph=g) as sess:
        sess.run(init_op)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                # sum video labels
                sess.run(accum_non_op)

        except tf.errors.OutOfRangeError:
            logging.info('Done the whole data set.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        sum_labels_val, total_num_videos_val = sess.run([sum_labels_onehot, total_num_videos])
        sess.close()

    labels_prior_prob_val = (smooth_para + sum_labels_val) / (smooth_para * 2 + total_num_videos_val)

    logging.debug('sum_labels_val: {}\n accum_num_videos_val: {}'.format(sum_labels_val, total_num_videos_val))
    logging.debug('compute_labels_prob: {}'.format(labels_prior_prob_val))

    return sum_labels_val, total_num_videos_val, labels_prior_prob_val


def find_k_nearest_neighbors(video_id_batch, video_batch, data_pipeline, is_train, k=8,
                             logdir='/tmp/ml-knn/find-knn'):
    """
    Return k-nearest neighbors. https://www.tensorflow.org/programmers_guide/reading_data.

    :param video_id_batch: Must be a value.
    :param video_batch: Must be a numpy array.
    :param data_pipeline:
    :param is_train: If True, exclude the most similar example (itself). 
    :param k: int.
    :param logdir: path to log dir.
    :return: k-nearest videos, representing by (video_ids, video_labels)
    """
    logging.info('Entering find knn...')
    num_videos = video_batch.shape[0]
    reader = data_pipeline.reader
    num_classes = reader.num_classes

    # If training, k = k + 1, to avoid the video itself. Otherwise, not necessary.
    _k = int(k)
    k = (_k + 1) if is_train else _k

    # Create a new graph to compute k-nearest neighbors for each video of video_batch.
    with tf.Graph().as_default() as graph:
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        global_step_inc_op = global_step.assign_add(1)

        # Define normalized video batch (features), top k similar videos' similarities and their labels sets.
        video_batch_normalized_initializer = tf.placeholder(tf.float32, video_batch.shape)
        video_batch_normalized = tf.Variable(initial_value=video_batch_normalized_initializer, trainable=False,
                                             collections=[], name='video_batch_outer')

        topk_sims_initializer = tf.placeholder(tf.float32, shape=[num_videos, k])
        topk_sims = tf.Variable(initial_value=topk_sims_initializer, collections=[], name='topk_sims')

        topk_labels_initializer = tf.placeholder(tf.bool, shape=[num_videos, k, num_classes])
        topk_labels = tf.Variable(initial_value=topk_labels_initializer, collections=[], name='topk_labels')

        # Generate example queue. Traverse the queue to traverse the data set.
        # Works as the inner loop of finding k-nearest neighbors.
        video_id_batch_inner, video_batch_inner, video_labels_batch_inner, num_frames_batch_inner = (
            get_input_data_tensors(data_pipeline, num_epochs=1, name_scope='inner_loop'))

        # normalization along the last dimension.
        video_batch_inner_normalized = tf.nn.l2_normalize(video_batch_inner, dim=-1,
                                                          name='video_batch_inner_normalized')

        # compute cosine similarities
        similarities = tf.matmul(video_batch_normalized, video_batch_inner_normalized, transpose_b=True,
                                 name='similarities')
        with tf.name_scope('batch_topk'):
            # top k similar videos per video in video_batch_normalized.
            # values and indices are in shape [batch_size, k].
            batch_topk_sims, batch_topk_sim_indices = tf.nn.top_k(similarities, k=k, name='batch_topk')

            batch_topk_labels = tf.gather(video_labels_batch_inner, batch_topk_sim_indices,
                                          name='batch_topk_labels')
        with tf.name_scope('update_topk_sims'):
            # Update topk_sims and labels.
            top2k_video_sims = tf.concat([topk_sims, batch_topk_sims], 1)
            updated_topk_sims, updated_topk_sims_indices = tf.nn.top_k(top2k_video_sims, k=k)
            update_topk_sims_op = tf.assign(topk_sims, updated_topk_sims,
                                            name='update_topk_sims')
        with tf.name_scope('update_topk_labels'):
            top_2k_video_labels = tf.concat([topk_labels, batch_topk_labels], 1, name='top_2k_video_labels')
            flatten_top2k_labels = tf.reshape(top_2k_video_labels, [-1, num_classes])
            idx_inc = tf.expand_dims(tf.range(0, num_videos * 2 * k, 2 * k, dtype=tf.int32), axis=1)
            idx_in_flatten = tf.add(updated_topk_sims_indices, idx_inc)
            update_topk_labels_op = tf.assign(topk_labels, tf.gather(flatten_top2k_labels, idx_in_flatten),
                                              name='update_topk_labels')

        # Update top k similar video sims and labels.
        # To avoid fetching useless data.
        with tf.control_dependencies([global_step_inc_op, update_topk_sims_op, update_topk_labels_op]):
            update_topk_non_op = tf.no_op()

        summary_op = tf.summary.merge_all()
        # Initialization of global and local variables (e.g., queue epoch).
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # A new graph needs a new session. Thus, create one.
    with tf.Session(graph=graph) as sess:
        sess.run(init_op)
        # initialize outer video batch. Current top k similarities are -2.0 (< minimum -1.0).
        sess.run([video_batch_normalized.initializer, topk_sims.initializer, topk_labels.initializer],
                 feed_dict={
                     video_batch_normalized_initializer: video_batch / np.clip(
                         np.linalg.norm(video_batch, axis=-1, keepdims=True), 1e-6, np.PINF),
                     topk_sims_initializer: np.full([num_videos, k], -2.0, dtype=np.float32),
                     topk_labels_initializer: np.zeros([num_videos, k, num_classes], dtype=np.bool)
                 })

        writer = tf.summary.FileWriter(logdir, graph=sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        count = 0
        try:
            while not coord.should_stop():

                if count % 100 == 0:
                    # Run results are numpy arrays. Update topk_sims and tok_video_labels.
                    _, global_step_val, summary = sess.run([update_topk_non_op, global_step, summary_op])

                    writer.add_summary(summary, global_step=global_step_val)
                else:
                    _ = sess.run(update_topk_non_op)

                count += 1

        except tf.errors.OutOfRangeError:
            logging.info('Done the whole data set - found k nearest neighbors.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        final_topk_labels = sess.run(topk_labels)

        sess.close()

    if is_train:
        return None, final_topk_labels[:, 1:]
    else:
        return None, final_topk_labels


def compute_prior_posterior_prob(k_list=[8], smooth_para=1.0, opt_hyper_para=False):
    if (not opt_hyper_para) and (len(k_list) != 1):
        raise ValueError('Only one k is needed. Check your argument.')

    model_dir = FLAGS.model_dir

    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)

    train_data_pattern = FLAGS.train_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern, batch_size=batch_size,
                                       num_readers=num_readers)

    # Step 1. Compute prior probabilities and store the results.
    start_time = time.time()
    sum_labels, accum_num_videos, labels_prior_prob = compute_prior_prob(train_data_pipeline, smooth_para=smooth_para)
    logging.info('Computing prior probability took {} s.'.format(time.time() - start_time))
    save_prior_prob(sum_labels, accum_num_videos, labels_prior_prob, model_dir)

    # Step 2. Compute posterior probabilities, actually likelihood function or sampling distribution.
    # Total number of classes.
    num_classes = reader.num_classes
    range_num_classes = range(num_classes)

    max_k = max(k_list)
    # For each possible class, define a count and counter_count to count.
    # Compute the posterior probability, namely, given a label l, counting the number of training examples that have
    # exactly j (0 <= j <= k) nearest neighbors that have label l and normalizing it.
    # Here, j is considered as a random variable.
    count_list, counter_count_list = [], []
    for k in k_list:
        count_list.append(np.zeros([k + 1, num_classes], dtype=np.float32))
        counter_count_list.append(np.zeros([k + 1, num_classes], dtype=np.float32))

    with tf.Graph().as_default() as g:
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        global_step_inc_op = global_step.assign_add(1)

        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (get_input_data_tensors(
            train_data_pipeline, num_epochs=1, name_scope='outer_loop'))

        tf.summary.scalar('global_step', global_step)

        summary_op = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    sess = tf.Session(graph=g)
    sess.run(init_op)

    writer = tf.summary.FileWriter(model_dir, graph=sess.graph)

    inner_reader = get_reader(model_type, feature_names, feature_sizes)

    # Be cautious to not be blocked by queue.
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tol_num_examples_processed = 0

    try:

        while not coord.should_stop():
            # Run training steps or whatever.
            start_time = time.time()
            video_id_batch_val, video_batch_val, video_labels_batch_val, global_step_val, summary = sess.run(
                [video_id_batch, video_batch, video_labels_batch, global_step_inc_op, summary_op])

            writer.add_summary(summary, global_step=global_step_val)

            logging.info('video_id_batch shape: {}, video_batch shape: {}'.format(video_id_batch_val.shape,
                                                                                  video_batch_val.shape))
            # Smaller batch size and less number of readers.
            _train_data_pipeline = DataPipeline(reader=inner_reader, data_pattern=train_data_pattern,
                                                batch_size=batch_size, num_readers=num_readers)
            # Pass values instead of tensors.
            top_max_k_video_ids, top_max_k_labels = find_k_nearest_neighbors(video_id_batch_val,
                                                                             video_batch_val,
                                                                             _train_data_pipeline,
                                                                             is_train=True,
                                                                             k=max_k)
            logging.info('Finding k nearest neighbors needs {} s.'.format(time.time() - start_time))
            # logging.debug('topk_video_ids: {}\ntopk_labels: {}'.format(topk_video_ids, topk_labels))

            # Update count_list and counter_count_list.
            for idx, k in enumerate(k_list):
                topk_labels = top_max_k_labels[:, :k]
                # batch_size * delta.
                deltas = topk_labels.astype(np.int32).sum(axis=1)
                # Update count and counter_count for each example.
                for delta, video_labels_val in zip(deltas, video_labels_batch_val):
                    inc = video_labels_val.astype(np.float32)
                    count_list[idx][delta, range_num_classes] += inc
                    counter_count_list[idx][delta, range_num_classes] += 1 - inc

                # logging.debug('count: {}\ncounter_count: {}'.format(count_list[idx], counter_count_list[idx]))

            now = time.time()
            tol_num_examples_processed += video_id_batch_val.shape[0]
            logging.info('Batch processing step {}, elapsed {} s, processed {} examples in total'.format(
                global_step_val, now - start_time, tol_num_examples_processed))

    except tf.errors.OutOfRangeError:
        logging.info('Done training -- one epoch limit reached.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    # Save models parameters.
    for k, count, counter_count in zip(k_list, count_list, counter_count_list):
        # Compute posterior probabilities.
        pos_prob_positive = (smooth_para + count) / (smooth_para * (k + 1) + count.sum(axis=0))
        pos_prob_negative = (smooth_para + counter_count) / (smooth_para * (k + 1) + counter_count.sum(axis=0))

        # Write to files for future use.
        save_posterior_prob(count, counter_count, pos_prob_positive, pos_prob_negative, k, model_dir)

    # Output the best k in validate set.
    if opt_hyper_para:
        validate_data_pattern = FLAGS.validate_data_pattern

        validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                              batch_size=batch_size, num_readers=num_readers)
        validate_ids, validate_data, validate_labels, _ = random_sample(0.1, mask=(True, True, True, False),
                                                                        data_pipeline=validate_data_pipeline)
        best_k = None
        best_validate_gap = np.NINF
        for k in k_list:
            pred_obj = Predict(train_data_pipeline, model_dir, k=k)
            num_validate_videos = validate_data.shape[0]
            if num_validate_videos >= 51200:
                one_third = num_validate_videos / 3
                two_third = num_validate_videos * 2 / 3
                predictions_1 = pred_obj.make_batch_predictions(validate_ids[:one_third],
                                                                validate_data[:one_third])
                validate_gap_1 = calculate_gap(predictions_1, validate_labels[:one_third])

                predictions_2 = pred_obj.make_batch_predictions(validate_ids[one_third:two_third],
                                                                validate_data[one_third:two_third])
                validate_gap_2 = calculate_gap(predictions_2, validate_labels[one_third:two_third])

                predictions_3 = pred_obj.make_batch_predictions(validate_ids[two_third:],
                                                                validate_data[two_third:])
                validate_gap_3 = calculate_gap(predictions_3, validate_labels[two_third:])

                validate_gap = (validate_gap_1 + validate_gap_2 + validate_gap_3) / 3.0
            else:
                predictions = pred_obj.make_batch_predictions(validate_ids, validate_data)
                validate_gap = calculate_gap(predictions, validate_labels)

            logging.info('k: {}, validate gap: {}'.format(k, validate_gap))
            if validate_gap > best_validate_gap:
                best_k = k
                best_validate_gap = validate_gap
        print('Best k: {}, with validate gap {}'.format(best_k, best_validate_gap))


class Predict(object):
    def __init__(self, train_data_pipeline, model_dir, k=8):
        """
        :param model_dir: The dir where model parameters are stored.
        :param k: The k in ml-knn. 
        """
        self.train_data_pipeline = train_data_pipeline
        self.model_dir = model_dir
        self.k = k

        reader = train_data_pipeline.reader
        # Total number of classes.
        num_classes = reader.num_classes
        self.range_num_classes = range(num_classes)

        # Load prior and posterior probabilities.
        self.sum_labels, self.accum_num_videos, self.labels_prior_prob = restore_prior_prob(folder=self.model_dir)
        self.count, self.counter_count, self.pos_prob_positive, self.pos_prob_negative = restore_posterior_prob(
            self.k, folder=self.model_dir)

    def make_batch_predictions(self, video_id_batch_val, video_batch_val):
        """
        Make predictions for a batch of videos.
        Return:
            Predictions probabilities as a Numpy array.
        """
        topk_video_ids, topk_labels = find_k_nearest_neighbors(video_id_batch_val,
                                                               video_batch_val, self.train_data_pipeline,
                                                               is_train=False, k=self.k)

        logging.debug('topk_video_ids: {}\ntopk_labels: {}'.format(topk_video_ids, topk_labels))

        # batch_size * delta.
        deltas = topk_labels.astype(np.int32).sum(axis=1)

        batch_predictions_prob = []
        for delta in deltas:
            positive_prob_numerator = np.multiply(self.labels_prior_prob,
                                                  self.pos_prob_positive[delta, self.range_num_classes])
            negative_prob_numerator = np.multiply(1.0 - self.labels_prior_prob,
                                                  self.pos_prob_negative[delta, self.range_num_classes])
            # predictions = positive_prob_numerator > negative_prob_numerator

            batch_predictions_prob.append(
                np.true_divide(positive_prob_numerator, positive_prob_numerator + negative_prob_numerator))

        return np.array(batch_predictions_prob, dtype=np.float32)

    def make_predictions(self, test_data_pipeline, output_file_loc, top_k=20):
        """
        Make predictions.
        :param test_data_pipeline
        :param output_file_loc: The file to which predictions should be written to. Supports gcloud file.
        :param top_k: See FLAGS.top_k.
        """
        with tf.Graph().as_default() as g:
            video_id_batch, video_batch, video_labels_batch, num_frames_batch = get_input_data_tensors(
                test_data_pipeline, num_epochs=1, name_scope='test_input')

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        with tf.Session(graph=g) as sess, gfile.Open(output_file_loc, "w+") as out_file:
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
                    video_id_batch_val, video_batch_val = sess.run(
                        [video_id_batch, video_batch])

                    logging.debug(
                        'video_id_batch_val: {}\nvideo_batch_val: {}'.format(video_id_batch_val, video_batch_val))

                    # Pass values instead of tensors.
                    batch_predictions_prob = self.make_batch_predictions(video_id_batch_val, video_batch_val)

                    # Write batch predictions to files.
                    for line in format_lines(video_id_batch_val, batch_predictions_prob, top_k):
                        out_file.write(line)
                    out_file.flush()

                    now = time.time()
                    processing_count += 1
                    num_examples_processed += video_id_batch_val.shape[0]
                    print('Batch processing step {}, elapsed {} seconds, processed {} examples in total'.format(
                        processing_count, now - start_time, num_examples_processed))

            except tf.errors.OutOfRangeError:
                logging.info('Done with inference. The predictions were written to {}'.format(output_file_loc))
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)

            sess.close()
            out_file.close()


def main(unused_argv):
    logging.set_verbosity(logging.INFO)

    is_train = FLAGS.is_train
    is_tuning_hyper_para = FLAGS.is_tuning_hyper_para

    if is_train:
        ks = FLAGS.ks
        k_list = [int(k.strip()) for k in ks.split(',')]

        compute_prior_posterior_prob(k_list=k_list, opt_hyper_para=is_tuning_hyper_para)
    else:
        model_dir = FLAGS.model_dir
        k = FLAGS.pred_k

        output_file = FLAGS.output_file
        top_k = FLAGS.top_k

        model_type = FLAGS.model_type
        batch_size = FLAGS.batch_size
        num_readers = FLAGS.num_readers
        feature_names = FLAGS.feature_names
        feature_sizes = FLAGS.feature_sizes

        test_data_pattern = FLAGS.test_data_pattern
        reader = get_reader(model_type, feature_names, feature_sizes)
        test_data_pipeline = DataPipeline(reader=reader, data_pattern=test_data_pattern,
                                          batch_size=batch_size, num_readers=num_readers)

        train_data_pattern = FLAGS.train_data_pattern
        inner_reader = get_reader(model_type, feature_names, feature_sizes)
        train_data_pipeline = DataPipeline(reader=inner_reader, data_pattern=train_data_pattern,
                                           batch_size=batch_size, num_readers=num_readers)

        pred_obj = Predict(train_data_pipeline, model_dir, k=k)
        pred_obj.make_predictions(test_data_pipeline, output_file, top_k=top_k)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train/trainaQ.tfrecord',
                        'File glob for the training data set.')

    flags.DEFINE_string('validate_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/validate/validateo*.tfrecord',
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    flags.DEFINE_string('test_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/test/test4*.tfrecord',
                        'Test data pattern, to be specified when making predictions.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_audio', 'Features to be used, separated by ,.')
    # 1024, 128
    flags.DEFINE_string('feature_sizes', '128', 'Dimensions of features to be used, separated by ,.')

    # Set by the memory limit. Larger values will reduce data passing times. For debug, use a small value, e.g., 1024.
    flags.DEFINE_integer('batch_size', 20480, 'Size of batch processing.')
    # For debug, use a single reader.
    flags.DEFINE_integer('num_readers', 3, 'Number of readers to form a batch.')

    # To find the best k in validate set, set it as True.
    # After getting the best k, setting this as False when using train and validate set to re-train the model.
    flags.DEFINE_boolean('is_tuning_hyper_para', False,
                         'Boolean variable indicating whether to perform hyper-parameter tuning.')

    # Separated by ,.
    flags.DEFINE_string('ks', '16', 'k nearest neighbors to tune.')

    flags.DEFINE_integer('pred_k', 16, 'The k nearest neighbor to make predictions.')

    flags.DEFINE_string('model_dir', '/tmp/ml-knn',
                        'The directory to which prior and posterior probabilities should be written.')

    flags.DEFINE_boolean('is_train', False, 'Boolean variable to indicate training or test.')

    flags.DEFINE_string('output_file', '/tmp/ml-knn/predictions.csv', 'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
