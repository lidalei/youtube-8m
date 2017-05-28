# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of util functions for training and evaluating.
"""

import numpy as np
import tensorflow as tf
from tensorflow import logging, gfile
import pickle
from os.path import join as path_join
from eval_util import calculate_gap

# Used to locate constants dir.
from inspect import getsourcefile
from os.path import abspath
from os.path import dirname

from collections import namedtuple


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary


def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
    """Add the global_step summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    global_step_info_dict: a dictionary of the evaluation metrics calculated for
      a mini-batch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
    this_hit_at_one = global_step_info_dict["hit_at_one"]
    this_perr = global_step_info_dict["perr"]
    this_loss = global_step_info_dict["loss"]
    examples_per_second = global_step_info_dict.get("examples_per_second", -1)

    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
        global_step_val)

    if examples_per_second != -1:
        summary_writer.add_summary(
            MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                        examples_per_second), global_step_val)

    summary_writer.flush()
    info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
            "| Examples_per_sec: {4:.3f}").format(
        global_step_val, this_hit_at_one, this_perr, this_loss,
        examples_per_second)
    return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
    """Add the epoch summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    epoch_info_dict: a dictionary of the evaluation metrics calculated for the
      whole epoch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
    epoch_id = epoch_info_dict["epoch_id"]
    avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
    avg_perr = epoch_info_dict["avg_perr"]
    avg_loss = epoch_info_dict["avg_loss"]
    aps = epoch_info_dict["aps"]
    gap = epoch_info_dict["gap"]
    mean_ap = np.mean(aps)

    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_MAP", mean_ap),
        global_step_val)
    summary_writer.add_summary(
        MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
        global_step_val)
    summary_writer.flush()

    info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
            "| MAP: {3:.3f} | GAP: {4:.3f} | Avg_Loss: {5:3f}").format(
        epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss)
    return info


def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):
    """Extract the list of feature names and the dimensionality of each feature
     from string of comma separated values.

  Args:
    feature_names: string containing comma separated list of feature names
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the feature names and list of the dimensionality of each feature.
    Elements in the first/second list are strings/integers.
  """
    list_of_feature_names = [
        feature_names.strip() for feature_names in feature_names.split(',')]
    list_of_feature_sizes = [
        int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
    if len(list_of_feature_names) != len(list_of_feature_sizes):
        logging.error("length of the feature names (={}) != length of feature sizes (={})"
                      .format(len(list_of_feature_names), len(list_of_feature_sizes)))

    return list_of_feature_names, list_of_feature_sizes


def partial_data_features_mean():
    """
    Load approximate features mean for computing variance with numerical stability.
    """
    folder = dirname(abspath(getsourcefile(lambda: 0)))
    with open(path_join(folder, 'constants/partial_data_features_mean.pickle'), 'rb') as pickle_file:
        try:
            features_mean = pickle.load(pickle_file)
        except:
            features_mean = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    # features_mean = {'mean_rgb': np.array([1024 floats]), 'mean_audio': np.array([128 floats])}
    return features_mean


def load_sum_labels():
    """
    Load number of videos per label in train data.
    """
    folder = dirname(abspath(getsourcefile(lambda: 0)))
    with open(path_join(folder, 'ml-knn-model/sum_labels.pickle'), 'rb') as pickle_file:
        try:
            sum_labels = pickle.load(pickle_file)
        except:
            sum_labels = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    return sum_labels


def load_features_mean_var(reader):
    """
    Load features mean in train data.
    """
    feature_names = reader.feature_names

    folder = dirname(abspath(getsourcefile(lambda: 0)))
    with open(path_join(folder, 'constants/all_data_features_mean.pickle'), 'rb') as f:
        # {'mean_rgb': features_mean[:1024], 'mean_audio': features_mean[1024:]}
        features_mean = pickle.load(f)
        mean_tuple = [features_mean[feature] for feature in feature_names]
        mean = np.concatenate(mean_tuple, axis=0)

    with open(path_join(folder, 'constants/all_data_features_var.pickle'), 'rb') as f:
        # {'mean_rgb': features_var[:1024], 'mean_audio': features_var[1024:]}
        features_var = pickle.load(f)
        var_tuple = [features_var[feature] for feature in feature_names]
        var = np.concatenate(var_tuple, axis=0)

    return mean, var


def save_prior_prob(sum_labels, accum_num_videos, labels_prior_prob, folder=''):
    # Create the directory if it does not exist.
    if not tf.gfile.Exists(folder):
        try:
            tf.gfile.MakeDirs(folder)
        except tf.errors.OpError:
            logging.error("Failed to create dir {}. Please manually create it.".format(folder))

    with open(path_join(folder, 'sum_labels.pickle'), 'wb') as pickle_file:
        pickle.dump(sum_labels, pickle_file)

    with open(path_join(folder, 'accum_num_videos.pickle'), 'wb') as pickle_file:
        pickle.dump(accum_num_videos, pickle_file)

    with open(path_join(folder, 'labels_prior_prob.pickle'), 'wb') as pickle_file:
        pickle.dump(labels_prior_prob, pickle_file)


def restore_prior_prob(folder=''):
    with open(path_join(folder, 'sum_labels.pickle'), 'rb') as pickle_file:
        try:
            sum_labels = pickle.load(pickle_file)
        except:
            sum_labels = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'accum_num_videos.pickle'), 'rb') as pickle_file:
        try:
            accum_num_videos = pickle.load(pickle_file)
        except:
            accum_num_videos = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'labels_prior_prob.pickle'), 'rb') as pickle_file:
        try:
            labels_prior_prob = pickle.load(pickle_file)
        except:
            labels_prior_prob = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    return sum_labels, accum_num_videos, labels_prior_prob


def save_posterior_prob(count, counter_count, pos_prob_positive, pos_prob_negative, k, folder=''):
    # Create the directory if it does not exist.
    if not tf.gfile.Exists(folder):
        try:
            tf.gfile.MakeDirs(folder)
        except tf.errors.OpError:
            logging.error("Failed to create dir {}. Please manually create it.".format(folder))

    with open(path_join(folder, 'count_{}.pickle'.format(k)), 'wb') as pickle_file:
        pickle.dump(count, pickle_file)

    with open(path_join(folder, 'counter_count_{}.pickle'.format(k)), 'wb') as pickle_file:
        pickle.dump(counter_count, pickle_file)

    with open(path_join(folder, 'pos_prob_positive_{}.pickle'.format(k)), 'wb') as pickle_file:
        pickle.dump(pos_prob_positive, pickle_file)

    with open(path_join(folder, 'pos_prob_negative_{}.pickle'.format(k)), 'wb') as pickle_file:
        pickle.dump(pos_prob_negative, pickle_file)


def restore_posterior_prob(k, folder=''):
    with open(path_join(folder, 'count_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            count = pickle.load(pickle_file)
        except:
            count = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'counter_count_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            counter_count = pickle.load(pickle_file)
        except:
            counter_count = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'pos_prob_positive_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            pos_prob_positive = pickle.load(pickle_file)
        except:
            pos_prob_positive = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    with open(path_join(folder, 'pos_prob_negative_{}.pickle'.format(k)), 'rb') as pickle_file:
        try:
            pos_prob_negative = pickle.load(pickle_file)
        except:
            pos_prob_negative = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    return count, counter_count, pos_prob_positive, pos_prob_negative


DataPipeline = namedtuple('DataPipeline', ['reader', 'data_pattern', 'batch_size', 'num_readers'])


def random_sample(sample_ratio, mask=(True, True, True, True), data_pipeline=None, name_scope='rnd_sample'):
    """
    Randomly sample sample_ratio examples from data that specified reader by and data_pattern.
    Args:
        sample_ratio: The ratio of examples to be sampled. Range (0, 1.0].
        mask: To keep which part or parts of video information, namely, id, features, labels and num of frames.
        data_pipeline: A namedtuple consisting of the following elements. reader, See readers.py.
            data_pattern, File Glob of data.
            batch_size, The size of a batch. The last a few batches might have less examples.
            num_readers, How many IO threads to enqueue example queue.
        name_scope: To distinguish from other tf graph part.
    Returns:
        Roughly the ratio of examples will be returned. If a part is not demanded, the corresponding part is None.
    Raises:
        ValueError, if sample_ratio is not larger than 0.0 or greater than 1.0. Or mask has not exactly 4 elements. Or
            mask does not have one True.
    """
    if (sample_ratio <= 0.0) or (sample_ratio > 1.0):
        raise ValueError('Invalid sample ratio: {}'.format(sample_ratio))

    if (len(mask) != 4) or all(not e for e in mask):
        raise ValueError('Invalid mask argument, require a tuple with exactly 4 boolean values and at least one True.')

    logging.info('Enter random_sample...')

    # Create the graph to traverse all data once.
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(data_pipeline, num_epochs=1, name_scope=name_scope))

        num_batch_videos = tf.shape(video_batch)[0]
        rnd_nums = tf.random_uniform([num_batch_videos])
        sample_mask = tf.less_equal(rnd_nums, sample_ratio)

        if mask[0]:
            video_id_partial_sample = tf.boolean_mask(video_id_batch, sample_mask)
        else:
            video_id_partial_sample = tf.no_op('no_video_id')

        if mask[1]:
            video_partial_sample = tf.boolean_mask(video_batch, sample_mask)
        else:
            video_partial_sample = tf.no_op('no_video_features')

        if mask[2]:
            video_labels_partial_sample = tf.boolean_mask(video_labels_batch, sample_mask)
        else:
            video_labels_partial_sample = tf.no_op('no_video_labels')

        if mask[3]:
            num_frames_partial_sample = tf.boolean_mask(num_frames_batch, sample_mask)
        else:
            num_frames_partial_sample = tf.no_op('no_video_num_frames')

        partial_sample = [video_id_partial_sample, video_partial_sample,
                          video_labels_partial_sample, num_frames_partial_sample]

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    graph.finalize()

    # Create a session for running operations in the Graph.
    sess = tf.Session(graph=graph)
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Write graph definition.
    # output_dir = FLAGS.output_dir
    # tf.train.write_graph(sess.graph, path_join(output_dir, 'rnd_sample'),
    #                      'sample_{}.pb'.format(int(time.time())), as_text=False)

    # Find num_centers_ratio of the total examples. Cannot use [[]] * 4, for Python will treat copy reference only.
    accum_sample = [[], [], [], []]
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Sample once.
            partial_sample_val = sess.run(partial_sample)

            # bool_mask might return empty numpy array.
            for idx, indicator in enumerate(mask):
                if indicator and (partial_sample_val[idx].size > 0):
                    accum_sample[idx].append(partial_sample_val[idx])

    except tf.errors.OutOfRangeError:
        logging.info('Done sampling -- one epoch finished.')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

    a_sample = [None, None, None, None]

    for idx, indicator in enumerate(mask):
        if indicator:
            a_sample[idx] = np.concatenate(accum_sample[idx], axis=0)

    logging.info('The sample result has shape {}.'.format([e.shape if e is not None else e for e in a_sample]))

    return a_sample


def compute_data_mean_var(data_pipeline=None, tr_data_fn=None, tr_data_paras=None):
    """
    Compute mean and variance per feature (column) and mean of each label.

    Note:
        From Spark StandardScaler documentation.
        * The "unit std" is computed using the corrected sample standard deviation
        * (https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation),
        * which is computed as the square root of the unbiased sample variance.
    Args:
        data_pipeline: A namedtuple consisting of the following elements.
            reader, video-level features reader or frame-level features reader.
            data_pattern, File Glob of data set.
            batch_size, How many examples to handle per time.
            num_readers, How many IO threads to prefetch examples.
        tr_data_fn: a function that transforms input data.
        tr_data_paras: Extra parameters needed to call tr_data_fn.

    Returns:
        Mean values of each feature column as a numpy array of rank 1.
        Standard deviations of each feature column as a numpy array of rank 1.
    """
    reader = data_pipeline.reader
    feature_names = reader.feature_names
    feature_sizes = reader.feature_sizes

    logging.info('Computing mean and std of {} features with sizes {}.'.format(
        feature_names, feature_sizes))

    # features_mean on partial data (600 + train files).
    # Note, can only be used locally, not in google cloud.
    try:
        par_features_mean = partial_data_features_mean()
    except IOError:
        logging.error('Cannot locate partial_data_features_mean data file.')
        par_features_mean = None

    # Total number of features.
    features_size = sum(feature_sizes)
    if par_features_mean is None:
        approx_raw_features_mean = np.zeros([features_size], dtype=np.float32)
    else:
        approx_raw_features_mean = np.concatenate([par_features_mean[e] for e in feature_names])

    # Transform may change features size.
    if tr_data_fn is not None:
        if tr_data_paras is None:
            tr_data_paras = {}
        else:
            if ('reshape' in tr_data_paras) and (tr_data_paras['reshape'] is True):
                feature_size = tr_data_paras['size']
                logging.warn('Data transform changes the features size to {}.'.format(feature_size))

        with tf.Graph().as_default() as g:
            approx_features_mean_op = tr_data_fn(approx_raw_features_mean, **tr_data_paras)
        g.finalize()
        sess = tf.Session(graph=g)
        approx_features_mean = sess.run(approx_features_mean_op)
        sess.close()
    else:
        approx_features_mean = approx_raw_features_mean

    # numerical stability with
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data.
    # Create the graph to traverse all data once.
    with tf.Graph().as_default() as graph:
        id_batch, raw_features_batch, labels_batch, num_frames_batch = (
            get_input_data_tensors(data_pipeline, num_epochs=1, name_scope='features_mean_std'))

        example_count = tf.Variable(initial_value=0.0, name='example_count')
        features_sum = tf.Variable(initial_value=tf.zeros([features_size]), name='features_sum')
        features_squared_sum = tf.Variable(initial_value=tf.zeros([features_size]), name='features_squared_sum')

        if tr_data_fn:
            features_batch = tr_data_fn(raw_features_batch, **tr_data_paras)
        else:
            features_batch = tf.identity(raw_features_batch)

        # Compute shift features sum and squared sum.
        shift = tf.constant(approx_features_mean, dtype=tf.float32, name='shift')
        shifted_features_batch = tf.subtract(features_batch, shift)

        batch_example_count = tf.cast(tf.shape(shifted_features_batch)[0], tf.float32)
        batch_features_sum = tf.reduce_sum(shifted_features_batch, axis=0, name='batch_features_sum')
        batch_features_squared_sum = tf.reduce_sum(tf.square(shifted_features_batch), axis=0,
                                                   name='batch_features_squared_sum')

        update_example_count = tf.assign_add(example_count, batch_example_count)
        update_features_sum = tf.assign_add(features_sum, batch_features_sum)
        update_features_squared_sum = tf.assign_add(features_squared_sum, batch_features_squared_sum)

        with tf.control_dependencies(
                [update_example_count, update_features_sum, update_features_squared_sum]):
            update_accum_non_op = tf.no_op()

        # Define final results. To be run after all data have been handled.
        features_mean = tf.add(tf.divide(features_sum, example_count), shift, name='features_mean')
        # Corrected sample standard deviation.
        features_variance = tf.divide(
            tf.subtract(features_squared_sum, tf.scalar_mul(example_count, tf.square(features_mean))),
            tf.subtract(example_count, 1.0), name='features_var')
        # features_std = tf.sqrt(features_variance, name='features_std')

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
    features_mean_val, features_var_val = sess.run([features_mean, features_variance])
    sess.close()

    return features_mean_val, features_var_val


def get_input_data_tensors(data_pipeline, shuffle=False, num_epochs=1, name_scope='input'):
    """
    Args:
        data_pipeline: DataPipeline tuple.
        shuffle: Boolean argument indicating whether shuffle examples.
        num_epochs: How many passes can be gone through the data.
        name_scope: For better visualization and organization.
    Returns: video_id_batch, video_batch, video_labels_batch, num_frames_batch

    """
    reader, data_pattern, batch_size, num_readers = data_pipeline
    return _get_input_data_tensors(reader=reader, data_pattern=data_pattern, batch_size=batch_size,
                                   num_readers=num_readers, shuffle=shuffle, num_epochs=num_epochs,
                                   name_scope=name_scope)


def _get_input_data_tensors(reader=None, data_pattern=None, batch_size=2048, num_readers=2, shuffle=False,
                            num_epochs=1, name_scope='input'):
    """Creates the section of the graph which reads the input data.

    Similar to the same-name function in train.py.
    Args:
        reader: A class which parses the input data.
        data_pattern: A 'glob' style path to the data files.
        batch_size: How many examples to process at a time.
        num_readers: How many I/O threads to use.
        shuffle: Boolean argument indicating whether shuffle examples.
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
        logging.info("Number of input files: {} within {}".format(len(files), name_scope))
        # Pass test data once. Thus, num_epochs is set as 1.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=shuffle, capacity=128)
        examples_and_labels = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

        # In shuffle_batch_join,
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        if shuffle:
            capacity = (num_readers + 1) * batch_size + 2048
            video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
                tf.train.shuffle_batch_join(examples_and_labels,
                                            batch_size=batch_size,
                                            capacity=capacity,
                                            min_after_dequeue=batch_size,
                                            allow_smaller_final_batch=True,
                                            enqueue_many=True))
        else:
            capacity = num_readers * batch_size + 2048
            video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
                tf.train.batch_join(examples_and_labels,
                                    batch_size=batch_size,
                                    capacity=capacity,
                                    allow_smaller_final_batch=True,
                                    enqueue_many=True))

        return video_id_batch, video_batch, video_labels_batch, num_frames_batch


def gap_fn(predictions=None, labels=None):
    """
    Make predictions and labels to be specified explicitly.
    :param predictions: Model output.
    :param labels: Targets or ground truth.
    :return: GAP - global average precision.
    """
    return calculate_gap(predictions, labels)


# if __name__ == '__main__':
    # features_mean = partial_data_features_mean()
    # print(features_mean)

    # sum_labels = load_sum_labels()
    # print(sum_labels)
    # from readers import get_reader
    # reader = get_reader('video', 'mean_rgb,mean_audio', '1024,128')
    # #
    # train_data_pipeline = DataPipeline(reader=reader, data_pattern='yt8m/video_level/*/*.tfrecord',
    #                                    batch_size=4096, num_readers=1)
    #
    # features_mean, features_var = compute_data_mean_var(data_pipeline=train_data_pipeline,
    #                                                     tr_data_fn=None, tr_data_paras=None)
    #
    # with open('constants/all_data_features_mean.pickle', 'wb') as f:
    #     pickle.dump({'mean_rgb': features_mean[:1024], 'mean_audio': features_mean[1024:]}, f)
    #
    # with open('constants/all_data_features_var.pickle', 'wb') as f:
    #     pickle.dump({'mean_rgb': features_var[:1024], 'mean_audio': features_var[1024:]}, f)

    # mean, var = load_features_mean_var(reader)
    # pass
