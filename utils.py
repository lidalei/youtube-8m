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
    folder = dirname(abspath(getsourcefile(lambda: 0)))
    with open(path_join(folder, 'constants/partial_data_features_mean.pickle'), 'rb') as pickle_file:
        try:
            features_mean = pickle.load(pickle_file)
        except:
            features_mean = pickle.load(pickle_file, fix_imports=True, encoding='latin1')

    # features_mean = {'mean_rgb': np.array([1024 floats]), 'mean_audio': np.array([128 floats])}
    return features_mean


DataPipeline = namedtuple('DataPipeline', ['reader', 'data_pattern', 'batch_size', 'num_readers'])


def random_sample(sample_ratio, mask=(True, True, True, True), data_pipeline=None):
    """
    Randomly sample sample_ratio examples from data that specified reader by and data_pattern.
    Args:
        sample_ratio: The ratio of examples to be sampled. Range (0, 1.0].
        mask: To keep which part or parts of video information, namely, id, features, labels and num of frames.
        data_pipeline: A namedtuple consisting of the following elements. reader, See readers.py.
            data_pattern, File Glob of data.
            batch_size, The size of a batch. The last a few batches might have less examples.
            num_readers, How many IO threads to enqueue example queue.
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

    # Create the graph to traverse all data once.
    with tf.Graph().as_default() as graph:
        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(data_pipeline, num_epochs=1, name_scope='rnd_sample'))

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

    # Find num_centers_ratio of the total examples.
    accum_sample = [[]] * 4
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

    a_sample = [None] * 4

    for idx, indicator in enumerate(mask):
        if indicator:
            a_sample[idx] = np.concatenate(accum_sample[idx], axis=0)

    logging.info('The sample result has shape {}.'.format([e.shape if e is not None else e for e in a_sample]))

    return a_sample


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
        logging.info("Number of input files: {}".format(len(files)))
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

# if __name__ == '__main__':
#     features_mean = partial_data_features_mean()
#     print(features_mean)
