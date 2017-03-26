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
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

import eval_util
import export_model
import losses
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Dataset flags.
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string(
        "train_data_pattern", "",
        "File glob for the training dataset. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                     "to use for training.")
    flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

    # Model flags.
    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_string(
        "model", "LogisticModel",
        "Which architecture to use for the model. Models are defined "
        "in models.py.")
    flags.DEFINE_bool(
        "start_new_model", False,
        "If set, this will not resume from a checkpoint and will instead create a"
        " new model instance.")

    # Training flags.
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float(
        "regularization_penalty", 1,
        "How much weight to give to the regularization loss (the label loss has "
        "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.01,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.95,
                       "Learning rate decay factor to be applied every "
                       "learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                       "Multiply current learning rate by learning_rate_decay "
                       "every learning_rate_decay_examples.")
    flags.DEFINE_integer("num_epochs", 5,
                         "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer("max_steps", None,
                         "The maximum number of iterations of the training loop.")
    flags.DEFINE_integer("export_model_steps", 1000,
                         "The period, in number of steps, with which the model "
                         "is exported for batch prediction.")

    # Other flags.
    flags.DEFINE_integer("num_readers", 8,
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    flags.DEFINE_bool("log_device_placement", False,
                      "Whether to write the device on which every op will run into the logs on startup.")


def validate_class_name(flag_value, category, modules, expected_superclass):
    """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
    candidates = [getattr(module, flag_value, None) for module in modules]
    for candidate in candidates:
        if not candidate:
            continue
        if not issubclass(candidate, expected_superclass):
            raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                                   (category, flag_value,
                                    expected_superclass.__name__))
        return True
    raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))


def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
    """Creates the section of the graph which reads the training data.

    Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

    Returns:
    A tuple containing the video ids tensor, features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

    Raises:
    IOError: If no files matching the given pattern were found.
    """
    logging.info("Using batch size of {} for training.".format(batch_size))
    with tf.name_scope("train_input"):
        # tf.train.match_filenames_once returns a tf variable.
        # tf.gfile.Glob(FLAGS.file_pattern) returns the list of files (dirs) matched.
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find training files. data_pattern='{}'.".format(data_pattern))
        logging.info("Number of training files: {}.".format(len(files)))
        # Create a file name queue with randomly shuffled orders.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=True)
        # list of (video_ids, features, labels, padding)
        training_data = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

        """

        min_after_dequeue: Minimum number elements in the queue after a dequeue,
        used to ensure a level of mixing of elements.

        allow_smaller_final_batch: If True, allow the final batch to be smaller
        if there are insufficient items left in the queue.

        If enqueue_many is False, each tensors_list[i] is assumed to represent a single example.
        An input tensor with shape [x, y, z] will be output as a tensor with shape [batch_size, x, y, z].

        If enqueue_many is True, tensors_list[i] is assumed to represent a batch of examples,
        where the first dimension is indexed by example, and all members of tensors_list[i] should
        have the same size in the first dimension. If an input tensor has shape [*, x, y, z],
        the output will have shape [batch_size, x, y, z].
        """
        # Returns a tensor with shape [None, ], where None represents the actual batch_size is not fixed.
        # Any graph that assumes fixed batch_size will fail.
        return tf.train.shuffle_batch_join(
            training_data,
            batch_size=batch_size,
            capacity=FLAGS.batch_size * 5,
            min_after_dequeue=FLAGS.batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True)


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    # (if None) === False, added by Sophie.
    # (e for e in iterable) returns a generator, which doesn't generate elements until being invoked.
    # return next(a for a in modules if a is not None)
    return next(a for a in modules if a)


def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
    """Creates the Tensorflow graph.

      This will only be called once in the life of
      a training model, because after the graph is created the model will be
      restored from a meta graph file rather than being recreated.

      Args:
        reader: The data file reader. It should inherit from BaseReader.
        model: The core model (e.g. logistic or neural net). It should inherit from BaseModel.
        train_data_pattern: glob path to the training data files.
        label_loss_fn: What kind of loss to apply to the model. It should inherit
                    from BaseLoss.
        batch_size: How many examples to process at a time.
        base_learning_rate: What learning rate to initialize the optimizer with.
        optimizer_class: Which optimization algorithm to use.
        clip_gradient_norm: Magnitude of the gradient to clip to.
        regularization_penalty: How much weight to give the regularization loss
                                compared to the label loss.
        num_readers: How many threads to use for I/O operations.
        num_epochs: How many passes to make over the data. 'None' means an
                    unlimited number of passes.
      """

    global_step = tf.Variable(0, trainable=False, name="global_step")

    # exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None).
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps).
    # Treat gradient descent as stochastic gradient descent using global_step = global_step (batch) * batch_size.
    # staircase=True decays the learning rate at discrete intervals.
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # create optimizer with decayed learning rate.
    optimizer = optimizer_class(learning_rate)

    # Get training data.
    # num_frames = [1, 1, ..., 1] whose shape is [batch_size].
    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_data_tensors(
            reader,
            train_data_pattern,
            batch_size=batch_size,
            num_readers=num_readers,
            num_epochs=num_epochs))
    tf.summary.histogram("model/input_raw", model_input_raw)

    # dimension of features, starting from zero, thus minus 1.
    feature_dim = len(model_input_raw.get_shape()) - 1
    # When making predictions (export_model.py), normalization is used, too.
    # normalize the features (the feature_dim, i.e., the last dimension) of each instance.
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

    with tf.name_scope("model"):
        # result is model output, batch_size * num_classes.
        result = model.create_model(
            model_input,
            num_frames=num_frames,
            vocab_size=reader.num_classes,
            labels=labels_batch)

        for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

        # Compute prediction loss.
        predictions = result["predictions"]
        if "loss" in result.keys():
            label_loss = result["loss"]
        else:
            label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)
        tf.summary.scalar("label_loss", label_loss)

        if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
        else:
            reg_loss = tf.constant(0.0)
        # reg_losses is a list of loss variables.
        reg_losses = tf.losses.get_regularization_losses()
        if reg_losses:
            # add_n adds all input tensors element-wise.
            reg_loss += tf.add_n(reg_losses)

        if regularization_penalty != 0:
            tf.summary.scalar("reg_loss", reg_loss)

        # Adds update_ops (e.g., moving average updates in batch normalization) as
        # a dependency to the train_op.
        # TODO, not understand.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if "update_ops" in result.keys():
            update_ops += result["update_ops"]
        if update_ops:
            with tf.control_dependencies(update_ops):
                barrier = tf.no_op(name="gradient_barrier")
                with tf.control_dependencies([barrier]):
                    label_loss = tf.identity(label_loss)

        # Incorporate the L2 weight penalties etc.
        final_loss = regularization_penalty * reg_loss + label_loss
        # Build training op.
        train_op = slim.learning.create_train_op(
            final_loss,
            optimizer,
            global_step=global_step,
            clip_gradient_norm=clip_gradient_norm)

        tf.add_to_collection("global_step", global_step)
        tf.add_to_collection("loss", label_loss)
        tf.add_to_collection("predictions", predictions)
        tf.add_to_collection("input_batch_raw", model_input_raw)
        tf.add_to_collection("input_batch", model_input)
        tf.add_to_collection("num_frames", num_frames)
        tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
        tf.add_to_collection("train_op", train_op)


class Trainer(object):
    """A Trainer to train a Tensorflow graph."""

    def __init__(self, cluster, task, train_dir, model, reader, model_exporter,
                 log_device_placement=True, max_steps=None,
                 export_model_steps=1000):
        """"Creates a Trainer.

        Args:
          cluster: A tf.train.ClusterSpec if the execution is distributed.
            None otherwise.
          task: A TaskSpec describing the job type and the task index.
          export_model_steps: Every export_model_steps to store the trained model.
        """

        self.cluster = cluster
        self.task = task
        self.is_master = (task.type == "master" and task.index == 0)
        self.train_dir = train_dir
        self.config = tf.ConfigProto(log_device_placement=log_device_placement)
        self.model = model
        self.reader = reader
        self.model_exporter = model_exporter
        self.max_steps = max_steps
        self.max_steps_reached = False
        self.export_model_steps = export_model_steps
        self.last_model_export_step = 0

        # StandardError only exists in Python 2, added by Sophie
        # The reason why is_master is False is task.index > 0.
        # This is useless. Always False.
        # if self.is_master and self.task.index > 0:
        #     raise StandardError("{}: Only one replica of master expected".format(task_as_string(self.task)))

    def run(self, start_new_model=False):
        """Performs training on the currently defined Tensorflow graph.

        Returns:
          A tuple of the training Hit@1 and the training PERR.
        """
        if self.is_master and start_new_model:
            # Training process is only recorded in master node.
            # Remove training directory. The function invoked will handle non-existing case.
            self.remove_training_directory(self.train_dir)

        target, device_fn = self.start_server_if_distributed()
        # The full path to the latest checkpoint or None if no checkpoint was found or start_new_model or ....
        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

        # Recover graph or build a new one.
        with tf.Graph().as_default() as graph:
            if meta_filename:
                saver = self.recover_model(meta_filename)

            with tf.device(device_fn):
                if not meta_filename:
                    saver = self.build_model(self.model, self.reader)

                global_step = tf.get_collection("global_step")[0]
                loss = tf.get_collection("loss")[0]
                predictions = tf.get_collection("predictions")[0]
                labels = tf.get_collection("labels")[0]
                train_op = tf.get_collection("train_op")[0]
                init_op = tf.global_variables_initializer()

        # A training helper that checkpoints models and computes summaries.
        # Supervisor is a small wrapper around a Coordinator, a Saver, and a SessionManager
        # that takes care of common needs of TensorFlow training programs.
        # https://www.tensorflow.org/programmers_guide/supervisor
        sv = tf.train.Supervisor(
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=15 * 60,
            save_summaries_secs=120,
            saver=saver)

        task_str = task_as_string(self.task)
        logging.info("{}: Starting managed session.".format(task_str))

        # Get a TensorFlow session managed by the supervisor.
        with sv.managed_session(target, config=self.config) as sess:

            try:
                logging.info("{}: Entering training loop.".format(task_str))
                while (not sv.should_stop()) and (not self.max_steps_reached):

                    batch_start_time = time.time()
                    # train_op returns None.
                    _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
                        [train_op, global_step, loss, predictions, labels])
                    seconds_per_batch = time.time() - batch_start_time

                    # early stopping.
                    if self.max_steps and self.max_steps <= global_step_val:
                        self.max_steps_reached = True

                    if self.is_master:
                        #
                        examples_per_second = labels_val.shape[0] / seconds_per_batch
                        hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
                        perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val, labels_val)
                        gap = eval_util.calculate_gap(predictions_val, labels_val)

                        logging.info(
                            "{}: training step | Hit@1: {}  PERR: {} GAP: {} Loss: {}".format(
                                global_step_val, hit_at_one, perr, gap, loss_val), task_str)

                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Hit@1", hit_at_one), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Perr", perr), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_GAP", gap), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("global_step/Examples/Second", examples_per_second), global_step_val)
                        sv.summary_writer.flush()

                        # Exporting the model every x steps
                        time_to_export = ((self.last_model_export_step == 0) or
                                          (global_step_val - self.last_model_export_step >= self.export_model_steps))

                        if self.is_master and time_to_export:
                            self.export_model(global_step_val, sv.saver, sv.save_path, sess)
                            self.last_model_export_step = global_step_val

                # Exporting the final model
                if self.is_master:
                    self.export_model(global_step_val, sv.saver, sv.save_path, sess)

            except tf.errors.OutOfRangeError:
                # Queue does not have enough examples any more, caused by reaching maximal epochs of queue.
                logging.info("{}: Done training -- epoch limit reached.".format(task_str))

        logging.info("{}: Exited training loop.".format(task_str))
        # Stop supervisor.
        sv.Stop()

    def export_model(self, global_step_val, saver, save_path, session):

        # If the model has already been exported at this step, return.
        if global_step_val == self.last_model_export_step:
            return

        # save() returns the path at which the variables were saved.
        last_checkpoint = saver.save(session, save_path, global_step_val)

        model_dir = "{}/export/step_{}".format(self.train_dir, global_step_val)
        logging.info("{}: Exporting the model at step {} to {}.".format(
            task_as_string(self.task), global_step_val, model_dir))

        self.model_exporter.export_model(
            model_dir=model_dir,
            global_step_val=global_step_val,
            last_checkpoint=last_checkpoint)

    def start_server_if_distributed(self):
        """Starts a server if the execution is distributed."""

        if self.cluster:
            logging.info("{}: Starting trainer within cluster {}.".format(task_as_string(self.task),
                                                                          self.cluster.as_dict()))
            # start a new server
            server = start_server(self.cluster, self.task)
            target = server.target
            device_fn = tf.train.replica_device_setter(
                ps_device="/job:ps",
                # "/job:{}/task:{}".format((self.task.type, self.task.index))
                worker_device=task_as_string(self.task),
                cluster=self.cluster)
        else:
            target = ""
            device_fn = ""
        return target, device_fn

    def remove_training_directory(self, train_dir):
        """Removes the training directory."""
        if tf.gfile.Exists(train_dir):
            try:
                logging.info("{}: Removing existing train dir.".format(task_as_string(self.task)))
                gfile.DeleteRecursively(train_dir)
            except:
                logging.error(
                    "{}: Failed to delete dir {} when starting a new model. Delete it manually and try again.".format(
                        task_as_string(self.task), train_dir))

    def get_meta_filename(self, start_new_model, train_dir):
        task_str = task_as_string(self.task)

        if start_new_model:
            logging.info("{}: Flag 'start_new_model' is set. Building a new model.".format(task_str))
            return None

        """
        Finds the filename of latest saved checkpoint file.
        Returns: The FULL path to the latest checkpoint or None if no checkpoint was found.
        """
        latest_checkpoint = tf.train.latest_checkpoint(train_dir)
        if not latest_checkpoint:
            logging.info("{}: No checkpoint file found. Building a new model.".format(task_str))
            return None

        meta_filename = latest_checkpoint + ".meta"
        if not gfile.Exists(meta_filename):
            logging.info("{}: No meta graph file found. Building a new model.".format(task_str))
            return None
        else:
            return meta_filename

    def recover_model(self, meta_filename):
        logging.info("{}: Restoring from meta graph file {}".format(task_as_string(self.task), meta_filename))
        # import_meta_graph returns A saver constructed from saver_def in MetaGraphDef or None.
        # A None value is returned if no variables exist in the MetaGraphDef (i.e., there are no variables to restore).
        # Call saver.restore(sess, 'full/path/to/meta_filename_without_suffix').
        return tf.train.import_meta_graph(meta_filename)

    def build_model(self, model, reader):
        """
        Find the model and build the graph.
        Returns a saver, keeping all checkpoints that are generated every 15 minutes.
        """

        label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
        optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

        # Build graph and add essential ops into global collection so that they can be accessed.
        build_graph(reader=reader,
                    model=model,
                    optimizer_class=optimizer_class,
                    clip_gradient_norm=FLAGS.clip_gradient_norm,
                    train_data_pattern=FLAGS.train_data_pattern,
                    label_loss_fn=label_loss_fn,
                    base_learning_rate=FLAGS.base_learning_rate,
                    learning_rate_decay=FLAGS.learning_rate_decay,
                    learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                    regularization_penalty=FLAGS.regularization_penalty,
                    num_readers=FLAGS.num_readers,
                    batch_size=FLAGS.batch_size,
                    num_epochs=FLAGS.num_epochs)

        return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)


def get_reader():
    """
    Convert feature_names and feature_sizes to lists of values.
    :return: A reader to read data from tfrecords.
    """
    # Convert csv format into a list of strings or list of integers. Added by Sophie.
    # For example, feature_names = 'mean_rgb, mean_audio' => ['mean_rgb', 'mean_audio'].
    # feature_sizes = '1024, 128' => [1024, 128].
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(FLAGS.feature_names, FLAGS.feature_sizes)

    if FLAGS.frame_features:
        # using frame-level features. Added by Sophie.
        reader = readers.YT8MFrameFeatureReader(feature_names=feature_names, feature_sizes=feature_sizes)
    else:
        # using video-level features. Added by Sophie.
        reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names, feature_sizes=feature_sizes)

    return reader


class ParameterServer(object):
    """A parameter server to serve variables in a distributed execution."""

    def __init__(self, cluster, task):
        """Creates a ParameterServer.

        Args:
          cluster: A tf.train.ClusterSpec if the execution is distributed.
            None otherwise.
          task: A TaskSpec describing the job type and the task index.
        """

        self.cluster = cluster
        self.task = task

    def run(self):
        """Starts the parameter server."""

        logging.info("{}: Starting parameter server within cluster {}.".format(task_as_string(self.task),
                                                                               self.cluster.as_dict()))
        server = start_server(self.cluster, self.task)
        server.join()


def start_server(cluster, task):
    """Creates a Server.

    Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
    """

    if not task.type:
        raise ValueError("{}: The task type must be specified.".format(task_as_string(task)))
    if task.index is None:
        raise ValueError("{}: The task index must be specified.".format(task_as_string(task)))

    """
    An in-process TensorFlow server, for use in distributed training.
    A tf.train.Server instance encapsulates a set of devices and a tf.Session target
    that can participate in distributed training. A server belongs to a cluster (specified by a tf.train.ClusterSpec),
    and corresponds to a particular task in a named job.
    The server can communicate with any other server in the same cluster.
    """
    # Create and start a server immediately.
    return tf.train.Server(
        tf.train.ClusterSpec(cluster),
        protocol="grpc",
        job_name=task.type,
        task_index=task.index)


def task_as_string(task):
    """
    Similar to toString in Java.
    :param task: {type:'', index: int}
    :return: a string, the path to this task.
    """
    return "/job:{}/task:{}".format(task.type, task.index)


def main(unused_argv):
    # Load the environment.
    env = json.loads(os.environ.get("TF_CONFIG", "{}"))

    # Load the cluster data from the environment.
    cluster_data = env.get("cluster", None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

    # Load the task data from the environment.
    # A task is described with a type and an index.
    task_data = env.get("task", None) or {"type": "master", "index": 0}
    # Create a new type. Its names is 'TaskSpec', inherits from object, and has two members, type and index.
    # Equivalent form, class TaskSpec(object):    type='master'    index=0.
    task = type("TaskSpec", (object,), task_data)

    # Tensorflow logging level. DEBUG, INFO, WARN (default), ERROR, FATAL. Right after import.
    logging.set_verbosity(tf.logging.INFO)
    logging.info("{}: Tensorflow version: {}.".format(task_as_string(task), tf.__version__))

    # Dispatch to a master, a worker, or a parameter server.
    if not cluster or task.type == "master" or task.type == "worker":
        # be cautious with () at the end. It means creating an object of the model class, e.g., LogisticModel.
        model = find_class_by_name(FLAGS.model, [frame_level_models, video_level_models])()
        # get train / validate / test reader to access data stored in tfrecords files.
        reader = get_reader()
        # FLAGS.frame_features, True represents using frame level features.
        # Export models for future prediction (and evaluation).
        model_exporter = export_model.ModelExporter(frame_features=FLAGS.frame_features, model=model, reader=reader)

        Trainer(cluster, task, FLAGS.train_dir, model, reader, model_exporter,
                FLAGS.log_device_placement, FLAGS.max_steps,
                FLAGS.export_model_steps).run(start_new_model=FLAGS.start_new_model)

    elif task.type == "ps":
        # ps, abbr. of ParameterServer.
        ParameterServer(cluster, task).run()

    else:
        raise ValueError("{}: Invalid task_type: {}.".format(task_as_string(task), task.type))


if __name__ == "__main__":
    app.run()
