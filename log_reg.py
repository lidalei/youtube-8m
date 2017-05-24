"""
One-vs-all logistic regression.

Note:
    1. Normalizing features will lead to much faster convergence but worse performance.
    2. Instead, standard scaling features will help achieve better performance.
    3. Initializing with linear regression will help get even better result.
    4. Bagging is implemented as training separately but combining inferences from multiple models. 
TODO:
    Add layers to form a neural network.
"""
import tensorflow as tf
import numpy as np

from readers import get_reader
from utils import get_input_data_tensors, DataPipeline, random_sample, load_sum_labels, load_features_mean_var
from tensorflow import flags, gfile, logging, app
from eval_util import calculate_gap
from linear_model import LinearClassifier, LogisticRegression

import time

from inference import format_lines

FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640


def gap_fn(predictions=None, labels=None):
    """
    Make predictions and labels to be specified explicitly.
    :param predictions: Model output.
    :param labels: Targets or ground truth.
    :return: GAP - global average precision.
    """
    return calculate_gap(predictions, labels)


def train(init_learning_rate, decay_steps, decay_rate=0.95, l2_reg_rate=0.01, epochs=None):
    """
    Training.

    Args:
        init_learning_rate: Initial learning rate.
        decay_steps: How many training steps to decay learning rate once.
        decay_rate: How much to decay learning rate.
        l2_reg_rate: l2 regularization rate.
        epochs: The maximal epochs to pass all training data.

    Returns:

    """
    output_dir = FLAGS.output_dir
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    validate_data_pattern = FLAGS.validate_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    init_with_linear_clf = FLAGS.init_with_linear_clf
    is_bootstrap = FLAGS.is_bootstrap

    # Increase num_readers.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=2 * num_readers)

    # Sample validate set for line search in linear classifier or logistic regression early stopping.
    _, validate_data, validate_labels, _ = random_sample(0.1, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline)

    # Set pos_weights for extremely imbalanced situation in one-vs-all classifiers.
    try:
        # Load sum_labels in training set, numpy float format to compute pos_weights.
        train_sum_labels = load_sum_labels()
        # num_neg / num_pos, assuming neg_weights === 1.0.
        pos_weights = np.sqrt((float(NUM_TRAIN_EXAMPLES) - train_sum_labels) / train_sum_labels)
        logging.info('Computing pos_weights based on sum_labels in train set successfully.')
    except:
        logging.error('Cannot load train sum_labels. Use default value.')
        pos_weights = None
    finally:
        # Set it as None to disable pos_weights.
        pos_weights = None

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    if init_with_linear_clf:
        # ...Start linear classifier...
        # Compute weights and biases of linear classifier using normal equation.
        # Linear search helps little.
        linear_clf = LinearClassifier(logdir=output_dir)
        linear_clf_weights, linear_clf_biases = linear_clf.fit(data_pipeline=train_data_pipeline,
                                                               l2_regs=[0.01],
                                                               validate_set=(validate_data, validate_labels),
                                                               line_search=False)

        logging.info('linear classifier weights and biases with shape {}, {}'.format(linear_clf_weights.shape,
                                                                                     linear_clf_biases.shape))
        logging.debug('linear classifier weights and {} biases: {}.'.format(linear_clf_weights,
                                                                            linear_clf_biases))
        # ...Exit linear classifier...
    else:
        linear_clf_weights, linear_clf_biases = None, None

    # Load train data mean and std.
    train_features_mean, train_features_var = load_features_mean_var(reader)

    # Run logistic regression.
    log_reg = LogisticRegression(logdir=output_dir)
    log_reg.fit(train_data_pipeline, train_features_mean_var=(train_features_mean, train_features_var),
                validate_set=(validate_data, validate_labels), validate_fn=gap_fn, bootstrap=is_bootstrap,
                init_learning_rate=init_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
                epochs=epochs, l2_reg_rate=l2_reg_rate, pos_weights=pos_weights,
                initial_weights=linear_clf_weights, initial_biases=linear_clf_biases)


def inference(train_model_dirs_list):
    out_file_location = FLAGS.output_file
    top_k = FLAGS.top_k
    test_data_pattern = FLAGS.test_data_pattern
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    # TODO, bagging, load several trained models and average the predictions.
    sess_list, video_input_batch_list, pred_prob_list = [], [], []
    for train_model_dir in train_model_dirs_list:
        # Load pre-trained graph and corresponding variables.
        g = tf.Graph()
        with g.as_default():
            latest_checkpoint = tf.train.latest_checkpoint(train_model_dir)
            if latest_checkpoint is None:
                raise Exception("unable to find a checkpoint at location: {}".format(train_model_dir))
            else:
                meta_graph_location = '{}{}'.format(latest_checkpoint, ".meta")
                logging.info("loading meta-graph: {}".format(meta_graph_location))
            pre_trained_saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

            # Create a session to restore model parameters.
            sess = tf.Session(graph=g)
            logging.info("restoring variables from {}".format(latest_checkpoint))
            pre_trained_saver.restore(sess, latest_checkpoint)
            # Get collections to be used in making predictions for test data.
            video_input_batch = tf.get_collection('video_input_batch')[0]
            pred_prob = tf.get_collection('predictions')[0]

            # Append session and input and predictions.
            sess_list.append(sess)
            video_input_batch_list.append(video_input_batch)
            pred_prob_list.append(pred_prob)

    # Get test data.
    test_data_pipeline = DataPipeline(reader=reader, data_pattern=test_data_pattern,
                                      batch_size=batch_size, num_readers=num_readers)

    test_graph = tf.Graph()
    with test_graph.as_default():
        video_id_batch, video_batch, labels_batch, num_frames_batch = (
            get_input_data_tensors(test_data_pipeline, shuffle=False, num_epochs=1, name_scope='test_input'))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Run test graph to get video batch and feed video batch to pre_trained_graph to get predictions.
    test_sess = tf.Session(graph=test_graph)
    with gfile.Open(out_file_location, "w+") as out_file:
        test_sess.run(init_op)

        # Be cautious to not be blocked by queue.
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=test_sess, coord=coord)

        processing_count, num_examples_processed = 0, 0
        out_file.write("VideoId,LabelConfidencePairs\n")

        try:

            while not coord.should_stop():
                # Run training steps or whatever.
                start_time = time.time()
                video_id_batch_val, video_batch_val = test_sess.run([video_id_batch, video_batch])
                logging.debug('video_id_batch_val: {}\nvideo_batch_val: {}'.format(video_id_batch_val, video_batch_val))

                batch_predictions_prob_list = []
                for sess, video_input_batch, pred_prob in zip(sess_list, video_input_batch_list, pred_prob_list):
                    feature_shape = video_input_batch.get_shape()[-1]
                    logging.info('Feature shape is {}.'.format(feature_shape))
                    if feature_shape == 128:
                        _video_batch = video_batch_val[:, -128:]
                    elif feature_shape == 1024:
                        _video_batch = video_batch_val[:, :1024]
                    else:
                        _video_batch = video_batch_val

                    batch_predictions_prob = sess.run(pred_prob, feed_dict={
                        video_input_batch: _video_batch})
                    batch_predictions_prob_list.append(batch_predictions_prob)

                batch_predictions_mean_prob = np.mean(np.stack(batch_predictions_prob_list, axis=0), axis=0)
                # Write batch predictions to files.
                for line in format_lines(video_id_batch_val, batch_predictions_mean_prob, top_k):
                    out_file.write(line)
                out_file.flush()

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

        test_sess.close()
        out_file.close()
        for sess in sess_list:
            sess.close()


def main(unused_argv):
    is_train = FLAGS.is_train
    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate
    l2_reg_rate = FLAGS.l2_reg_rate

    train_epochs = FLAGS.train_epochs
    is_tuning_hyper_para = FLAGS.is_tuning_hyper_para

    # Where training checkpoints are stored.
    train_model_dirs = FLAGS.train_model_dirs

    logging.set_verbosity(logging.INFO)

    if is_train:
        if is_tuning_hyper_para:
            raise NotImplementedError('Implementation is under progress.')
        else:
            train(init_learning_rate, decay_steps, decay_rate=decay_rate, l2_reg_rate=l2_reg_rate, epochs=train_epochs)
    else:
        train_model_dirs_list = [e.strip() for e in train_model_dirs.split(',')]
        inference(train_model_dirs_list)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train/traina*.tfrecord',
                        'File glob for the training data set.')

    flags.DEFINE_string('validate_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/validate/validateq*.tfrecord',
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    flags.DEFINE_string('test_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/test/test4*.tfrecord',
                        'Test data pattern, to be specified when making predictions.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '128', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_boolean('is_train', True, 'Boolean variable to indicate training or test.')

    flags.DEFINE_bool('is_bootstrap', False, 'Boolean variable indicating using bootstrap or not.')

    flags.DEFINE_boolean('init_with_linear_clf', True,
                         'Boolean variable indicating whether to init logistic regression with linear classifier.')

    flags.DEFINE_float('init_learning_rate', 0.01, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_float('l2_reg_rate', 0.01, 'l2 regularization rate.')

    flags.DEFINE_integer('train_epochs', 20, 'Training epochs, one epoch means passing all training data once.')

    flags.DEFINE_boolean('is_tuning_hyper_para', False,
                         'Boolean variable indicating whether to perform hyper-parameter tuning.')

    # Added current timestamp.
    flags.DEFINE_string('output_dir', '/tmp/video_level',
                        'The directory where intermediate and model checkpoints should be written.')

    # Separated by , (csv separator), e.g., log_reg_rgb,log_reg_audio. Used in bagging.
    flags.DEFINE_string('train_model_dirs', '/tmp/video_level/log_reg',
                        'The directories where to load trained logistic regression models.')

    flags.DEFINE_string('output_file', '/tmp/video_level/log_reg/predictions.csv',
                        'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
