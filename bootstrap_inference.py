"""
This file contains BootstrapInference class. This class is used to make predictions and 
 averaging predictions from many models to form the final prediction.
Note: The function format_lines is copied from inference.py in the same parent folder.
    It is just for decoupling file dependency.
"""
import tensorflow as tf
from tensorflow import logging, gfile, flags, app

import numpy as np
from utils import get_input_data_tensors, DataPipeline
from readers import get_reader

from os.path import join as path_join
import time

FLAGS = flags.FLAGS


def format_lines(video_ids, predictions, top_k=20):
    batch_size = len(video_ids)
    for video_index in range(batch_size):
        top_indices = np.argpartition(predictions[video_index], -top_k)[-top_k:]
        line = [(class_index, predictions[video_index][class_index])
                for class_index in top_indices]
        line = sorted(line, key=lambda p: -p[1])
        yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair for pair in line) + "\n"


class BootstrapInference(object):
    def __init__(self, train_model_dirs_list):
        # Bagging, load several trained models and average the predictions.
        self.train_model_dirs_list = train_model_dirs_list

        self.sess_list = []
        self.video_input_batch_list = []
        self.pred_prob_list = []
        self.phase_train_pl_list = []

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
                video_input_batch = tf.get_collection('raw_features_batch')[0]
                pred_prob = tf.get_collection('predictions')[0]
                phase_train_pl = tf.get_collection('phase_train_pl')

                # Append session and input and predictions.
                self.sess_list.append(sess)
                self.video_input_batch_list.append(video_input_batch)
                self.pred_prob_list.append(pred_prob)
                if len(phase_train_pl) >= 1:
                    self.phase_train_pl_list.append({phase_train_pl[0]: False})
                else:
                    self.phase_train_pl_list.append({})

    def __del__(self):
        for sess in self.sess_list:
            sess.close()

    def transform(self, test_data_pipeline, out_file_location, top_k=20):
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
                    logging.debug('video_id_batch_val: {}\nvideo_batch_val: {}'.format(
                        video_id_batch_val, video_batch_val))

                    batch_predictions_prob_list = []
                    for sess, video_input_batch, pred_prob, phase_train_pl in zip(
                            self.sess_list, self.video_input_batch_list,
                            self.pred_prob_list, self.phase_train_pl_list):
                        feature_shape = video_input_batch.get_shape()[-1]
                        # logging.info('Feature shape is {}.'.format(feature_shape))
                        if feature_shape == 128:
                            _video_batch = video_batch_val[:, -128:]
                        elif feature_shape == 1024:
                            _video_batch = video_batch_val[:, :1024]
                        else:
                            _video_batch = video_batch_val

                        batch_predictions_prob = sess.run(pred_prob, feed_dict=dict(
                            {video_input_batch: _video_batch}, **phase_train_pl
                        ))
                        batch_predictions_prob_list.append(batch_predictions_prob)

                    batch_predictions_mean_prob = np.mean(np.stack(batch_predictions_prob_list, axis=0), axis=0)
                    # Write batch predictions to files.
                    for line in format_lines(video_id_batch_val, batch_predictions_mean_prob, top_k):
                        out_file.write(line)
                    out_file.flush()

                    now = time.time()
                    processing_count += 1
                    num_examples_processed += video_id_batch_val.shape[0]
                    print('Batch processing step {}, elapsed {} s, processed {} examples in total'.format(
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


def main(unsed_argv):
    logging.set_verbosity(logging.INFO)
    # Where training checkpoints are stored.
    train_model_dirs = FLAGS.train_model_dirs
    out_file_location = FLAGS.output_file
    top_k = FLAGS.top_k
    test_data_pattern = FLAGS.test_data_pattern
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    train_model_dirs_list = [e.strip() for e in train_model_dirs.split(',')]
    # Get test data.
    test_data_pipeline = DataPipeline(reader=reader, data_pattern=test_data_pattern,
                                      batch_size=batch_size, num_readers=num_readers)

    # Make inference.
    inference = BootstrapInference(train_model_dirs_list)
    inference.transform(test_data_pipeline, out_file_location, top_k=top_k)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    flags.DEFINE_string('test_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/test/test*.tfrecord',
                        'Test data pattern, to be specified when making predictions.')

    flags.DEFINE_string('feature_names', 'mean_rgb,mean_audio', 'Features to be used, separated by ,.')

    flags.DEFINE_string('feature_sizes', '1024,128', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_integer('batch_size', 4096, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    # Separated by , (csv separator), e.g., log_reg_rgb,log_reg_audio. Used in bagging.
    flags.DEFINE_string('train_model_dirs', '/tmp/video_level/mlp',
                        'The directories where to load trained logistic regression models.')

    flags.DEFINE_string('output_file', '/tmp/video_level/predictions_{}.csv'.format(int(time.time())),
                        'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
