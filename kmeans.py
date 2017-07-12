import tensorflow as tf
import numpy as np

from tensorflow import logging
from utils import get_input_data_tensors

import time


class KMeans(object):
    def __init__(self, initial_centers, data_pipeline=None, metric='cosine', return_mean_clu_dist=False):
        """
        Args:
            initial_centers: A list of centers (as a numpy array).
            data_pipeline: A namedtuple consisting the following elements.
                reader, Video-level features reader or frame-level features reader.
                data_pattern, tf data Glob. Supports *, ? and [] wildcards.
                batch_size, How many examples to read per batch.
                num_readers, How many IO threads to read examples.
            metric: Distance metric, support euclidean and cosine.
            return_mean_clu_dist: boolean. If True, compute mean distance per cluster. Else, return None.
        Returns:
            Optimized centers and corresponding average cluster-center distance and mean distance per cluster.
        Raises:
            ValueError if distance is not euclidean or cosine.
        """
        self.current_centers = np.copy(initial_centers)
        self.data_pipeline = data_pipeline
        self.metric = metric
        self.return_mean_clu_dist = return_mean_clu_dist

        if (self.metric == 'euclidean') or (self.metric == 'cosine'):
            logging.info('Perform k-means clustering using {} distance.'.format(metric))
        else:
            raise ValueError('Only euclidean and cosine distance metrics are supported.')

        # Normalize current centers if distance metric is cosine.
        if self.metric == 'cosine':
            normalized_centers = self.current_centers / np.clip(
                np.linalg.norm(self.current_centers, axis=-1, keepdims=True), 1e-6, np.PINF)
            self.current_centers = normalized_centers

        self.graph = None
        # Attributes of the graph, tensor type.
        self.current_centers_initializer = None
        self.current_centers_init_op = None
        self.per_clu_sum_initializer = None
        self.per_clu_sum_init_op = None
        self.total_dist = None
        self.per_clu_sum = None
        self.per_clu_count = None
        self.per_clu_total_dist = None
        self.update_non_op = None
        self.init_op = None

        # Build iteration graph - initialize all attributes related to the graph.
        self.build_iter_graph()

        initialize_success = self.check_graph_initialized()
        if initialize_success:
            logging.info('Succeeded initializing a Tensorflow graph to perform k-means.')
        else:
            raise ValueError('Failed to initialize a Tensorflow Graph to perform k-means.')

        # clustering objective function.
        self.mean_dist = np.PINF
        self.per_clu_mean_dist = None

    def build_iter_graph(self):
        num_centers = self.current_centers.shape[0]

        # Create the graph to traverse all training data once.
        graph = tf.Graph()
        with graph.as_default():
            # Define current centers as a variable in graph and use placeholder to hold large number of centers.
            current_centers_initializer = tf.placeholder(tf.float32, shape=self.current_centers.shape,
                                                         name='centers_initializer')
            # Setting collections=[] keeps the variable out of the GraphKeys.GLOBAL_VARIABLES collection
            # used for saving and restoring checkpoints.
            current_centers = tf.Variable(initial_value=current_centers_initializer,
                                          trainable=False, collections=[], name='current_centers')

            # Objective function. POSSIBLE ISSUE, overflow in initial iteration.
            total_dist = tf.Variable(initial_value=0.0, dtype=tf.float32, name='total_distance')
            # Define sum per clu as Variable and use placeholder to hold large number of centers.
            per_clu_sum_initializer = tf.placeholder(tf.float32, shape=self.current_centers.shape,
                                                     name='per_clu_sum_initializer')
            per_clu_sum = tf.Variable(initial_value=per_clu_sum_initializer,
                                      trainable=False, collections=[], name='per_cluster_sum')
            per_clu_count = tf.Variable(initial_value=tf.zeros([num_centers]), dtype=tf.float32, name='per_clu_count')
            if self.return_mean_clu_dist:
                per_clu_total_dist = tf.Variable(initial_value=tf.zeros([num_centers]), name='per_clu_total_dist')
            else:
                per_clu_total_dist = tf.Variable(initial_value=0.0, dtype=tf.float32, name='per_clu_total_dist')

            # Construct data read pipeline.
            video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
                get_input_data_tensors(self.data_pipeline, num_epochs=1, name_scope='k_means_reader'))

            # Assign video batch to current centers (clusters).
            if self.metric == 'euclidean':
                with tf.device('/cpu:0'):
                    # sub is very large.
                    # Make use of broadcasting feature.
                    expanded_current_centers = tf.expand_dims(current_centers, axis=0)
                    expanded_video_batch = tf.expand_dims(video_batch, axis=1)

                    sub = tf.subtract(expanded_video_batch, expanded_current_centers)
                    # element-wise square.
                    squared_sub = tf.square(sub)
                    # Compute distances with centers video-wisely. Shape [batch_size, num_initial_centers].
                    # negative === -.
                    neg_dist = tf.negative(tf.sqrt(tf.reduce_sum(squared_sub, axis=-1, name='distance')))
                # Compute assignments and the distance with nearest centers video-wisely.
                neg_topk_nearest_dist, topk_assignments = tf.nn.top_k(neg_dist, k=1)
                nearest_topk_dist = tf.negative(neg_topk_nearest_dist)
                # Remove the last dimension due to k.
                nearest_dist = tf.squeeze(nearest_topk_dist, axis=[-1], name='nearest_dist')
                assignments = tf.squeeze(topk_assignments, axis=[-1], name='assignment')

                # Compute new centers sum and number of videos that belong to each cluster within this video batch.
                batch_per_clu_sum = tf.unsorted_segment_sum(video_batch, assignments, num_centers,
                                                            name='batch_per_clu_sum')

            else:
                normalized_video_batch = tf.nn.l2_normalize(video_batch, -1)
                cosine_sim = tf.matmul(normalized_video_batch, current_centers, transpose_b=True, name='cosine_sim')
                nearest_topk_cosine_sim, topk_assignments = tf.nn.top_k(cosine_sim, k=1)
                nearest_topk_dist = tf.subtract(1.0, nearest_topk_cosine_sim)
                # Remove the last dimension due to k.
                nearest_dist = tf.squeeze(nearest_topk_dist, axis=[-1], name='nearest_dist')
                assignments = tf.squeeze(topk_assignments, axis=[-1], name='assignment')

                # Compute new centers sum and number of videos that belong to each cluster with this video batch.
                batch_per_clu_sum = tf.unsorted_segment_sum(normalized_video_batch, assignments, num_centers,
                                                            name='batch_per_clu_sum')

            batch_per_clu_count = tf.unsorted_segment_sum(tf.ones_like(video_id_batch, dtype=tf.float32),
                                                          assignments, num_centers,
                                                          name='batch_per_clu_count')
            # Update total distance, namely objective function.
            if self.return_mean_clu_dist:
                batch_per_clu_total_dist = tf.unsorted_segment_sum(nearest_dist, assignments, num_centers,
                                                                   name='batch_per_clu_total_dist')
                update_per_clu_total_dist = tf.assign_add(per_clu_total_dist, batch_per_clu_total_dist,
                                                          name='update_per_clu_total_dist')

                total_batch_dist = tf.reduce_sum(batch_per_clu_total_dist, name='total_batch_dist')
            else:
                update_per_clu_total_dist = tf.no_op()
                total_batch_dist = tf.reduce_sum(nearest_dist)

            update_total_dist = tf.assign_add(total_dist, total_batch_dist, name='update_total_dist')
            update_per_clu_sum = tf.assign_add(per_clu_sum, batch_per_clu_sum, name='update_per_clu_sum')
            update_per_clu_count = tf.assign_add(per_clu_count, batch_per_clu_count, name='update_per_clu_count')

            # Avoid unnecessary fetches.
            with tf.control_dependencies(
                    [update_total_dist, update_per_clu_sum, update_per_clu_count, update_per_clu_total_dist]):
                update_non_op = tf.no_op()

            # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        graph.finalize()

        writer = tf.summary.FileWriter('/tmp/kmeans', graph=graph)
        writer.flush()
        writer.close()

        # Update the corresponding attributes of the class.
        self.graph = graph
        self.current_centers_initializer = current_centers_initializer
        self.current_centers_init_op = current_centers.initializer
        self.per_clu_sum_initializer = per_clu_sum_initializer
        self.per_clu_sum_init_op = per_clu_sum.initializer
        self.total_dist = total_dist
        self.per_clu_sum = per_clu_sum
        self.per_clu_count = per_clu_count
        self.per_clu_total_dist = per_clu_total_dist
        self.update_non_op = update_non_op
        self.init_op = init_op

    def check_graph_initialized(self):
        """
        To check if all graph operations and the graph itself are initialized successfully.

        Return:
            True if graph and all graph ops are not None, otherwise False.
        """
        graph_ops = [self.current_centers_initializer, self.current_centers_init_op, self.per_clu_sum_initializer,
                     self.per_clu_sum_init_op, self.total_dist, self.per_clu_sum, self.per_clu_count,
                     self.per_clu_total_dist, self.update_non_op, self.init_op]

        return (self.graph is not None) and (graph_ops.count(None) == 0)

    def kmeans_iter(self):
        logging.info('Entering k-means iter ...')
        # Create a new session due to closed queue cannot be reopened.
        sess = tf.Session(graph=self.graph)
        sess.run(self.init_op)

        # initialize current centers variable in tf graph.
        sess.run(self.current_centers_init_op,
                 feed_dict={self.current_centers_initializer: self.current_centers})

        # initializer per_clu_sum in tf graph.
        sess.run(self.per_clu_sum_init_op,
                 feed_dict={self.per_clu_sum_initializer: np.zeros_like(self.current_centers)})

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                _ = sess.run(self.update_non_op)

        except tf.errors.OutOfRangeError:
            logging.info('One k-means iteration done. One epoch limit reached.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        # Get final values.
        final_total_dist, final_per_clu_sum, final_per_clu_count, final_per_clu_total_dist = sess.run(
            [self.total_dist, self.per_clu_sum, self.per_clu_count, self.per_clu_total_dist])
        logging.info('Exiting k-means iter ...')
        sess.close()

        # Remove smaller clusters.
        threshold = np.percentile(final_per_clu_count, 5)
        # Deal with empty cluster situation.
        nonzero_indices = np.greater_equal(final_per_clu_count, threshold)
        per_nonempty_clu_count = final_per_clu_count[nonzero_indices]
        # Expand to each feature to make use of broadcasting.
        per_nonempty_clu_feat_count = np.expand_dims(per_nonempty_clu_count, axis=1)
        per_nonempty_clu_sum = final_per_clu_sum[nonzero_indices]

        updated_centers = per_nonempty_clu_sum / per_nonempty_clu_feat_count

        if self.return_mean_clu_dist:
            per_nonempty_clu_total_dist = final_per_clu_total_dist[nonzero_indices]
            # Objective function value.
            total_nonempty_num_points = np.sum(per_nonempty_clu_count)
            total_nonempty_dist = np.sum(per_nonempty_clu_total_dist)
            mean_dist = total_nonempty_dist / total_nonempty_num_points
            # Numpy array divide element-wisely.
            per_nonempty_clu_mean_dist = per_nonempty_clu_total_dist / per_nonempty_clu_count
        else:
            # Objective function value.
            total_num_points = np.sum(final_per_clu_count)
            mean_dist = final_total_dist / total_num_points
            per_nonempty_clu_mean_dist = None

        # Numpy array divide element-wisely.
        return updated_centers, mean_dist, per_nonempty_clu_mean_dist

    def fit(self, max_iter=100, tol=0.01):
        """
        This function works as sk-learn estimator fit.
        :param max_iter: 
        :param tol: Percentage not improved one iteration, stop iteration.
        :return: Update current centers and current objective function value (member variables).
        """
        for iter_count in xrange(max_iter):
            start_time = time.time()
            new_centers, new_mean_dist, new_per_clu_mean_dist = self.kmeans_iter()
            print('The {}-th iteration took {} s.'.format(iter_count + 1, time.time() - start_time))

            # There are empty centers (clusters) being removed.
            need_rebuild_graph = new_centers.shape[0] != self.current_centers.shape[0]

            # Update current centers and mean distance per cluster.
            # Normalize current centers if distance metric is cosine.
            if self.metric == 'cosine':
                self.current_centers = new_centers / np.clip(
                    np.linalg.norm(new_centers, axis=-1, keepdims=True), 1e-6, np.PINF)
            else:
                self.current_centers = new_centers

            self.per_clu_mean_dist = new_per_clu_mean_dist

            # Converged, break!
            if not np.isinf(self.mean_dist) and np.abs(self.mean_dist - new_mean_dist) / self.mean_dist < tol:
                # Update current objective function value.
                self.mean_dist = new_mean_dist
                logging.info('Done k-means clustering. Final centers have shape {}. Final mean dist is {}.'.format(
                    self.current_centers.shape, self.mean_dist))
                break
            else:
                # Update current objective function value.
                self.mean_dist = new_mean_dist

            if need_rebuild_graph:
                # Re-build graph using updated current centers.
                self.build_iter_graph()
                initialize_success = self.check_graph_initialized()
                if initialize_success:
                    logging.info('Succeeded re-initializing a Tensorflow graph to perform k-means.')
                else:
                    raise ValueError('Failed to re-initialize a Tensorflow Graph to perform k-means.')

            logging.debug('new_centers: {}'.format(self.current_centers))
            logging.info('new_centers shape: {}'.format(self.current_centers.shape))
            logging.info('New mean point-center distance: {}'.format(self.mean_dist))


def mini_batch_kmeans():
    raise NotImplementedError('Not implemented. Batch kmeans works fast enough now.')
