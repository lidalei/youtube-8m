import tensorflow as tf
import numpy as np

from tensorflow import logging
from utils import get_input_data_tensors, MakeSummary


class LinearClassifier(object):
    def __init__(self, logdir='/tmp'):
        """
        Args:
             logdir: Path to the log dir.
        """
        self.logdir = logdir
        self.weights = None
        self.biases = None
        self.rmse = np.NINF

    def fit(self, data_pipeline=None, tr_data_fn=None, tr_data_paras=None,
            l2_regs=None, validate_set=None, line_search=True):
        """
        Compute weights and biases of linear classifier using normal equation. With line search for best l2_reg.
        Args:
            data_pipeline: A namedtuple consisting of the following elements.
                reader, video-level features reader or frame-level features reader.
                data_pattern, File Glob of data set.
                batch_size, How many examples to handle per time.
                num_readers, How many IO threads to prefetch examples.
            tr_data_fn: a function that transforms input data.
            tr_data_paras: Other parameters should be passed to tr_data_fn. A dictionary.
            l2_regs: An array, each element represents how much the linear classifier weights should be penalized.
            validate_set: (data, labels) with dtype float32. The data set (numpy arrays) used to choose the best l2_reg.
                Sampled from whole validate set if necessary. If line_search is False, this argument is simply ignored.
            line_search: Boolean argument representing whether to do boolean search.

        Returns: Weights and biases fit on the given data set, where biases are appended as the last row.

        """
        logging.info('Entering linear classifier ...')

        reader = data_pipeline.reader
        num_classes = reader.num_classes
        batch_size = reader.batch_size
        feature_names = reader.feature_names
        feature_sizes = reader.feature_sizes
        feature_size = sum(feature_sizes)
        logging.info('Linear regression uses {} features with dims {}.'.format(feature_names, feature_sizes))

        if line_search:
            # Both l2_regs and validate_set are required.
            if l2_regs is None:
                raise ValueError('There is no l2_regs to do line search.')
            else:
                logging.info('l2_regs is {}.'.format(l2_regs))

            if validate_set is None:
                raise ValueError('There is no validate_set to do line search for l2_reg.')
            else:
                validate_data, validate_labels = validate_set

        else:
            # Simply fit the training set. Make l2_regs have only one element. And ignore validate_set.
            if l2_regs is None:
                l2_regs = 0.001
            if isinstance(l2_regs, list):
                l2_regs = l2_regs[0]
            logging.info('No line search, l2_regs is {}.'.format(l2_regs))
            if validate_set is None:
                # Important! To make the graph construction successful.
                validate_data = np.zeros([1, feature_size], dtype=np.float32)
                validate_labels = np.zeros([1, num_classes], dtype=np.float32)
            else:
                validate_data, validate_labels = validate_set

        # Check validate data and labels shape.
        logging.info('validate set: data has shape {}, labels has shape {}.'.format(
            validate_data.shape, validate_labels.shape))
        if (validate_data.shape[-1] != feature_size) or (validate_labels.shape[-1] != num_classes):
            raise ValueError('validate set shape does not conforms with training set.')

        # TO BE CAUTIOUS! THE FOLLOWING MAY HAVE TO DEAL WITH FEATURE SIZE CHANGE.
        # Check extra data transform function arguments.
        # If transform changes the features size, change it.
        if tr_data_fn is not None:
            if tr_data_paras is None:
                tr_data_paras = {}
            else:
                if ('reshape' in tr_data_paras) and (tr_data_paras['reshape'] is True):
                    feature_size = tr_data_paras['size']
                    logging.warn('Data transform changes the features size to {}.'.format(feature_size))

        # Method - append an all-one col to X by using block matrix multiplication (all-one col is treated as a block).
        # Create the graph to traverse all data once.
        with tf.Graph().as_default() as graph:
            global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')
            global_step_inc_op = tf.assign_add(global_step, 1)

            # X.transpose * X
            norm_equ_1_initializer = tf.placeholder(tf.float32, shape=[feature_size, feature_size])
            norm_equ_1 = tf.Variable(initial_value=norm_equ_1_initializer, collections=[], name='X_Tr_X')

            # X.transpose * Y
            norm_equ_2_initializer = tf.placeholder(tf.float32, shape=[feature_size, num_classes])
            norm_equ_2 = tf.Variable(initial_value=norm_equ_2_initializer, collections=[], name='X_Tr_Y')

            example_count = tf.Variable(initial_value=0.0, name='example_count')
            features_sum = tf.Variable(initial_value=tf.zeros([feature_size]), name='features_sum')
            labels_sum = tf.Variable(initial_value=tf.zeros([num_classes]), name='labels_sum')

            id_batch, raw_features_batch, labels_batch, num_frames_batch = (
                get_input_data_tensors(data_pipeline, num_epochs=1, name_scope='input'))
            if tr_data_fn is None:
                transformed_features_batch = tf.identity(raw_features_batch)
            else:
                transformed_features_batch = tr_data_fn(raw_features_batch, **tr_data_paras)

            with tf.name_scope('batch_increment'):
                transformed_features_batch_tr = tf.matrix_transpose(transformed_features_batch, name='X_Tr')
                float_labels_batch = tf.cast(labels_batch, tf.float32)
                batch_norm_equ_1 = tf.matmul(transformed_features_batch_tr, transformed_features_batch,
                                             name='batch_norm_equ_1')
                # batch_norm_equ_1 = tf.add_n(tf.map_fn(lambda x: tf.einsum('i,j->ij', x, x),
                #                                       transformed_features_batch_tr), name='X_Tr_X')
                batch_norm_equ_2 = tf.matmul(transformed_features_batch_tr, float_labels_batch,
                                             name='batch_norm_equ_2')
                batch_example_count = tf.cast(tf.shape(transformed_features_batch)[0], tf.float32,
                                              name='batch_example_count')
                batch_features_sum = tf.reduce_sum(transformed_features_batch, axis=0,
                                                   name='batch_features_sum')
                batch_labels_sum = tf.reduce_sum(float_labels_batch, axis=0,
                                                 name='batch_labels_sum')

            with tf.name_scope('update_ops'):
                update_norm_equ_1_op = tf.assign_add(norm_equ_1, batch_norm_equ_1)
                update_norm_equ_2_op = tf.assign_add(norm_equ_2, batch_norm_equ_2)
                update_example_count = tf.assign_add(example_count, batch_example_count)
                update_features_sum = tf.assign_add(features_sum, batch_features_sum)
                update_labels_sum = tf.assign_add(labels_sum, batch_labels_sum)

            with tf.control_dependencies([update_norm_equ_1_op, update_norm_equ_2_op, update_example_count,
                                          update_features_sum, update_labels_sum, global_step_inc_op]):
                update_equ_non_op = tf.no_op(name='unified_update_op')

            with tf.name_scope('solution'):
                # After all data being handled, compute weights.
                l2_reg_ph = tf.placeholder(tf.float32, shape=[])
                l2_reg_term = tf.diag(tf.fill([feature_size], l2_reg_ph), name='l2_reg')
                # X.transpose * X + lambda * Id, where d is the feature dimension.
                norm_equ_1_with_reg = tf.add(norm_equ_1, l2_reg_term)

                # Concat other blocks to form the final norm equation terms.
                final_norm_equ_1_top = tf.concat([norm_equ_1_with_reg, tf.expand_dims(features_sum, 1)], 1)
                final_norm_equ_1_bot = tf.concat([features_sum, tf.expand_dims(example_count, 0)], 0)
                final_norm_equ_1 = tf.concat([final_norm_equ_1_top, tf.expand_dims(final_norm_equ_1_bot, 0)], 0,
                                             name='norm_equ_1')
                final_norm_equ_2 = tf.concat([norm_equ_2, tf.expand_dims(labels_sum, 0)], 0,
                                             name='norm_equ_2')

                # The last row is the biases.
                weights_biases = tf.matrix_solve(final_norm_equ_1, final_norm_equ_2, name='weights_biases')

                weights = weights_biases[:-1]
                biases = weights_biases[-1]

            with tf.name_scope('validate_loss'):
                validate_x_pl = tf.placeholder(tf.float32, shape=[None, feature_size], name='validate_data')

                validate_y_pl = tf.placeholder(tf.float32, shape=[None, num_classes], name='validate_labels')

                if tr_data_fn is None:
                    validate_x_transformed = tf.identity(validate_x_pl)
                else:
                    validate_x_transformed = tr_data_fn(validate_x_pl, **tr_data_paras)

                predictions = tf.matmul(validate_x_transformed, weights) + biases
                loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions, validate_y_pl)), name='rmse')
                # pred_labels = tf.greater_equal(predictions, 0.0, name='pred_labels')

            summary_op = tf.summary.merge_all()

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(),
                               name='init_glo_loc_var')

        sess = tf.Session(graph=graph)
        # Initialize variables.
        sess.run(init_op)
        sess.run([norm_equ_1.initializer, norm_equ_2.initializer], feed_dict={
            norm_equ_1_initializer: np.zeros([feature_size, feature_size], dtype=np.float32),
            norm_equ_2_initializer: np.zeros([feature_size, num_classes], dtype=np.float32)
        })

        summary_writer = tf.summary.FileWriter(self.logdir, graph=sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _, summary, global_step_val = sess.run([update_equ_non_op, summary_op, global_step])
                summary_writer.add_summary(summary, global_step=global_step_val)
        except tf.errors.OutOfRangeError:
            logging.info('Finished normal equation terms computation -- one epoch done.')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

        if line_search:
            # Do true search.
            best_weights_val, best_biases_val = None, None
            best_l2_reg = 0
            min_loss = np.PINF

            for l2_reg in l2_regs:
                # Compute regularized weights.
                weights_val, biases_val = sess.run([weights, biases], feed_dict={l2_reg_ph: l2_reg})
                # Compute validation loss.
                num_validate_videos = validate_data.shape[0]
                split_indices = np.linspace(0, num_validate_videos, num_validate_videos / batch_size,
                                            dtype=np.int32)
                loss_vals = []
                for i in xrange(len(split_indices) - 1):
                    start_ind = split_indices[i]
                    end_ind = split_indices[i + 1]

                    ith_loss_val = sess.run(loss,
                                            feed_dict={validate_x_pl: validate_data[start_ind:end_ind],
                                                       validate_y_pl: validate_labels[start_ind:end_ind]})

                    loss_vals.append(ith_loss_val * (end_ind - start_ind))

                validate_loss_val = sum(loss_vals) / num_validate_videos

                logging.info('l2_reg {} leads to rmse loss {}.'.format(l2_reg, validate_loss_val))
                if validate_loss_val < min_loss:
                    best_weights_val, best_biases_val = weights_val, biases_val
                    min_loss = validate_loss_val
                    best_l2_reg = l2_reg

        else:
            # Extract weights and biases of num_classes linear classifiers. Each column corresponds to a classifier.
            best_weights_val, best_biases_val, loss_val = sess.run(
                [weights, biases, loss], feed_dict={l2_reg_ph: l2_regs})
            best_l2_reg, min_loss = l2_regs, None if not validate_set else loss_val

        sess.close()

        logging.info('The best l2_reg is {} with rmse loss {}.'.format(best_l2_reg, min_loss))
        logging.info('Exiting linear classifier ...')

        self.weights = best_weights_val
        self.biases = best_biases_val
        self.rmse = min_loss


class LogisticRegression(object):
    def __init__(self, logdir='/tmp', max_train_steps=1000000):
        """
        Args:
             logdir: The dir where intermediate results and model checkpoints should be written.
        """
        self.logdir = logdir
        self.max_train_steps = max_train_steps

        # Member variables used to construct graph.
        self.train_data_pipeline = None
        self.raw_feature_size = None
        self.feature_size = None
        self.num_classes = None
        self.batch_size = None
        self.tr_data_fn = None
        self.tr_data_paras = dict()
        self.bootstrap = False
        self.init_learning_rate = 0.01
        self.decay_steps = 40000
        self.decay_rate = 0.95
        self.epochs = None
        self.l1_reg_rate = None
        self.l2_reg_rate = 0.01
        self.pos_weights = None
        self.initial_weights = None
        self.initial_biases = None

        self.graph = None
        # Member variables associated with graph.
        self.saver = None
        self.global_step = None
        self.init_op = None
        self.train_op = None
        self.summary_op = None
        self.raw_features_batch = None
        self.labels_batch = None
        self.loss = None
        self.pred_prob = None

    def _build_graph(self):
        """
        Build graph.

        Returns:
            A saver object. It can be used in constructing a Supervisor object.
        Note:
            To avoid contaminating default graph.
            This function must be wrapped into a with tf.Graph().as_default() as graph contextmanager.
        """
        # Build logistic regression graph and optimize it.
        # Set seed to keep whole data sampling consistency, though impossible due to system variation.
        # seed = np.random.randint(2 ** 28)
        # tf.set_random_seed(seed)

        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')

        id_batch, raw_features_batch, labels_batch, num_frames_batch = (
            get_input_data_tensors(self.train_data_pipeline, shuffle=True, num_epochs=self.epochs,
                                   name_scope='input'))

        # Define num_classes logistic regression models parameters.
        if self.initial_weights is None:
            weights = tf.Variable(initial_value=tf.truncated_normal(
                [self.feature_size, self.num_classes], stddev=1.0 / np.sqrt(self.feature_size)),
                dtype=tf.float32, name='weights')
        else:
            weights = tf.Variable(initial_value=self.initial_weights, dtype=tf.float32, name='weights')
        # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
        tf.summary.histogram('model/weights', weights)

        if self.initial_biases is None:
            biases = tf.Variable(initial_value=tf.zeros([self.num_classes]), name='biases')
        else:
            biases = tf.Variable(initial_value=self.initial_biases, name='biases')

        tf.summary.histogram('model/biases', biases)

        if self.tr_data_fn is None:
            transformed_features_batch = tf.identity(raw_features_batch)
        else:
            transformed_features_batch = self.tr_data_fn(raw_features_batch, **self.tr_data_paras)

        output = tf.add(tf.matmul(transformed_features_batch, weights), biases, name='output')

        float_labels = tf.cast(labels_batch, tf.float32, name='float_labels')
        pred_prob = tf.nn.sigmoid(output, name='pred_probability')

        with tf.name_scope('train'):
            if self.pos_weights is None:
                loss_per_ex_label = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=float_labels, logits=output, name='x_entropy_per_ex_label')
            else:
                loss_per_ex_label = tf.nn.weighted_cross_entropy_with_logits(
                    targets=float_labels, logits=output, pos_weight=self.pos_weights,
                    name='x_entropy_per_ex_label')

            # Sum over label set.
            loss_per_ex = tf.reduce_sum(loss_per_ex_label, axis=1, name='loss_per_ex')

            #  In addition to class weighting, example weighting is supported.
            if self.bootstrap:
                num_videos = tf.shape(loss_per_ex)[0]
                sample_indices = tf.random_uniform([num_videos], maxval=num_videos, dtype=tf.int32,
                                                   name='sample_indices')
                example_weights = tf.unsorted_segment_sum(tf.ones([num_videos]), sample_indices, num_videos,
                                                          name='example_weights')
                # bootstrap-weighted loss.
                weighted_loss_per_ex = tf.multiply(loss_per_ex, example_weights, name='weighted_loss_per_ex')
                # Mean over batch.
                loss = tf.reduce_mean(weighted_loss_per_ex, name='x_entropy')
            else:
                # Mean over batch.
                loss = tf.reduce_mean(loss_per_ex, name='x_entropy')

            tf.summary.scalar('loss/xentropy', loss)

            # Add regularization.
            reg_losses = []
            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            to_regularize = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.l1_reg_rate:
                l1_reg_losses = [tf.reduce_sum(tf.abs(w)) for w in to_regularize]
                l1_reg_loss = tf.add_n(l1_reg_losses, name='l1_reg_loss')
                tf.summary.scalar('loss/l1_reg_loss', l1_reg_loss)
                reg_losses.append(tf.multiply(self.l1_reg_rate, l1_reg_loss))

            if self.l2_reg_rate:
                l2_reg_losses = [tf.reduce_sum(tf.square(w)) for w in to_regularize]
                l2_reg_loss = tf.add_n(l2_reg_losses, name='l2_loss')
                tf.summary.scalar('loss/l2_reg_loss', l2_reg_loss)
                reg_losses.append(tf.multiply(self.l2_reg_rate, l2_reg_loss))

            if len(reg_losses) > 0:
                reg_loss = tf.add_n(reg_losses, name='reg_loss')
            else:
                reg_loss = tf.constant(0.0, name='zero_reg_loss')

            final_loss = tf.add(loss, reg_loss, name='final_loss')

        with tf.name_scope('optimization'):
            # RMSPropOptimizer
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.init_learning_rate)
            # Decayed learning rate.
            # rough_num_examples_processed = tf.multiply(global_step, self.batch_size)
            # adap_learning_rate = tf.train.exponential_decay(self.init_learning_rate, rough_num_examples_processed,
            #                                                 self.decay_steps, self.decay_rate, staircase=True,
            #                                                 name='adap_learning_rate')
            # tf.summary.scalar('learning_rate', adap_learning_rate)
            # GradientDescentOptimizer
            # optimizer = tf.train.GradientDescentOptimizer(adap_learning_rate)
            # MomentumOptimizer
            # optimizer = tf.train.MomentumOptimizer(adap_learning_rate, 0.9, use_nesterov=True)
            train_op = optimizer.minimize(final_loss, global_step=global_step)

        summary_op = tf.summary.merge_all()
        # summary_op = tf.constant(1.0)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Used for restoring training checkpoints.
        tf.add_to_collection('global_step', global_step)
        tf.add_to_collection('init_op', init_op)
        tf.add_to_collection('train_op', train_op)
        tf.add_to_collection('summary_op', summary_op)
        # Add to collection. In inference, get collection and feed it with test data.
        tf.add_to_collection('raw_features_batch', raw_features_batch)
        tf.add_to_collection('labels_batch', labels_batch)
        tf.add_to_collection('loss', loss)
        tf.add_to_collection('predictions', pred_prob)

        # To save global variables and savable objects, i.e., var_list is None.
        # Using rbf transform will also save centers and scaling factors.
        saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=0.15)

        return saver

    def _restore_graph(self):
        """
        Restore graph def.
        Returns:
             A saver previously created when building this graph.
        Note:
            To avoid contaminating default graph.
            This function must be wrapped into a with tf.Graph().as_default() as graph contextmanager.
        """
        # Load pre-trained graph.
        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)
        if latest_checkpoint is None:
            raise Exception("unable to find a checkpoint at location: {}".format(self.logdir))
        else:
            meta_graph_location = '{}{}'.format(latest_checkpoint, ".meta")
            logging.info("loading meta-graph: {}".format(meta_graph_location))
        # Recreates a Graph saved in a MetaGraphDef proto, docs.
        pre_trained_saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)

        return pre_trained_saver

    def _check_graph_initialized(self):
            """
            To check if all graph operations and the graph itself are initialized successfully.

            Return:
                True if graph and all graph ops are not None, otherwise False.
            """
            graph_ops = [self.saver, self.global_step, self.init_op, self.train_op, self.summary_op,
                         self.raw_features_batch, self.labels_batch, self.loss, self.pred_prob]

            return (self.graph is not None) and (graph_ops.count(None) == 0)

    def fit(self, train_data_pipeline, start_new_model=False,
            tr_data_fn=None, tr_data_paras=None,
            validate_set=None, validate_fn=None,
            bootstrap=False, init_learning_rate=0.01, decay_steps=40000, decay_rate=0.95, epochs=None,
            l1_reg_rate=None, l2_reg_rate=0.01, pos_weights=None, initial_weights=None, initial_biases=None):
        """
        Logistic regression fit function.
        Args:
            train_data_pipeline: A namedtuple consisting of reader, data_pattern, batch_size and num_readers.
            start_new_model: If True, start a new model instead of restoring from existing checkpoints.
            tr_data_fn: a function that transforms input data.
            tr_data_paras: Other parameters should be passed to tr_data_fn. A dictionary.
            validate_set: If not None, check validation loss regularly. Else, ignored.
            validate_fn: The function to check the performance of learned model parameters on validate set.
            bootstrap: If True, sampling training examples with replacement by differential weighting.
            init_learning_rate: Decayed gradient descent parameter.
            decay_steps: Decayed gradient descent parameter.
            decay_rate: Decayed gradient descent parameter.
            epochs: Maximal epochs to use.
            l1_reg_rate: None, not impose l1 regularization.
            l2_reg_rate: l2 regularization rate.
            pos_weights: For imbalanced binary classes. Here, num_pos << num_neg, the weights should be > 1.0.
                If None, treated as 1.0 for all binary classifiers.
            initial_weights: If not None, the weights will be initialized with it.
            initial_biases: If not None, the biases will be initialized with it.
        Returns: None.
        """
        reader = train_data_pipeline.reader
        batch_size = train_data_pipeline.batch_size
        num_classes = reader.num_classes
        feature_names = reader.feature_names
        feature_sizes = reader.feature_sizes
        logging.info('Logistic regression uses {} features with dims {}.'.format(feature_names, feature_sizes))

        raw_feature_size = sum(feature_sizes)

        self.train_data_pipeline = train_data_pipeline
        self.raw_feature_size = raw_feature_size
        self.feature_size = raw_feature_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.tr_data_fn = tr_data_fn
        self.tr_data_paras = tr_data_paras
        self.bootstrap = bootstrap
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.epochs = epochs
        self.l1_reg_rate = l1_reg_rate
        self.l2_reg_rate = l2_reg_rate
        self.pos_weights = pos_weights
        self.initial_weights = initial_weights
        self.initial_biases = initial_biases

        # Check extra data transform function arguments.
        # If transform changes the features size, change it.
        if self.tr_data_fn is not None:
            if self.tr_data_paras is None:
                self.tr_data_paras = dict()
            else:
                if ('reshape' in self.tr_data_paras) and (self.tr_data_paras['reshape'] is True):
                    self.feature_size = self.tr_data_paras['size']
                    logging.warn('Data transform changes the features size to {}.'.format(
                        self.feature_size))

            logging.debug('Data transform arguments are {}.'.format(self.tr_data_paras))
        else:
            self.tr_data_paras = dict()

        start_new_model = start_new_model or (not tf.gfile.Exists(self.logdir))

        # This is NECESSARY to avoid contaminating default graph.
        # Alternatively, we can define a member graph variable. When building a new graph or
        # restoring a graph, wrap the code into a similar contextmanager.
        self.graph = tf.Graph()
        with self.graph.as_default():
            if start_new_model:
                logging.info('Starting a new model...')
                # Start new model, delete existing checkpoints.
                if tf.gfile.Exists(self.logdir):
                    try:
                        tf.gfile.DeleteRecursively(self.logdir)
                    except tf.errors.OpError:
                        logging.error('Failed to delete dir {}.'.format(self.logdir))
                    else:
                        logging.info('Succeeded to delete train dir {}.'.format(self.logdir))
                else:
                    # Do nothing.
                    pass

                # Build graph, namely building a graph and initialize member variables associated with graph.
                self.saver = self._build_graph()
            else:
                self.saver = self._restore_graph()

            # After either building a graph or restoring a graph, graph is CONSTRUCTED successfully.
            # Get collections to be used in training.
            self.global_step = tf.get_collection('global_step')[0]
            self.init_op = tf.get_collection('init_op')[0]
            self.train_op = tf.get_collection('train_op')[0]
            self.summary_op = tf.get_collection('summary_op')[0]
            self.raw_features_batch = tf.get_collection('raw_features_batch')[0]
            self.labels_batch = tf.get_collection('labels_batch')[0]
            self.loss = tf.get_collection('loss')[0]
            self.pred_prob = tf.get_collection('predictions')[0]

        if self._check_graph_initialized():
            logging.info('Succeeded to initialize logistic regression Graph.')
        else:
            logging.error('Failed to initialize logistic regression Graph.')

        # Start or restore training.
        # To avoid summary causing memory usage peak, manually save summaries.
        sv = tf.train.Supervisor(graph=self.graph, init_op=self.init_op, logdir=self.logdir,
                                 global_step=self.global_step, summary_op=None,
                                 save_model_secs=600, saver=self.saver)

        with sv.managed_session() as sess:
            logging.info("Entering training loop...")
            for step in xrange(self.max_train_steps):
                if sv.should_stop():
                    # Save the final model and break.
                    self.saver.save(sess, save_path='{}_{}'.format(sv.save_path, 'final'))
                    break

                if step % 500 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    if validate_fn is not None:
                        _, summary, train_pred_prob_batch, train_labels_batch, global_step_val = sess.run(
                            [self.train_op, self.summary_op, self.pred_prob, self.labels_batch, self.global_step],
                            options=run_options, run_metadata=run_metadata)

                        # Evaluate on train data.
                        train_per = validate_fn(predictions=train_pred_prob_batch, labels=train_labels_batch)
                        sv.summary_writer.add_summary(
                            MakeSummary('train/{}'.format(validate_fn.func_name), train_per),
                            global_step_val)
                        logging.info('Step {}, train {}: {}.'.format(global_step_val,
                                                                     validate_fn.func_name, train_per))
                    else:
                        _, summary, global_step_val = sess.run(
                            [self.train_op, self.summary_op, self.global_step],
                            options=run_options, run_metadata=run_metadata)

                    # Add train summary.
                    sv.summary_computed(sess, summary, global_step=global_step_val)
                    sv.summary_writer.add_run_metadata(run_metadata, 'step{}'.format(global_step_val))

                    # Compute validate loss and performance (validate_fn).
                    if validate_set is not None:
                        validate_data, validate_labels = validate_set

                        # Compute validation loss.
                        num_validate_videos = validate_data.shape[0]
                        split_indices = np.linspace(0, num_validate_videos, num_validate_videos / (4 * batch_size),
                                                    dtype=np.int32)

                        validate_loss_vals, validate_pers = [], []
                        for i in xrange(len(split_indices) - 1):
                            start_ind = split_indices[i]
                            end_ind = split_indices[i + 1]

                            if validate_fn is not None:
                                ith_validate_loss_val, ith_predictions = sess.run(
                                    [self.loss, self.pred_prob], feed_dict={
                                        self.raw_features_batch: validate_data[start_ind:end_ind],
                                        self.labels_batch: validate_labels[start_ind:end_ind]})

                                ith_validate_per = validate_fn(predictions=ith_predictions,
                                                               labels=validate_labels[start_ind:end_ind])
                                validate_loss_vals.append(ith_validate_loss_val * (end_ind - start_ind))
                                validate_pers.append(ith_validate_per * (end_ind - start_ind))
                            else:
                                ith_validate_loss_val = sess.run(
                                    self.loss, feed_dict={
                                        self.raw_features_batch: validate_data[start_ind:end_ind],
                                        self.labels_batch: validate_labels[start_ind:end_ind]})

                                validate_loss_vals.append(ith_validate_loss_val * (end_ind - start_ind))

                        validate_loss_val = sum(validate_loss_vals) / num_validate_videos
                        # Add validate summary.
                        sv.summary_writer.add_summary(
                            MakeSummary('validate/xentropy', validate_loss_val), global_step_val)

                        if validate_fn is not None:
                            validate_per = sum(validate_pers) / num_validate_videos
                            sv.summary_writer.add_summary(
                                MakeSummary('validate/{}'.format(validate_fn.func_name), validate_per),
                                global_step_val)
                            logging.info('Step {}, validate {}: {}.'.format(global_step_val,
                                                                            validate_fn.func_name, validate_per))

                elif step % 200 == 0:
                    _, summary, global_step_val = sess.run(
                        [self.train_op, self.summary_op, self.global_step])
                    sv.summary_computed(sess, summary, global_step=global_step_val)
                else:
                    sess.run(self.train_op)

            logging.info("Exited training loop.")

        # Session will close automatically when with clause exits.
        # sess.close()
        sv.stop()
