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
import time

from readers import get_reader
from utils import get_input_data_tensors, DataPipeline, random_sample, load_sum_labels, load_features_mean_var
from tensorflow import flags, gfile, logging, app
from eval_util import calculate_gap

from os.path import join as path_join
import numpy as np

from inference import format_lines

FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640

MAX_TRAIN_STEPS = 1000000


def linear_classifier(data_pipeline=None, tr_data_fn=None, l2_regs=None,
                      validate_set=None, line_search=True):
    """
    Compute weights and biases of linear classifier using normal equation. With line search for best l2_reg.
    Args:
        data_pipeline: A namedtuple consisting of the following elements.
            reader, video-level features reader or frame-level features reader.
            data_pattern, File Glob of data set.
            batch_size, How many examples to handle per time.
            num_readers, How many IO threads to prefetch examples.
        tr_data_fn: a function that transforms input data.
        l2_regs: An array, each element represents how much the linear classifier weights should be penalized.
        validate_set: (data, labels) with dtype float32. The data set (numpy arrays) used to choose the best l2_reg.
            Sampled from whole validate set if necessary. If line_search is False, this argument is simply ignored.
        line_search: Boolean argument representing whether to do boolean search.

    Returns: Weights and biases fit on the given data set, where biases are appended as the last row.

    """
    logging.info('Entering linear classifier ...')
    output_dir = FLAGS.output_dir

    reader = data_pipeline.reader
    num_classes = reader.num_classes
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
            logging.info('validate_data has shape {}, validate_labels has shape {}.'.format(validate_data.shape,
                                                                                            validate_labels.shape))

            if (validate_data.shape[-1] != feature_size) or (validate_labels.shape[-1] != num_classes):
                raise ValueError('validate_set shape does not conforms with training set.')
    else:
        # Simply fit the training set. Make l2_regs have only one element. And ignore validate_set.
        if l2_regs is None:
            l2_regs = 0.001
        logging.info('No line search, l2_regs is {}.'.format(l2_regs))
        # Important! To make the graph construction successful.
        validate_data = np.zeros([1, feature_size], dtype=np.float32)
        validate_labels = np.zeros([1, num_classes], dtype=np.float32)

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

        video_count = tf.Variable(initial_value=0.0, name='video_count')
        features_sum = tf.Variable(initial_value=tf.zeros([feature_size]), name='features_sum')
        labels_sum = tf.Variable(initial_value=tf.zeros([num_classes]), name='labels_sum')

        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(data_pipeline, num_epochs=1, name_scope='input'))
        if tr_data_fn is None:
            video_batch_transformed = tf.identity(video_batch)
        else:
            video_batch_transformed = tr_data_fn(video_batch)

        with tf.name_scope('batch_increment'):
            video_batch_transformed_tr = tf.matrix_transpose(video_batch_transformed, name='X_Tr')
            video_labels_batch_cast = tf.cast(video_labels_batch, tf.float32)
            batch_norm_equ_1 = tf.matmul(video_batch_transformed_tr, video_batch_transformed,
                                         name='batch_norm_equ_1')
            # batch_norm_equ_1 = tf.add_n(tf.map_fn(lambda x: tf.einsum('i,j->ij', x, x),
            #                                       video_batch_transformed), name='X_Tr_X')
            batch_norm_equ_2 = tf.matmul(video_batch_transformed_tr, video_labels_batch_cast,
                                         name='batch_norm_equ_2')
            batch_video_count = tf.cast(tf.shape(video_batch)[0], tf.float32, name='batch_video_count')
            batch_features_sum = tf.reduce_sum(video_batch, axis=0, name='batch_features_sum')
            batch_labels_sum = tf.reduce_sum(video_labels_batch_cast, axis=0, name='batch_labels_sum')

        with tf.name_scope('update_ops'):
            update_norm_equ_1_op = tf.assign_add(norm_equ_1, batch_norm_equ_1)
            update_norm_equ_2_op = tf.assign_add(norm_equ_2, batch_norm_equ_2)
            update_video_count = tf.assign_add(video_count, batch_video_count)
            update_features_sum = tf.assign_add(features_sum, batch_features_sum)
            update_labels_sum = tf.assign_add(labels_sum, batch_labels_sum)

        with tf.control_dependencies([update_norm_equ_1_op, update_norm_equ_2_op, update_video_count,
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
            final_norm_equ_1_bot = tf.concat([features_sum, tf.expand_dims(video_count, 0)], 0)
            final_norm_equ_1 = tf.concat([final_norm_equ_1_top, tf.expand_dims(final_norm_equ_1_bot, 0)], 0,
                                         name='norm_equ_1')
            final_norm_equ_2 = tf.concat([norm_equ_2, tf.expand_dims(labels_sum, 0)], 0,
                                         name='norm_equ_2')

            # The last row is the biases.
            weights_biases = tf.matrix_solve(final_norm_equ_1, final_norm_equ_2, name='weights_biases')

            weights = weights_biases[:-1]
            biases = weights_biases[-1]

        with tf.name_scope('validate_loss'):
            validate_x_initializer = tf.placeholder(tf.float32, shape=validate_data.shape)
            validate_x = tf.Variable(initial_value=validate_x_initializer, trainable=False, collections=[],
                                     name='validate_data')

            validate_y_initializer = tf.placeholder(tf.float32, shape=validate_labels.shape)
            validate_y = tf.Variable(initial_value=validate_y_initializer, trainable=False, collections=[],
                                     name='validate_labels')

            predictions = tf.matmul(validate_x, weights) + biases
            loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(predictions, validate_y)), name='rmse')
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

    summary_writer = tf.summary.FileWriter(path_join(output_dir, 'linear_classifier'), graph=sess.graph)

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
        # Initialize validate data and labels in the graph.
        sess.run([validate_x.initializer, validate_y.initializer], feed_dict={
            validate_x_initializer: validate_data,
            validate_y_initializer: validate_labels
        })

        # Do true search.
        best_weights_val, best_biases_val = None, None
        best_l2_reg = 0
        min_loss = np.PINF

        for l2_reg in l2_regs:
            weights_val, biases_val, loss_val = sess.run([weights, biases, loss], feed_dict={l2_reg_ph: l2_reg})
            logging.info('l2_reg {} leads to rmse loss {}.'.format(l2_reg, loss_val))
            if loss_val < min_loss:
                best_weights_val, best_biases_val = weights_val, biases_val
                min_loss = loss_val
                best_l2_reg = l2_reg

    else:
        # Extract weights and biases of num_classes linear classifiers. Each column corresponds to a classifier.
        best_weights_val, best_biases_val = sess.run([weights, biases], feed_dict={l2_reg_ph: l2_regs})
        best_l2_reg, min_loss = l2_regs, None

    sess.close()

    logging.info('The best l2_reg is {} with rmse loss {}.'.format(best_l2_reg, min_loss))
    logging.info('Exiting linear classifier ...')

    return best_weights_val, best_biases_val


def log_reg_fit(train_data_pipeline, train_features_mean_var=None, validate_set=None,
                bootstrap=False, init_learning_rate=0.01, decay_steps=40000, decay_rate=0.95,
                epochs=None, l2_reg_rate=0.01, pos_weights=None, initial_weights=None, initial_biases=None):
    """
    Logistic regression.
    Args:
        train_data_pipeline: A namedtuple consisting of reader, data_pattern, batch_size and num_readers.
        train_features_mean_var: For train data standardization.
        validate_set: If not None, check validation loss regularly. Else, ignored.
        bootstrap: If True, sampling training examples with replacement by differential weighting.
        init_learning_rate: Decayed gradient descent parameter.
        decay_steps: Decayed gradient descent parameter.
        decay_rate: Decayed gradient descent parameter.
        epochs: Maximal epochs to use.
        l2_reg_rate: l2 regularizer rate.
        pos_weights: For imbalanced binary classes. Here, num_pos << num_neg, the weights should be > 1.0.
            If None, treated as 1.0 for all binary classifiers.
        initial_weights: If not None, the weights will be initialized with it.
        initial_biases: If not None, the biases will be initialized with it.
    Returns: None.
    """
    output_dir = FLAGS.output_dir
    # The dir where intermediate results and model checkpoints should be written.
    log_dir = path_join(output_dir, 'log_reg')

    reader = train_data_pipeline.reader
    batch_size = train_data_pipeline.batch_size
    num_classes = reader.num_classes
    feature_names = reader.feature_names
    feature_sizes = reader.feature_sizes
    feature_size = sum(feature_sizes)
    logging.info('Logistic regression uses {} features with dims {}.'.format(feature_names, feature_sizes))

    # Sample validate set.
    if validate_set is not None:
        validate_data, validate_labels = validate_set
    else:
        validate_data = np.zeros([1, feature_size], np.float32)
        validate_labels = np.zeros([1, num_classes], np.float32)

    # Build logistic regression graph and optimize it.
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')

        video_id_batch, video_batch, video_labels_batch, num_frames_batch = (
            get_input_data_tensors(train_data_pipeline, shuffle=True, num_epochs=epochs, name_scope='train_input'))

        # Define num_classes logistic regression models parameters.
        if initial_weights is None:
            weights = tf.Variable(initial_value=tf.truncated_normal([feature_size, num_classes]),
                                  dtype=tf.float32, name='weights')
        else:
            weights = tf.Variable(initial_value=initial_weights, dtype=tf.float32, name='weights')

        tf.summary.histogram('log_reg_weights', weights)

        if initial_biases is None:
            biases = tf.Variable(initial_value=tf.zeros([num_classes]), name='biases')
        else:
            biases = tf.Variable(initial_value=initial_biases, name='biases')

        tf.summary.histogram('log_reg_biases', biases)

        if train_features_mean_var is None:
            # normalized_video_batch = tf.nn.l2_normalize(video_batch, -1, name='normalized_video_batch')
            # For program consistency.
            features_mean = tf.Variable(initial_value=0.0, trainable=False, name='features_mean')
            features_var = tf.Variable(initial_value=1.0, trainable=False, name='features_var')
            output = tf.add(tf.matmul(video_batch, weights), biases, name='output')
        else:
            mean, var = train_features_mean_var
            features_mean = tf.Variable(initial_value=mean, trainable=False, name='features_mean')
            features_var = tf.Variable(initial_value=var, trainable=False, name='features_var')
            standardized_video_batch = tf.nn.batch_normalization(video_batch,
                                                                 mean=features_mean, variance=features_var,
                                                                 offset=None, scale=None, variance_epsilon=1e-12,
                                                                 name='standardized_video_batch')
            output = tf.add(tf.matmul(standardized_video_batch, weights), biases, name='output')

        float_labels = tf.cast(video_labels_batch, tf.float32, name='float_labels')
        pred_prob = tf.nn.sigmoid(output, name='pred_probability')

        with tf.name_scope('train_loss'):
            if pos_weights is None:
                loss_per_ex_label = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=float_labels, logits=output, name='x_entropy_per_ex_label')
            else:
                loss_per_ex_label = tf.nn.weighted_cross_entropy_with_logits(
                    targets=float_labels, logits=output, pos_weight=pos_weights, name='x_entropy_per_ex_label')

            # Sum over label set.
            loss_per_ex = tf.reduce_sum(loss_per_ex_label, axis=1, name='loss_per_ex')

            # Mean over batch.
            #  In addition to class weighting, example weighting is supported.
            if bootstrap:
                num_videos = tf.shape(loss_per_ex)[0]
                sample_indices = tf.random_uniform([num_videos], maxval=num_videos, dtype=tf.int32,
                                                   name='sample_indices')
                example_weights = tf.unsorted_segment_sum(tf.ones([num_videos]), sample_indices, num_videos,
                                                          name='example_weights')
                # bootstrap-weighted loss.
                weighted_loss_per_ex = tf.multiply(loss_per_ex, example_weights, name='weighted_loss_per_ex')
                loss = tf.reduce_mean(weighted_loss_per_ex, name='x_entropy')
            else:
                loss = tf.reduce_mean(loss_per_ex, name='x_entropy')

            # Add regularizer.
            weights_l2_loss_per_label = tf.reduce_sum(tf.square(weights), axis=0, name='weights_l2_loss_per_label')
            weights_l2_loss = tf.reduce_sum(weights_l2_loss_per_label, name='weights_l2_loss')

            final_loss = tf.add(loss, tf.multiply(l2_reg_rate, weights_l2_loss), name='final_loss')

            tf.summary.histogram('weights_l2_loss_per_label', weights_l2_loss_per_label)
            tf.summary.scalar('weights_l2_loss', weights_l2_loss)
            tf.summary.scalar('xentropy', loss)

        with tf.name_scope('optimization'):
            # Decayed learning rate.
            rough_num_examples_processed = tf.multiply(global_step, batch_size)
            adap_learning_rate = tf.train.exponential_decay(init_learning_rate, rough_num_examples_processed,
                                                            decay_steps, decay_rate, staircase=True,
                                                            name='adap_learning_rate')
            optimizer = tf.train.GradientDescentOptimizer(adap_learning_rate)
            train_op = optimizer.minimize(final_loss, global_step=global_step)

            tf.summary.scalar('learning_rate', adap_learning_rate)

        with tf.name_scope('validate'):
            validate_data_initializer = tf.placeholder(tf.float32, shape=validate_data.shape)
            validate_labels_initializer = tf.placeholder(tf.bool, shape=validate_labels.shape)
            validate_data_var = tf.Variable(initial_value=validate_data_initializer, trainable=False,
                                            collections=[], name='data')
            validate_labels_var = tf.Variable(initial_value=validate_labels_initializer, trainable=False,
                                              collections=[], name='labels')
            with tf.control_dependencies([validate_data_var.initializer, validate_labels_var.initializer]):
                set_validate_non_op = tf.no_op('set_validate_set')

            float_validate_labels = tf.cast(validate_labels_var, tf.float32, name='float_labels')

            if train_features_mean_var is None:
                validate_pred = tf.add(tf.matmul(validate_data_var, weights), biases)
            else:
                standardized_validate_data = tf.nn.batch_normalization(validate_data_var,
                                                                       mean=features_mean, variance=features_var,
                                                                       offset=None, scale=None, variance_epsilon=1e-12,
                                                                       name='standardized_validate_data')
                validate_pred = tf.add(tf.matmul(standardized_validate_data, weights), biases, name='validate_pred')

            validate_pred_prob = tf.nn.sigmoid(validate_pred, name='validate_pred_prob')
            validate_loss_per_ex_label = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=float_validate_labels, logits=validate_pred, name='x_entropy_per_ex_label')

            validate_loss_per_label = tf.reduce_mean(validate_loss_per_ex_label, axis=0,
                                                     name='x_entropy_per_label')

            validate_loss = tf.reduce_sum(validate_loss_per_label, name='x_entropy')

            tf.summary.histogram('xentropy_per_label', validate_loss_per_label)
            tf.summary.scalar('xentropy', validate_loss)

        # Add to collection. In inference, get collection and feed it with test data.
        tf.add_to_collection('video_input_batch', video_batch)
        tf.add_to_collection('predictions', pred_prob)

        summary_op = tf.summary.merge_all()

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Save trainable variables only.
    saver = tf.train.Saver(var_list=[weights, biases, global_step, features_mean, features_var],
                           max_to_keep=20, keep_checkpoint_every_n_hours=0.2)
    # To avoid summary causing memory usage peak, manually save summaries.
    sv = tf.train.Supervisor(graph=graph, init_op=init_op, logdir=log_dir, global_step=global_step, summary_op=None,
                             save_model_secs=600, saver=saver)

    with sv.managed_session() as sess:
        logging.info("Entering training loop...")
        # Set validate set.
        # normalized_validate_data = validate_data / np.clip(
        #     np.linalg.norm(validate_data , axis=-1, keepdims=True), 1e-6, np.PINF)
        sess.run(set_validate_non_op, feed_dict={validate_data_initializer: validate_data,
                                                 validate_labels_initializer: validate_labels})
        logging.info('Set validate set in the graph for future use.')
        for step in xrange(1, MAX_TRAIN_STEPS):
            if sv.should_stop():
                # Save the final model and break.
                saver.save(sess, save_path='{}_{}'.format(sv.save_path, 'final'))
                break

            if step % 1000 == 0:
                _, summary, validate_loss_val, global_step_val, validate_pred_prob_val = sess.run(
                    [train_op, summary_op, validate_loss, global_step, validate_pred_prob])
                # global_step will be found automatically.
                sv.summary_computed(sess, summary, global_step=global_step_val)

                validate_gap = calculate_gap(validate_pred_prob_val, validate_labels)
                logging.info('Step {}: validate gap: {}.'.format(global_step_val, validate_gap))
            elif step % 100 == 0:
                _, summary, validate_loss_val, global_step_val = sess.run(
                    [train_op, summary_op, validate_loss, global_step])
                # global_step will be found automatically.
                sv.summary_computed(sess, summary, global_step=global_step_val)
            else:
                sess.run(train_op)

    logging.info("Exited training loop.")
    # Session will close automatically when with clause exits.
    # sess.close()
    sv.stop()


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
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    validate_data_pattern = FLAGS.validate_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers
    init_with_linear_clf = FLAGS.init_with_linear_clf
    is_bootstrap = FLAGS.is_bootstrap

    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=num_readers)

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

    if init_with_linear_clf:
        # ...Start linear classifier...
        # Compute weights and biases of linear classifier using normal equation.
        # Linear search helps little.
        # Increase num_readers.
        _train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                            batch_size=batch_size, num_readers=4)
        linear_clf_weights, linear_clf_biases = linear_classifier(data_pipeline=_train_data_pipeline,
                                                                  l2_regs=[0.01, 0.1],
                                                                  validate_set=(validate_data, validate_labels),
                                                                  line_search=True)
        logging.info('linear classifier weights and biases with shape {}, {}'.format(linear_clf_weights.shape,
                                                                                     linear_clf_biases.shape))
        logging.debug('linear classifier weights and {} biases: {}.'.format(linear_clf_weights,
                                                                            linear_clf_biases))
        # ...Exit linear classifier...
    else:
        linear_clf_weights, linear_clf_biases = None, None

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)
    # Compute train data mean and std.
    train_features_mean, train_features_var = load_features_mean_var(reader)

    # Run logistic regression.
    log_reg_fit(train_data_pipeline, train_features_mean_var=(train_features_mean, train_features_var),
                validate_set=(validate_data, validate_labels), bootstrap=is_bootstrap,
                init_learning_rate=init_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate,
                epochs=epochs, l2_reg_rate=l2_reg_rate, pos_weights=pos_weights,
                initial_weights=linear_clf_weights, initial_biases=linear_clf_biases)


def inference(train_model_dir):
    out_file_location = FLAGS.output_file
    top_k = FLAGS.top_k
    test_data_pattern = FLAGS.test_data_pattern
    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    # TODO, bagging, load several trained models and average the predicstions.
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

                batch_predictions_prob = sess.run(pred_prob, feed_dict={video_input_batch: video_batch_val})

                # Write batch predictions to files.
                for line in format_lines(video_id_batch_val, batch_predictions_prob, top_k):
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
    train_model_dir = FLAGS.train_model_dir

    logging.set_verbosity(logging.INFO)

    if is_train:
        if is_tuning_hyper_para:
            raise NotImplementedError('Implementation is under progress.')
        else:
            train(init_learning_rate, decay_steps, decay_rate=decay_rate, l2_reg_rate=l2_reg_rate, epochs=train_epochs)
    else:
        inference(train_model_dir)


if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        '/Users/Sophie/Documents/youtube-8m-data/train/traina*.tfrecord',
                        'File glob for the training dataset.')

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

    # Set by the memory limit (52GB).
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
    flags.DEFINE_string('train_model_dir', '/tmp/video_level/log_reg',
                        'The directories where to load trained logistic regression models.')

    flags.DEFINE_string('output_file', '/tmp/video_level/log_reg/predictions.csv',
                        'The file to save the predictions to.')

    flags.DEFINE_integer('top_k', 20, 'How many predictions to output per video.')

    app.run()
