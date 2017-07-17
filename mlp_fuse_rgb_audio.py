"""
Multi-layer Perceptron (MLP).

Note:
    1. Normalizing features will lead to much faster convergence but worse performance.
    2. Instead, standard scaling features will help achieve better performance.
    
Credit.
    batch_norm_layer come from https://stackoverflow.com/a/44020133 and
    https://github.com/pkmital/tensorflow_tutorials/blob/master/python/libs/batch_norm.py.
    Thanks a lot!
"""
import tensorflow as tf
import numpy as np

from readers import get_reader
from utils import DataPipeline, random_sample, load_features_mean_var, load_sum_labels
from utils import MakeSummary, get_input_data_tensors
from tensorflow import flags, logging, app
from utils import gap_fn

from os.path import join as path_join

FLAGS = flags.FLAGS
NUM_TRAIN_EXAMPLES = 4906660
# TODO
NUM_VALIDATE_EXAMPLES = None
NUM_TEST_EXAMPLES = 700640


def batch_norm_layer(x, is_training, decay=0.999, epsilon=1e-5, scope='bn'):
    """
    Performs a batch normalization layer

    Args:
        x: input tensor
        is_training: python boolean value
        decay: the moving average decay
        epsilon: the variance epsilon - a small float number to avoid dividing by 0
        scope: scope name

    Returns:
        The ops of a batch normalization layer
    """
    with tf.name_scope(scope):
        shape = x.get_shape().as_list()
        size = shape[-1]
        # beta: a trainable shift value
        beta = tf.Variable(initial_value=tf.zeros([size]), trainable=True, name='beta')
        # gamma: a trainable scale factor
        gamma = tf.Variable(initial_value=tf.ones([size]), trainable=True, name='gamma')

        # tf.nn.moments == Calculate the mean and the variance of the tensor x.
        # The last dimension contains values to compute mean.
        batch_mean, batch_var = tf.nn.moments(x, range(len(shape)-1), name='moments')

        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        # apply creates the shadow variables, and add ops to maintain moving averages of mean and variance.
        maintain_averages_op = ema.apply([batch_mean, batch_var])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_averages_op)

        # Inference uses population average and variance.
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        mean, var = tf.cond(
            is_training, lambda: (batch_mean, batch_var), lambda: (ema_mean, ema_var)
        )

        bn = tf.nn.batch_normalization(x, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    return bn


def train(train_data_pipeline, epochs=None, pos_weights=None, l1_reg_rate=None, l2_reg_rate=None,
          init_learning_rate=0.01, decay_steps=NUM_TRAIN_EXAMPLES, decay_rate=0.95, bootstrap=False,
          validate_set=None, validate_fn=None, logdir='/tmp/mlp_fuse'):
    """
    Args:
        train_data_pipeline:
        epochs: The maximal epochs to pass all training data.
        pos_weights:
        l1_reg_rate:
        l2_reg_rate: l2 regularization rate.
        init_learning_rate: Initial learning rate.
        decay_steps: How many training steps to decay learning rate once.
        decay_rate: How much to decay learning rate.
        bootstrap: To sample data with replacement or not.
        validate_set:
        validate_fn:
        logdir:
    """
    reader = train_data_pipeline.reader
    num_classes = reader.num_classes
    feature_sizes = reader.feature_sizes
    # Assume mean_rgb and mean_audio are used.
    feature_size = sum(feature_sizes)
    batch_size = train_data_pipeline.batch_size

    # Load data mean and variance.
    features_mean, features_var = load_features_mean_var(reader)

    with tf.Graph().as_default() as g:
        global_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='global_step')

        id_batch, raw_features_batch, labels_batch, num_frames_batch = (
            get_input_data_tensors(train_data_pipeline, shuffle=True, num_epochs=epochs, name_scope='input'))

        # Used for dropout and batch normalization.
        phase_train_pl = tf.placeholder(tf.bool, [], name='phase_train_pl')

        with tf.name_scope('standard_scale'):
            mean = tf.Variable(initial_value=features_mean, trainable=False, name='features_mean')
            var = tf.Variable(initial_value=features_var, trainable=False, name='features_var')
            standardized = tf.nn.batch_normalization(raw_features_batch, mean=mean, variance=var,
                                                     offset=None, scale=None, variance_epsilon=1e-12,
                                                     name='standardized')

        prev_layer_activation = standardized
        # First Hidden layers---#
        layer_idx = 1

        # mean_rgb
        layer_name_rgb = 'hidden_{}_rgb'.format(layer_idx)
        layer_size_rgb = 2048

        with tf.name_scope(layer_name_rgb):
            # Initialize weights based on fan-in.
            weights = tf.Variable(initial_value=tf.truncated_normal(
                [1024, layer_size_rgb], stddev=1.0 / np.sqrt(1024)), name='weights')
            biases = tf.Variable(initial_value=tf.zeros([layer_size_rgb]), name='biases')

            inner_product = tf.matmul(prev_layer_activation[:, :1024], weights) + biases

            # Add batch normalization. It doesn't change tensor shape.
            bn = batch_norm_layer(inner_product, phase_train_pl, scope='bn_{}'.format(layer_idx))

            hidden_activation_rgb = tf.tanh(bn)

            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)

            # Add to summary.
            # tf.summary.histogram('model/weights', weights)
            # tf.summary.histogram('model/biases', biases)
            # tf.summary.histogram('model/activation', hidden_activation_rgb)

        # mean_audio
        layer_name_audio = 'hidden_{}_audio'.format(layer_idx)
        layer_size_audio = 256

        with tf.name_scope(layer_name_audio):
            # Initialize weights based on fan-in.
            weights = tf.Variable(initial_value=tf.truncated_normal(
                [128, layer_size_audio], stddev=1.0 / np.sqrt(128)), name='weights')
            biases = tf.Variable(initial_value=tf.zeros([layer_size_audio]), name='biases')

            inner_product = tf.matmul(prev_layer_activation[:, 1024:], weights) + biases

            # Add batch normalization. It doesn't change tensor shape.
            bn = batch_norm_layer(inner_product, phase_train_pl, scope='bn_{}'.format(layer_idx))

            hidden_activation_audio = tf.tanh(bn)

            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)

            # Add to summary.
            # tf.summary.histogram('model/weights', weights)
            # tf.summary.histogram('model/biases', biases)
            # tf.summary.histogram('model/activation', hidden_activation_audio)
        # ----End first layer.

        prev_layer_size = layer_size_rgb + layer_size_audio
        prev_layer_activation = tf.concat([hidden_activation_rgb, hidden_activation_audio], 1,
                                          name='hidden_{}_activation'.format(layer_idx))
        keep_prob = tf.cond(phase_train_pl, lambda: tf.constant(0.5, name='keep_prob'),
                            lambda: tf.constant(1.0, name='keep_prob'))
        prev_layer_activation = tf.nn.dropout(prev_layer_activation, keep_prob)

        # Second Hidden layers---#
        layer_idx = 2
        layer_name = 'hidden_{}'.format(layer_idx)
        layer_size = 1200

        with tf.name_scope(layer_name):
            # Initialize weights based on fan-in.
            weights = tf.Variable(initial_value=tf.truncated_normal(
                [prev_layer_size, layer_size], stddev=1.0 / np.sqrt(prev_layer_size)), name='weights')
            biases = tf.Variable(initial_value=tf.zeros([layer_size]), name='biases')

            inner_product = tf.matmul(prev_layer_activation, weights) + biases

            # Add batch normalization. It doesn't change tensor shape.
            bn = batch_norm_layer(inner_product, phase_train_pl, scope='bn_{}'.format(layer_idx))

            hidden_activation = tf.tanh(bn)

            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)

            # Add to summary.
            # tf.summary.histogram('model/weights', weights)
            # tf.summary.histogram('model/biases', biases)
            # tf.summary.histogram('model/activation', hidden_activation)
        # ----End Second layer.
        prev_layer_size = layer_size
        prev_layer_activation = hidden_activation

        # One-vs-all logistic regression layer.
        with tf.name_scope('one_vs_all_log_reg'):
            # Define num_classes logistic regression models parameters.
            weights = tf.Variable(initial_value=tf.truncated_normal(
                [prev_layer_size, num_classes], stddev=1.0 / np.sqrt(prev_layer_size)),
                dtype=tf.float32, name='weights')
            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights)
            # tf.summary.histogram('model/weights', weights)

            biases = tf.Variable(initial_value=tf.zeros([num_classes]), name='biases')
            # tf.summary.histogram('model/biases', biases)

            output = tf.add(tf.matmul(prev_layer_activation, weights), biases, name='output')

            float_labels = tf.cast(labels_batch, tf.float32, name='float_labels')
            pred_prob = tf.nn.sigmoid(output, name='pred_probability')

        with tf.name_scope('train'):
            if pos_weights is None:
                loss_per_ex_label = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=float_labels, logits=output, name='x_entropy_per_ex_label')
            else:
                loss_per_ex_label = tf.nn.weighted_cross_entropy_with_logits(
                    targets=float_labels, logits=output, pos_weight=pos_weights,
                    name='x_entropy_per_ex_label')

            # Sum over label set.
            loss_per_ex = tf.reduce_sum(loss_per_ex_label, axis=1, name='loss_per_ex')

            #  In addition to class weighting, example weighting is supported.
            if bootstrap:
                num_examples = tf.shape(loss_per_ex)[0]
                sample_indices = tf.random_uniform([num_examples], maxval=num_examples, dtype=tf.int32,
                                                   name='sample_indices')
                example_weights = tf.unsorted_segment_sum(tf.ones([num_examples]), sample_indices, num_examples,
                                                          name='example_weights')
                # bootstrap-weighted loss.
                weighted_loss_per_ex = tf.multiply(loss_per_ex, example_weights, name='weighted_loss_per_ex')
                # Mean over batch.
                loss = tf.reduce_mean(weighted_loss_per_ex, name='x_entropy')
            else:
                # Mean over batch.
                loss = tf.reduce_mean(loss_per_ex, name='x_entropy')

            tf.summary.scalar('loss/xentropy', loss)

            # Before computing gradient, update batch mean and variance. From train.py.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                with tf.control_dependencies(update_ops):
                    barrier = tf.no_op(name="gradient_barrier")
                    with tf.control_dependencies([barrier]):
                        loss = tf.identity(loss)

            # Add regularization.
            reg_losses = []
            # tf.GraphKeys.REGULARIZATION_LOSSES contains all variables to regularize.
            to_regularize = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if l1_reg_rate:
                l1_reg_losses = [tf.reduce_mean(tf.abs(w)) for w in to_regularize]
                l1_reg_loss = tf.add_n(l1_reg_losses, name='l1_reg_loss')
                tf.summary.scalar('loss/l1_reg_loss', l1_reg_loss)
                reg_losses.append(tf.multiply(l1_reg_rate, l1_reg_loss))

            if l2_reg_rate:
                l2_reg_losses = [0.5 * tf.reduce_mean(tf.square(w)) for w in to_regularize]
                l2_reg_loss = tf.add_n(l2_reg_losses, name='l2_loss')
                tf.summary.scalar('loss/l2_reg_loss', l2_reg_loss)
                reg_losses.append(tf.multiply(l2_reg_rate, l2_reg_loss))
            if len(reg_losses) > 0:
                reg_loss = tf.add_n(reg_losses, name='reg_loss')
            else:
                reg_loss = tf.constant(0.0, name='zero_reg_loss')

            final_loss = tf.add(loss, reg_loss, name='final_loss')

        with tf.name_scope('optimization'):
            # Decayed learning rate.
            # rough_num_examples_processed = tf.multiply(global_step, batch_size)
            # adap_learning_rate = tf.train.exponential_decay(init_learning_rate,
            #                                                 rough_num_examples_processed,
            #                                                 decay_steps, decay_rate, staircase=True,
            #                                                 name='adap_learning_rate')
            # tf.summary.scalar('learning_rate', adap_learning_rate)
            # GradientDescentOptimizer
            # optimizer = tf.train.GradientDescentOptimizer(adap_learning_rate)
            # MomentumOptimizer
            # optimizer = tf.train.MomentumOptimizer(adap_learning_rate, 0.9, use_nesterov=True)
            # RMSPropOptimizer
            optimizer = tf.train.RMSPropOptimizer(learning_rate=init_learning_rate)
            # Encapsulate optimizer inside the MovingAverageOptimizer.
            opt = tf.contrib.opt.MovingAverageOptimizer(optimizer)
            train_op = opt.minimize(final_loss, global_step=global_step)

        summary_op = tf.summary.merge_all()
        # summary_op = tf.constant(1.0)

        # num_epochs needs local variables to be initialized. Put this line after all other graph construction.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Add to collection. In inference, get collection and feed it with test data.
        tf.add_to_collection('phase_train_pl', phase_train_pl)
        tf.add_to_collection('raw_features_batch', raw_features_batch)
        tf.add_to_collection('predictions', pred_prob)

        # To save global variables and savable objects, i.e., var_list is None.
        # Using rbf transform will also save centers and scaling factors.
        # saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=0.15)
        saver = opt.swapping_saver(max_to_keep=50, keep_checkpoint_every_n_hours=0.15)

    # Start or restore training.
    # To avoid summary causing memory usage peak, manually save summaries.
    sv = tf.train.Supervisor(graph=g, init_op=init_op, logdir=logdir,
                             global_step=global_step, summary_op=None,
                             save_model_secs=600, saver=saver)

    with sv.managed_session() as sess:
        logging.info("Entering training loop...")
        for step in xrange(1000000):
            if sv.should_stop():
                # Save the final model and break.
                saver.save(sess, save_path='{}_{}'.format(sv.save_path, 'final'))
                break

            if step % 800 == 0:
                if validate_fn is not None:
                    _, summary, train_pred_prob_batch, train_labels_batch, global_step_val = sess.run(
                        [train_op, summary_op, pred_prob, labels_batch, global_step],
                        feed_dict={phase_train_pl: True})

                    # Evaluate on train data.
                    train_per = validate_fn(predictions=train_pred_prob_batch, labels=train_labels_batch)
                    sv.summary_writer.add_summary(
                        MakeSummary('train/{}'.format(validate_fn.func_name), train_per),
                        global_step_val)
                    logging.info('Step {}, train {}: {}.'.format(global_step_val,
                                                                 validate_fn.func_name, train_per))
                else:
                    _, summary, global_step_val = sess.run(
                        [train_op, summary_op, global_step], feed_dict={phase_train_pl: True})

                # Add train summary.
                sv.summary_computed(sess, summary, global_step=global_step_val)

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
                                [loss, pred_prob], feed_dict={
                                    raw_features_batch: validate_data[start_ind:end_ind],
                                    labels_batch: validate_labels[start_ind:end_ind]})

                            ith_validate_per = validate_fn(predictions=ith_predictions,
                                                           labels=validate_labels[start_ind:end_ind])
                            validate_loss_vals.append(ith_validate_loss_val * (end_ind - start_ind))
                            validate_pers.append(ith_validate_per * (end_ind - start_ind))
                        else:
                            ith_validate_loss_val = sess.run(loss, feed_dict={
                                raw_features_batch: validate_data[start_ind:end_ind],
                                labels_batch: validate_labels[start_ind:end_ind],
                                phase_train_pl: False})

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
            else:
                sess.run(train_op, feed_dict={phase_train_pl: True})

        logging.info("Exited training loop.")

    # Session will close automatically when with clause exits.
    # sess.close()
    sv.stop()


def main(unused_argv):
    logging.set_verbosity(logging.INFO)

    start_new_model = FLAGS.start_new_model
    logdir = FLAGS.logdir

    init_learning_rate = FLAGS.init_learning_rate
    decay_steps = FLAGS.decay_steps
    decay_rate = FLAGS.decay_rate
    l1_reg_rate = FLAGS.l1_reg_rate
    l2_reg_rate = FLAGS.l2_reg_rate
    is_bootstrap = FLAGS.is_bootstrap
    train_epochs = FLAGS.train_epochs

    model_type, feature_names, feature_sizes = FLAGS.model_type, FLAGS.feature_names, FLAGS.feature_sizes
    reader = get_reader(model_type, feature_names, feature_sizes)
    train_data_pattern = FLAGS.train_data_pattern
    validate_data_pattern = FLAGS.validate_data_pattern
    batch_size = FLAGS.batch_size
    num_readers = FLAGS.num_readers

    # Increase num_readers.
    validate_data_pipeline = DataPipeline(reader=reader, data_pattern=validate_data_pattern,
                                          batch_size=batch_size, num_readers=num_readers)

    # Sample validate set for line search in linear classifier or logistic regression early stopping.
    _, validate_data, validate_labels, _ = random_sample(0.05, mask=(False, True, True, False),
                                                         data_pipeline=validate_data_pipeline)

    train_data_pipeline = DataPipeline(reader=reader, data_pattern=train_data_pattern,
                                       batch_size=batch_size, num_readers=num_readers)

    if start_new_model and tf.gfile.Exists(logdir):
        logging.info('Starting a new model...')
        # Start new model, delete existing checkpoints.
        try:
            tf.gfile.DeleteRecursively(logdir)
        except tf.errors.OpError:
            logging.error('Failed to delete dir {}.'.format(logdir))
        else:
            logging.info('Succeeded to delete train dir {}.'.format(logdir))

    # Set pos_weights for extremely imbalanced situation in one-vs-all classifiers.
    try:
        # Load sum_labels in training set, numpy float format to compute pos_weights.
        train_sum_labels = load_sum_labels()
        # num_neg / num_pos, assuming neg_weights === 1.0.
        pos_weights = np.sqrt(float(NUM_TRAIN_EXAMPLES) / train_sum_labels - 1.0)
        logging.info('Computing pos_weights based on sum_labels in train set successfully.')
    except IOError:
        logging.error('Cannot load train sum_labels. Use default value.')
        pos_weights = None
    finally:
        logging.warn('Not to use positive weights.')
        pos_weights = None

    train(train_data_pipeline, epochs=train_epochs, pos_weights=pos_weights, l1_reg_rate=l1_reg_rate,
          l2_reg_rate=l2_reg_rate, init_learning_rate=init_learning_rate, bootstrap=is_bootstrap,
          validate_set=(validate_data, validate_labels), validate_fn=gap_fn, logdir=logdir)

if __name__ == '__main__':
    flags.DEFINE_string('model_type', 'video', 'video or frame level model')

    flags.DEFINE_string('yt8m_home', '/Users/Sophie/Documents/youtube-8m-data',
                        'YT8M dataset home.')
    # Set as '' to be passed in python running command.
    flags.DEFINE_string('train_data_pattern',
                        path_join(FLAGS.yt8m_home, 'train_validate/train*.tfrecord'),
                        'File glob for the training data set.')

    flags.DEFINE_string('validate_data_pattern',
                        path_join(FLAGS.yt8m_home, 'train_validate/validate*.tfrecord'),
                        'Validate data pattern, to be specified when doing hyper-parameter tuning.')

    # mean_rgb,mean_audio
    flags.DEFINE_string('feature_names', 'mean_rgb,mean_audio', 'Features to be used, separated by ,.')

    # 1024,128
    flags.DEFINE_string('feature_sizes', '1024,128', 'Dimensions of features to be used, separated by ,.')

    flags.DEFINE_integer('batch_size', 1024, 'Size of batch processing.')
    flags.DEFINE_integer('num_readers', 2, 'Number of readers to form a batch.')

    flags.DEFINE_bool('start_new_model', True, 'To start a new model or restore from output dir.')

    flags.DEFINE_float('init_learning_rate', 0.001, 'Float variable to indicate initial learning rate.')

    flags.DEFINE_integer('decay_steps', NUM_TRAIN_EXAMPLES,
                         'Float variable indicating no. of examples to decay learning rate once.')

    flags.DEFINE_float('decay_rate', 0.95, 'Float variable indicating how much to decay.')

    flags.DEFINE_float('l1_reg_rate', None, 'l1 regularization rate.')

    flags.DEFINE_float('l2_reg_rate', None, 'l2 regularization rate.')

    flags.DEFINE_bool('is_bootstrap', False, 'Boolean variable indicating using bootstrap or not.')

    flags.DEFINE_integer('train_epochs', 20, 'Training epochs, one epoch means passing all training data once.')

    # Added current timestamp.
    flags.DEFINE_string('logdir', '/tmp/video_level/mlp_fuse',
                        'The directory where intermediate and model checkpoints should be written.')

    app.run()
