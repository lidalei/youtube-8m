import tensorflow as tf

import readers
import utils


from tensorflow import flags, gfile, logging

FLAGS = flags.FLAGS

flags.DEFINE_string('model_type', 'video', 'video or frame level model')

# TODO, change according to running environment. Set as '' to be passed in python running command.
flags.DEFINE_string(
    "train_data_pattern", "/Users/Sophie/Documents/youtube-8m-data/train/train*.tfrecord",
    "File glob for the training dataset. If the files refer to Frame Level "
    "features (i.e. tensorflow.SequenceExample), then set --reader_type "
    "format. The (Sequence)Examples are expected to have 'rgb' byte array "
    "sequence feature as well as a 'labels' int64 context feature.")

flags.DEFINE_string('feature_names', 'mean_rgb', 'features to be used, separated by ,.')

flags.DEFINE_string('feature_sizes', '1024', 'dimensions of features to be used, separated by ,.')

flags.DEFINE_integer('batch_size', 1024, 'size of batch processing')


def get_reader():
    """
    similar to train.get_reader()
    :return:
    """
    feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(FLAGS.feature_names, FLAGS.feature_sizes)

    if FLAGS.model_type == 'video':
        return readers.YT8MAggregatedFeatureReader(feature_sizes=feature_sizes, feature_names=feature_names)
    elif FLAGS.model_type == 'frame':
        return readers.YT8MFrameFeatureReader(feature_sizes=feature_sizes, feature_names=feature_names)
    else:
        raise NotImplementedError('Not supported model type. Supported ones are video and frame.')


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1, num_epochs=1):
    """Creates the section of the graph which reads the input data.

    Similar to the same-name function in train.py.
    Args:
        reader: A class which parses the input data.
        data_pattern: A 'glob' style path to the data files.
        batch_size: How many examples to process at a time.
        num_readers: How many I/O threads to use.
        num_epochs: How many passed to go through the data files.

    Returns:
        A tuple containing the features tensor, labels tensor, and optionally a
        tensor containing the number of frames per video. The exact dimensions
        depend on the reader being used.

    Raises:
        IOError: If no files matching the given pattern were found.
    """
    # Adapted from namesake function in inference.py.
    with tf.name_scope("input"):
        files = gfile.Glob(data_pattern)
        if not files:
            raise IOError("Unable to find input files. data_pattern='{}'".format(data_pattern))
        logging.info("number of input files: {}".format(len(files)))
        # Pass test data once. Thus, num_epochs is set as 1.
        filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs, shuffle=False)
        examples_and_labels = [reader.prepare_reader(filename_queue) for _ in range(num_readers)]

        video_id_batch, video_batch, video_labels, num_frames_batch = (
            tf.train.batch_join(examples_and_labels,
                                batch_size=batch_size,
                                allow_smaller_final_batch=True,
                                enqueue_many=True))
        return video_id_batch, video_batch, video_labels, num_frames_batch


if __name__ == '__main__':
    reader = get_reader()

    batch_data = get_input_data_tensors(reader, FLAGS.train_data_pattern)

    command = raw_input("Help:")
    if command == ':q':
        exit(0)