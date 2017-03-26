from subprocess import check_output
import tensorflow as tf
import csv

VIDEO_LEVEL_DATA_FODLER = "/Users/Sophie/Documents/youtube-8m-data/train/"

CSV_FILE_PATH = 'train.csv'
with open(CSV_FILE_PATH, 'w') as f:
    fieldnames = ['video_id', 'mean_rgb', 'mean_audio', 'labels']
    csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
    csv_writer.writeheader()

    train_tf_files = check_output(["ls", VIDEO_LEVEL_DATA_FODLER]).decode("UTF-8").split("\n")
    for file_name in train_tf_files:
        if file_name.endswith("tfrecord"):
            print("file_name: {}".format(file_name))
            for example in tf.python_io.tf_record_iterator(VIDEO_LEVEL_DATA_FODLER + file_name):
                tf_example_feature = tf.train.Example.FromString(example).features.feature

                video_id = tf_example_feature['video_id'].bytes_list.value[0].decode('UTF-8')
                labels = tf_example_feature['labels'].int64_list.value
                mean_rgb = tf_example_feature['mean_rgb'].float_list.value
                mean_audio = tf_example_feature['mean_audio'].float_list.value
                csv_writer.writerow({
                    'video_id': video_id, 'mean_rgb': ':'.join([str(e) for e in mean_rgb]),
                    'mean_audio': ':'.join([str(e) for e in mean_audio]), 'labels': ':'.join([str(e) for e in labels])
                })
f.close()
