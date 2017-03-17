"""
This Python file contains pre trained neural networks.
 They are used to extract discriminative features from image.
 https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py
"""

import os
import tensorflow as tf
import numpy as np


class InceptionV3:
    def __init__(self, model_dir='inception-2015-12-05'):
        self.sess = tf.Session()
        self.model_dir = model_dir

        # load graph from file
        graph_def_file_path = os.path.join(self.model_dir, 'classify_image_graph_def.pb')
        with tf.gfile.FastGFile(graph_def_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def_tensors = tf.import_graph_def(graph_def, return_elements=['softmax:0', 'pool_3:0'],
                                                    name='classify_image_graph_def')
        self.softmax_tensor = graph_def_tensors[0]
        self.pool_3_tensor = graph_def_tensors[1]

    def extract_feature(self, image_path=''):
        if image_path == '':
            image_path = os.path.join(self.model_dir, 'cropped_panda.jpg')
        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        return np.squeeze(self.sess.run(self.pool_3_tensor, {'classify_image_graph_def/DecodeJpeg/contents:0': image_data}))

    def make_prediction(self, image_path=''):
        if image_path == '':
            image_path = os.path.join(self.model_dir, 'cropped_panda.jpg')

        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        return np.squeeze(self.sess.run(self.softmax_tensor, {'classify_image_graph_def/DecodeJpeg/contents:0': image_data}))


if __name__ == '__main__':

    inceptionV3 = InceptionV3()
    print('Features: {}'.format(inceptionV3.extract_feature()))
    print('# Features: {}'.format(len(inceptionV3.extract_feature())))
    print('Predictions: {}'.format(inceptionV3.make_prediction()))
