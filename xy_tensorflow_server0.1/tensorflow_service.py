#!/usr/bin/python
# -*-coding:UTF-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --warm_up_image_file argument to specify any jpg image to warm-up
the TensorFlow model.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qsl

FLAGS = tf.app.flags.FLAGS

# global variables used to prevent TensorFlow from initializing for multi times
sess = tf.Session()
softmax_tensor = None

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
  'model_dir', '/tmp/imagenet',
  """Path to retrained_graph.pb, """
  """retrained_labels.txt, and """
  """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('warm_up_image_file', '',
                           """Absolute path to warm-up image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


# pylint: enable=line-too-long


class MyRequestHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    # e.g. "/?image_path=/root/mobike.jpg"
    path = self.path
    # e.g. "/root/mobike.jpg"
    image_path = parse_qsl(path[2:])[0][1]
    print('-------------------------------------------')
    print('Will process image: {}\n'.format(image_path))

    prediction_result = run_inference_on_image(image_path)
    message_return_to_client = ''
    #

    # send response status code
    self.send_response(200)

    # send headers
    self.send_header('Content-type', 'text/html')
    self.end_headers()

    # send message back to client, write content as utf-8 data
    self.wfile.write(bytes(prediction_result, "utf8"))
#    return prediction_result
    return


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
        FLAGS.model_dir, 'retrained_labels.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)



def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
          FLAGS.model_dir, 'retrained_graph.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def warm_up_model(image):
  """Warm-up TensorFlow model, to increase the inference speed of each time."""

  # the image used to warm-up TensorFlow model
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  global sess, softmax_tensor
  softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

  print('Warm-up start')
  for i in range(1):
    print('Warm-up for time {}'.format(i))
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

  print('Warm-up finished')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()


  global softmax_tensor, sess

  # record the start time of the actual prediction
  start_time = time.time()

  predictions = sess.run(softmax_tensor,
                         {'DecodeJpeg/contents:0': image_data})
  label_lines = [line.rstrip() for line
                 in tf.gfile.GFile("retrained_labels.txt")]
  a = []
  # a list which contains the content to return to HTTP client
  top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
  for node_id in top_k:
    human_string = label_lines[node_id]
    score = predictions[0][node_id]
    print("Prediction used time:{} S".format(time.time() - start_time))
    resuit = ('%s (score = %.5f)' % (human_string, score))
    return resuit

  #print('%s (score = %.5f)' % (human_string, score))




def main(_):
  warm_up_model(FLAGS.warm_up_image_file)

  server_address = ('127.0.0.1', 8080)
  httpd = HTTPServer(server_address, MyRequestHandler)
  print('TensorFlow service started')
  httpd.serve_forever()


if __name__ == '__main__':
  tf.app.run()
