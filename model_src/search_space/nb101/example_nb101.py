# Copyright 2019 The Google Research Authors.
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

"""Runnable example, as shown in the README.md."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
#from nasbench import api

from model_src.search_space.nb101.nb101_builder import build_model
from model_src.search_space.nb101.model_spec_nb101 import ModelSpec
import model_src.search_space.nb101.config_nb101 as config

import tensorflow as tf


INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


def nb101_model_maker(ops, adj_mat):
    model_spec = ModelSpec(adj_mat, ops)
    example_config = config.build_config()
    example_config['use_tpu'] = False
    return build_model(model_spec, example_config)


def main(argv):
  del argv  # Unused


  # Create an Inception-like module (5x5 convolution replaced with two 3x3
  # convolutions).
  model_spec = ModelSpec(
      # Adjacency matrix of the module
      matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
              [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]],   # output layer
      # Operations at the vertices of the module, matches order of matrix
      ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

  example_config = config.build_config()
  example_config['use_tpu'] =  False
  
  model1 = build_model(model_spec,example_config)
  image_batch = tf.ones([1,32,32,3])
  x = tf.identity(image_batch,"input")
  out = model1(x)
  with  tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(out))
  sess.close()

  # Load the data from file (this will take some time)
  # NASBENCH_TFRECORD = '/home/fabian/Code/NASB101/nasbench/nasbench_full.tfrecord'
  # NASBENCH_TFRECORD = '/home/fabian/Code/NASB101/nasbench/nasbench_only108.tfrecord'
  # nasbench = api.NASBench(NASBENCH_TFRECORD)

  #   # Query this model from dataset, returns a dictionary containing the metrics
  #   # associated with this model.
  # print('Querying an Inception-like model.')
  # data = nasbench.query(model_spec)
  # print(data)
  

if __name__ == '__main__':
  app.run(main)
