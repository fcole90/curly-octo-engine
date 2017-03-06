import sys

import numpy as np
import tensorflow as tf

import visual_data_simulation.simulation_setup as sim_setup
import tools.generic_helpers as gh_tools

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


