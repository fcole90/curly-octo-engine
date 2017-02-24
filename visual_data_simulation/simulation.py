import numpy as np
import tensorflow as tf

import visual_data_simulation.simulation_setup as setup

"""
Simulation that uses only the visual part of the data.

Input: [movie, user]
Output: [rating]

Learning: supervised
Function: softmax
Cost: cross-entropy
"""

# --- Definitions ---
__COLOR_INPUT_SIZE__ = 30
__USER_INPUT_SIZE__ = 6
__INPUT_SIZE__ = __COLOR_INPUT_SIZE__ + __USER_INPUT_SIZE__
__RATING_SIZE__ = 5
__LEARNING_RATE__ = 0.5
# -------------------

# ---Setup ---
# None here means that it can have any length
x = tf.placeholder(tf.float32, [None, __INPUT_SIZE__])
W = tf.Variable(tf.zeros([__INPUT_SIZE__, __RATING_SIZE__]))
b = tf.Variable(tf.zeros([__RATING_SIZE__]))
# ------------

# --- Model ---
y = tf.matmul(x, W) + b
# -------------

# --- Training ---
# Correct output
y_ = tf.placeholder(tf.float32, [None, __RATING_SIZE__])
# Cross Entropy function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# ----------------

# --- Session ---
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(10000):
    # run random batch of 100 data points
    batch_xs, batch_ys = setup.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: setup.__input_data_test__(),
                                        y_: setup.__train_data_test__()}))

