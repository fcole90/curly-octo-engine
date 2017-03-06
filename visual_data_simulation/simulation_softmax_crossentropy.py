import numpy as np
import tensorflow as tf

import visual_data_simulation.simulation_setup as sim_setup
import tools.generic_helpers as gh_tools

"""
Simulation that uses only the visual part of the data.

Input: [user + movie]
Output: [rating] (one hot)

Learning: supervised
Function: softmax
Cost: cross-entropy
"""

# --- Definitions ---
__COLOR_INPUT_SIZE__ = 30
__USER_INPUT_SIZE__ = 6
__INPUT_SIZE__ = __USER_INPUT_SIZE__ + __COLOR_INPUT_SIZE__
__RATING_SIZE__ = 5
__LEARNING_RATE__ = 0.5
# -------------------

# ---Setup ---
# None here means that it can have any length
setup = sim_setup.Setup()
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
train_step = tf.train.GradientDescentOptimizer(0.35).minimize(cross_entropy)
# ----------------

# --- Session ---
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

iterations = 100000
for i in range(iterations):
    # run random batch of some data points
    batch_xs, batch_ys = setup.next_batch(10000)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%20 == 0 or i == iterations - 1:
        gh_tools.updating_text("\r[{0:.2f}%]".format((i + 1) / iterations * 100))
        # sys.stdout.write("\r[{0:.2f}%]".format((i + 1) / iterations * 100))
        # if i != iterations - 1:
        #     sys.stdout.flush()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

squared_error = tf.square(tf.subtract(tf.argmax(y, 1), tf.argmax(y_, 1)))
rmse = tf.sqrt(tf.reduce_mean(tf.cast(squared_error, tf.float32)))


correct_prediction_sugg = tf.equal(tf.greater_equal(tf.argmax(y, 1), 3),
                                   tf.greater_equal(tf.argmax(y_, 1), 3))
accuracy_sugg = tf.reduce_mean(tf.cast(correct_prediction_sugg, tf.float32))

experiment_feed_dict = {x: sim_setup.Setup.get_only_part(0, setup.dataset['test']),
                        y_: sim_setup.Setup.get_only_part(1, setup.dataset['test'])}

print("Accuracy: ", end='')
print(sess.run(accuracy, feed_dict=experiment_feed_dict))
print("RMSE: ", end='')
print(sess.run(rmse, feed_dict=experiment_feed_dict))

print("Accuracy on suggest/not suggest: ", end='')
print(sess.run(accuracy_sugg, feed_dict=experiment_feed_dict))
