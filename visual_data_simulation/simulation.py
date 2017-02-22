import numpy as np
import tensorflow as tf

import image_dataset.color_dataset as color

"""
Simulation that uses only the visual part of the data.

Input: [movie, user]
Output: [rating]

Learning: supervised
Function: softmax
Cost: cross-entropy
"""

def get_random_batch(data:iter, size:int = 100) -> tf.constant:
    """Returns a random batch of data"""
    batch = np.random.choice(data, size)
    np.ndarray(batch)


# --- Definitions ---
__COLOR_INPUT_SIZE__ = 0 # todo
__USER_INPUT_SIZE__ = 0 # todo
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
y_ = tf.placeholder(tf.float32, [None, 10])
# Cross Entropy function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# ----------------

# --- Session ---
with tf.InteractiveSession() as sess:
    for _ in range(1000):
        # run random batch of 100 data points
        batch_xs, batch_ys = None, None # mnist.train.next_batch(100) # todo
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

