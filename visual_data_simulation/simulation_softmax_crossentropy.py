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
__ONE_HOT_DATA__ = "one_hot"
__DECIMAL_DATA__ = "decimal"
__COLOR_INPUT_SIZE__ = 30
__USER_INPUT_SIZE__ = 6
__INPUT_SIZE__ = __USER_INPUT_SIZE__ + __COLOR_INPUT_SIZE__

# ---  Settings ---
__LABELS_DATA_TYPE__ = __ONE_HOT_DATA__
__LABELS_SIZE__ = 1 if __LABELS_DATA_TYPE__ is __DECIMAL_DATA__ else 5

batch_size = 200
__LEARNING_RATE__ = 0.01
__ALPHA__ = 0.5
myOptimizer = tf.train.GradientDescentOptimizer


iterations = 10000000

# -------------------

setup = sim_setup.Setup(user_data_function="clusters",
                        use_cache=True,
                        labels_data_type=__LABELS_DATA_TYPE__)
# None here means that it can have any length
x = tf.placeholder(tf.float32, [None, __INPUT_SIZE__])
x2 = tf.placeholder(tf.float32, [None, __LABELS_SIZE__])


# ---Setup ---

W = tf.Variable(tf.random_normal([__INPUT_SIZE__, __LABELS_SIZE__]))
b = tf.Variable(tf.constant([0.01] * __LABELS_SIZE__))
def first_step(x):
    return (tf.add(tf.matmul(x, W), b))

W2 = tf.Variable(tf.random_normal([__LABELS_SIZE__, __LABELS_SIZE__]))
b2 = tf.Variable(tf.constant([0.01] * __LABELS_SIZE__))
def second_step(x):
    return tf.add(tf.matmul(x, W2), b2)
# ------------

# --- Model ---

y1 = first_step(x)
y2 = second_step(y1)

y = y1
# -------------

# --- Training ---
# Correct output
y_ = tf.placeholder(tf.float32, [None, __LABELS_SIZE__])

# --- Differentiable argmax ---


def max_amplifier(x, amp=10, axis=None, transpose=False):
    """
    Amplifies to the greatest element of a tensor while reducing the others.

    The formula is: (x / max(x))^amp

    Parameters
    ----------
    x: tf.Tensor
        The tensor to amplify. Should have numeric type.

    amp: int (optional)
        The power of the amplification.

    axis: (optional)
        The dimensions to reduce. If None (the default), reduces all dimensions.

    Returns
    -------
    The amplified tensor.

    """
    pos_x = tf.add(x, tf.abs(tf.reduce_min(x)))
    amplified_x = tf.pow(tf.divide(tf.transpose(pos_x),
                                   tf.add(tf.reduce_max(pos_x, axis=axis),
                                          0.1e-100))
                         , amp)

    if transpose:
        return amplified_x
    else:
        return tf.transpose(amplified_x)


def differentiable_argmax(x, power=5, decoder=None, axis=None):
    if not decoder:
        decoder = tf.constant([list(range(5))], dtype=tf.float32)
    amplified_x = max_amplifier(x, power, axis=axis, transpose=True)
    return tf.matmul(decoder, amplified_x)


# Cross Entropy function
if __LABELS_DATA_TYPE__ is __ONE_HOT_DATA__:
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y1))
    mse_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(differentiable_argmax(y1, axis=1, power=100),
                                          differentiable_argmax(y_, axis=1, power=100)))))
    alpha = __ALPHA__
    beta = 1.0 - alpha
    combo = tf.add(tf.multiply(mse_loss, alpha),
                   tf.multiply(cross_entropy, beta))
    train_step = tf.train.GradientDescentOptimizer(__LEARNING_RATE__).minimize(combo)
    train_step2 = None#tf.train.GradientDescentOptimizer(__LEARNING_RATE__).minimize(mse_loss)
else:
    mse_cost = tf.cast(tf.reduce_mean(tf.square(tf.subtract(y2, y_))), tf.float32)
    train_step = myOptimizer(__LEARNING_RATE__).minimize(mse_cost)
    train_step2 = None
# ----------------

# --- Session ---
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# --- Testing functions ---
if __LABELS_DATA_TYPE__ is __ONE_HOT_DATA__:
    error = tf.subtract(tf.argmax(y, 1), tf.argmax(y_, 1))
    d_error = tf.subtract(differentiable_argmax(y, axis=1, power=100),
                          differentiable_argmax(y_, axis=1, power=100))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction_sugg = tf.equal(tf.greater_equal(tf.argmax(y, 1), 3),
                                       tf.greater_equal(tf.argmax(y_, 1), 3))


else:
    error = tf.scalar_mul(5, tf.subtract(y, y_))
    d_error = error
    correct_prediction = tf.equal(tf.ceil(tf.scalar_mul(5, y)),
                                  tf.ceil(tf.scalar_mul(5,y_)))
    correct_prediction_sugg = tf.equal(tf.greater_equal(y, 0.6),
                                       tf.greater_equal(y_, 0.6))


rmse = tf.sqrt(tf.reduce_mean(tf.cast(tf.square(error), tf.float32)))
d_rmse = tf.sqrt(tf.reduce_mean(tf.cast(tf.square(d_error), tf.float32)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_sugg = tf.reduce_mean(tf.cast(correct_prediction_sugg, tf.float32))

experiment_feed_dict = {x: sim_setup.Setup.get_only_part(0, setup.dataset['test']),
                        y_: sim_setup.Setup.get_only_part(1, setup.dataset['test'])}
# --------------------------

# --- Execution Loop ---

min_acc = (1.0, -1)
max_acc = (0.0, -1)
min_acc_sugg = (1.0, -1)
max_acc_sugg = (0.0, -1)
min_rmse = (10, -1)
max_rmse = (0, -1)

print("Batch size:", batch_size)
print("Learning rate:", __LEARNING_RATE__)
print("Labels data type:", __LABELS_DATA_TYPE__)
print("Optimizer:", myOptimizer.__name__)

for i in range(iterations):
    # run random batch of some data points
    batch_xs, batch_ys = setup.next_batch(batch_size, use_permutation=True)
    train_dict = {x: batch_xs, y_: batch_ys}
    sess.run(train_step, feed_dict = train_dict)
    # print(sess.run(combo, feed_dict = train_dict))
    if train_step2:
        sess.run(train_step2, feed_dict={x2: sess.run(y1, feed_dict=train_dict), y_: batch_ys})
        experiment_feed_dict[x2] = sess.run(y1, experiment_feed_dict)


    # # Update percentage
    # if i%20 == 0 or i == iterations - 1:
    #     gh_tools.updating_text("\r[{0:.2f}%]".format((i + 1) / iterations * 100))

    # Update estimators
    if i % 20 == 0 or i == iterations - 1:

        update = lambda x_new, cmp, x_old: x_new if cmp(x_new[0], x_old[0]) else x_old
        lt = lambda x_1, x_2: x_1 < x_2
        gt = lambda x_1, x_2: x_1 > x_2

        acc_run = sess.run(accuracy, feed_dict=experiment_feed_dict)
        rmse_run = sess.run(rmse, feed_dict=experiment_feed_dict)
        d_rmse_run = sess.run(d_rmse, feed_dict=experiment_feed_dict)
        acc_sugg_run = sess.run(accuracy_sugg, feed_dict=experiment_feed_dict)

        print("[{:10d}]".format(i), "Acc:", "{:2.4f}".format(acc_run), end=' | ')
        print("Acc s/ns:", "{:2.4f}".format(acc_sugg_run), end=' | ')
        print("d_RMSE: {:2.4f}".format(d_rmse_run), end=' | ')
        print("RMSE:", "{:2.4f}".format(rmse_run), end=' |   | ')

        is_good = " GOOD" if rmse_run < min_rmse[0] else ""

        min_acc = update((acc_run, i), lt, min_acc)
        max_acc = update((acc_run, i), gt, max_acc)
        min_acc_sugg = update((acc_sugg_run, i), lt, min_acc_sugg)
        max_acc_sugg = update((acc_sugg_run, i), gt, max_acc_sugg)
        min_rmse = update((rmse_run, i), lt, min_rmse)
        max_rmse = update((rmse_run, i), gt, max_rmse)

        print("B. acc: {:2.4f} | B. s/ns: {:2.4f} |"
               " B. RMSE: {:2.4f}{:s}".format(max_acc[0],
                                                 max_acc_sugg[0],
                                                 min_rmse[0],
                                                 is_good))

# ----------------------

