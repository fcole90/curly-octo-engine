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

class Settings:
    def __init__(self):
        pass

    def print_all(self):
        for key in self.__dict__.keys():
            print(key, self.__dict__[key], sep=": ")


# --- Definitions ---
# Constant values, should not be changed.

__ONE_HOT_DATA__ = "one_hot"
__DECIMAL_DATA__ = "decimal"
__COLOR_INPUT_SIZE__ = 30
__USER_INPUT_SIZE__ = 6
__INPUT_SIZE__ = __USER_INPUT_SIZE__ + __COLOR_INPUT_SIZE__

# ---  Settings ---
# Values that should be tuned to obtain the best results.
s = Settings()

s.use_cache = True
s.user_data_representation = "clusters"
s.labels_data_type = __ONE_HOT_DATA__
__LABELS_SIZE__ = 1 if s.labels_data_type is __DECIMAL_DATA__ else 5

s.batch_size = 2000
s.learning_rate = 0.005
# Using combined loss, alpha is the percentage of rmse_loss, the rest is for cross entropy
s.alpha = 1.2
# Using combined loss, beta is the percentage of cross entropy, if None becomes (1 - alpha).
s.beta = 0.8
s.optimizer = tf.train.GradientDescentOptimizer
s.iterations = int(10.0e+6)

# -------------------

setup = sim_setup.Setup(user_data_function=s.user_data_representation,
                        use_cache=s.use_cache,
                        labels_data_type=s.labels_data_type)

# None means that it can have any length
x = tf.placeholder(tf.float32, [None, __INPUT_SIZE__])
x2 = tf.placeholder(tf.float32, [None, __LABELS_SIZE__])


# --- Neuron functions ---
def first_layer(x):
    W = tf.Variable(tf.random_normal([__INPUT_SIZE__, __LABELS_SIZE__]))
    b = tf.Variable(tf.constant([0.01] * __LABELS_SIZE__))
    return tf.add(tf.matmul(x, W), b)


def inner_layer(x):
    W2 = tf.Variable(tf.random_normal([__LABELS_SIZE__, __LABELS_SIZE__]))
    b2 = tf.Variable(tf.constant([0.01] * __LABELS_SIZE__))
    return tf.add(tf.matmul(x, W2), b2)
# ------------

# --- Network connections ---

y1 = first_layer(x)
y2 = inner_layer(y1)

# Set one function as output function.
y = y1
# -------------

# --- Training ---
# Correct output
y_ = tf.placeholder(tf.float32, [None, __LABELS_SIZE__])

# --- Differentiable argmax ---
# todo: this shall be put in a separate library
def tf_max_amplifier(x, amp=10, axis=None, transpose=False):
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


def tf_differentiable_argmax(x, power=5, decoder=None, axis=None):
    # todo: the decoder size constant, should be dynamic
    if not decoder:
        decoder = tf.constant([list(range(__LABELS_SIZE__))], dtype=tf.float32)
    amplified_x = tf_max_amplifier(x, power, axis=axis, transpose=True)
    return tf.matmul(decoder, amplified_x)


# --- Loss functions ---
def cross_entropy(y, y_):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y1))

def root_mean_squared_error(y, y_):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, y_))))

def combine(loss1, loss2, alpha, beta=None):
    if not beta:
        beta = 1.0 - alpha
    return tf.add(tf.multiply(loss1, alpha),
                  tf.multiply(loss2, beta))

# ----------------------

if s.labels_data_type is __ONE_HOT_DATA__:
    cross_entropy_loss = cross_entropy(y, y_)
    rmse_loss = root_mean_squared_error(tf_differentiable_argmax(y1, axis=1, power=100),
                                        tf_differentiable_argmax(y_, axis=1, power=100))

    combo = combine(rmse_loss, cross_entropy_loss, s.alpha, s.beta)
    s.minimize_loss = "combined"
    train_step = tf.train.GradientDescentOptimizer(s.learning_rate).minimize(combo)

else:
    rmse_loss = tf.cast(tf.reduce_mean(tf.square(tf.subtract(y2, y_))), tf.float32)
    train_step = s.optimizer(s.learning_rate).minimize(rmse_loss)

# ----------------


# --- Estimators setup (RMSE, Accuracies) ---
if s.labels_data_type is __ONE_HOT_DATA__:
    # Error using standard argmax
    error = tf.subtract(tf.argmax(y, 1), tf.argmax(y_, 1))
    # Error using differentiable argmax
    d_error = tf.subtract(tf_differentiable_argmax(y, axis=1, power=1000),
                          tf_differentiable_argmax(y_, axis=1, power=1000))
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_yesno_prediction = tf.equal(tf.greater_equal(tf.argmax(y, 1), 3),
                                        tf.greater_equal(tf.argmax(y_, 1), 3))


else:
    error = tf.scalar_mul(5, tf.subtract(y, y_))
    # Here differentiable argmax is not in use
    d_error = error
    correct_prediction = tf.equal(tf.ceil(tf.scalar_mul(5, y)),
                                  tf.ceil(tf.scalar_mul(5,y_)))
    correct_yesno_prediction = tf.equal(tf.greater_equal(y, 0.6),
                                        tf.greater_equal(y_, 0.6))

# Estimators operations
rmse_error = root_mean_squared_error(tf.cast(error, tf.float32), 0.0)
d_rmse_error = root_mean_squared_error(tf.cast(d_error, tf.float32), 0.0)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
yesno_accuracy = tf.reduce_mean(tf.cast(correct_yesno_prediction, tf.float32))

# Feed dictionary for the validation set
valid_set_feed_dict = {x: sim_setup.Setup.get_only_part(0, setup.dataset['validation']),
                       y_: sim_setup.Setup.get_only_part(1, setup.dataset['validation'])}
# --------------------------


# --- Session ---
with tf.Session().as_default() as sess:
    tf.global_variables_initializer().run()

    # Print settings data for reference
    s.print_all()

    # Create a dict for the estimators
    data = dict()

    # Accuracy on rate prediction
    data['acc'] = {
        'min': dict(val=1.0, index=-1),
        'max': dict(val=0.0, index=-1),
        "current": dict(val=None, index=-1)
    }

    # Accuracy on suggest/not suggest base (suggests if rate >= 3).
    data['yesno_acc'] = {
        'min': dict(val=1.0, index=-1),
        'max': dict(val=0.0, index=-1),
        "current": dict(val=None, index=-1)
    }

    # Root mean squared error
    data['rmse'] = {
        'min': dict(val=10.0, index=-1),
        'max': dict(val=0.0, index=-1),
        "current": dict(val=None, index=-1)
    }

    # Root mean squared error using diff_argmax
    data['d_rmse'] = {
        'min': dict(val=10.0, index=-1),
        'max': dict(val=0.0, index=-1),
        "current": dict(val=None, index=-1)
    }

    # --- Execution Loop ---

    for i in range(s.iterations):

        # Run the experiment on a random batch of data points
        batch_xs, batch_ys = setup.next_batch(s.batch_size, use_permutations=True)
        train_dict = {x: batch_xs, y_: batch_ys}
        sess.run(train_step, feed_dict = train_dict)

        # # Update percentage
        # if i%20 == 0 or i == iterations - 1:
        #     gh_tools.updating_text("\r[{0:.2f}%]".format((i + 1) / iterations * 100))

        # Update the estimators
        if i % 20 == 0 or i == s.iterations - 1:

            # Update the estimators
            data["acc"]["current"]["val"] = sess.run(accuracy, feed_dict=valid_set_feed_dict)
            data["yesno_acc"]["current"]["val"] = sess.run(yesno_accuracy, feed_dict=valid_set_feed_dict)
            data["rmse"]["current"]["val"] = sess.run(rmse_error, feed_dict=valid_set_feed_dict)
            data["d_rmse"]["current"]["val"] = sess.run(d_rmse_error, feed_dict=valid_set_feed_dict)

            # Display current values
            print("[{:10d}]".format(i), "Acc:", "{:2.4f}".format(data["acc"]["current"]["val"]), end=' | ')
            print("Acc s/ns:", "{:2.4f}".format(data["yesno_acc"]["current"]["val"]), end=' | ')
            print("d_RMSE: {:2.4f}".format(data["d_rmse"]["current"]["val"]), end=' | ')
            print("RMSE:", "{:2.4f}".format(data["rmse"]["current"]["val"]), end=' |   | ')

            # I like having a lower RMSE!
            is_good = " GOOD" if data["rmse"]["current"]["val"] < data["rmse"]["min"]["val"] else ""

            # Update the max and min of the estimators
            for key in data.keys():
                if data[key]["current"]["val"] < data[key]["min"]["val"]:
                    data[key]["min"] = dict(val=data[key]["current"]["val"], index=i)
                if data[key]["current"]["val"] > data[key]["max"]["val"]:
                    data[key]["max"] = dict(val=data[key]["current"]["val"], index=i)

            # Display the best values
            print("B. acc: {:2.4f} | B. s/ns: {:2.4f} |"
                   " B. RMSE: {:2.4f}{:s}".format(data["acc"]["max"]["val"],
                                                  data["yesno_acc"]["max"]["val"],
                                                  data["rmse"]["min"]["val"],
                                                  is_good))

    # --- End loop ---
# --- End session ---

