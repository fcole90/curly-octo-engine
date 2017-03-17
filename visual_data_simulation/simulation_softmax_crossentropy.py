import numpy as np
import tensorflow as tf

import visual_data_simulation.simulation_setup as sim_setup

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
s.allow_negative_data = True
s.user_data_representation = "clusters"
s.movie_amount_limit = 0 # set to 0 for no limit
s.use_permutations = True
s.limit_permutations = 30
s.labels_data_type = __DECIMAL_DATA__
__LABELS_SIZE__ = 1 if s.labels_data_type is __DECIMAL_DATA__ else 5

s.batch_size = 256
s.learning_rate = 0.00001
s.iterations = int(10.0e+6)
s.hidden_layers = 0
s.optimizer = tf.train.AdamOptimizer()

# -------------------

setup = sim_setup.Setup(user_data_function=s.user_data_representation,
                        use_cache=s.use_cache,
                        labels_data_type=s.labels_data_type,
                        allow_negative_data=s.allow_negative_data,
                        movie_amount_limit=s.movie_amount_limit)

# None means that it can have any length
x = tf.placeholder(tf.float32, [None, __INPUT_SIZE__])
layers = list()

# --- Neuron functions ---
def first_layer(x):
    W = tf.Variable(tf.random_normal([__INPUT_SIZE__, __LABELS_SIZE__], mean=0.0, stddev=0.8))
    b = tf.Variable(tf.constant([0.0] * __LABELS_SIZE__))
    return dict(W=W, a=tf.nn.sigmoid(tf.add(tf.matmul(x, W), b)))


def hidden_layer(x):
    W = tf.Variable(tf.random_normal([__LABELS_SIZE__, __LABELS_SIZE__], mean=0.0, stddev=0.8))
    b = tf.Variable(tf.constant([0.0] * __LABELS_SIZE__))
    return dict(W=W, a=tf.nn.sigmoid(tf.add(tf.matmul(x, W), b)))

# ------------

# --- Network connections ---
layers.append(first_layer(x))

for i in range(s.hidden_layers):
    layers.append(hidden_layer(layers[i]["a"]))

y = layers[-1]["a"]


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
                                          0.1e-20))
                         , amp)

    # amplified_x = tf.cond(tf.is_nan(amplified_x),
    #                         lambda: tf.zeros_like(amplified_x),
    #                       lambda: amplified_x)

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
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name="cross_entropy")

def root_mean_squared_error(y, y_):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, y_))))

def error_suggestion_yesno(y, y_, val=3.00000000000000001):
    correct_yesno_prediction = tf.maximum(tf.sign(tf.multiply(tf.subtract(y, val),
                                                     tf.subtract(y_, val))),
                                          0.0)
    return tf.subtract(1.0, tf.reduce_mean(correct_yesno_prediction))

def regularization(network):
    loss = 0.0
    for layer in network:
        loss = tf.add(loss, tf.nn.l2_loss(layer['W']))
    return loss

# ----------------------

# --- Loss ---
if s.labels_data_type is __ONE_HOT_DATA__:
    pseudo_rate_y = tf_differentiable_argmax(y, axis=1, power=2)
    pseudo_rate_y_ = tf_differentiable_argmax(y_, axis=1, power=2)

    cross_entropy_loss = cross_entropy(y, y_)
    l2_yesno_loss = tf.nn.l2_loss(error_suggestion_yesno(pseudo_rate_y, pseudo_rate_y_) * 10)
    l2_loss = tf.nn.l2_loss(tf.subtract(pseudo_rate_y, pseudo_rate_y_))
    weight_loss = regularization(network=layers)


    s.l2_loss_perc = 0.07
    s.cross_entropy_loss_perc = 1.0 * 4.0
    s.yesno_loss_perc = 0.01

    s.regularization = 0.1 / (s.hidden_layers + 1) ** 2

    s.loss = 1.0
    s.loss = tf.multiply(s.loss, tf.multiply(s.l2_loss_perc, l2_loss))
    s.loss = tf.multiply(s.loss, tf.multiply(s.cross_entropy_loss_perc, cross_entropy_loss))
    # s.loss = tf.multiply(s.loss, tf.multiply(s.yesno_loss_perc, l2_yesno_loss))
    s.loss = tf.add(s.loss, tf.multiply(s.regularization, weight_loss))

    train_step = s.optimizer.minimize(s.loss)

else:
    l2_loss = tf.cast(tf.reduce_mean(tf.square(tf.subtract(y, y_))), tf.float32)

    s.loss = tf.nn.l2_loss(tf.subtract(y, y_))
    # s.loss = s.loss * tf.nn.l2_loss(error_suggestion_yesno(y, y_, 0.6)*10)
    s.loss = tf.add(s.loss, regularization(network=layers))
    train_step = s.optimizer.minimize(s.loss)

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
valid_set_feed_dict = {x: list(setup.dataset['validation'][0]),
                       y_: list(setup.dataset['validation'][1])}
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
        batch_xs, batch_ys = setup.next_batch(s.batch_size,
                                              use_permutations=s.use_permutations,
                                              limit_permutations=s.limit_permutations)
        train_dict = {x: batch_xs, y_: batch_ys}
        sess.run([train_step], feed_dict = train_dict)

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

            current_loss = sess.run(s.loss, feed_dict = train_dict)

            # Display current values
            print("[{:10d}]".format(i), "Acc:", "{:2.4f}".format(data["acc"]["current"]["val"]), end=' | ')
            print("Acc s/ns:", "{:2.4f}".format(data["yesno_acc"]["current"]["val"]), end=' | ')
            print("d_RMSE: {:2.4f}".format(data["d_rmse"]["current"]["val"]), end=' | ')
            print("RMSE:", "{:2.4f}".format(data["rmse"]["current"]["val"]), end=' | ')
            print("Loss:", "{:07.4f}".format(current_loss), end=' |   | ')

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

