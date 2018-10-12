from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import tensorflow as tf
import sys
import tarfile
import re
import errno
import random
import datetime

from utils import *

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
from RUM import RUMCell
from baselineModels.GORU import GORUCell
from baselineModels.EUNN import EUNNCell


def recall_data(T, n_data):
	"""
	Creates the recall data.
	"""
    # character
    n_category = int(T // 2)

    input1 = []
    for i in range(n_data):
        x0 = np.arange(1, n_category + 1)
        np.random.shuffle(x0)
        input1.append(x0[:T // 2])
    input1 = np.array(input1)
    # number
    input2 = np.random.randint(
        n_category + 1, high=n_category + 11, size=(n_data, T // 2))

    # question mark
    input3 = np.zeros((n_data, 2))
    seq = np.stack([input1, input2], axis=2)
    seq = np.reshape(seq, [n_data, T])
    # answer
    ind = np.random.randint(0, high=T // 2, size=(n_data))
    input4 = np.array([[input1[i][ind[i]]] for i in range(n_data)])

    x = np.concatenate((seq, input3, input4), axis=1).astype('int32')
    y = np.array([input2[i][ind[i]] for i in range(n_data)]) - n_category - 1

    return x, y


def next_batch(data_x, data_y, step, batch_size):
    data_size = data_x.shape[0]
    start = step * batch_size % data_size
    end = start + batch_size
    if end > data_size:
        end = end - data_size
        batch_x = np.concatenate((data_x[start:, ], data_x[:end, ]))
        batch_y = np.concatenate((data_y[start:], data_y[:end]))
    else:
        batch_x = data_x[start:end, ]
        batch_y = data_y[start:end]
    return batch_x, batch_y


def main(
        model,
        T,
        n_iter,
        n_batch,
        n_hidden,
        capacity,
        comp,
        FFT,
        learning_rate,
        decay,
        learning_rate_decay,
        norm,
        grid_name,
        activation):
    learning_rate = float(learning_rate)
    decay = float(decay)

    # --- Set data params ----------------
    n_input = int(T / 2) + 10 + 1
    n_output = 10
    n_train = 100000
    n_valid = 10000
    n_test = 20000

    n_steps = T + 3
    n_classes = 10

    # --- Create graph and compute gradients ----------------------
    x = tf.placeholder("int32", [None, n_steps])
    y = tf.placeholder("int64", [None])

    input_data = tf.one_hot(x, n_input, dtype=tf.float32)

    # --- Input to hidden layer ----------------------
    if model == "LSTM":
        cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "GRU":
        cell = GRUCell(n_hidden)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "RUM":
        if activation == "relu":
            act = relu
        elif activation == "sigmoid":
            act = sigmoid
        elif activation == "tanh":
            act = tanh
        elif activation == "softsign":
            act = softsign
        cell = RUMCell(n_hidden, T_norm=norm, activation=act)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
        """
		hidden_out = hidden_out[:,:,:50]
		costh = hidden_out[:,:,-1]
		print(colored(hidden_out,'red'))
		print(colored(costh, 'green'))
		
		costh_mean_dist = tf.reduce_mean(costh, axis=0)
		costh_hist = tf.summary.histogram('costh',costh_mean_dist)
		print(colored(costh_normalized_dist,'yellow'))
		"""

    elif model == "ARUM":
        cell = ARUMCell(n_hidden, T_norm=norm)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "ARUM2":
        cell = ARUM2Cell(n_hidden, T_norm=norm)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "RNN":
        cell = BasicRNNCell(n_hidden)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "EUNN":
        cell = EUNNCell(n_hidden, capacity, FFT, comp)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    elif model == "GORU":
        cell = GORUCell(n_hidden, capacity, FFT)
        hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

    # --- Hidden Layer to Output ----------------------
    # important `tanh` prevention from blow up
    V_init_val = np.sqrt(6.) / np.sqrt(n_output + n_input)

    V_weights = tf.get_variable("V_weights", shape=[
                                n_hidden, n_classes], dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[
                             n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    hidden_out = tf.unstack(hidden_out, axis=1)[-1]
    temp_out = tf.matmul(hidden_out, V_weights)
    output_data = tf.nn.bias_add(temp_out, V_bias)

    # --- evaluate process ----------------------
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=output_data, labels=y))
    correct_pred = tf.equal(tf.argmax(output_data, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # --- Initialization ----------------------
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    print("\n###")
    sumz = 0
    for i in tf.global_variables():
        print(i.name, i.shape, np.prod(np.array(i.get_shape().as_list())))
        sumz += np.prod(np.array(i.get_shape().as_list()))
    print("# parameters: ", sumz)
    print("###\n")

    folder = "./output/recall/T=" + str(T) + '/' + model
    filename = folder + "_h=" + str(n_hidden)
    filename = filename + "_lr=" + str(learning_rate)
    filename = filename + "_norm=" + str(norm)
    filename = filename + ".txt"
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not os.path.exists(os.path.dirname(folder + "/modelCheckpoint/")):
        try:
            print(folder + "/modelCheckpoint/")
            os.makedirs(os.path.dirname(folder + "/modelCheckpoint/"))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    f = open(filename, 'w')
    f.write("########\n\n")
    f.write("## \tModel: %s with N=%d" % (model, n_hidden))
    f.write("########\n\n")

    # --- Training Loop ----------------------
    saver = tf.train.Saver()
    mx2 = 0
    step = 0

    train_x, train_y = recall_data(T, n_train)
    val_x, val_y = recall_data(T, n_valid)
    test_x, test_y = recall_data(T, n_test)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=False)) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
            "./logs/recall/" + grid_name, sess.graph)

        sess.run(init)

        steps = []
        losses = []
        accs = []

        while step < n_iter:
            batch_x, batch_y = next_batch(train_x, train_y, step, n_batch)

            # acc, loss = \
            # sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            # costh_val = sess.run([costh], feed_dict={x: batch_x, y: batch_y})
            # print(colored(costh_val,'green'))
            # print(colored("###",'yellow'))
            # acc, loss, costh_h = \
            # sess.run([accuracy, cost, costh_hist], feed_dict={x: batch_x, y: batch_y})
            acc, loss = sess.run([accuracy, cost], feed_dict={
                                 x: batch_x, y: batch_y})
            # writer.add_summary(costh_h, step)

            print("Iter " + str(step) + ", Minibatch Loss= " +
                  "{:.6f}".format(loss) + ", Training Accuracy= " +
                  "{:.5f}".format(acc))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            steps.append(step)
            losses.append(loss)
            accs.append(acc)
            step += 1
            if step % 1000 == 999:
                acc = sess.run(accuracy, feed_dict={x: val_x, y: val_y})
                loss = sess.run(cost, feed_dict={x: val_x, y: val_y})

                print("Validation Loss= " +
                      "{:.6f}".format(loss) + ", Validation Accuracy= " +
                      "{:.5f}".format(acc))
                f.write("%d\t%f\t%f\n" % (step, loss, acc))

            if step % 1000 == 1:

                saver.save(sess, folder + "/modelCheckpoint/step=" + str(step))
                # if model == "GRU": tmp = "gru"
                # if model == "RUM": tmp = "RUM"
                # if model == "EUNN": tmp = "eunn"
                # if model == "GORU": tmp = "goru"

                # kernel = [v for v in tf.global_variables() if v.name == "rnn/" + tmp + "_cell/gates/kernel:0"][0]
                # bias = [v for v in tf.global_variables() if v.name == "rnn/" + tmp + "_cell/gates/bias:0"][0]
                # k, b = sess.run([kernel, bias])
                # np.save(folder + "/kernel_" + str(step), k)
                # np.save(folder + "/bias_" + str(step), b)

        print("Optimization Finished!")

        # --- test ----------------------

        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
        f.write("Test result: Loss= " + "{:.6f}".format(test_loss) +
                ", Accuracy= " + "{:.5f}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="recall task")
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument("model", default='LSTM',
                        help='Model name: LSTM, EUNN, GRU, GORU')
    parser.add_argument('-T', type=int, default=50,
                        help='Information sequence length')
    parser.add_argument('attention', type=str,
                        default="False", help='is attn. mechn.')
    parser.add_argument('--n_iter', '-I', type=int,
                        default=10000, help='training iteration number')
    parser.add_argument('--n_batch', '-B', type=int,
                        default=32, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int,
                        default=256, help='hidden layer size')
    parser.add_argument('--capacity', '-L', type=int, default=2,
                        help='Tunable style capacity, only for EUNN, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False",
                        help='Complex domain or Real domain. Default is False: real domain')
    parser.add_argument('--FFT', '-F', type=str, default="False",
                        help='FFT style, default is False')
    parser.add_argument('--learning_rate', '-R', default=0.001, type=str)
    parser.add_argument('--norm', '-N', default=None, type=float)
    parser.add_argument('--update_gate', '-U', default="True",
                        type=str, help='is there update gate?')
    parser.add_argument('--activation', '-A', default="relu",
                        type=str, help='specify activation')
    parser.add_argument('--lambd', '-LA', default=0,
                        type=int, help='lambda for RUM model')
    parser.add_argument('--layer_norm', '-LN', default="False",
                        type=str, help='is there layer normalization?')
    parser.add_argument('--zoneout', '-Z', default="False",
                        type=str, help='is there zoneout?')

    args = parser.parse_args()
    dicts = vars(args)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    for i in dicts:
        if (dicts[i] == "False"):
            dicts[i] = False
        elif dicts[i] == "True":
            dicts[i] = True

    kwargs = {
        'model': dicts['model'],
        'T': dict['T'],
        'n_iter': dicts['n_iter'],
        'n_batch': dicts['n_batch'],
        'n_hidden': dicts['n_hidden'],
        'capacity': dicts['capacity'],
        'comp': dicts['comp'],
        'FFT': dicts['FFT'],
        'learning_rate': dicts['learning_rate'],
        'norm': dicts['norm'],
        'update_gate': dicts['update_gate'],
        'activation': dicts['activation'],
        'lambd': dicts['lambd'],
        'layer_norm': dicts['layer_norm'],
        'zoneout': dicts['zoneout'],
    }

    main(**kwargs)
