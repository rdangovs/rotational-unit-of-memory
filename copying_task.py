from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import tensorflow as tf
import sys
from utils import *
import shutil

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from RUM import RUMCell
from baselineModels.GORU import GORUCell
from baselineModels.EUNN import EUNNCell


def random_variable(shape, dev):
    """init a random variable"""
    initial = tf.truncated_normal(shape, stddev=dev)
    return tf.Variable(initial)


def copying_data(T, n_data, n_sequence):
    """generating the data"""
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, T - 1))
    zeros2 = np.zeros((n_data, T))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int64')

    return x, y


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
        update_gate,
        activation,
        lambd):

    learning_rate = float(learning_rate)
    decay = float(decay)

    # --- Set data params ----------------
    n_input = 10
    n_output = 9
    n_sequence = 10
    n_train = n_iter * n_batch
    n_test = n_batch

    n_steps = T + 20
    n_classes = 9

    # --- Create data --------------------

    train_x, train_y = copying_data(T, n_train, n_sequence)
    test_x, test_y = copying_data(T, n_test, n_sequence)

    # --- Create graph and compute gradients ----------------------
    x = tf.placeholder("int32", [None, n_steps])
    y = tf.placeholder("int64", [None, n_steps])

    input_data = tf.one_hot(x, n_input, dtype=tf.float32)

    # input to hidden
    if model == "LSTM":
        cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
    elif model == "GRU":
        cell = GRUCell(
            n_hidden, kernel_initializer=tf.orthogonal_initializer())
    elif model == "RUM":
        if activation == "relu":
            act = tf.nn.relu
        elif activation == "sigmoid":
            act = tf.nn.sigmoid
        elif activation == "tanh":
            act = tf.nn.tanh
        elif activation == "softsign":
            act = tf.nn.softsign
        cell = cell = RUMCell(n_hidden,
                              eta_=norm,
                              update_gate=update_gate,
                              lambda_=lambd,
                              activation=act)
    elif model == "ARUM":
        cell = ARUMCell(n_hidden, T_norm=norm)
    elif model == "EUNN":
        cell = EUNNCell(n_hidden, capacity, FFT, comp)
    elif model == "GORU":
        cell = GORUCell(n_hidden, capacity, FFT)
    elif model == "RNN":
        cell = BasicRNNCell(n_hidden)
    hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

    # hidden to output
    V_init_val = np.sqrt(6.) / np.sqrt(n_output + n_input)
    V_weights = tf.get_variable("V_weights", shape=[
                                n_hidden, n_classes], dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[
                             n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    hidden_out_list = tf.unstack(hidden_out, axis=1)
    temp_out = tf.stack([tf.matmul(i, V_weights) for i in hidden_out_list])
    output_data = tf.nn.bias_add(tf.transpose(temp_out, [1, 0, 2]), V_bias)

    # evaluate process
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=output_data, labels=y))
    tf.summary.scalar('cost', cost)
    correct_pred = tf.equal(tf.argmax(output_data, 2), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # initialization
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate, decay=decay).minimize(cost)
    init = tf.global_variables_initializer()

    # save
    filename = model + "_H" + str(n_hidden) + "_" + \
        ("L" + str(lambd) + "_" if lambd else "") + \
        ("E" + str(eta) + "_" if norm else "") + \
        ("A" + activation + "_" if activation else "") + \
        ("U_" if update_gate else "") + \
        (str(capacity) if model in ["EUNN", "GORU"] else "") + \
        ("FFT_" if model in ["EUNN", "GORU"] and FFT else "") + \
        "B" + str(n_batch)
    save_path = os.path.join('train_log', 'copying', 'T' + str(T), filename)

    if os.path.exists(save_path):
        print(colored(
            "Directory exists. Enter a string in [Y, yes, y] to override it.", "red"))
        inp = raw_input("Enter key here: ")
        if inp in ["Y", "yes", "y"]:
            print(colored("OK: overriding...", "red"))
            shutil.rmtree(save_path)
        else:
            print(colored("Invalid key: exiting...", "blue"))
            exit()
    log(kwargs, save_path)

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    parameters_profiler()

    # --- Training Loop ----------------------
    saver = tf.train.Saver()
    mx2 = 0
    step = 0
    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(save_path, sess.graph)

        steps = []
        losses = []
        accs = []

        while step < n_iter:
            batch_x = train_x[step * n_batch: (step + 1) * n_batch]
            batch_y = train_y[step * n_batch: (step + 1) * n_batch]
            summ, acc, loss = sess.run([merged_summary, accuracy, cost], feed_dict={
                                       x: batch_x, y: batch_y})
            train_writer.add_summary(summ, step)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            print(col("Iter " + str(step) + ", Minibatch Loss: " +
                      "{:.6f}".format(loss) + ", Training Accuracy: " +
                      "{:.5f}".format(acc), 'b'))
            steps.append(step)
            losses.append(loss)
            accs.append(acc)
            if step % 1000 == 0:
                saver.save(sess, save_path)
            step += 1

        print(col("Optimization Finished!", 'b'))

        # test
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
        print("finish this!")  # finish this


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copying Task")
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument("model", default='DRUM',
                        help='Model name: LSTM, LSTSM, LSTRM, LSTUM, EURNN, GRU, GRRU, GORU, GRRU')
    parser.add_argument('-T', type=int, default=500,
                        help='Information sequence length')
    parser.add_argument('--n_iter', '-I', type=int,
                        default=20000, help='training iteration number')
    parser.add_argument('--n_batch', '-B', type=int,
                        default=128, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int,
                        default=250, help='hidden layer size')
    parser.add_argument('--capacity', '-L', type=int, default=2,
                        help='Tunable style capacity, only for EURNN, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False",
                        help='Complex domain or Real domain. Default is False: real domain')
    parser.add_argument('--FFT', '-F', type=str,
                        default="False", help='FFT style, default is False')
    parser.add_argument('--learning_rate', '-R', default=0.001, type=str)
    parser.add_argument('--decay', '-D', default=0.9, type=str)
    parser.add_argument('--learning_rate_decay', '-RD',
                        default="False", type=str)
    parser.add_argument('--norm', '-norm', default=None, type=float)
    parser.add_argument('--update_gate', '-U', default=1,
                        type=bool, help='is there update gate?')
    parser.add_argument('--activation', '-A', default="relu",
                        type=str, help='specify activation')
    parser.add_argument('--lambd', '-LA', default=0,
                        type=int, help='lambda for RUM model')

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
        'T': dicts['T'],
        'n_iter': dicts['n_iter'],
        'n_batch': dicts['n_batch'],
        'n_hidden': dicts['n_hidden'],
        'capacity': dicts['capacity'],
        'comp': dicts['comp'],
        'FFT': dicts['FFT'],
        'learning_rate': dicts['learning_rate'],
        'decay': dicts['decay'],
        'learning_rate_decay': dicts['learning_rate_decay'],
        'norm': dicts['norm'],
        'update_gate': dicts['update_gate'],
        'activation': dicts['activation'],
        'lambd': dicts['lambd']
    }

    main(**kwargs)
