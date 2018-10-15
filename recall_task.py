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
    """ Creates the recall data. """

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
        norm,
        update_gate,
        activation,
        lambd,
        layer_norm,
        zoneout):

    learning_rate = float(learning_rate)

    # data params
    n_input = int(T / 2) + 10 + 1
    n_output = 10
    n_train = 100000
    n_valid = 10000
    n_test = 20000

    n_steps = T + 3
    n_classes = 10

    # graph and gradients
    x = tf.placeholder("int32", [None, n_steps])
    y = tf.placeholder("int64", [None])

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
                              activation=act,
                              use_layer_norm=layer_norm,
                              use_zoneout=zoneout)
    elif model == "EUNN":
        cell = EUNNCell(n_hidden, capacity, FFT, comp)
    elif model == "GORU":
        cell = GORUCell(n_hidden, capacity, FFT)
    elif model == "RNN":
        cell = BasicRNNCell(n_hidden)

    hidden_out, _ = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)

    # RESEARCH RELATED
    # hidden_out = hidden_out[:,:,:50]
    # costh = hidden_out[:,:,-1]
    # print(colored(hidden_out,'red'))
    # print(colored(costh, 'green'))

    # costh_mean_dist = tf.reduce_mean(costh, axis=0)
    # costh_hist = tf.summary.histogram('costh',costh_mean_dist)
    # print(colored(costh_normalized_dist,'yellow'))

    # hidden to output
    V_init_val = np.sqrt(6.) / np.sqrt(n_output + n_input)

    V_weights = tf.get_variable("V_weights", shape=[
                                n_hidden, n_classes], dtype=tf.float32, initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
    V_bias = tf.get_variable("V_bias", shape=[
                             n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

    hidden_out = tf.unstack(hidden_out, axis=1)[-1]
    temp_out = tf.matmul(hidden_out, V_weights)
    output_data = tf.nn.bias_add(temp_out, V_bias)

    # evaluate process
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=output_data, labels=y))
    tf.summary.scalar('cost', cost)
    correct_pred = tf.equal(tf.argmax(output_data, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # initialization
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # save
    filename = model + "_H" + str(n_hidden) + "_" + \
        ("L" + str(lambd) + "_" if lambd else "") + \
        ("E" + str(eta) + "_" if norm else "") + \
        ("A" + activation + "_" if activation else "") + \
        ("U_" if update_gate else "") + \
        ("Z_" if zoneout and model == "RUM" else "") + \
        ("ln_" if layer_norm and model == "RUM" else "") + \
        (str(capacity) if model in ["EUNN", "GORU"] else "") + \
        ("FFT_" if model in ["EUNN", "GORU"] and FFT else "") + \
        "B" + str(n_batch)
    save_path = os.path.join('train_log', 'recall', 'T' + str(T), filename)

    file_manager(save_path)

    # what follows is task specific
    filepath = os.path.join(save_path, "eval.txt")
    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    f = open(filepath, 'w')
    f.write(col("validation \n", 'r'))

    log(kwargs, save_path)

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    parameters_profiler()

    # train
    saver = tf.train.Saver()
    step = 0

    train_x, train_y = recall_data(T, n_train)
    val_x, val_y = recall_data(T, n_valid)
    test_x, test_y = recall_data(T, n_test)

    with tf.Session() as sess:
        sess.run(init)
        train_writer = tf.summary.FileWriter(save_path, sess.graph)

        steps = []
        losses = []
        accs = []

        while step < n_iter:
            batch_x, batch_y = next_batch(train_x, train_y, step, n_batch)

            # RESEARCH RELATED
            # acc, loss = \
            # sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            # costh_val = sess.run([costh], feed_dict={x: batch_x, y: batch_y})
            # print(colored(costh_val,'green'))
            # print(colored("###",'yellow'))
            # acc, loss, costh_h = \
            # sess.run([accuracy, cost, costh_hist], feed_dict={x: batch_x, y:
            # batch_y})

            acc, loss = sess.run([accuracy, cost], feed_dict={
                x: batch_x, y: batch_y})
            # writer.add_summary(costh_h, step) # RESEARCH RELATED

            print(col("Iter " + str(step) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(acc), 'g'))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            steps.append(step)
            losses.append(loss)
            accs.append(acc)
            if step % 1000 == 0:
                summ = sess.run(merged_summary, feed_dict={x: val_x, y: val_y})
                acc = sess.run(accuracy, feed_dict={x: val_x, y: val_y})
                loss = sess.run(cost, feed_dict={x: val_x, y: val_y})
                train_writer.add_summary(summ, step)

                print("Validation Loss= " +
                      "{:.6f}".format(loss) + ", Validation Accuracy= " +
                      "{:.5f}".format(acc))
                f.write(col("%d\t%f\t%f\n" % (step, loss, acc), 'y'))
                f.flush

            if step % 1000 == 1:
                print(col("saving graph and metadata in " + save_path, "b"))
                saver.save(sess, os.path.join(save_path, "model"))

            step += 1

        print(col("Optimization Finished!", 'b'))

        # test
        test_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
        test_loss = sess.run(cost, feed_dict={x: test_x, y: test_y})
        f.write(col("Test result: Loss= " + "{:.6f}".format(test_loss) +
                    ", Accuracy= " + "{:.5f}".format(test_acc), 'g'))

        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="recall task")
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument("model", default='LSTM',
                        help='Model name: LSTM, EUNN, GRU, GORU')
    parser.add_argument('-T', type=int, default=50,
                        help='Information sequence length')
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
        'T': dicts['T'],
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
