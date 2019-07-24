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


def process_vis(weights, num_points, n_hidden=100, cell="RUM"):
    """
    helper function for processing the placeholder weights for visualization
    """
    if cell == "RUM":
        feed_temp_target = weights[
            :, :(n_hidden + 10) * n_hidden]
        feed_temp_target = np.reshape(feed_temp_target,
                                      (num_points, n_hidden + 10, n_hidden))
        feed_temp_target_bias = weights[
            :, (n_hidden + 10) * n_hidden:(n_hidden + 10) * n_hidden + n_hidden]
        # no need for reshape
        feed_temp_embed = weights[:, - 10 * n_hidden:]
        feed_temp_embed = np.reshape(
            feed_temp_embed, (num_points, 10, n_hidden))
        return feed_temp_target, feed_temp_target_bias, feed_temp_embed
    else:  # cell is either `eunn` or `goru`
        feed_temp_theta0 = weights[:, :n_hidden // 2]
        feed_temp_theta1 = weights[:, -(n_hidden // 2 - 1):]
        return feed_temp_theta0, feed_temp_theta1


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
        zoneout,
        visualization_experiment):

    learning_rate = float(learning_rate)

    # data params
    n_input = 10
    n_output = 9
    n_sequence = 10
    n_train = n_iter * n_batch
    n_test = n_batch

    n_steps = T + 20
    n_classes = 9

    # create data
    train_x, train_y = copying_data(T, n_train, n_sequence)
    test_x, test_y = copying_data(T, n_test, n_sequence)

    # graph and gradients
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
        # activation
        if activation == "relu":
            act = tf.nn.relu
        elif activation == "sigmoid":
            act = tf.nn.sigmoid
        elif activation == "tanh":
            act = tf.nn.tanh
        elif activation == "softsign":
            act = tf.nn.softsign
        if visualization_experiment:
            # placeholder
            temp_target = tf.placeholder("float32", [n_hidden + 10, n_hidden])
            temp_target_bias = tf.placeholder("float32", [n_hidden])
            temp_embed = tf.placeholder("float32", [10, n_hidden])

        cell = RUMCell(n_hidden,
                      eta_=norm,
                      update_gate=update_gate,
                      lambda_=lambd,
                      activation=act,
                      use_layer_norm=layer_norm,
                      use_zoneout=zoneout,
                      visualization=visualization_experiment,
                      temp_target=temp_target if visualization_experiment else None,
                      temp_target_bias=temp_target_bias if visualization_experiment else None,
                      temp_embed=temp_embed if visualization_experiment else None)
    elif model == "EUNN":
        if visualization_experiment:
            # placeholder
            temp_theta0 = tf.placeholder("float32", [n_hidden // 2])
            temp_theta1 = tf.placeholder("float32", [n_hidden // 2 - 1])
        cell = EUNNCell(n_hidden, capacity, FFT, comp, name="eunn")
    elif model == "GORU":
        if visualization_experiment:
            # placeholder
            temp_theta0 = tf.placeholder("float32", [n_hidden // 2])
            temp_theta1 = tf.placeholder("float32", [n_hidden // 2 - 1])
        cell = GORUCell(n_hidden, capacity, FFT,
                        temp_theta0=temp_theta0, temp_theta1=temp_theta1)
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
        ("VE_" if model in ["EUNN", "GORU", "RUM"] and visualization_experiment else "") + \
        "B" + str(n_batch)
    save_path = os.path.join(
        '../../train_log', 'copying', 'T' + str(T), filename)

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
    f.write("accuracies \n")

    log(kwargs, save_path)

    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    parameters_profiler()

    # train
    saver = tf.train.Saver()
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
            if visualization_experiment:
                """ initiative to write simpler code """

                if model == "RUM":
                    number_of_weights = (n_hidden + 10) * \
                        n_hidden + n_hidden + 10 * n_hidden
                elif model in ["GORU", "EUNN"]:
                    # assuming that n_hidden is even.
                    number_of_weights = n_hidden - 1

                print(col("strating linear visualization", 'b'))
                num_points = 200

                coord, weights = generate_points_for_visualization(
                    number_of_weights, num_points)

                processed_placeholders = process_vis(
                    weights, num_points, n_hidden=n_hidden, cell=model)
                if model == "RUM":
                    feed_temp_target, feed_temp_target_bias, feed_temp_embed = processed_placeholders

                else:
                    feed_temp_theta0, feed_temp_theta1 = processed_placeholders

                collect_losses = []
                for i in range(num_points):
                    if model == "RUM":
                        loss = sess.run(cost, feed_dict={x: batch_x,
                                                         y: batch_y,
                                                         temp_target: feed_temp_target[i],
                                                         temp_target_bias: feed_temp_target_bias[i],
                                                         temp_embed: feed_temp_embed[i]})
                    elif model in ["EUNN", "GORU"]:
                        loss = sess.run(cost, feed_dict={
                                        x: batch_x,
                                        y: batch_y,
                                        temp_theta0: feed_temp_theta0[i],
                                        temp_theta1: feed_temp_theta1[i]})

                    print(col("iter: " + str(i) + " loss: " + str(loss), 'y'))
                    collect_losses.append(loss)
                np.save(os.path.join(save_path, "linear_height"),
                        np.array(collect_losses))
                np.save(os.path.join(save_path, "linear_coord"),
                        np.array(coord))
                print(col("done with linear visualization", 'b'))

                #####################

                print(col("strating contour visualization", 'b'))
                num_points = 20
                coord, weights = generate_points_for_visualization(
                    number_of_weights, num_points, type_vis="contour")
                np.save(os.path.join(save_path, "contour_coord"),
                        np.array(coord))
                processed_placeholders = process_vis(
                    weights, num_points ** 2, n_hidden=n_hidden, cell=model)
                if model == "RUM":
                    feed_temp_target, feed_temp_target_bias, feed_temp_embed = processed_placeholders
                else:
                    feed_temp_theta0, feed_temp_theta1 = processed_placeholders

                collect_contour = np.empty((num_points, num_points))
                for i in range(num_points):
                    for j in range(num_points):
                        if model == "RUM":
                            loss = sess.run(cost, feed_dict={
                                x: batch_x,
                                y: batch_y,
                                temp_target: feed_temp_target[i * num_points + j],
                                temp_target_bias: feed_temp_target_bias[i * num_points + j],
                                temp_embed: feed_temp_embed[i * num_points + j]})
                        elif model in ["GORU", "EUNN"]:
                            loss = sess.run(cost, feed_dict={
                                x: batch_x,
                                y: batch_y,
                                temp_theta0: feed_temp_theta0[i * num_points + j],
                                temp_theta1: feed_temp_theta1[i * num_points + j]})
                        collect_contour[i, j] = loss
                        print(col("iter: " + str(i) + "," +
                                  str(j) + " loss: " + str(loss), 'y'))
                np.save(os.path.join(save_path, "contour_height"),
                        np.array(collect_contour))

                print(col("exiting visualization experiment", 'r'))
                exit()

            summ, acc, loss = sess.run([merged_summary, accuracy, cost], feed_dict={
                x: batch_x, y: batch_y})
            train_writer.add_summary(summ, step)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            print(col("Iter " + str(step) + ", Minibatch Loss: " +
                      "{:.6f}".format(loss) + ", Training Accuracy: " +
                      "{:.5f}".format(acc), 'g'))
            steps.append(step)
            losses.append(loss)
            accs.append(acc)
            if step % 200 == 0:
                f.write(col("%d\t%f\t%f\n" % (step, loss, acc), 'y'))
                f.flush()

            if step % 1000 == 0:
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
    parser.add_argument('-T', type=int, default=500,
                        help='Information sequence length')
    parser.add_argument('--n_iter', '-I', type=int,
                        default=10000, help='training iteration number')
    parser.add_argument('--n_batch', '-B', type=int,
                        default=32, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int,
                        default=100, help='hidden layer size')
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
    parser.add_argument('--visualization_experiment', '-VE', default="False",
                        type=str, help='is there experiment?')

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
        'visualization_experiment': dicts['visualization_experiment']
    }

    main(**kwargs)
