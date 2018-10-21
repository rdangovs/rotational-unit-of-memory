# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import sys
import os
from termcolor import colored

import tensorflow as tf

import auxiliary as aux
import reader
import configs

from baselineModels import LNLSTM
from baselineModels import FSRNN, GORU, EUNN
from tensorflow.contrib.rnn import MultiRNNCell

from utils import *

import RUM


flags = tf.flags

flags.DEFINE_string(
    "model", "ptb",
    "A type of model. Check configs file to know which models are available.")
flags.DEFINE_string("data_path", '../../data/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", '../../train_log/ptb/',
                    "Model output directory.")
flags.DEFINE_string("gpu", None,
                    "Using GPUs.")
flags.DEFINE_string(
    "mode", "train",
    "A type of mode. Train or test.")
flags.DEFINE_string(
    "restore", "False",
    "Restore? True or False?")
flags.DEFINE_boolean(
    "parameters", False,
    "Show # parameters")
flags.DEFINE_integer(
    "fast_size", None,
    "hidden size of the fast cell")
flags.DEFINE_float(
    "eta", None,
    "eta for time normalization")


FLAGS = flags.FLAGS
FLAGS.save_path = os.path.join(FLAGS.save_path, FLAGS.model)
file_manager(FLAGS.save_path)


if FLAGS.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config, input_):
        self._input = input_

        # prelim
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        emb_size = config.embed_size
        vocab_size = config.vocab_size
        F_size = FLAGS.fast_size if FLAGS.fast_size else config.cell_size
        if config.cell not in ["rum", "lstm"]:
            S_size = config.hyper_size

        # embedding
        emb_init = aux.orthogonal_initializer(1.0)
        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, emb_size], initializer=emb_init, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        # construct Fast and Slow states
        if config.cell not in ["rum", "lstm"]:
            F_cells = [LNLSTM.LN_LSTMCell(F_size, use_zoneout=True, is_training=is_training,
                                          zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
                       for _ in range(config.fast_layers)]
        if config.cell == "fs-lstm":
            S_cell = LNLSTM.LN_LSTMCell(S_size, use_zoneout=True, is_training=is_training,
                                        zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
        elif config.cell == "fs-rum":
            S_cell = RUM.RUMCell(S_size,
                                 # eta_=config.T_norm,
                                 eta=FLAGS.eta,
                                 use_zoneout=config.use_zoneout,
                                 use_layer_norm=config.use_layer_norm,
                                 is_training=is_training)
        elif config.cell == "fs-goru":
            with tf.variable_scope("goru"):
                S_cell = GORU.GORUCell(hidden_size=S_size)
        # test pure RUM/LSTM models (room for experiments)
        if config.cell == "rum":
            if config.activation == "tanh":
                act = tf.nn.tanh
            elif config.activation == "sigmoid":
                act = tf.nn.sigmoid
            elif config.activation == "softsign":
                act = tf.nn.softsign
            elif config.activation == "relu":
                act = tf.nn.relu

            def rum_cell():
                return RUM.RUMCell(F_size,
                                   # eta_=config.T_norm,
                                   eta_=FLAGS.eta,
                                   use_zoneout=config.use_zoneout,
                                   use_layer_norm=config.use_layer_norm,
                                   is_training=is_training,
                                   update_gate=config.update_gate,
                                   lambda_=0,
                                   activation=act)
            mcell = MultiRNNCell([rum_cell() for _ in range(
                config.num_layers)], state_is_tuple=True)
            self._initial_state = mcell.zero_state(batch_size, tf.float32)
            state = self._initial_state
            print(colored(mcell, "yellow"))
        elif config.cell == "lstm":
            def lstm_cell():
                return LNLSTM.LN_LSTMCell(F_size, use_zoneout=True, is_training=is_training,
                                          zoneout_keep_h=config.zoneout_h, zoneout_keep_c=config.zoneout_c)
            mcell = MultiRNNCell([lstm_cell() for _ in range(
                config.num_layers)], state_is_tuple=True)
            self._initial_state = mcell.zero_state(batch_size, tf.float32)
            state = self._initial_state
            print(colored(mcell, "yellow"))
        else:
            FS_cell = FSRNN.FSRNNCell(
                F_cells, S_cell, config.keep_prob, is_training)
            self._initial_state = FS_cell.zero_state(batch_size, tf.float32)
            state = self._initial_state
            print(colored(FS_cell, "yellow"))

        outputs = []
        print(colored('generating graph', "blue"))
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                if config.cell not in ["rum", "lstm"]:
                    out, state = FS_cell(inputs[:, time_step, :], state)
                else:
                    out, state = mcell(inputs[:, time_step, :], state)
                outputs.append(out)

        print(colored('graph generated', "blue"))
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, F_size])

        # Output layer and cross entropy loss

        out_init = aux.orthogonal_initializer(1.0)
        softmax_w = tf.get_variable(
            "softmax_w", [F_size, vocab_size], initializer=out_init, dtype=tf.float32)
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        tf.summary.scalar('cost', cost)

        self._final_state = state

        if not is_training:
            return

        # Create the parameter update ops if training

        self._lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(
                cost, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            config.max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        feed_dict[model.initial_state] = state

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print(colored("%.3f BPC: %.3f speed: %.0f characters per second" %
                          (step * 1.0 / model.input.epoch_size, costs / (iters * 0.69314718056),
                           iters * model.input.batch_size / (time.time() - start_time)), "green"))

        sys.stdout.flush()

    return costs / (iters * 0.69314718056)


def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")

    config = configs.get_config(FLAGS.model)
    eval_config = configs.get_config(FLAGS.model)
    valid_config = configs.get_config(FLAGS.model)
    print(colored(config.batch_size, "blue"))
    eval_config.batch_size = 1
    valid_config.batch_size = 20

    raw_data = reader.ptb_raw_data(FLAGS.data_path + config.dataset + '/')
    train_data, valid_data, test_data, _ = raw_data

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(
                config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                m = PTBModel(is_training=True, config=config,
                             input_=train_input)

        parameters_profiler()

        with tf.name_scope("Valid"):
            valid_input = PTBInput(
                config=config, data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = PTBModel(is_training=False,
                                  config=config, input_=valid_input)

        with tf.name_scope("Test"):
            test_input = PTBInput(config=eval_config,
                                  data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = PTBModel(is_training=False, config=eval_config,
                                 input_=test_input)

        # merged_summary = tf.summary.merge_all()
        saver = tf.train.Saver(tf.trainable_variables())

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            # train_writer = tf.summary.FileWriter(FLAG.save_path, session.graph)
            threads = tf.train.start_queue_runners(sess=session, coord=coord)
            if FLAGS.restore == "True":
                saver.restore(session, os.path.join(
                    FLAGS.save_path, 'model.ckpt'))
            if FLAGS.mode == "train":
                previous_val = 9999
                if FLAGS.restore == "True":
                    f = open(FLAGS.save_path + 'train-and-valid.txt', 'r')
                    x = f.readlines()[2]
                    x = x.rstrip()
                    x = x.split(" ")
                    previous_val = float(x[1])
                    print(colored("previous validation is %f\n" %
                                  (previous_val), "green"))
                    f.close()
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i +
                                                      1 - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)

                    print(colored("Epoch: %d Learning rate: %.3f" %
                                  (i + 1, session.run(m.lr)), "green"))
                    train_perplexity = run_epoch(
                        session, m, eval_op=m.train_op, verbose=True)
                    print(colored("Epoch: %d Train BPC: %.4f" %
                                  (i + 1, train_perplexity), "green"))
                    valid_perplexity = run_epoch(session, mvalid)
                    print(colored("Epoch: %d Valid BPC: %.4f" %
                                  (i + 1, valid_perplexity), "green"))
                    sys.stdout.flush()

                    if i == 180:
                        config.learning_rate *= 0.1

                    if valid_perplexity < previous_val:
                        print(colored("Storing weights", "blue"))
                        saver.save(session, os.path.join(
                            FLAGS.save_path, 'model.ckpt'))
                        f = open(FLAGS.save_path + 'train-and-valid.txt', 'w')
                        f.write("Epoch %d\nTrain %f\nValid %f\n" %
                                (i, train_perplexity, valid_perplexity))
                        f.close()
                        previous_val = valid_perplexity
                        counter_val = 0
                    elif config.dataset == 'enwik8':
                        counter_val += 1
                        if counter_val == 2:
                            config.learning_rate *= 0.1
                            counter_val = 0

            print(colored("Loading best weights", "blue"))
            saver.restore(session, os.path.join(FLAGS.save_path, 'model.ckpt'))
            test_perplexity = run_epoch(session, mtest)
            print(colored("Test Perplexity: %.4f" % test_perplexity, "green"))
            f = open(FLAGS.save_path + 'test_2.txt', 'w')
            f.write("Test %f\n" % (test_perplexity))
            f.close()
            sys.stdout.flush()
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.app.run()
