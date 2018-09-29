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
from termcolor import colored

from tensorflow.contrib.rnn import BasicLSTMCell, BasicRNNCell, GRUCell
from RUM import RUMCell
from baselineModels.GORU import GORUCell
from baselineModels.EUNN import EUNNCell

from tensorflow.contrib.layers import fully_connected

# preprocess data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q,
            answer in data if not max_length or len(flatten(story)) < max_length]

    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen, attention):
    xs = []
    qs = []
    ys = []

    x_len = []
    q_len = []
    vocab_length = len(word_idx) + 1

    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        q = [word_idx[w] for w in query]

        len_x = len(x)
        if len_x < story_maxlen:
            x = [0] * (story_maxlen - len_x) + x

        len_q = len(q)
        q_len.append(len_q)
        for i in range(len_q, query_maxlen):
            q.append(0)

        if not attention:
            y = word_idx[answer]
        else:
            # does the scheme that me and Preslav discussed
            ind = word_idx[answer]
            y = x + [ind]
            numbers = range(ind) + range(ind + 1, vocab_length)
            r = random.choice(numbers)
            x = x + [r]
            len_x += 1
        x_len.append(len_x)
        xs.append(x)
        qs.append(q)
        ys.append(y)

    return xs, qs, ys, x_len, q_len


def main(model,
         qid,
         data_path,
         level,
         attention,
         margin,
         n_iter,
         n_batch,
         n_hidden,
         n_embed,
         capacity,
         comp,
         FFT,
         learning_rate,
         norm,
         grid_name,
         update_gate,
         activation,
         lambd):

    # preprocessing
    learning_rate = float(learning_rate)
    margin = float(margin)
    tar = tarfile.open(data_path)

    name_str = [
        'single-supporting-fact',
        'two-supporting-facts',
        'three-supporting-facts',
        'two-arg-relations',
        'three-arg-relations',
        'yes-no-questions',
        'counting',
        'lists-sets',
        'simple-negation',
        'indefinite-knowledge',
        'basic-coreference',
        'conjunction',
        'compound-coreference',
        'time-reasoning',
        'basic-deduction',
        'basic-induction',
        'positional-reasoning',
        'size-reasoning',
        'path-finding',
        'agents-motivations',
    ]

    challenge = 'tasks_1-20_v1-2/en-10k/qa' + \
        str(qid) + '_' + name_str[qid - 1] + '_{}.txt'

    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])

    vocab = sorted(vocab)

    # if attention:
    #     # generates the complements of the vocabulary
    #     complement_vocabs = {}
    #     for x in vocab:
    #         complement_vocabs[x] = list(vocab)
    #         complement_vocabs[x].remove(x)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    train_x, train_q, train_y, train_x_len, train_q_len = vectorize_stories(
        train, word_idx, story_maxlen, query_maxlen, attention)
    test_x, test_q, test_y, test_x_len, test_q_len = vectorize_stories(
        test, word_idx, story_maxlen, query_maxlen, attention)
    if attention:
        story_maxlen += 1
    n_data = len(train_x)
    n_val = int(0.1 * n_data)

    val_x = train_x[-n_val:]
    val_q = train_q[-n_val:]
    val_y = train_y[-n_val:]
    val_x_len = train_x_len[-n_val:]
    val_q_len = train_q_len[-n_val:]
    train_x = train_x[:-n_val]
    train_q = train_q[:-n_val]
    train_y = train_y[:-n_val]
    train_q_len = train_q_len[:-n_val]
    train_x_len = train_x_len[:-n_val]

    n_train = len(train_x)

    print(colored('vocab = {}'.format(vocab), 'yellow'))
    print(colored('x.shape = {}'.format(np.array(train_x).shape), 'yellow'))
    print(colored('xq.shape = {}'.format(np.array(train_q).shape), 'yellow'))
    print(colored('y.shape = {}'.format(np.array(train_y).shape), 'yellow'))
    print(colored('story_maxlen, query_maxlen = {}, {}'.format(
        story_maxlen, query_maxlen), 'yellow'))

    print(colored("building model", "blue"))

    # model
    sentence = tf.placeholder("int32", [None, story_maxlen])
    question = tf.placeholder("int32", [None, query_maxlen])
    if not attention:
        answer_holder = tf.placeholder("int64", [None])
    else:
        answer_holder = tf.placeholder("int64", [None, story_maxlen])

    n_output = n_hidden
    n_input = n_embed
    n_classes = vocab_size

    with tf.variable_scope("embedding"):
        embed_init_val = np.sqrt(6.) / np.sqrt(vocab_size)
        embed = tf.get_variable("embedding", [vocab_size, n_embed], initializer=tf.random_normal_initializer(
            -embed_init_val, embed_init_val), dtype=tf.float32)
        encoded_story = tf.nn.embedding_lookup(embed, sentence)
        encoded_question = tf.nn.embedding_lookup(embed, question)
    if attention:
        encoded_answer = tf.nn.embedding_lookup(embed, answer_holder)
    else:
        merged = tf.concat([encoded_story, encoded_question], axis=1)

    # defined the rnn cell
    if model == "LSTM":
        cell = BasicLSTMCell(n_hidden, state_is_tuple=True, forget_bias=1)
    elif model == "GRU":
        cell = GRUCell(n_hidden)
    elif model == "RUM":
        if activation == "relu":
            act = tf.nn.relu
        elif activation == "sigmoid":
            act = tf.nn.sigmoid
        elif activation == "tanh":
            act = tf.nn.tanh
        elif activation == "softsign":
            act = tf.nn.softsign
        cell = RUMCell(n_hidden,
                       eta_=norm,
                       update_gate=update_gate,
                       lambda_=lambd,
                       activation=act)
    elif model == "EUNN":
        cell = EUNNCell(n_hidden, capacity, FFT, comp, name="eunn")
    elif model == "GORU":
        cell = GORUCell(n_hidden, capacity, FFT)

    if not attention:
        merged, _ = tf.nn.dynamic_rnn(cell, merged, dtype=tf.float32)
        print("merged:", colored(merged, 'green'))
        # hidden to output
        with tf.variable_scope("hidden_to_output"):
            V_init_val = np.sqrt(6.) / np.sqrt(n_output + n_input)
            V_weights = tf.get_variable("V_weights", shape=[
                                        n_hidden, n_classes], dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-V_init_val, V_init_val))
            V_bias = tf.get_variable("V_bias", shape=[
                                     n_classes], dtype=tf.float32, initializer=tf.constant_initializer(0.01))

            merged_list = tf.unstack(merged, axis=1)[-1]
            temp_out = tf.matmul(merged_list, V_weights)
            final_out = tf.nn.bias_add(temp_out, V_bias)

        with tf.variable_scope("loss"):
            # evaluate process
            cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=final_out, labels=answer_holder))
            correct_pred = tf.equal(tf.argmax(final_out, 1), answer_holder)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    else:
        output_q, _ = tf.nn.dynamic_rnn(
            cell, encoded_question, dtype=tf.float32)
        output_x, _ = tf.nn.dynamic_rnn(cell, encoded_story, dtype=tf.float32)
        output_a, _ = tf.nn.dynamic_rnn(cell, encoded_answer, dtype=tf.float32)
        print("output_x:", colored(output_x, 'green'))
        print("output_q:", colored(output_q, 'green'))
        print("output_a:", colored(output_a, 'green'))
        with tf.variable_scope("attention"):
            ref_attn = tf.get_variable("ref_attention", shape=[
                                       n_hidden, 1], dtype=tf.float32)
            W_attn = tf.get_variable("W_attention", shape=[
                                     n_hidden, n_hidden], dtype=tf.float32)
            b_attn = tf.get_variable("b_attention", shape=[
                                     n_hidden], dtype=tf.float32, initializer=tf.constant_initializer(0.01))
            output_q_p = tf.reshape(output_q, [-1, n_hidden])
            output_x_p = tf.reshape(output_x, [-1, n_hidden])
            output_a_p = tf.reshape(output_a, [-1, n_hidden])
            h_q = tf.nn.tanh(tf.matmul(output_q_p, W_attn) + b_attn)
            h_x = tf.nn.tanh(tf.matmul(output_x_p, W_attn) + b_attn)
            h_a = tf.nn.tanh(tf.matmul(output_a_p, W_attn) + b_attn)
            h_q = tf.matmul(h_q, ref_attn)
            h_x = tf.matmul(h_x, ref_attn)
            h_a = tf.matmul(h_a, ref_attn)
            h_q = tf.reshape(h_q, [-1, query_maxlen, 1])
            h_x = tf.reshape(h_x, [-1, story_maxlen, 1])
            h_a = tf.reshape(h_a, [-1, story_maxlen, 1])
            alphas_q = tf.nn.softmax(h_q, axis=1)
            alphas_x = tf.nn.softmax(h_x, axis=1)
            alphas_a = tf.nn.softmax(h_a, axis=1)
            # get context vectors
            c_q = tf.reduce_sum(alphas_q * output_q, axis=1)
            c_x = tf.reduce_sum(alphas_x * output_x, axis=1)
            c_a = tf.reduce_sum(alphas_a * output_a, axis=1)

        with tf.variable_scope("loss"):
            normalize_q = tf.nn.l2_normalize(c_q, 1)
            normalize_x = tf.nn.l2_normalize(c_x, 1)
            normalize_a = tf.nn.l2_normalize(c_a, 1)
            cos_similarity_qa = tf.reduce_sum(
                tf.multiply(normalize_q, normalize_a), axis=1)
            cos_similarity_qx = tf.reduce_sum(
                tf.multiply(normalize_q, normalize_x), axis=1)
            prelim_loss = tf.nn.relu(
                margin - cos_similarity_qa + cos_similarity_qx)
            cost = tf.reduce_mean(prelim_loss)

    # initialization
    tf.summary.scalar('cost', cost)
    if not attention:
        tf.summary.scalar('accuracy', accuracy)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

   # --- save result ----------------------
    # + "_lambda=" + str(learning_rate) + "_beta=" + str(decay)
    folder = "./output/babi/" + str(qid) + '/' + model
    filename = folder + "_h=" + str(n_hidden)
    filename = filename + "_lr=" + str(learning_rate)
    filename = filename + "_norm=" + str(norm)

    # --- Training Loop ----------------------
    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    step = 0
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=False)) as sess:

        train_writer = tf.summary.FileWriter(
            os.path.join('train_log', 'babi', 'word', str(qid), model), sess.graph)
        saver = tf.train.Saver()
        sess.run(init)

        steps = []
        losses = []
        accs = []

        # val_dict = {sentence: val_x,
        #             question: val_q, answer_holder: val_y}
        # val_dicts = []
        # val_answers = []
        # val_cores = []
        # for i in range(len(val_x)):
        #     core = val_y[i][:-1]
        #     answer = val_y[i][-1]
        #     val_answers.append(answer)
        #     for j in range(vocab_size):
        #         val_dicts.append(
        #             {sentence: val_x, question: val_q, answer_holder: core + [j]})
        # print(val_answers)
        # print(colored(len(val_answers), "yellow"))
        # input()

        while step < n_iter:
            a = int(step % (n_train / n_batch))
            batch_x = train_x[a * n_batch: (a + 1) * n_batch]
            batch_q = train_q[a * n_batch: (a + 1) * n_batch]
            batch_y = train_y[a * n_batch: (a + 1) * n_batch]

            train_dict = {sentence: batch_x,
                          question: batch_q, answer_holder: batch_y}
            summ, loss = sess.run(
                [merged_summary, cost], feed_dict=train_dict)
            train_writer.add_summary(summ, step)
            sess.run(optimizer, feed_dict=train_dict)

            if not attention:
                acc = sess.run(accuracy, feed_dict=train_dict)
                print(colored("Iter " + str(step) + ", Minibatch Loss= " +
                              "{:.6f}".format(loss) + ", Training Accuracy= " +
                              "{:.5f}".format(acc), 'green'))
            else:
                print(colored("Iter " + str(step) + ", Minibatch Loss= " +
                              "{:.6f}".format(loss), 'green'))
            steps.append(step)
            losses.append(loss)
            if not attention:
                accs.append(acc)
            step += 1

            if step % 200 == 1:
                val_dicts = []

                val_dict = {sentence: val_x,
                            question: val_q, answer_holder: val_y}
                if not attention:
                    val_acc = sess.run(accuracy, feed_dict=val_dict)
                val_loss = sess.run(prelim_loss, feed_dict=val_dict)
                print(val_loss)
                print(val_loss.shape)
                raw_input()
                if not attention:
                    print("Validation Loss= " +
                          "{:.6f}".format(val_loss) + ", Validation Accuracy= " +
                          "{:.5f}".format(val_acc))

        print(colored("Optimization Finished!", 'blue'))

        # --- test ----------------------
        test_dict = {sentence: test_x, question: test_q, answer_holder: test_y}
        test_acc = sess.run(accuracy, feed_dict=test_dict)
        test_loss = sess.run(cost, feed_dict=test_dict)
        f.write("Test result: Loss= " + "{:.6f}".format(test_loss) +
                ", Accuracy= " + "{:.5f}".format(test_acc))
        print("Test result: Loss= " + "{:.6f}".format(test_loss) +
              ", Accuracy= " + "{:.5f}".format(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="bAbI Task")
    parser.add_argument("model", default='LSTM',
                        help='Model name: LSTM, EUNN, GRU, GORU')
    parser.add_argument('qid', type=int, default=20, help='Test set')
    parser.add_argument('level', type=str, default="word",
                        help='level: word or sentence')
    parser.add_argument('attention', type=str,
                        default=False, help='is attn. mechn.')
    parser.add_argument('--margin', '-M', type=str,
                        default=0.2, help='margin for attention')
    parser.add_argument('--n_iter', '-I', type=int,
                        default=10000, help='training iteration number')
    parser.add_argument(
        '--data_path', default="../RUM-TF-2/data/tasks_1-20_v1-2.tar.gz", type=str)
    parser.add_argument('--n_batch', '-B', type=int,
                        default=32, help='batch size')
    parser.add_argument('--n_hidden', '-H', type=int,
                        default=256, help='hidden layer size')
    parser.add_argument('--n_embed', '-E', type=int,
                        default=64, help='embedding size')
    parser.add_argument('--capacity', '-L', type=int, default=2,
                        help='Tunable style capacity, only for EUNN, default value is 2')
    parser.add_argument('--comp', '-C', type=str, default="False",
                        help='Complex domain or Real domain. Default is False: real domain')
    parser.add_argument('--FFT', '-F', type=str, default="True",
                        help='FFT style, default is False')
    parser.add_argument('--learning_rate', '-R', default=0.001, type=str)
    # parser.add_argument('--decay', '-D', default=0.9, type=str)
    parser.add_argument('--norm', '-norm', default=None, type=float)
    parser.add_argument('--grid_name', '-GN', default=None,
                        type=str, help='specify folder to save to')
    parser.add_argument('--update_gate', '-U', default=1,
                        type=bool, help='is there update gate?')
    parser.add_argument('--activation', '-A', default="relu",
                        type=str, help='specify activation')
    parser.add_argument('--lambd', '-LA', default=0,
                        type=int, help='lambda for RUM model')

    args = parser.parse_args()
    dicts = vars(args)

    for i in dicts:
        if (dicts[i] == "False"):
            dicts[i] = False
        elif dicts[i] == "True":
            dicts[i] = True

    kwargs = {
        'model': dicts['model'],
        'qid': dicts['qid'],
        'level': dicts['level'],
        'attention': dicts['attention'],
        'margin': dicts['margin'],
        'data_path': dicts['data_path'],
        'n_iter': dicts['n_iter'],
        'n_batch': dicts['n_batch'],
        'n_hidden': dicts['n_hidden'],
        'n_embed': dicts['n_embed'],
        'capacity': dicts['capacity'],
        'comp': dicts['comp'],
        'FFT': dicts['FFT'],
        'learning_rate': dicts['learning_rate'],
        # 'decay': dicts['decay'],
        'norm': dicts['norm'],
        'grid_name': dicts['grid_name'],
        'update_gate': dicts['update_gate'],
        'activation': dicts['activation'],
        'lambd': dicts['lambd']
    }

    main(**kwargs)
