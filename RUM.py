import tensorflow as tf
import numpy as np
import baselineModels.auxiliary as aux

from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.contrib.layers import fully_connected


def rotation_components(x, y, eps=1e-12, costh=None):
    """Components for the operator Rotation(x,y)
       Together with `rotation_operator` achieves best memory complexity: O(N_batch * N_hidden)

    Args: 
            x: a tensor from where we want to start 
            y: a tensor at which we want to finish 
            eps: the cutoff for the normalizations (avoiding division by zero)
    Returns: 
            Five components: u, v, [u,v] and `2x2 rotation by theta`, cos(theta)
    """

    size_batch = tf.shape(x)[0]
    hidden_size = tf.shape(x)[1]

    # construct the 2x2 rotation
    u = tf.nn.l2_normalize(x, 1, epsilon=eps)
    if costh == None:
        costh = tf.reduce_sum(u * tf.nn.l2_normalize(y, 1, epsilon=eps), 1)
    sinth = tf.sqrt(1 - costh ** 2)
    step1 = tf.reshape(costh, [size_batch, 1])
    step2 = tf.reshape(sinth, [size_batch, 1])
    Rth = tf.reshape(
        tf.concat([step1, -step2, step2, step1], axis=1), [size_batch, 2, 2])

    # get v and concatenate u and v
    v = tf.nn.l2_normalize(
        y - tf.reshape(tf.reduce_sum(u * y, 1), [size_batch, 1]) * u, 1, epsilon=eps)
    step3 = tf.concat([tf.reshape(u, [size_batch, 1, hidden_size]),
                       tf.reshape(v, [size_batch, 1, hidden_size])],
                      axis=1)

    # do the batch matmul
    step4 = tf.reshape(u, [size_batch, hidden_size, 1])
    step5 = tf.reshape(v, [size_batch, hidden_size, 1])
    return step4, step5, step3, Rth, costh


def rotation_operator(x, y, hidden_size, eps=1e-12):
    """Rotational matrix tensor between two tensors: R(x,y) is orthogonal and takes x to y. 

    Args: 
            x: a tensor from where we want to start 
            y: a tensor at which we want to finish 
            hidden_size: the hidden size 
            eps: the cutoff for the normalizations (avoiding division by zero)
    Returns: 
            A pair: `a tensor, which is the orthogonal rotation operator R(x,y)`, cos(theta)

    Comment: 
            For your research you may decide that you want to use only `rotation` when lambda!=0,
            but this will come at a trade-off with increased latency. Currently, we are investigating 
            optimal implementation.
    """
    step4, step5, step3, Rth, costh = rotation_components(x, y, eps=eps)
    size_batch = tf.shape(step4)[0]
    return (tf.eye(hidden_size, batch_shape=[size_batch]) -
            tf.matmul(step4, tf.transpose(step4, [0, 2, 1])) -
            tf.matmul(step5, tf.transpose(step5, [0, 2, 1])) +
            tf.matmul(tf.matmul(tf.transpose(step3, [0, 2, 1]), Rth), step3)), costh


def rotate(v1, v2, v, costh=None):
    """Rotates v via the rotation R(v1,v2)

    Args: 
            v: a tensor, which is the vector we want to rotate
            == to define rotation matrix R(v1,v2) == 
            v1: a tensor from where we want to start 
            v2: a tensor at which we want to finish 

    Returns: 
            A pair: `rotated vector R(v1,v2)[v]`, cos(theta)
    """
    size_batch = tf.shape(v1)[0]
    hidden_size = tf.shape(v1)[1]

    U = rotation_components(v1, v2, costh=costh)
    h = tf.reshape(v, [size_batch, hidden_size, 1])

    return (v + tf.reshape(
        - tf.matmul(U[0], tf.matmul(tf.transpose(U[0], [0, 2, 1]), h))
        - tf.matmul(U[1], tf.matmul(tf.transpose(U[1], [0, 2, 1]), h))
        + tf.matmul(tf.transpose(U[2], [0, 2, 1]),
                    tf.matmul(U[3], tf.matmul(U[2], h))),
        [size_batch, hidden_size]
    )), U[4]


class RUMCell(RNNCell):
    """Rotational Unit of Memory

    lambda = 0; 
    uses `rotate` to implement the `Rotation` efficiently.
    """

    def __init__(self,
                 hidden_size,
                 lambda_=0,
                 eta_=None,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 eps=1e-12,
                 use_zoneout=False,
                 zoneout_keep_h=0.9,
                 use_layer_norm=False,
                 is_training=False,
                 # following arguments are for ablation studies
                 # and further research
                 update_gate=True,
                 trainable_rot=True,
                 track_angle=False,
                 # research on visualization
                 visualization=False,
                 temp_target=None,
                 temp_target_bias=None,
                 temp_embed=None
                 ):
        """RUM init

        Args:
                hidden_size: number of neurons in hidden state
                lambda_: lambda parameter for the associative memory
                eta_: eta parameter for the norm for the time normalization
                acitvation: activation of the temporary new state
                reuse: reuse setting
                kernel_initializer: init for kernel
                bias_initializer: init for bias
                eps: the cutoff for the normalizations
                use_zoneout: zoneout, True or False
                use_layer_norm: batch normalization, True or False
                is_training: marker for the zoneout
                update_gate: use update gate, True or False
                trainable_rot: use trainable rotation, True or False,
                track_angle: keep track of the angle, True or False
                visualization: whether to visualize the energy landscape
                temp_target: a placeholder to feed in for visualization 
                temp_target_bias: a placeholder to feed in for visualization
                temp_embed: a placeholder to feed in for visualization
        """
        super(RUMCell, self).__init__(_reuse=reuse)
        self._hidden_size = hidden_size
        if lambda_ not in [0, 1]:
            raise ValueError(
                "For now we only support lambda=0,1. Feel free \
                to experiment with other values for lambda:)")
        self._lambda = lambda_
        self._eta = eta_
        self._activation = activation or tf.nn.relu
        self._kernel_initializer = kernel_initializer or aux.orthogonal_initializer(
            1.0)
        self._bias_initializer = bias_initializer
        self._eps = eps
        self._use_zoneout = use_zoneout
        self._zoneout_keep_h = zoneout_keep_h
        self._use_layer_norm = use_layer_norm
        self._is_training = is_training
        self._update_gate = update_gate
        self._trainable_rot = trainable_rot
        self._track_angle = track_angle
        self._visualization = visualization
        self._temp_target = temp_target
        self._temp_target_bias = temp_target_bias
        self._temp_embed = temp_embed

    @property
    def state_size(self):
        return self._hidden_size * (self._hidden_size * self._lambda + 1)
        # sanity check: if lambda_=0, then the state size
        # is simply self._hidden_size:)

    @property
    def output_size(self):
        if self._track_angle:
            return self._hidden_size + 1
        return self._hidden_size

    def call(self, inputs, state):
        if self._lambda != 0:
            # extract the associative memory and the state
            size_batch = tf.shape(state)[0]
            assoc_mem, state = tf.split(
                state, [self._hidden_size * self._hidden_size, self._hidden_size], 1)
            assoc_mem = tf.reshape(
                assoc_mem, [size_batch, self._hidden_size, self._hidden_size])
        with tf.variable_scope("gates"):
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                dtype = [a.dtype for a in [inputs, state]][0]
                bias_ones = tf.constant_initializer(1.0, dtype=dtype)
            if self._visualization:
                r = tf.matmul(
                    tf.concat([inputs, state], axis=1), self._temp_target) + self._temp_target_bias
            else:
                r = fully_connected(inputs=tf.concat([inputs, state], axis=1),
                                    num_outputs=self._hidden_size,
                                    activation_fn=None,
                                    biases_initializer=bias_ones,
                                    weights_initializer=aux.rum_ortho_initializer(),
                                    trainable=self._trainable_rot)
            # no update gate if there is no update gate
            if self._update_gate:
                u = fully_connected(inputs=tf.concat([inputs, state], axis=1),
                                    num_outputs=self._hidden_size,
                                    activation_fn=tf.nn.sigmoid,
                                    biases_initializer=bias_ones,
                                    weights_initializer=aux.rum_ortho_initializer(),
                                    trainable=self._trainable_rot)
            if self._use_layer_norm:
                if self._update_gate:
                    concat = tf.concat([r, u], 1)
                    concat = aux.layer_norm_all(
                        concat, 2, self._hidden_size, "ln_r_u")
                    r, u = tf.split(concat, 2, 1)
                else:
                    r = aux.layer_norm_all(
                        r, 1, self._hidden_size, "ln_r")
        with tf.variable_scope("candidate"):
            if self._visualization:
                x_emb = tf.matmul(inputs, self._temp_embed)
            else:
                x_emb = fully_connected(inputs=inputs,
                                        num_outputs=self._hidden_size,
                                        activation_fn=None,
                                        biases_initializer=self._bias_initializer,
                                        weights_initializer=self._kernel_initializer,
                                        trainable=True)
            if self._lambda == 0:
                state_new, costh = rotate(x_emb, r, state)
            else:
                tmp_rotation, costh = rotation_operator(
                    x_emb, r, self._hidden_size)
                Rt = tf.matmul(assoc_mem, tmp_rotation)
                state_new = tf.reshape(tf.matmul(Rt, tf.reshape(
                    state, [size_batch, self._hidden_size, 1])), [size_batch, self._hidden_size])
            if self._use_layer_norm:
                c = self._activation(aux.layer_norm(x_emb + state_new, "ln_c"))
            else:
                c = self._activation(x_emb + state_new)
        new_h = u * state + (1 - u) * c if self._update_gate else c
        if self._eta != None:
            new_h = tf.nn.l2_normalize(
                new_h, 1, epsilon=self._eps) * self._eta
        if self._use_zoneout:
            new_h = aux.rum_zoneout(
                new_h, state, self._zoneout_keep_h, self._is_training)
        if self._lambda == 0:
            new_state = new_h
        else:
            Rt = tf.reshape(
                Rt, [size_batch, self._hidden_size * self._hidden_size])
            new_state = tf.concat([Rt, new_h], 1)
        if self._track_angle:
            # keep track of the angle at the current time step:
            # append it to the output
            costh = tf.reshape(costh, [-1, 1])
            return tf.concat([costh, new_h], axis=1), new_state
        return new_h, new_state

    def zero_state(self, batch_size, dtype):
        if self._lambda == 0:
            h = tf.zeros([batch_size, self._hidden_size], dtype=dtype)
        else:
            e = tf.eye(self._hidden_size, batch_shape=[batch_size])
            e = tf.reshape(
                e, [batch_size, self._hidden_size * self._hidden_size])
            c = tf.zeros([batch_size, self._hidden_size], dtype=dtype)
            h = tf.concat([e, c], 1)
        return h
