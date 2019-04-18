# rotational-unit-of-memory

Rotational Unit of Memory Official Tensorflow Repo. 

Official website: http://super-ms.mit.edu/rum.html

If this work is useful to you, please cite: https://www.mitpressjournals.org/doi/full/10.1162/tacl_a_00258

Previous work: https://github.com/jingli9111/RUM-Tensorflow

Paper will be presented at NAACL-HLT 2019: https://naacl2019.org


# Usage

RUM is used as a drop-in replacement of the standard RNN cells, GRUs, LSTMs, etc. To use it in your research you need two lines of code:

```
from RUM import RUMCell
rnn_cell = RUMCell(hidden_size)
```

You can further inspect the arguments of RUMCell that account for its tunability:

```
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
```            

Note that the Rotation operation can be used in context outside of RNNs. For example, to rotate a vector `v` by a rotation `R(v1,v2)` encoded by the vectors `v1` and `v2` you need two lines of code: 

```
from RUM import rotate 
new_v = rotate(v1, v2, v)
```

You can also play with the `rotation_operator` and `rotation_components` functions in `RUM.py`.

# Tasks

Please inspect the content of `tasks/` for training and inference options. A simple example for training on the copying tasks is:

```
python tasks/copying/copying_task.py RUM
```

The code in `tasks/LM/` is based on [1] and `tasks/summarization/` is based on [2].

# License

This project is licensed under the terms of the MIT license.

# References

[1] https://github.com/amujika/Fast-Slow-LSTM

[2] https://github.com/abisee/pointer-generator
