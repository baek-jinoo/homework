import tensorflow as tf
import numpy as np

class CloningModel(tf.keras.Model):
    def __init__(self, action_dim, state_dim, layers=[60, 40]):
        super(CloningModel, self).__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim

        previous_dim = self._state_dim
        self.mlp_weights = []
        self.biases = []
        self.activations = []
        for i, layer in enumerate(layers):
            w = tf.get_variable(dtype=tf.float32,
                    shape=[previous_dim, layer],
                    name=f'w{i}',
                    initializer=tf.contrib.layers.xavier_initializer())
            self.mlp_weights.append(w)
            setattr(self, f'w{i}', w)
            b = tf.get_variable(dtype=tf.float32,
                    shape=[layer],
                    name=f'b{i}',
                    initializer=tf.constant_initializer(0.))
            self.biases.append(b)
            setattr(self, f'b{i}', b)
            self.activations.append(tf.nn.relu)
            previous_dim = layer

        w = tf.get_variable(dtype=tf.float32,
                shape=[previous_dim, self._action_dim],
                name=f'w{len(layers)}',
                initializer=tf.contrib.layers.xavier_initializer())
        self.mlp_weights.append(w)
        setattr(self, f'w{len(layers)}', w)
        b = tf.get_variable(dtype=tf.float32,
                shape=[self._action_dim],
                name=f'b{len(layers)}',
                initializer=tf.constant_initializer(0.))
        self.biases.append(b)
        setattr(self, f'b{len(layers)}', b)
        self.activations.append(None)

    def call(self, states_placeholder, training=True):
        if isinstance(states_placeholder, np.ndarray):
            states_placeholder = states_placeholder.astype(np.float32)
        previous_layer = states_placeholder
        for i, (weight, bias, activation) in enumerate(zip(self.mlp_weights, self.biases, self.activations)):
            previous_layer = tf.matmul(previous_layer, weight) + bias
            if i < len(self.mlp_weights) - 1:
                previous_layer = tf.layers.batch_normalization(previous_layer, training=training)
                previous_layer = tf.layers.dropout(previous_layer, rate=0.5)
            if activation is not None:
                previous_layer = activation(previous_layer)

        output_layer = previous_layer
        return output_layer

