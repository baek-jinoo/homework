import tensorflow as tf
import numpy as np

class CloningModel(tf.keras.Model):
    def __init__(self, action_dim, state_dim, input_mean, input_std, layers=[60, 40]):
        super(CloningModel, self).__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim

        previous_dim = self._state_dim
        self.mlp_weights = []
        self.biases = []
        self.activations = []
        self.batch_norms = []

        self.input_mean = tf.Variable(input_mean, dtype=tf.float32)
        self.input_std = tf.Variable(input_std, dtype=tf.float32)

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
            batch_norm = tf.layers.BatchNormalization()
            setattr(self, f'batch_norm{i}', batch_norm)
            self.batch_norms.append(batch_norm)
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
        self.batch_norms.append(None)

    def call(self, states_placeholder, training=True):
        if isinstance(states_placeholder, np.ndarray):
            states_placeholder = states_placeholder.astype(np.float32)
        previous_layer = states_placeholder
        previous_layer = (previous_layer - self.input_mean) / self.input_std
        for i, (weight, bias, activation, batch_norm) in enumerate(zip(self.mlp_weights, self.biases, self.activations, self.batch_norms)):
            previous_layer = tf.matmul(previous_layer, weight) + bias
            if batch_norm is not None:
                pass
                #previous_layer = batch_norm(previous_layer, training=training)
            if i < len(self.mlp_weights) - 1:
                pass
                #previous_layer = tf.layers.dropout(previous_layer, rate=0.5)
            if activation is not None:
                previous_layer = activation(previous_layer)

        output_layer = previous_layer
        return output_layer

