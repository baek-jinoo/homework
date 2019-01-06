import tensorflow as tf
import gym
import pickle
import argparse
import os
import numpy as np
import tf_util


class Model(tf.keras.Model):

    def __init__(self, action_dim, state_dim, layers=[60, 40]):

        super(Model, self).__init__()
        self._action_dim = action_dim
        self._state_dim = state_dim

        previous_dim = self._state_dim
        self.__weights = []
        self.biases = []
        self.activations = []
        for i, layer in enumerate(layers):
            w = tf.get_variable(dtype=tf.float32,
                    shape=[previous_dim, layer],
                    name=f'w{i}',
                    initializer=tf.contrib.layers.xavier_initializer())
            self.__weights.append(w)
            b = tf.get_variable(dtype=tf.float32,
                    shape=[layer],
                    name=f'b{i}',
                    initializer=tf.constant_initializer(0.))
            self.biases.append(b)
            self.activations.append(tf.nn.relu)
            previous_dim = layer

        w = tf.get_variable(dtype=tf.float32,
                shape=[previous_dim, self._action_dim],
                name=f'w{len(layers)}',
                initializer=tf.contrib.layers.xavier_initializer())
        self.__weights.append(w)
        b = tf.get_variable(dtype=tf.float32,
                shape=[self._action_dim],
                name=f'b{len(layers)}',
                initializer=tf.constant_initializer(0.))
        self.biases.append(b)
        self.activations.append(tf.math.sigmoid)

        self.my_variables = self.__weights + self.biases

    def call(self, states_placeholder, training=True):
        previous_layer = states_placeholder.astype(np.float32)
        for i, (weight, bias, activation) in enumerate(zip(self.__weights, self.biases, self.activations)):
            previous_layer = tf.matmul(previous_layer, weight) + bias
            if i < len(self.__weights) - 1:
                previous_layer = tf.layers.batch_normalization(previous_layer, training=training)
                previous_layer = tf.layers.dropout(previous_layer, rate=0.5)
            if activation is not None:
                previous_layer = activation(previous_layer)

        output_layer = previous_layer
        return output_layer


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

def train(model, inputs, outputs, optimizer):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.GradientTape() as t:
            current_loss = loss(model(inputs), outputs)
    grads = t.gradient(current_loss, model.my_variables)
    optimizer.apply_gradients(zip(grads, model.my_variables),
            global_step=tf.train.get_or_create_global_step())
    return (current_loss, grads)

def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # import data from pickle
    expert_data = None
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        print(f)
        expert_data = pickle.load(f)

    if expert_data is None:
        raise Exception('Failed to load expert data')

    observations = np.asarray(expert_data['observations'])
    actions = np.asarray(expert_data['actions'])
    actions = np.squeeze(actions)
    if args.verbose:
        print('actions.shape', actions.shape)
        print('observations.shape', observations.shape)

    # initialize params for the agent
    action_dim = actions.shape[-1]
    state_dim = observations.shape[-1]

    model = Model(action_dim, state_dim, layers=[400, 200, 100])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    batch_size = 2048
    training_steps = 12000
    losses = []
    for training_step in range(training_steps):
        indices = np.random.randint(observations.shape[0], size=batch_size)
        batch_actions = actions[indices]
        batch_observations = observations[indices]
        inputs = batch_observations
        outputs = batch_actions
        loss, _ = train(model, inputs, outputs, optimizer)
        losses.append(loss.numpy())
        if training_step % 100 == 0:
            print(f'{training_step} loss:', loss.numpy())

if __name__ == '__main__':
    main()
