import tensorflow as tf
import gym
import pickle
import argparse
import os
import numpy as np
import tf_util


class Model():

    def __init__(self, action_dim, state_dim, layers=[60, 40]):
        self._action_dim = action_dim
        self._state_dim = state_dim

        self._first_weight_dim = layers[0]
        w1 = tf.get_variable(dtype=tf.float32,
                shape=[self._state_dim, self._first_weight_dim],
                name='w1',
                initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(dtype=tf.float32,
                shape=[self._first_weight_dim],
                name='b1',
                initializer=tf.constant_initializer(0.))

        self._second_weight_dim = layers[1]
        w2 = tf.get_variable(dtype=tf.float32,
                shape=[self._first_weight_dim, self._second_weight_dim],
                name='w2',
                initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(dtype=tf.float32,
                shape=[self._second_weight_dim],
                name='b2',
                initializer=tf.constant_initializer(0.))

        self._third_weight_dim = self._action_dim
        w3 = tf.get_variable(dtype=tf.float32,
                shape=[self._second_weight_dim, self._third_weight_dim],
                name='w3',
                initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable(dtype=tf.float32,
                shape=[self._third_weight_dim],
                name='b3',
                initializer=tf.constant_initializer(0.))

        self.weights = [w1, w2, w3]
        self.biases = [b1, b2, b3]
        self.variables = [w1, w2, w3, b1, b2, b3]
        self.activations = [tf.nn.relu, tf.nn.relu, tf.math.sigmoid]

    def __call__(self, states_placeholder):
        previous_layer = states_placeholder.astype(np.float32)
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            previous_layer = tf.matmul(previous_layer, weight) + bias
            if activation is not None:
                previous_layer = activation(previous_layer)

        output_layer = previous_layer
        return output_layer


def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

def train(model, inputs, outputs, optimizer):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    grads = t.gradient(current_loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables),
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

    model = Model(action_dim, state_dim, layers=[200, 100])
    optimizer = tf.train.AdamOptimizer()

    batch_size = 512
    training_steps = 10000
    for training_step in range(training_steps):
        indices = np.random.randint(observations.shape[0], size=batch_size)
        batch_actions = actions[indices]
        batch_observations = observations[indices]
        inputs = batch_observations
        outputs = batch_actions
        loss, _ = train(model, inputs, outputs, optimizer)
        if training_step % 10 == 0:
            print(f'{training_step} loss:', loss.numpy())

if __name__ == '__main__':
    main()
