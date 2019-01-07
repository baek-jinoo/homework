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

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))
    #return tf.losses.huber_loss(desired_y, predicted_y)

def train(model, inputs, outputs, optimizer):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.GradientTape() as t:
            current_loss = loss(model(inputs), outputs)
    grads = t.gradient(current_loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables),
            global_step=tf.train.get_or_create_global_step())
    return (current_loss, grads)

def print_graph(action_dim, state_dim, _, observations):
    logdir = "./tb/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    g_1 = tf.Graph()
    with g_1.as_default():
        model = Model(action_dim, state_dim, layers=[400, 200, 100])
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

        _ = model(tf.placeholder(dtype=tf.float32, shape=[None, observations.shape[-1]]))
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./tb/", sess.graph)
            writer.close()

def run_eager_train(action_dim, state_dim, actions, observations):
    tf.enable_eager_execution()

    model = Model(action_dim, state_dim, layers=[400, 200, 100])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    global_step = tf.train.get_or_create_global_step()

    logdir = "./tb/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    batch_size = 2048
    training_steps = 1000
    losses = []
    for training_step in range(training_steps):
        with tf.contrib.summary.record_summaries_every_n_global_steps(10):

            indices = np.random.randint(observations.shape[0], size=batch_size)
            batch_actions = actions[indices]
            batch_observations = observations[indices]
            inputs = batch_observations
            outputs = batch_actions
            loss, _ = train(model, inputs, outputs, optimizer)
            losses.append(loss.numpy())

            tf.contrib.summary.scalar('loss', loss)
            tf.contrib.summary.scalar('global_step', global_step)

        if training_step % 100 == 0:
            print(f'{training_step} loss:', loss.numpy())

    #import tempfile
    #losses_f = '[' + ','.join([str(x) for x in losses]) + ']'
    #with tempfile.NamedTemporaryFile(delete=False) as f:
    #    f.write(losses_f.encode('utf-8'))
    #    f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--print_graph', action='store_true')

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

    if args.print_graph:
        print_graph(action_dim, state_dim, actions, observations)
    else:
        run_eager_train(action_dim, state_dim, actions, observations)

if __name__ == '__main__':
    main()
