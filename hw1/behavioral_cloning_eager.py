import tensorflow as tf
import gym
import pickle
import argparse
import os
import numpy as np
import tf_util
from cloning_model import CloningModel

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
        model = CloningModel(action_dim, state_dim, layers=[400, 200, 100])
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

        _ = model(tf.placeholder(dtype=tf.float32, shape=[None, observations.shape[-1]]))
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("./tb/", sess.graph)
            writer.close()

def run_eager_train(action_dim, state_dim, actions, observations, envname):
    tf.enable_eager_execution()

    model = CloningModel(action_dim, state_dim, layers=[400, 200, 100])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    global_step = tf.train.get_or_create_global_step()

    logdir = "./tb/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()

    batch_size = 2048
    training_steps = 800
    losses = []

    checkpoint_dir = f'./checkpoints_{envname}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

    root = tf.train.Checkpoint(optimizer=optimizer,
            model=model,
            optimizer_step=tf.train.get_or_create_global_step())
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
            root.save(checkpoint_prefix)
    root.save(checkpoint_prefix)

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
        run_eager_train(action_dim, state_dim, actions, observations, args.envname)

if __name__ == '__main__':
    main()
