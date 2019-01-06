import tensorflow as tf
import gym
import pickle
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    #parser.add_argument(

    args = parser.parse_args()

    # import data from pickle
    expert_data = None
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        print(f)
        expert_data = pickle.load(f)

    if expert_data is None:
        raise Exception('Failed to load expert data')
    print(expert_data)
    observations = np.asarray(expert_data['observations'])
    actions = np.asarray(expert_data['actions'])
    actions = np.squeeze(actions)
    if verbose:
        print('actions.shape', actions.shape)
        print('observations.shape', observations.shape)

    # initialize params for the agent
    action_dim = actions.shape[-1]
    state_dim = observations.shape[-1]
    states_placeholer = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name='states')
    actions_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name='actions')

    first_weight_dim = 60
    w1 = tf.get_variable(dtype=tf.float32,
            shape=[state_dim, first_weight_dim],
            name='w1',
            initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(dtype=tf.float32,
            shape=[first_weight_dim],
            name='b1',
            initializer=tf.constant_initializer(0.))

    second_weight_dim = 40
    w2 = tf.get_variable(dtype=tf.float32,
            shape=[first_weight_dim, second_weight_dim],
            name='w2',
            initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(dtype=tf.float32,
            shape=[second_weight_dim],
            name='b2',
            initializer=tf.constant_initializer(0.))

    third_weight_dim = action_dim
    w3 = tf.get_variable(dtype=tf.float32,
            shape=[second_weight_dim, third_weight_dim],
            name='w3',
            initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(dtype=tf.float32,
            shape=[third_weight_dim],
            name='b3',
            initializer=tf.constant_initializer(0.))

    weights = [w1, w2, w3]
    biases = [b1, b2, b3]
    activations = [tf.nn.relu, tf.nn.relu, tf.math.sigmoid]

    # create the model
    previous_layer = states_placeholer
    for weight, bias, activation in zip(weights, biases, activations):
        previous_layer = tf.matmul(previous_layer, weight) + bias
        if activation is not None:
            previous_layer = activation(previous_layer)

    output_layer = previous_layer

    # loss function
    loss = tf.reduce_mean(tf.square(output_layer - actions_placeholder) * 0.5)

    # optimizer
    optim = tf.train.AdamOptimizer(learning_rate = 3e-4).minimize(loss)

    batch_size = 32
    training_steps = 100

    tf_util.initialize()
    saver = tf.train.Saver()

    lowest_loss = float('inf')
    for training_step in range(training_steps):
        indices = np.random.randint(observations.shape[0], size=batch_size)
        batch_actions = actions[indices]
        batch_observations = observations[indices]

        with tf.Session():
            session = get_session()

            _, loss = session.run([optim, loss],
                    feed_dict={actions_placeholder:batch_actions,
                        states_placeholer:batch_observations})

            if training_step % 10 == 0:
                print(f'{training_step} loss:', loss)
                if lowest_loss > loss:
                    lowest_loss = loss
                    saver.save(session, f'./intermediate_models/{args.envname}_{loss}_{training_step}.ckpt')

    # generate std and var
    # print them out

if __name__ == '__main__':
    main()
