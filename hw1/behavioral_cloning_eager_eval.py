import tensorflow as tf
import argparse
import os
from cloning_model import CloningModel
import numpy as np
import gym

def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--max_timesteps', type=int)

    args = parser.parse_args()

    def action_obs_dims_hack():
        expert_data = None
        import pickle
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
            print(f)
            expert_data = pickle.load(f)

        if expert_data is None:
            raise Exception('Failed to load expert data')

        observations = np.asarray(expert_data['observations'])
        actions = np.asarray(expert_data['actions'])
        actions = np.squeeze(actions)

        # initialize params for the agent
        action_dim = actions.shape[-1]
        state_dim = observations.shape[-1]
        return (action_dim, state_dim)

    checkpoint_dir = f'./checkpoints_{args.envname}'
    print(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    action_dim, state_dim = action_obs_dims_hack()
    model = CloningModel(action_dim, state_dim, [400, 200, 100])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    root = tf.train.Checkpoint(model=model, optimizer= optimizer)
    status = root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    env = gym.make(args.envname)
    obs = env.reset()
    max_steps = args.max_timesteps or env.spec.timestep_limit

    done = False

    rewards = []
    accum_r = 0
    steps = 0

    while not done:
        next_action = model(obs[None, :])
        obs, reward, done, _ = env.step(next_action)
        accum_r += reward
        steps += 1
        rewards.append(reward)
        if steps % 10 == 0:
            print(f'{steps}/{max_steps}')

    print('steps', steps)
    print('totalr', accum_r)
    print('std:', np.std(rewards))
    print('mean:', np.mean(rewards))


if __name__ == '__main__':
    main()
