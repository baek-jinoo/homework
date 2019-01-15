import tensorflow as tf
import argparse
import os
from cloning_model import CloningModel
import numpy as np
import gym
import tempfile
import pickle
from mujoco_py.generated import const

def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int)

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
    model = CloningModel(action_dim, state_dim, np.zeros(state_dim), np.zeros(state_dim), [400, 200, 100])
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)

    root = tf.train.Checkpoint(model=model, optimizer= optimizer)
    print(tf.train.latest_checkpoint(checkpoint_dir))
    status = root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    rewards = []
    total_steps = []
    for i in range(args.num_rollouts):
        obs = env.reset()

        done = False

        accum_r = 0
        steps = 0

        if args.render:
            env.render(mode='rgb_array')
            if 'reacher' not in args.envname.lower():
                env.unwrapped.viewer.cam.type = const.CAMERA_FIXED
                env.unwrapped.viewer.cam.fixedcamid = 0

            f = tempfile.NamedTemporaryFile()
            print('named temp file', f.name)
            filename = f.name
            f.close()
            f = open(filename, 'ab')

        while not done:
            next_action = model.predict(obs[None, :])
            obs, reward, done, _ = env.step(next_action)
            accum_r += reward
            steps += 1

            if args.render:
                pickle.dump(env.render(mode='rgb_array'), f)
            if steps % 40 == 0:
                print(f'{steps}/{max_steps}')
                #print(next_action)

        if args.render:
            f.close()

        rewards.append(accum_r)
        total_steps.append(steps)
        print('steps', steps)
        print('totalr', accum_r)
    print('rewards', rewards)
    print('steps', total_steps)
    print('std:', np.std(rewards))
    print('mean:', np.mean(rewards))


if __name__ == '__main__':
    main()
