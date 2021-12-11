import numpy as np
import gym
import os

def qlearning_dataset(env):

    dataset = {k: [] for k in ['observations', 'actions', 'next_observations', 'rewards', 'terminals']}

    len_sample = 1000000
    i_sample = 0

    while 1:
        observation = env.reset()
        for i_step in range(500):
            action = env.action_space.sample()
            next_observation, reward, terminal, info = env.step(action)
            for k, v in zip([k for k in dataset], [observation, action, next_observation, reward, terminal]):
                dataset[k].append(v)

            i_sample += 1
            if i_sample == len_sample:
                break

            if terminal:
                break

        if i_sample == len_sample:
            break

    assert len(dataset['observations']) == len_sample
    dataset = {k: np.array(dataset[k]) for k in dataset}

    return dataset

def load_data():

    dataset = {k: [] for k in ['observations', 'actions', 'next_observations', 'rewards', 'terminals']}

    len_sample = 1000000
    i_sample = 0

    save_path = os.path.join('..', 'DDPG.data', 'trajectory-expert')
    for i, f in enumerate(os.listdir(save_path)):
        trajectory = np.load(os.path.join(save_path, f), allow_pickle=True)

        assert np.array(trajectory).shape[0] < 500  # 断言没有达到最大步数，即是 terminal 为 True 导致的退出。

        for i_step, trajectory_step in enumerate(trajectory):

            terminal = False
            if i_step == np.array(trajectory).shape[0] - 1:
                terminal = True

            observation, action, next_observation, reward = trajectory_step['state'], trajectory_step['action'], trajectory_step['next_state'], trajectory_step['reward']
            for k, v in zip([k for k in dataset], [observation, action, next_observation, reward, terminal]):
                dataset[k].append(v)

            i_sample += 1
            if i_sample == len_sample:
                break

        if i_sample == len_sample:
            break
    print(i, i_sample)
    assert len(dataset['observations']) == len_sample
    dataset = {k: np.array(dataset[k]) for k in dataset}

    return dataset

def make(env_name):

    env = gym.make(env_name)

    def get_normalized_score(score):
        return score

    env.get_normalized_score = get_normalized_score

    return env

def main():

    env = gym.make('Hopper-v2')
    qlearning_dataset(env)

if __name__ == '__main__':
    main()
