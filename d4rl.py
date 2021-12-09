import numpy as np
import gym

def qlearning_dataset(env):

    dataset = {k: [] for k in ['observations', 'actions', 'next_observations', 'rewards', 'terminals']}

    for i_episode in range(200):
        observation = env.reset()
        for i_step in range(500):
            action = env.action_space.sample()
            next_observation, reward, terminal, info = env.step(action)
            for k, v in zip([k for k in dataset], [observation, action, next_observation, reward, terminal]):
                dataset[k].append(v)

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
