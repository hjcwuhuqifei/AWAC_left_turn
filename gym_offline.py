# import filter_env
import time

import matplotlib.pyplot as plt
import gym
import gc
import numpy as np
import gym_env.envs

gc.enable()

# ENV_NAME = 'intersection-left-v0'
ENV_NAME = 'carla-v1'

EPISODES = 1000
TEST = 5


def main():
    # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    env = gym.make(ENV_NAME)
    env = env.unwrapped

    ons = env.reset()
    for i in range(110):

        obs, rew, done, _ = env.step([-0.5])
        env.render()
        # env.plot_collision_for_test()
        time.sleep(0.1)
        if done:
            env.close()
            break

    # env.close()
    # env.seed(1)
    # agent = DDPG(env)
    # agent.load_network()
    # # dic = np.load('my_data.npy').item()
    # num = 0
    # for episode in range(EPISODES + 1):
    #     state = env.reset()
    #     # print "episode:",episode
    #     # Train
    #     for step in range(800):
    #         action = agent.noise_action(state)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.perceive(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             break
    #     # print(episode)
    #     # Testing:
    #     # if episode % 100 == 0 and episode > 100:
    #     if episode % 20 == 0 and episode >= 20:
    #         total_reward = 0
    #         num = num + 1
    #         for i in range(TEST):
    #             state = env.reset()
    #             for j in range(800):
    #                 env.render()
    #                 action = agent.action(state)  ##direct action for test
    #                 # print("before:",action)
    #                 state, reward, done, _ = env.step(action)
    #                 total_reward += reward
    #                 if done:
    #                     break
    #         env.close()
    #         ave_reward = total_reward / TEST
    #         if ave_reward > -300:
    #             agent.save_network(episode)
    #
    #         plt.plot(episode, ave_reward, '*', color='#000000')
    #         # dic['{0}'.format(num)].append(ave_reward)
    #
    #         plt.pause(0.01)
    #
    #         print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
    # # np.save('my_data.npy', dic)


if __name__ == '__main__':
    main()
    print("finish")
