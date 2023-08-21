import math
import numpy as np
import pickle
import gym
from pathlib import Path
from gym_env.envs.offline_gym import get_reward

argoverse_scenario_dir = Path(
        '/home/haojiachen/桌面/interaction-dataset/left_turn_data/')
all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.pkl"))
scenario_file_lists = (all_scenario_files[:2])
scenarios = []
observations_ = []
for scenario_file_list in all_scenario_files:
    scenario = pickle.load(open(scenario_file_list, 'rb'))
    scenarios += scenario['left_turn_tracks']
    observations_ += scenario['states']

observations = []
next_observations = []
actions = []
rewards = []
terminals = []
ego_v_state = []
surround1_v_state = []
surround2_v_state = []
max_ego_v = 0

for observation in observations_:
    for i in range(len(observation)):
        state = observation[i]
        ego_v = math.sqrt(state[2] ** 2 + state[3] ** 2)
        surround1_v = math.sqrt(state[9] ** 2 + state[10] ** 2)
        surround2_v = math.sqrt(state[14] ** 2 + state[15] ** 2)
        
        ego_v_state.append(ego_v)
        if ego_v >max_ego_v:
            max_ego_v = ego_v
        surround1_v_state.append(surround1_v)
        surround2_v_state.append(surround2_v)

        if i != len(observation) - 1:
            next_state = observation[i + 1]
            next_ego_v = math.sqrt(next_state[2] ** 2 + next_state[3] ** 2)
            action = min(next_ego_v - 4, 4) / 4

            reward = 0.5 * (action * 4 + 4)/20

            terminal = 0
        else:
            action = actions[-1][0]

            terminal = 1
            reward = 0.5 * (action * 4+ 4 )/20 + 10

        rewards.append(reward)
        observations.append(state)
        actions.append([action])
        terminals.append(terminal)

        # reward = get_reward(observation)
for i in range(1, len(observations)):
    next_observations.append(observations[i])
next_observations.append(observations[0])

equal_ego_v = sum(ego_v_state) / len(ego_v_state)
equal_surround1_v_state = sum(surround1_v_state) / len(surround1_v_state)
equal_surround2_v_state = sum(surround2_v_state) / len(surround2_v_state)

observations = np.array(observations, dtype='float32')
next_observations = np.array(next_observations, dtype='float32')
actions = np.array(actions, dtype='float32')
rewards = np.array(rewards, dtype='float32')
terminals = np.array(terminals, dtype='float32')

b = rewards.shape[0]

# Create expert + td3
# dataset_from_td3 = pickle.load(open( '/home/haojiachen/桌面/offline_rl/dataset_from_td3_10', 'rb'))

# rewards_td3 = np.array(dataset_from_td3['reward'])
# terminals_td3 = np.array(dataset_from_td3['done'])
# observations_td3 = np.array(np.squeeze(dataset_from_td3['state']))
# next_state_td3 = np.array(np.squeeze(dataset_from_td3['next_state']))
# actions_td3 = np.array(dataset_from_td3['action'])

# rewards_ = np.concatenate((rewards,rewards_td3))
# observations_ = np.concatenate((observations,observations_td3))
# next_observations_ = np.concatenate((next_observations,next_state_td3))
# actions_ = np.concatenate((actions,actions_td3))
# terminals_ = np.concatenate((terminals,terminals_td3))

dataset = {}
dataset['observations'] = observations
dataset['next_observations'] = next_observations
dataset['actions'] = actions
dataset['rewards'] = rewards
dataset['terminals'] = terminals

save_path = "/home/haojiachen/桌面/AWAC_for_biye/AWAC/left_turn_data_and_scen/"

pickle.dump(dataset, open(save_path + 'dataset', 'wb'))

pickle.dump(scenarios, open(save_path + 'scenarios', 'wb'))