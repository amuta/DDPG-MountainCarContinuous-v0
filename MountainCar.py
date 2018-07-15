import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

# Importing important things
from DDPG import DDPG
from ReplayBuffer import ReplayBuffer
from OUNoise import OUNoise
import gym
import numpy as np
import os

def preprocess_state(state):
    s = np.array(state)  # mapping the state values to [-1,1]
    s[0] = (state[0] - 0.30) / 0.9
    s[1] = (state[1] - 0.00) / 0.07
    return s

env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
action_low = -1
action_high = 1
buffer_size = 16000
batch_size = 1024
agent = DDPG(action_low, action_high, buffer_size, batch_size)
agent.actor_local.reset_weights()
agent.actor_target.reset_weights()
agent.critic_local.reset_weights()
agent.critic_target.reset_weights()
noise = OUNoise(mu=0.3)
# np.random.seed(2018)
learning_epochs = int(buffer_size)
episodes = 800
epochs = 2000
epsilon = 1
total_epochs = 0
start_act = 50
learning = True
acting = False
stuck = False
done = False
stop_train = False
final_flag = True
rewards_hist = []
last_reset = 0
max_reward = -np.inf
last_max = 0

print('Start learning phase...')
for i_episode in range(1, episodes):
    mem_len = agent.memory.__len__()
    mem_full = mem_len >= buffer_size
    if mem_len >= learning_epochs:
        learning = False
    if i_episode > start_act:
        acting = True

    
    if done and acting and i_episode > (last_reset+10):
        # epochs = max(epochs-15, 200)
        epsilon = max(epsilon-0.1, 0)
    else:
        # epochs = min(epochs+15, 3000)
        epsilon = min(epsilon+0.05, 1)
        if stuck > 30 and i_episode > (last_reset+5*buffer_size/epochs):
            print('Stuck!! Reseting learner weights and memory...')
            epsilon = 1
            agent.build_models()
            stuck = 0
            # agent.memory.reset()
            last_reset = i_episode
            agent.actor_local.reset_weights()
            agent.actor_target.reset_weights()
            agent.critic_local.reset_weights()
            agent.critic_target.reset_weights()
    # if last_max > 30 and epsilon < 0.1:
        # epsilon = 0.08
    if stop_train:
        done = False

    state = env.reset()
    state = preprocess_state(state)
    noise.reset()
    agent.reset_episode(state, learning, 1, True)
    epoch = 0
    
    rewards = []
    agent_actions = []
    noise_actions = []


    while True:
        agent_action = agent.act(state)[0]
        action = np.clip((1 - epsilon) * agent_action + epsilon * noise.sample(), -1, 1)

        if epoch > epochs:
            break

        agent_actions.append(agent_action)
        noise_actions.append(action)
        next_state, reward, done, info = env.step(action)
#         reward = done * 100 - 1
        next_state = preprocess_state(next_state)
        rewards.append(reward)
        agent.step(state, action, reward, next_state, done, learning)
        state = next_state
        
        
        total_epochs += 1
        epoch += 1
        
        if done:
            break
        
    noise.reset()
    final_reward = np.sum(rewards)
    
    num_epochs = len(rewards)
    mean_action = np.mean(agent_actions)
    mean_noise = np.mean(noise_actions)
    std_action = np.std(agent_actions)
    std_noise = np.std(noise_actions)
    rewards_hist.append(final_reward)

    if len(rewards_hist) > 30:
        mean_hist = np.mean(rewards_hist[len(rewards_hist)-20:])
        # early stop if mean of 15 episodes > 90

        if mean_hist > max_reward:
            max_reward = mean_hist
            last_max = 0
        else:
            last_max += 1

        # CHECK SUCCESS
        if final_reward > 90 and epsilon < 0.05:
            break
        if mean_hist < 60 and mem_full:
            stuck += 1
        else:
            stuck = min(stuck - 1, 0)

    else:
        mean_hist = np.mean(rewards_hist)
    
    

    if i_episode % 1 == 0:
        print('Ep:{: 4d} R:{: 4.2f} epo:{:04d}/{:04d} amu:{: 1.3f} asd:{: 1.3f} nmu:{: 1.3f} nsd:{: 1.3f}'
        ' eps:{: 1.3f} mem:{: 5d} d:{} hs:{: 4.2f} mrh: {: 4.2f}'.
              format(i_episode, final_reward, num_epochs, epochs, mean_action, std_action, 
                mean_noise, std_noise, epsilon, mem_len, done, mean_hist, max_reward))



# testing the model

render = False
input('lets take a look at the model')
tests = 100
success = 100
rewards_hist = []
for i_episode in range(tests):
    state = env.reset()
    state = preprocess_state(state)
    noise.reset()
    agent.reset_episode(state, learning, 10, False)
    epoch = 0
    rewards = []

    while True:
        if epoch > 5000:
            success -= 1
            break
        action = [np.clip(agent.act(state)[0], -1, 1)]
        next_state, reward, done, info = env.step(action)
        next_state = preprocess_state(next_state)
        rewards.append(reward)
        state = next_state
        if i_episode > 95:
            env.render()

        total_epochs += 1
        epoch += 1
        
        if done:
            break
    final_reward = np.sum(rewards)
    num_epochs = len(rewards)
    rewards_hist.append(final_reward)

    
    print('Ep:{: 4d} R:{: 4.2f} epo:{:04d}'.format(i_episode, final_reward, num_epochs))
print('MEAN REWARD:{: 4.2f}'.format(np.mean(rewards_hist)))