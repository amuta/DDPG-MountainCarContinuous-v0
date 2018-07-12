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
    s = np.array(state)
    s[0] = state[0] * 10
    s[1] = state[1] * 60
    return s

env = gym.make('MountainCarContinuous-v0')
env = env.unwrapped
action_low = -1
action_high = 1
buffer_size = 25000
batch_size = 1024
agent = DDPG(action_low, action_high, buffer_size, batch_size)
noise = OUNoise()
# np.random.seed(2018)
learning_epochs = int(buffer_size)
episodes = 1500
epochs = 1500
epsilon = 1
total_epochs = 0
start_act = 30
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

    
    if done and acting:
        # epochs = max(epochs-15, 200)
        epsilon = max(epsilon-0.05, 0)
    else:
        # epochs = min(epochs+15, 3000)
        epsilon = min(epsilon+0.05, 1)
        if stuck > 20 and i_episode > (last_reset+5*buffer_size/epochs):
            print('Stuck!! Reseting learner weights and memory...')
            epsilon = 1
            agent.build_models()
            stuck = 0
            agent.memory.reset()
            last_reset = i_episode
            # agent.actor_local.normalize(0.1)
            # agent.actor_target.normalize(0.05)
            # agent.critic_local.normalize(0.1)
            # agent.critic_target.normalize(0.05)
    if last_max > 30 and epsilon < 0.1:
        epsilon = min(epsilon+0.01, 1)
    if stop_train:
        done = False

    state = env.reset()
    state = preprocess_state(state)
    

    noise.reset()
    agent.reset_episode(state, learning, 1, True)
    epoch = 0
    
    rewards = []
    agent_actions = []
#     for epoch in range(epochs):
    while True:
        
        agent_action = agent.act(state)[0]
        agent_actions.append(agent_action)
        # if acting:
        action = np.clip((1 - epsilon) * agent_action + epsilon * noise.sample(), -1, 1)
        # else:
            # action = noise.sample()
        if epoch > epochs:
            # remove bad episode from memory
            # for pops in range(epoch):
            #     agent.memory.memory.pop()
            break
#             action = agent_action + noise.sample() * max(epsilon,0.01)
        
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
    std_action = np.std(agent_actions)
    rewards_hist.append(final_reward)

    if len(rewards_hist) > 30:
        mean_hist = np.mean(rewards_hist[len(rewards_hist)-20:])
        # early stop if mean of 15 episodes > 90

        if mean_hist > max_reward:
            max_reward = mean_hist
            last_max = 0
        else:
            last_max += 1

        if mean_hist > 90:
            stop_train = True
            if np.mean(rewards_hist[len(rewards_hist)-30:]) > 90:
                break
        if mean_hist < 83:
            stop_train = False
        if mean_hist < 60 and mem_full:
            stuck += 1

        else:
            stuck = min(stuck - 1, 0)

    else:
        mean_hist = np.mean(rewards_hist)
    
    

    if i_episode % 1 == 0:
        print("Ep:{:3d} R:{:4.2f} epo:{:4d}/{:4d} mu:{:1.3f} sd:{:1.3f} eps:{:1.3f} mem:{:5d} d:{} hs:{:4.2f} mrh: {:4.2f}".
              format(i_episode, final_reward, num_epochs, epochs, mean_action,
               std_action, epsilon, mem_len, done, mean_hist, max_reward))


# testing the model
input('lets take a look at the model')
tests = 10
for i_episode in range(tests):

    state = env.reset()
    state = preprocess_state(state)

    noise.reset()
    agent.reset_episode(state, learning, 10, done)
    epoch = 0
    rewards = []
    agent_actions = []
    #     for epoch in range(epochs):
    while True:
        
        agent_action = agent.act(state)[0]
        agent_actions.append(agent_action)
        # if acting:
        action = [np.clip(agent_action, -1, 1)]
        # else:
            # action = noise.sample()
        if epoch > 3000:
            break
        
        next_state, reward, done, info = env.step(action)
    #         reward = done * 100 - 1
        next_state = preprocess_state(next_state)
        rewards.append(reward)
        agent.step(state, action, reward, next_state, done, learning)
        state = next_state
        env.render()
        
        
        total_epochs += 1
        epoch += 1
        
        if done:
            break
    final_reward = np.sum(rewards)
    num_epochs = len(rewards)
    mean_action = np.mean(agent_actions)
    std_action = np.std(agent_actions)
    
    print("Ep:{:3d} R:{:4.2f} epo:{:4d}/{:4d} mu:{:1.3f} sd:{:1.3f} eps:{:1.3f} d:{} ".format(i_episode, 
        final_reward, epoch, epochs, mean_action, std_action, epsilon, done))