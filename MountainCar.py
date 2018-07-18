import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Importing important things
from DDPG import DDPG
from ReplayBuffer import ReplayBuffer
from OUNoise import OUNoise
import gym
import numpy as np
import msvcrt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint


env_raw = gym.make('MountainCarContinuous-v0')
env = env_raw.unwrapped
agent = DDPG(env)

max_epochs = 100
max_steps = 1000
solved = False
rewards_hist = []
test_hist = []

def preprocess_state(state):
    s = np.array(state)  # mapping the state values to [-1,1]
    s[0] = ((state[0] + 1.2) / 1.8)*2-1
    s[1] = ((state[1] + 0.07) / 0.14)*2-1
    return s


def plot_Q(agent, num):
    state_step = 0.2
    action_step = 0.2
    plot_range = np.arange(-1,1+state_step,state_step)
    action_range = np.arange(-1,1+action_step,action_step)
    shape = plot_range.shape[0]
    matrix_Q = np.ones((shape,shape))
    matrix_mQ = np.ones((shape,shape))
    matrix_sQ = np.ones((shape,shape))
    matrix_A = np.ones((shape,shape))
    for i in range(shape):
        for j in range(shape):
            pos = plot_range[j]
            vel = plot_range[i]
            state = np.array([pos,vel]).reshape(-1,2)
            # print(state)
            best_Q = -np.inf
            Q_list = []
            for a in action_range:
                action = np.array(a).reshape(-1,1)
                Q_list.append(agent.critic_local.model.predict([state, action]))
            matrix_Q[i][j] = np.max(Q_list)
            matrix_sQ[i][j] = np.std(Q_list)
            matrix_mQ[i][j] = action_range[np.argmax(Q_list)]
            # prefered action
            matrix_A[i][j] = agent.actor_local.model.predict(state)
    extent = [plot_range[0], plot_range[-1], plot_range[0], plot_range[-1]]

    fig, ax = plt.subplots(2,2, sharex=True)
    ax[0, 0].set_title('Q value max' + str(num))
    ax[0, 0].set_ylabel('Velocity')
    ax[0, 0].set_xlabel('Position')
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax[0, 0].imshow(matrix_Q, extent=extent, origin='lower')
    plt.colorbar(im, cax=cax)


    ax[0, 1].set_title('Q value std')
    ax[0, 1].set_ylabel('Velocity')
    ax[0, 1].set_xlabel('Position')
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax[0, 1].imshow(matrix_sQ, extent=extent, origin='lower')
    plt.colorbar(im, cax=cax)

    ax[1, 0].set_title('Action with Q max')
    ax[1, 0].set_ylabel('Velocity')
    ax[1, 0].set_xlabel('Position')
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax[1, 0].imshow(matrix_mQ, extent=extent, origin='lower')
    plt.colorbar(im, cax=cax)

    ax[1, 1].set_title('Predicted Action')
    ax[1, 1].set_ylabel('Velocity')
    ax[1, 1].set_xlabel('Position')
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax[1, 1].imshow(matrix_A, extent=extent, origin='lower')
    plt.colorbar(im, cax=cax)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                        wspace=0.4)

    plt.show()

for epoch in range(1, max_epochs):
    steps = 0
    state = preprocess_state(agent.reset_episode())
    rewards_list = []
    actions_list = []

    while True:
        
        action, pure_action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = preprocess_state(next_state)
        agent.step(action, reward, next_state, done)
        state = next_state
        rewards_list.append(reward)
        actions_list.append(pure_action)


        # if epoch > 30:
        #     env.render()

        if steps > max_steps or done:
            break
        steps += 1

    test = True
    p_done = done

    final_test_r = 0
    if test :
        test_rewards = []
        test_steps = 0
        state = agent.reset_episode()
        while True:
            _, pure_action = agent.act(preprocess_state(state))
            state, reward, done, info = env.step(pure_action)
            test_rewards.append(reward)
            # if epoch > max_epochs - 1:
                # env.render()
            if test_steps > max_steps or done:
                break
            test_steps += 1
        final_test_r = np.sum(test_rewards)

    final_reward = np.sum(rewards_list)
    mean_action = np.mean(actions_list)
    std_action = np.std(actions_list)
    rewards_hist.append(final_reward)    
    test_hist.append(final_test_r)  

    if epoch > 50:
        mean_rewards = np.mean(rewards_hist[epoch-50:])
        mean_test = np.mean(test_hist[epoch-50:])

        if mean_test > 90 and np.min(mean_test) > 80:
            print('Solved!')
            solved = True
    else:
        mean_rewards = np.mean(rewards_hist)
        mean_test = np.mean(test_hist)

    print('Epo:{:4d} Ste:{:5d} Don:{:1d} Rew:{: 5.1f} Hst:{: 5.1f} '
        'Act:{: .3f}/{: .3f} Tst:{: 5.1f} Don:{:1d} THst:{: 5.1f}' .format(epoch,
                                                   steps,
                                                   int(p_done),
                                                   final_reward,
                                                   mean_rewards,
                                                   mean_action,
                                                   std_action,
                                                   final_test_r,
                                                   int(done),
                                                   mean_test))
    

    plot = True
    if epoch == max_epochs-1 or solved:
        env.close()
        random_num = np.random.randint(0,1000)
        print(random_num)
        plot_Q(agent, random_num)
        break

pprint(vars(agent))