import gymnasium as gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.layers import Dense
from tensorflow import keras
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='ReLU')
        self.d2 = Dense(64, activation='ReLU')
        self.d3 = Dense(32, activation='ReLU')
        self.d4 = Dense(5, activation='linear')
        self.optimizer = keras.optimizers.Adam(0.001)

        self.M = []  # 리플레이 버퍼

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

#pendulum 환경 구성
env = gym.make('Pendulum-v1', g=10)
model = DQN()

memory = deque(maxlen=4000) #############################################

for j in range(6):
    episode = 500 #######################################################
    step = 200 #############################################################
    reward_mean_graph=[]
    last_x_position_graph=[]
    for i in range(episode):

        state = env.reset()[0] #state = [x,y,angular velocity]
        state = state.reshape(1,3) #state = 1x3
        eps = ((5-(j/3))/10)* np.exp(-i/100) ##########################################
        reward_mean = []
        last_x_position = 0

        for t in range(step):
            #e-greedy
            if np.random.rand() < eps:
                action = np.random.choice([-2,-1,0,1,2])
            else:
                action = np.argmax(model.call(state))-2
            #다음 상태 및 보상
            next_state, reward = env.step((action,))[0:2]
            if t == step-1:
                last_x_position = next_state[0]
            next_state = next_state.reshape(1,3)
            reward_mean.append(reward)

            memory.append((state, action, reward, next_state))
            state = next_state

        if i > 20:
            minibatch = random.sample(memory, 64) #############################################################
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch]) 
            next_states = np.array([x[3][0] for x in minibatch])

            actions_to_index = np.array([k+2 for k in actions])

            target_y = model.call(states).numpy()
            target_y[range(64), actions_to_index] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1) #############################################################

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        m = np.mean(reward_mean)
        reward_mean_graph.append(m)
        last_x_position_graph.append(last_x_position)
        if last_x_position > 0.999 :
            model.save_weights('pendulum_g10.h5')
        print(f'episode : {i} reward_mean: {m:.3f} eps : {eps:.3f} last_x_position : {last_x_position:.3f}')


    plt.scatter(np.arange(0,episode), reward_mean_graph, label='reward_mean_scatter')
    reward_mean_graph = np.array(reward_mean_graph)
    rmgm = reward_mean_graph.reshape(int(episode/10),10)
    rmgm = np.mean(rmgm, axis=1)
    rmgm = rmgm.reshape(1,int(episode/10))[0]
    plt.plot(np.arange(0,episode,10), rmgm, color='red', linewidth=2.0, label='reward_mean')
    plt.legend(loc='lower right')
    plt.show()

    last_x_postion_mean_graph = np.array(last_x_position_graph)
    last_x_position_mean_graph = last_x_postion_mean_graph.reshape(int(episode/10),10)
    last_x_position_mean_graph = np.mean(last_x_position_mean_graph, axis=1)
    last_x_position_mean_graph = last_x_position_mean_graph.reshape(1,int(episode/10))[0]
    plt.plot(np.arange(0,episode,10), last_x_position_mean_graph, color='green', linewidth=2.0, label='last_x_position')
    plt.legend(loc='lower right')
    plt.show()

env.close()
