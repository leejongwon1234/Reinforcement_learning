import gym
import numpy as np
import random
from collections import deque #저장 도와주는 라이브러리
from keras.layers import Dense
import tensorflow as tf
from keras import Model

class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(32,activation ='tanh', input_dim = 2)
        self.d3 = Dense(16,activation ='tanh')
        self.d4 = Dense(3,activation ='linear')
        self.optimizer = tf.keras.optimizers.Adam(0.001)


    def call(self, x):
        x = self.d1(x)
        x = self.d3(x)
        x = self.d4(x)

        return x
    
model = DQN()

memory_b = deque(maxlen=5000)
# MountainCar 환경 구성
env = gym.make("MountainCar-v0")
#scores
scores = []

for i in range(8000): #에피소드 스텝
  state = env.reset()[0] #env.reset()은 (state,{})를 반환 #state는 크기가 2인 1차원 배열
  state = state.reshape(1, 2) #state를 1x2인 2차원배열로 변환

  # e-greedy 점점 줄어들게 설정
  eps = np.exp(-i/2000)
  score = 0
  total_reward=0

  for t in range(200): # 타임 스텝
    # Inference: e-greedy
    if np.random.rand() < eps:
        action = np.random.randint(0, 3)
    else:
        action = np.argmax(np.array(model.call(state))[0])
    
    #다음 상태 및 보상
    next_state, reward, done = env.step(action)[0:3]
    next_state = next_state.reshape(1, 2)

    if done :
        reward = 1000
    elif next_state[0][0] >= 0.4:
        reward = 128
    elif next_state[0][0] >= 0.3:
        reward = 64
    elif next_state[0][0] >= 0.2:
        reward = 32
    elif next_state[0][0] >= 0.1:
        reward = 16
    elif next_state[0][0] >= 0.05:
        reward = 8
    elif next_state[0][0] >= 0:
        reward = 4
    elif next_state[0][0] >= -0.1:
        reward = 2
    elif next_state[0][0] >= -0.2:
        reward = 1
    elif next_state[0][0] >= -0.3:
        reward = 0
    else:
        reward = -1


    memory_b.append((state, action, reward, next_state, done))
    state = next_state
    total_reward += reward
    score -= 1

    if done :
        print('Episode', i,'total reward ',total_reward,'Score', score,'Step', t)
        scores.append(score)
        break
    elif t == 199:
        print('Episode', i,'total reward ',total_reward,'Score', score)
        scores.append(score)
        break

  if i > 64:
        minibatch = random.sample(memory_b, 128)

        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch]) 
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        target_y = model.call(states).numpy()
        target_y[range(128), actions] = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1)

        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#학습 결과
import matplotlib.pyplot as plt
plt.scatter(range(len(scores)),scores)
plt.show()