import gymnasium as gym #gymnasium 라이브러리 : 강화학습 환경을 제공하는 라이브러리
import numpy as np 
import random

#딥러닝 모델을 만들어주는 라이브러리
from keras.layers import Dense #완전 연결 계층
from keras import Model #모델을 만들어주는 라이브러리
import tensorflow as tf

#a메모리를 저장하는 라이브러리
from collections import deque #저장 도와주는 라이브러리 : 메모리! 다양한 방식으로 저장 할 수 있는데 나는 이거 좋아함

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        #4x32x16x2
        self.d1 = Dense(64,input_dim = 4, activation='ReLU') #입력층 : 4개 - 128
        self.d2 = Dense(2, activation='linear') #출력층 : 2개 : 0,1 : 0은 왼쪽, 1은 오른쪽
        self.optimizer = tf.keras.optimizers.Adam(0.01) #옵티마이저 : Adam, 학습률 : 0.01

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return x
    
model = DQN() #모델 생성

#cartpole 환경 생성
env = gym.make('CartPole-v1') #환경 생성
memory = deque(maxlen=5000) #메모리 생성 : 최대 4000개까지 저장 가능
episode =  1000 #에피소드 500번을 이용해서 학습
step = 500
score = []
for i in range(episode):
    state = env.reset()[0] #환경 초기화
    state = state.reshape(1,4)
    eps =np.exp(-i/30) #epsilon-greedy : 그냥 다양한 감소함수 쓰시면 됩니다.

    for t in range(step):
        #epsilon-greedy : 전략적 방법으로 action을 선택
        if np.random.rand() < eps:
            action = np.random.randint(0,2)
        else:
            action = np.argmax(model.call(state))
        #다음 상태 및 보상 : 환경한테 물어본다.
        next_state, reward, done, _ , _  = env.step(action)

        if done : 
            reward = -10

        next_state = next_state.reshape(1,4)
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done or t == step-1:
            print('Episode', i, 'Score', t + 1)
            score.append(t + 1)
            break

    if i > 50 :
        minibatch = random.sample(memory,32)
        states = np.array([x[0][0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3][0] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])

        target_y = model.call(states).numpy()
        target_y[range(32), actions] = rewards + (1-dones)*0.95*np.max(model.call(next_states).numpy(), axis=1)


        #경사하강법을 통한 업데이트 - 역전파 : 100퍼센트 이해하긴 어렵다.
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))



import matplotlib.pyplot as plt

plt.plot(score)
plt.show()
