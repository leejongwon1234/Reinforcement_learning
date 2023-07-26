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
        self.d1 = Dense(1024, input_dim=4, activation='ReLU')
        self.d2 = Dense(64, activation='ReLU')
        self.d3 = Dense(32, activation='ReLU')
        self.d4 = Dense(2, activation='linear')
        self.optimizer = keras.optimizers.Adam(0.001)

        self.M = []  # 리플레이 버퍼

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x
    
#cartpole 환경 구성
env = gym.make( 'CartPole-v1')
model = DQN()

#수치 설정
episode = 1000
step = 2000
memory_length = 5000
minibatch_size = 256
trys = 10

#리플레이 버퍼
memory = deque(maxlen=memory_length)

#보상 함수
def reward_function(next_state): #state 1x4
    position = next_state[0][0]
    velocity = next_state[0][1]
    angle = next_state[0][2]
    angular_velocity = next_state[0][3]

    if position <0.5 and velocity>0 and angle>0.05:
        if abs(angular_velocity)<0.5:
            return velocity*angle*200
        elif abs(angular_velocity)<1.5:
            return velocity*angle*100
        else:
            return velocity*angle*50
    elif position >-0.5 and velocity<0 and angle<-0.05:
        if abs(angular_velocity)<0.5:
            return velocity*angle*200
        elif abs(angular_velocity)<1.5:
            return velocity*angle*100
        else:
            return velocity*angle*50
    else:
        return 0.05

#학습
for k in range(trys):
    for i in range(episode):
        #시작 포지션을 -1로 설정
        state = env.reset()[0] #env.reset()은 (state,{})를 반환 #[position, velocity, angle, angular velocity]
        if i%2 == 0:
            env.state[0] -= 1
            env.state[1] += 0.5
            state[0] -= 1
            state[1] += 0.5
            cart_goal = 'right'
        else:
            env.state[0] += 1
            env.state[1] -= 0.5
            state[0] += 1
            state[1] -= 0.5
            cart_goal = 'left'
        state = state.reshape(1,4) #state = 1x4
        eps = (5/10)* np.exp(-i/200) #e-greedy 점점 줄어들게 설정
        total_reward = 0
        score = 0

        for t in range(step):
            #e-greedy
            if np.random.rand() < eps:
                action = np.random.randint(0,2)
            else:
                action = np.argmax(model.call(state))
            #다음 상태 및 보상
            next_state, reward, done = env.step(action)[0:3]

            if cart_goal == 'left' and next_state[0] < -0.3:
                score += 1
                cart_goal = 'right'

            if cart_goal == 'right' and next_state[0] > 0.3:
                score += 1
                cart_goal = 'left'

            next_state = next_state.reshape(1,4)
            #reward structure
            if done:
                reward = 0
            else:
                reward = reward*reward_function(next_state)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done or t == step-1:
                print('Episode {}, total_reward : {:.3f}, time : {}, score : {}'.format(i, total_reward, t+1, score))
                break
        
        if i > 50:
            minibatch = random.sample(memory, minibatch_size)
            states = np.array([x[0][0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch]) 
            next_states = np.array([x[3][0] for x in minibatch])
            dones = np.array([x[4] for x in minibatch])

            targets = rewards + 0.95 * np.max(model.call(next_states).numpy(), axis=1) * (1-dones)
            target_y = model.call(states).numpy()
            target_y[range(minibatch_size), actions] = targets

            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(target_y - model.call(states)))
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if (i%100==0 or i%100==1) and i>600:
            env = gym.make('CartPole-v1', render_mode="human")
        else:
            env = gym.make('CartPole-v1')
        
        if score >= 5:
            #가중치 저장
            model.save_weights('moving_cartpole_idea3.h5')


#환경 close
env.close()