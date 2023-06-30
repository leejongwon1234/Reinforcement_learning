import gym
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# 뉴럴 네트워크 모델 만들기
class DQN(Model):
    def __init__(self):
        super(DQN, self).__init__()
        self.d1 = Dense(128, input_dim=3, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(5, activation='linear')

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

# pendulum 환경 구성
env = gym.make('Pendulum-v1', g=10, render_mode='human')

model = DQN()
model(np.zeros((1, 3)))  # 모델 호출하여 변수 생성

# 저장된 모델 로드
model.load_weights('pendulum_g10.h5')

episode = 10  # 테스트할 에피소드 수
step = 200

for i in range(episode):
    state = env.reset()[0]
    state = state.reshape(1, 3)
    reward_mean = []
    last_x_position = 0

    for t in range(step):
        # 테스트 시에는 epsilon-greedy 방법을 사용하지 않습니다.
        # 최적의 액션을 선택하기 위해 argmax 함수를 사용합니다.
        action = np.argmax(model(state)) - 2

        # 다음 상태 및 보상
        next_state, reward = env.step((action,))[0:2]
        next_state = next_state.reshape(1, 3)
        if t == step - 1:
            last_x_position = next_state[0][0]
        reward_mean.append(reward)

        state = next_state

    print(f'Test Episode: {i} Avg. Reward: {np.mean(reward_mean):.3f} Last x position: {last_x_position:.3f}')

env.close()


