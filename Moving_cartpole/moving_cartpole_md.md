 # 강화학습 - moving cartpole
----------------------------------------------------

> ### 1. 사고과정과 아이디어
> ### 2. 1번 : model 2개를 통한 moving_cartpole 구현
> ### 3. 2번 : 특정 state에 큰 보상을 주어 moving_cartpole 구현
> ### 4. 3번 : 특정 position 범위에 대해 보상을 주어 moving_cartpole 구현
> ### 5. 4번 : model은 모든 경우에 대해 쓰러지지 않도록 학습하고, 환경을 조작하여 moving_cartpole 구현
> ### 6. 5번 : state의 각 변수마다 함수를 만들어 곱하는 reward를 이용하여 moving_cartpole 구현
> ### 7. 시행착오, 해결 방법, 경험 공유
#
## 1. 사고과정과 아이디어
#

 - 사고과정 :
   -
   - agent는 0(왼쪽으로 힘을 가함)과 1(오른쪽으로 힘을 가함)중 action을 선택한다. : DQN을 이용하여 더 큰 Q_VALUE를 가진 ACTION을 취한다.
   - Environment는 agent의 action을 받아 next_state, reward, done 값을 반환한다.
   - episode가 끝날 때 마다 학습한다. : 미니배치학습
   - state는 4가지 변수를 가진다.  [position,velocity,angle,angular_velocity]
   - reward는 1(쓰러지지 않거나 벗어나지 않음) or -1(쓰러지거나 벗어남)이다.

   ### 결론 : reward 구조를 변형시켜 막대가 쓰러지지 않을 뿐 아니라 좌우로 이동하도록 학습시켜야 한다.
#
 - 학습 아이디어 :
   - 
   - episode가 시작할 때의 초깃값을 변형시킨다. (짝수번째 episode는 왼쪽에서 시작, 홀수번째 episode는 오른쪽에서 시작) = 더 빠른 속도로 학습할 수 있다.
   - 학습시킬 메모리 버퍼를 두개로 만들어서, 짝수번째 episode에서 발생한 정보와 홀수번째 episode에서 발생한 정보를 따로 저장하여 학습시킬때 각 메모리에서 절반을 뽑아 학습시킨다.

   >저장하는 경우
   ```python
            if i%2 == 0:
                memory1.append((state, action, reward, next_state, done))
            else:
                memory2.append((state, action, reward, next_state, done))
   ```
   >학습하는 경우
   ```python
        if i > 80:
            if i%2 == 0:
                minibatch = random.sample(memory1, minibatch_size)
            else:
                minibatch = random.sample(memory2, minibatch_size)
   ```
   - 좌우로 이동할 때 마다 score를 측정하여 score>=6이상이고, 이전까지의 score들 중 max score와 같거나 크면 가중치를 저장한다. = episode가 진행되는 도중에 목표에 도달 했지만, episode가 끝났을때 오히려 목적에 맞게 학습되지 않는 경우를 방지함.
   ```python
        if score >= max_score:
            max_score = score
            #가중치 저장
            if score >= 6:
                model.save_weights('moving_cartpole_idea5_3.h5')
                print(' model saved')
        print(' max_score : {}'.format(max_score))
    ```
#

 - 구현 아이디어 :
   -
   1. model을 2개 이용한다. :  model A 는 어떠한 경우가 있더라도 오른쪽으로, model B는 어떠한 경우가 있더라도 왼쪽으로 이동하도록 학습한 후, inference에서 abs(position)이 0.5에 도달한 경우에 model을 교체하는 방식으로 구현한다.

   2. abs(position)<0.1인 경우에 velocity와 angle이 같은 방향인 특정 state인 경우 큰 reward를 부여하여 그 모양을 동작하려면 왕복운동을 할 수 밖에 없도록 구현한다.

   3. 특정 position 범위까지 velocity와 angle이 같은 방향인 state인 경우에 velocity*angle에 비례하는 보상을 부여하여 더 큰 보상을 받기위해 더 빠른 속도,더 큰 각도로 이동해야하는 상황을 설정하여 왕복운동을 구현한다.

   4. model은 어떠한 상황에도 완벽히 서 있는 경우만 학습하고, inference단계에서 velocity나 angular_velocity를 조작하여 왕복운동을 하게 만든다.

   5. position,velocity,angle,angular_velocity를 이용한 함수를 만들어 곱하는 보상 구조를 구현한다.
#
## 2. 1번 : model 2개를 통한 moving_cartpole 구현
#
```진짜 무조건 성공 시킬 수 있을 것 같은데 시간이 없어서 시도도 못해봤습니다. ```
#
## 3. 2번 : 특정 state에 큰 보상을 주어 moving_cartpole 구현
#
 - 보상 코드
```python
#보상 함수
def reward_function(next_state): #state 1x4
    position = next_state[0][0]
    velocity = next_state[0][1]
    angle = next_state[0][2]
    angular_velocity = next_state[0][3]

    if abs(position) <0.2 and 0.08<abs(angle)<0.15 and abs(angular_velocity)<2:
        return 1000
    elif abs(position) <0.4 and 0.08<abs(angle)<0.15 and abs(angular_velocity)<3:
        return 500
    else:
        return 0.5
```
- 결과

```실패 : 특정 state를 찾지 못하고 쓰러지지 않는 방향으로 학습한다.```
![idea2_result](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/c02cbd46-4fad-4839-a4b9-f3ad14dc5bc8)
```세부 조건을 조금 변경시키면 성공할 것 같지만.. 시간이 없었다.```
#
## 4. 3번 : 특정 position 범위에 대해 보상을 주어 moving_cartpole 구현
#
 - 보상 코드
```python
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
```
 - inference 결과
#
![idea3_result](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/2065b51b-4bdd-4497-8cdd-7381d2afa830)
# 
## 5. 4번 : model은 모든 경우에 대해 쓰러지지 않도록 학습하고, 환경을 조작하여 moving_cartpole 구현
#
 - 코드
```python
    #초기state 설정
    env.reset()
    env.state[1] = np.random.uniform(-1, 1)
    env.state[2] = np.random.uniform(-0.08, 0.08)
    env.state[3] = np.random.uniform(-0.5, 0.5)
    state = env.state
    #많은 초깃값에 대해 흔들림 없이 쓰러지지 않는 경우를 학습한다.
```
- 결과

```실패 : render_mode = 'human' 상태에서는 episode가 진행되는 동안 state값을 변경할 수 없다.``` 
#
## 6. 5번 : state의 각 변수마다 함수를 만들어 곱하는 reward를 이용하여 moving_cartpole 구현
#
 - 보상 코드
```python
#보상 함수
def position_function(position):
    if abs(position) < 2:
        return 1
    else:
        return (3-abs(position))
    
def velocity_angle_function(position,velocity,angle):
    if position<1 and velocity>0 and angle>0.05:
        return velocity*angle*100
    elif position>-1 and velocity<0 and angle<-0.05:
        return velocity*angle*100
    else:
        return 0.3
    
def angular_velocity_function(angular_velocity):
    if abs(angular_velocity) < 0.5:
        return 5
    elif abs(angular_velocity) < 1:
        return 3
    elif abs(angular_velocity) < 1.5:
        return 2
    else:
        return 1


def reward_function(next_state): #state 1x4
    position = next_state[0][0]
    velocity = next_state[0][1]
    angle = next_state[0][2]
    angular_velocity = next_state[0][3]

    return position_function(position)*velocity_angle_function(position,velocity,angle)*angular_velocity_function(angular_velocity)
```
- 결과
![idea5_result](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/d8d4f122-6947-4b04-97fa-15bae586300e)
#

---------------
### 번외 : 처음 짠 코드로 성공시킨 가중치를 저장해 뒀는데 그 코드위에 보상을 자꾸 바꾸다 보니 처음 짠 코드가 사라졌습니다. 그리고 어떻게 했는지 기억이 안나서 inference만 올립니다.
![idea000_result](https://github.com/leejongwon1234/Reinforcement_learning/assets/123047859/2568fa64-defa-414a-a2c5-857df67f9bd1)

#
## 7. 시행착오, 해결 방법, 피드백, 경험 공유 등등
#
<pre>
1. 학습하는 과정에서 성공하는 경우가 많고, 성공 이후 목표하는 방향으로 학습하지 않는 경우가 많았기 때문에, 성공했을 때의 가중치를 저장하는 것이 중요했던 것 같습니다.


2. 다른 문제들과는 다르게 episode의 양이 충분해야 했던 것 같습니다. 특히, 코드를 잘 짰어도 어떤 경우에는 성공하고 어떤 경우에는 실패하기도 하므로 자신의 코드에 성공할 거라는 믿음이 있을때 충분한 episode를 여러번 시도해봐야 했던 것 같습니다.


3. reward를 부여할 때 음수(penalty)와 양수(reward)를 섞는 것은 좋지 않은 듯 합니다.

예를 들어, 
 1. 쓰러진 경우에는 penalty-1을 부여하고 다른 조건인 경우에 reward(+)를 주었다. 
 2. 비슷한 상태 A와 a가 있다.
 3. 상태 A에서 왼쪽으로 힘을 주었더니 막대가 쓰러졌다.
라고 가정해봅시다.

이 data를 학습하게 되면 target은 state A일 때 [-1,0.44]가 될 것이고, 이에 가깝게 역전파를 통해 가중치가 학습될 것입니다. 그렇다면 행렬 가중치 연산으로 인해 state a인 경우에 [-0.11, 0.33] 정도의 Q 값을 가지게 될것이고, 오른쪽으로 힘을 주어야 하는 action을 취하게 될 것입니다. 
즉, 비슷한 state라면 reward 구조를 어떻게 구성하는지는 중요치 않고, 쓰러지지 않는 방향으로의 action만 취하게 될것입니다.
이렇다 보니 아마 reward에 양수와 음수를 섞어서 부여하게 되면 cart가 막대를 세우는 데에만 학습하는 양상을 띄는 것 같습니다.
이는 이번 moving_cartpole의 목표를 방해하는 요소였습니다.


4. 경험상 state 변수중 position,velocity 보다 angular_velocity값이 매우 핵심적인 요소였습니다.
angular_velocity는 막대가 움직이는 각속도이므로 절댓값이 특정 값보단 작은 것이 moving_cartpole의 안정성에 중요한 요소였습니다. 따라서 보상을 부여할 때 agent가 적절한 angular_velocity를 찾을 수 있도록 넓은 범위를 부여하되, 절댓값이 특정값보다 커지면 보상을 매우 작게 주어야 학습이 잘 이루어 졌습니다.


5.moving_cartpole은 결국 카트가 움직여야 하므로 가만히 막대를 세우는 행위는 잘못된 행위임을 인지해야 했습니다. 따라서 보상을 줄때 곱셈을 잘 이용하는 방향이 효과적이었습니다. 

예를 들어, 속도와 각도가 클수록 더 큰 보상을 주고 싶어서 덧셈을 이용했다고 가정해봅시다.
reward = position_function + velocity_function + angle_function
이 경우 속도와 각도가 클수록 더 큰 보상을 받는 건 맞지만, 만약 속도와 각도를 0에 가깝게 만들어도 position_fucntion값을 받게 될것입니다. 즉, 탐험을 충분히 많이 하여 더 큰 보상을 받는 경우를 찾거나 예측할 수 있도록 해야 학습이 될 것입니다.

하지만, 속도와 각도가 클수록 더 큰 보상을 주고 싶어서 곱셉을 이용했다고 가정해봅시다.
reward = position_function*velocity_function*angle_function
이 경우 속도와 각도가 클수록 더 큰 보상을 받을 뿐 아니라, 속도와 각도 중 어느 하나라도 0에 가까워지면 reward는 급격히 낮아집니다. 즉, 가만히 있는 행위는 넘어지는 것과 비슷할 정도로 판단 되게 되는 거죠. 이는 moving_cartpole 문제 해결에 큰 도움이 되었습니다. 
