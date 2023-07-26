#강화학습 - moving cartpole
----------------------------------------------------
*목차
> ### 1. 사고과정과 아이디어

> ### 2. 1번 : model 2개를 통한 moving_cartpole 구현

> ### 3. 2번 : 특정 state에 큰 보상을 주어 moving_cartpole 구현

> ### 4. 3번 : 특정 position 범위에 대해 보상을 주어 moving_cartpole 구현

> ### 5. 4번 : model은 모든 경우에 대해 쓰러지지 않도록 학습하고, 환경을 조작하여 moving_cartpole 구현

> ### 6. 5번 : state의 각 변수마다 함수를 만들어 곱하는 reward를 이용하여 moving_cartpole 구현

> ### 7. 시행착오, 해결 방법, 경험 공유

#
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
<pre>
</pre>
 - 구현 코드
```python
```
- 테스트
```python
```
- 테스트 결과
<pre>
</pre>
## 3. g=5
<pre>
</pre>
 - 구현 코드
```python
```

<pre>
</pre>

## 5.시행착오 및 해결 방법, 피드백
<pre>
</pre>
### 1. 보상을 어떻게 처리 할 것인가?
<pre>
</pre>
>>이유는 알 수 없지만, reward를 임의로 조정하는 것 보다 환경에서 주어진 reward를 사용하는 것이 더 학습을 잘했다.

 - reward가 0.01보다 클때 reward+=1을 해주는 구조(0.01보다 큰 부분에서 불연속)로 변경해보았지만, 잘 학습하지 못했다.
 
 - np.square(reward)*(-1)을 이용하여 reward의 절댓값이 클수록 더 큰 penalty를 주는 구조로 변경한 경우 학습을 잘 했지만, 수렴하는 데에 더 많은 episode를 필요로 했고, reward 구조를 변경할 이유가 없었다.

 - np.sqrt(reward)*(-2)를 이용하여 reward의 절댓값이 작을수록 급격하게 reward가 커지는 방향으로 변경하는 경우 학습을 잘 했지만, 수렴하는 데에 더 많은 episode를 필요로 했고, reward구조를 변경할 이유가 없었다.
<pre>
</pre>
### 2.가중치를 언제 저장할 것인가?
<pre>
</pre>
>>last_x_postion이 0.999보다 큰 경우 성공이라고 생각하여 이 때의 가중치를 저장하고, 업데이트 하는 방향으로 구현하여 overfitting이나 원인불명의 학습실패를 방지하려 했지만, 결과론적으로 수렴이 되기 때문에 학습이 마무리 된 뒤, 가중치를 저장해도 문제가 없었을 것이다.
<pre>
</pre>
### 3.저장한 가중치를 어떻게 로드 하여 테스트 하는가?
<pre>
</pre>
>>단순히 모델을 구현하여 가중치를 로드 하니 계속 오류가 발생하여 고생했다. 이리저리 찾아보다가 모델 전부를 구현하고, 모델을 호출하여 변수를 생성한 뒤에 로드를 해야 문제없이 로드가 되었다.
```python
model = DQN()
model(np.zeros((1, 3)))  ### 모델 호출하여 변수 생성

# 저장된 모델 로드
model.load_weights('pendulum_g10.h5')
```
<pre>
</pre>
### 4.적절한 학습을 위해 코드를 어떤 구조로 구성해야할까?
<pre>
</pre>
>>g=1인 경우, 스윙이 필요없이 한쪽방향으로 힘을 가하여 막대를 세울 수 있기에 큰 고민이 필요없이 성공할 수 있지만, g가 5보다 커지는 경우 특정 상황에서 스윙이 필요하기 때문에 단순히 epsilon값이 줄어드는 방향으로 episode만 늘려서는 학습이 잘 되지 않았다. 

>>이를 위해 episode500인 학습을 3번 시도하는 방향으로 구조를 작성했고, 시도가 바뀔 때 마다 epsilon값을 reset(약간의 변형은 존재)하여 최적의 결과를 탐험하게 시도시켜 성공시켰다.
<pre>
</pre>
