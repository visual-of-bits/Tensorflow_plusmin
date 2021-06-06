import tensorflow as tf
import numpy as np
import random

#데이터를 넣는 변수
question=np.zeros((6000,3))
answer=np.zeros((6000))

#수의 크기의 스케일
scale=100

# +-문제의 문제를 준비(6000개)
def preparedata():
    for i in range(0,6000):
        #+-를 렌덤으로 정하기
        question[i][2]=int(random.randrange(1,3))
        #각 항의 수를 렌덤으로 정한다
        for j in range(0,2):
            question[i][j]=int(random.randrange(-scale,scale))/scale
    #+인지 -인지를 보고 정답을 연수에 넣는다
    for i in range(0,6000):
        if(question[i][2]==1):
            answer[i]=question[i][0]+question[i][1]
        elif(question[i][2]==2):
            answer[i]=question[i][0]-question[i][1]


#데이터의 분비
preparedata()

#모델 정의하기
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(3),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(256),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(512),
  tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam',
              loss='MeanSquaredError',
              metrics=['MeanSquaredError'])

#학습을 20번 반복하기
model.fit(question, answer, epochs=20)


network_answer=model.predict(question)


#답 출력해보기
print("네트워크의 문제에 대한 출력")
for i in range(0,100):
        if(question[i][2]==1):
            strc="+"
        elif(question[i][2]==2):
            strc="-"
        #문제의 출력
        print(str(i)+"번 문제: "+str(question[i][0]*scale)+" "+strc+" "+str(question[i][1]*scale)+"= ?")
        print("네트워크가 생각해낸 답: "+str(int(network_answer[i][0]*scale)))
        print("진짜 정답: "+str(int(answer[i]*scale)))
        print()