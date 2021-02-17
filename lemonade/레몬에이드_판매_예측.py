# -*- coding: utf-8 -*-
"""레몬에이드 판매 예측.ipynb"""

import pandas as pd

#파일들로부터 데이터 읽어 오기
lemonade_dir='https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade=pd.read_csv(lemonade_dir)

#데이터 확인
display(lemonade)

#데이터의 모양 확인
print(lemonade.shape)

#데이터 칼럼이름 확인
print(lemonade.columns)

#독립변수와 종속변수 분리
#a:독립변수 , b: 종속 변수
a=lemonade[['온도']]
b=lemonade[['판매량']]

print(a.shape,b.shape)

import tensorflow as tf
import pandas as pd

lemonade_dir='https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade=pd.read_csv(lemonade_dir)

#display(lemonade.head())

a=lemonade[['온도']]
b=lemonade[['판매량']]
print(a.shape,b.shape)

#모델 만들기
#독립변수의 컬럼 갯수가 1개이므로 1
X=tf.keras.layers.Input(shape=[1])
#종속변수의 컬럼 갯수가 1개이므로 1 
Y=tf.keras.layers.Dense(1)(X) 
model=tf.keras.models.Model(X,Y)
model.compile(loss='mse')

model.fit(a,b,epochs=3000,verbose=1)

print(model.predict(a))

model.predict([[15]])

