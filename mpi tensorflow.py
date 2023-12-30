


import tensorflow as tf


from typing_extensions import Required
from mpi4py import MPI

# MPI 초기화
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import os

# 1. MNIST 데이터셋 로드 및 전처리
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# 2. 데이터 분산
x_train_split = tf.split(x_train, size, axis=0)
y_train_split = tf.split(y_train, size, axis=0)

# 3. 분산 전략 정의
strategy = tf.distribute.MirroredStrategy()

# 4. 분산된 데이터셋 생성
train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split[rank], y_train_split[rank])).shuffle(60000).batch(64)

# 5. 모델 구성
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # 6. 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 7. 모델 훈련
model.fit(train_dataset, epochs=5)

# 8. 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print('테스트 정확도:', test_acc)

# 9. 특정 노드에서만 실행되는 코드
if rank == 0:
    # 모델의 가중치 저장
    model.save_weights('model_weights.h5')
    print("Model weights saved by rank 0.")
