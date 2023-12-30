from os import startfile
import tensorflow as tf
import horovod.tensorflow as hvd
import time

start=time.time()

# Horovod 초기화
hvd.init()

# Horovod: CPU 사용
tf.config.experimental.set_visible_devices([], 'GPU')

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
x_train, x_test = x_train/255.0, x_test/255.0

# 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Horovod: CPU 사용 시 GPU 수에 따라 학습률 조정
optimizer = tf.keras.optimizers.Adam(0.001 * hvd.size())

# Horovod로 옵티마이저 래핑
optimizer = hvd.DistributedOptimizer(optimizer)

# 모델 컴파일
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Horovod: 초기 변수 상태를 랭크 0에서 모든 다른 프로세스로 브로드캐스트
# 이는 훈련이 랜덤 가중치로 시작되거나 체크포인트에서 복원될 때 모든 작업자의 일관된 초기화를 보장하기 위해 필요합니다.
if hvd.rank() == 0:
    model.fit(x_train, y_train, epochs=1)

# Horovod: 스텝 수 조정
model.fit(x_train, y_train, epochs=5 // hvd.size(), steps_per_epoch=100)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)

# 랭크 0 프로세스에서만 출력
if hvd.rank() == 0:
    print('테스트 정확도:', test_acc)

print('걸린 시간:',time.time()-start)