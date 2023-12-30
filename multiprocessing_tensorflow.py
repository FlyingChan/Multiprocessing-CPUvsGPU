import tensorflow as tf
from multiprocessing import Process, Queue
import time

start=time.time()

def train_model(x_train, y_train, queue):
    # 3. 모델 구성
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # 4. 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 5. 모델 훈련
    model.fit(x_train, y_train, epochs=5)

    # 훈련이 끝나면 테스트 정확도를 큐에 넣음
    test_loss, test_acc = model.evaluate(x_test, y_test)
    queue.put(test_acc)

if __name__ == "__main__":
    # 1. MNIST 데이터셋 임포트
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 2. 데이터 전처리
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 프로세스 간 통신을 위한 큐
    test_acc_queue = Queue()

    # 프로세스 리스트
    processes = []

    # 프로세스 생성 및 시작
    for i in range(4):  # 4개의 프로세스로 나눔
        start = i * len(x_train) // 4
        end = (i + 1) * len(x_train) // 4
        process = Process(target=train_model, args=(x_train[start:end], y_train[start:end], test_acc_queue))
        processes.append(process)
        process.start()

    # 모든 프로세스 완료 대기
    for process in processes:
        process.join()

    # 각 프로세스에서 전달된 테스트 정확도를 모아 출력
    test_accs = [test_acc_queue.get() for _ in processes]
    average_test_acc = sum(test_accs) / len(test_accs)
    
    print('전체 평균 테스트 정확도:', average_test_acc)
    print('시간:',time.time()-start)