import tensorflow as tf
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 1. MNIST 데이터셋 임포트 (only load on rank 0)
if rank == 0:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    x_train, y_train, x_test, y_test = None, None, None, None

# Broadcast the data from rank 0 to all other ranks
x_train = comm.bcast(x_train, root=0)
y_train = comm.bcast(y_train, root=0)
x_test = comm.bcast(x_test, root=0)
y_test = comm.bcast(y_test, root=0)

# 2. 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0

# Divide the data across MPI ranks
local_batch_size = len(x_train) // size
local_x_train = x_train[rank * local_batch_size: (rank + 1) * local_batch_size]
local_y_train = y_train[rank * local_batch_size: (rank + 1) * local_batch_size]

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
# Divide the batch size by the number of MPI processes
epochs = 5
local_batch_size //= size

for epoch in range(epochs):
    # Train the model on the local data
    model.fit(local_x_train, local_y_train, batch_size=local_batch_size)

    # Synchronize after each epoch
    comm.Barrier()

# 6. 정확도 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print('테스트 정확도:', test_acc)