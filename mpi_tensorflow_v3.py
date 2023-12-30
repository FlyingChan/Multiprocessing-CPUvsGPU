import tensorflow as tf
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 행렬 곱셈을 위한 간단한 데이터 생성
matrix_A = tf.constant([[1.0, 2.0], [3.0, 4.0]])
matrix_B = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# 데이터 분배
local_matrix_A = tf.split(matrix_A, size, axis=0)[rank]
local_matrix_B = matrix_B

# 로컬 텐서플로우 그래프 생성
local_result = tf.matmul(local_matrix_A, local_matrix_B)

# 결과 수집
result = None
if rank == 0:
    result = tf.TensorArray(tf.float32, size)
result = comm.gather(local_result, root=0)

# 최종 결과 출력
if rank == 0:
    final_result = tf.concat(result, axis=0)
    print("Final Result:")
    print(final_result)