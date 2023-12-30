from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
size  = comm.Get_size()
rank = comm.Get_rank()

x=-1
dx=0.0000001
iters=int((1-(-1))/dx)
N = iters // size + (iters % size > rank)
start = comm.scan(N) - N
A=0.
x=x+dx*start
for i in range(N):
        A=A+math.sqrt(1-x**2)*dx
        x=x+dx
A = comm.reduce(A, op=MPI.SUM, root=0)

if rank==0:
    tpi=2*A
    error = abs(tpi - math.pi)
    print ("pi is approximately %.16f, "
                "error is %.16f" % (tpi, error))