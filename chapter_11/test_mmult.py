#
#  file:  test_mmult.py
#
#  Generate data for curves.py to compare the exponent
#  when fitting NumPy's matrix multiplication and the naive
#  algorithm.
#
#  RTK, 07-Sep-2022
#  Last update:  07-Sep-2022
#
################################################################
import numpy as np
import matplotlib.pylab as plt
import time

def mmult(A,B):
    """Naive matrix multiplication"""
    n,m = A.shape
    p = B.shape[1]
    C = np.zeros((n,p), dtype=A.dtype)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                C[i,j] += A[i,k]*B[k,j]
    return C

#  Naive
out = "3\np[0]*x**p[1]+p[2]\n"
tm = []
n = [2,5,10,20,30,40,50,60,70,80,90,100]
for i in n:
    v = []
    for j in range(10):
        s = time.time()
        A = np.random.random(size=(i,i))
        B = np.random.random(size=(i,i))
        C = mmult(A,B)
        v.append(time.time()-s)
    tm.append(np.array(v).mean())
    out += ("%0.4f %4d\n" % (tm[-1],i))

f = open("test_mmult_naive.txt","w")
f.write(out)
f.close()

tm = np.array(tm)
plt.plot(n,tm, marker='o', fillstyle='none', color='k')
plt.xlabel("Matrix size")
plt.ylabel("Mean multiplication time (n=10)")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("test_mmult_naive.png", dpi=300)
plt.close()

#  NumPy
out = "3\np[0]*x**p[1]+p[2]\n"
tm = []
n = [10,100,500,1000,1500,2000,2500,3000,3500,4000]
for i in n:
    v = []
    for j in range(10):
        s = time.time()
        A = np.random.random(size=(i,i))
        B = np.random.random(size=(i,i))
        C = A @ B
        v.append(time.time()-s)
    tm.append(np.array(v).mean())
    out += ("%0.4f %4d\n" % (tm[-1],i))

f = open("test_mmult_numpy.txt","w")
f.write(out)
f.close()

tm = np.array(tm)
plt.plot(n,tm, marker='o', fillstyle='none', color='k')
plt.xlabel("Matrix size")
plt.ylabel("Mean multiplication time (n=10)")
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
plt.savefig("test_mmult_numpy.png", dpi=300)
plt.close()

