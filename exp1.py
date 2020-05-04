import mckean
import numpy as np

h = 0.02
L = int(1 / h)
N = 1000
M = 100

# Matlab result: mean=0.4084 std=0.0087

result_mc = np.zeros(M)
for j in range(M):
    X, deltaW = mckean.genproc_1dim_ex(L, h, N, mckean.a, mckean.b)
    result_mc[j] = np.mean(mckean.f(X[:, -1], X[:, -1]))

print(np.mean(result_mc))
print(np.std(result_mc))
