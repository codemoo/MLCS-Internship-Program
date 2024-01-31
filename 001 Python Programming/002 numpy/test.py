import numpy as np

x = np.arange(200)


0.7
0.2
0.1

a = int(100 * 0.7)
b = int(a + 100 * 0.2)
c = int(b + 100 * 0.1)

print(a, b, c)
# print(x)

y = np.split(x, [a, b, c])

print(y)