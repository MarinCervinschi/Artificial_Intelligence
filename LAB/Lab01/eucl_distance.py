import numpy as np
import math, time


def vanila_d(a, b):
    return [math.sqrt(sum([(ai - b[i]) ** 2 for i, ai in enumerate(row)])) for row in a]


def numpy_d(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


if __name__ == "__main__":
    a = np.random.randint(0, 10000000, size=(1000, 3)).tolist()
    b = np.random.randint(0, 10000000, size=3).tolist()

    a1 = np.array(a)
    b1 = np.array(b)

    start_time = time.time()
    vanila_d(a, b)
    print("vanila_d runtime: %.10f seconds" % (time.time() - start_time))

    start_time = time.time()
    numpy_d(a1, b1)
    print("numpy_d runtime: %.10f seconds" % (time.time() - start_time))
