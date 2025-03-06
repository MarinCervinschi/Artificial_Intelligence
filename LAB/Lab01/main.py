import numpy as np

""" 
• Write a function that takes a 1d numpy array and computes its reverse vector
"""


def reverse(n):
    return n[::-1]


""" 
• Given the following square array, compute the product of the elements on its
diagonal.
"""


def diagonal_product(n):
    return np.prod(np.diag(n))


"""
• Create a random vector of size (3, 6) and find its mean value.
"""


def rand_vector():
    return np.mean(np.random.rand(3, 6))


""" 
• Given two arrays a and b, compute how many time an item of a is higher than the
corresponding element of b.
"""


def confront(a, b):
    return a > b


""" 
• Create and normalize the following matrix (use min-max normalization).
"""


def normalize(n):
    return (n - n.min()) / (n.max() - n.min())


if __name__ == "__main__":
    n = np.array([1, 2, 3, 4])
    n1 = np.array([[1, 3, 8], [-1, 3, 0], [-3, 9, 2]])

    a = np.array([1, 5, 6, 8, 2, -3, 13, 23, 0, -10, -9, 7]).reshape((3, 4))
    b = np.array([-3, 0, 8, 1, -20, -9, -1, 32, 7, 7, 7, 7]).reshape((3, 4))

    nn = np.array([[0.35, -0.27, 0.56], [0.15, 0.65, 0.42], [0.73, -0.78, -0.08]])

    print(f"--> Reverse of {n} = {reverse(n)}")
    print(f"--> Diagonal product of \n{n1} = {diagonal_product(n1)}")
    print(f"--> Mean of a random vector of size (3, 6) = {rand_vector()}")
    print(
        f"--> Confront matrix a > matrix b = {np.sum(confront(a, b))} value of a are > than b"
    )
    print(f"--> Min max normalization of matrix = \n{nn} = \n\n{normalize(nn)}")
