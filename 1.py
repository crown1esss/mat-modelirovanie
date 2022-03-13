import numpy as np


def analitical(x):
    return 3 / 4 * x


def k(x, y):
    return x * y


def H(a, b, n):
    h = (abs(b - a)) / n
    return h


def test(A, X):
    print(np.linalg.solve(A, X))
    print(analitical(X))


def solution(a, b, n, h):
    X = np.zeros(n)
    for i in range(n):
        X[i] = h * i + (h / 2) + a
    A = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i][j] = X[i] * X[j] * h + 1
            else:
                A[i][j] = X[i] * X[j] * h
    for i in range(n):
        print(A[i])
    return A, X


def iteration(A, b):
    eps = 10e-7
    n_iter = 10000
    max_A = np.max(A)
    max_b = np.max(b)
    max_a_b = np.max(np.array(max_A, max_b))
    A = A / max_a_b
    b = b / max_a_b
    u = np.random.normal(0, 1, A.shape[0])
    B = np.diag(np.ones(A.shape[0])) - A
    for i in range(n_iter):
        u_res = (B @ u) + b
        dif = u - u_res
        if (np.sqrt(dif @ dif)) / (np.sqrt(b @ b)) < eps:
            break
        u = u_res.copy()

    return u_res


def main():
    a = 0
    b = 1
    n = 3
    h = H(a, b, n)
    A, X = solution(a, b, n, h)
    iter = iteration(A, X)
    print('testing ....')
    test(A, X)
    print('iteration ....')
    print(iter)


if __name__ == '__main__':
    main()
