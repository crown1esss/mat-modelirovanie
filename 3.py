import numpy as np


def H(a, b, n):
    h = (abs(b - a)) / n
    return h


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


def Atr(A):
    return np.conj(A.T)


def grad(A, b):
    n_iter = 10000
    eps = 10e-7
    max_A = np.max(A)
    max_b = np.max(b)
    max_a_b = np.max(np.array(max_A, max_b))
    A = A / max_a_b
    b = b / max_a_b
    u = np.random.normal(0, 1, A.shape[0])
    for i in range(n_iter):
        rk = A @ u - b
        g = Atr(A) @ rk
        u_res = u - (g @ g / ((A @ g) @ (A @ g))) * g
        diff = u - u_res
        if np.sqrt(diff @ diff) / np.sqrt(b @ b) < eps:
            break
        u = u_res.copy()
    return (u_res)


def analitical(x):
    return (3 / 4 * x)




def test(A, X):
    print('результат проверки')
    print(np.linalg.solve(A, X))
    print(analitical(X))


def main():
    a = 0
    b = 1
    n = 3
    h = H(a, b, n)
    A, X = solution(a, b, n, h)
    gradient = grad(A, X)
    print('gradient : ')
    print(gradient)
    test(A, X)


if __name__ == '__main__':
    main()
