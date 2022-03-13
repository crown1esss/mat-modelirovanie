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


def analitical(x):
    return (3 / 4 * x)


def test(A, X):
    print('результат проверки')
    print(np.linalg.solve(A, X))
    print(analitical(X))


def grad_2(A, b):
    n_iter = 10000
    eps = 10e-7
    U_0 = np.random.normal(0, 1, A.shape[0])
    r_0 = A @ U_0 - b
    g = A.conj().T @ r_0
    U_prev = U_0 - (g @ g / ((A @ g) @ (A @ g))) * g
    for i in range(n_iter):
        r_k = A @ U_prev - b
        g = A.conj().T @ r_k
        nestac = np.linalg.solve(np.matrix([[(r_k - r_0) @ (r_k - r_0), g @ g],
                                            [g @ g, (A @ g) @ (A @ g)]]),
                                 np.array([0, g @ g]))
        U_current = U_prev - nestac[0] * (U_prev - U_0) - nestac[1] * g
        diff = U_prev - U_current
        if np.sqrt(diff @ diff) / np.sqrt(b @ b) < eps:
            break
        U_0 = U_prev.copy()
        U_prev = U_current.copy()
        r_0 = r_k.copy()

    return U_current


def main():
    a = 0
    b = 1
    n = 3
    h = H(a, b, n)
    A, X = solution(a, b, n, h)
    gradient = grad_2(A, X)
    print('gradient 2 : ')
    print(gradient)
    test(A, X)


if __name__ == '__main__':
    main()
