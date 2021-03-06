import numpy as np


# находим H
def H(a,b,n):
  h =  (abs(b-a)) / n
  return h


# аналитический метод для проверки
def analitical(x):
   return (3 / 4 * x)


# проверка
def test(A,X):
  print('результат проверки')
  print(np.linalg.solve(A, X))
  print(analitical(X))


# делаем матрицу
def solution(a,b,n,h):
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
  return A , X


#  итерационный метод
def iteration(A,b):
  eps = 10e-7
  n_iter = 10000
  u = np.random.normal(0,1,A.shape[0])

  max_A = np.max(A)
  max_b = np.max(b)
  max_a_b = np.max(np.array(max_A, max_b))
  A = A/max_a_b
  b = b/max_a_b
  B = np.diag(np.ones(A.shape[0])) - A

  for i in range(n_iter):
    u_res = (B @ u) + b
    dif = u - u_res
    if (np.sqrt( dif @ dif)) / (np.sqrt(b @ b)) < eps:
      break
    u = u_res.copy()

  return u_res


# метод бисопряженных градиентов 
def bisjoint_gradient_method(A,b):
  eps = 10e-7
  n_iter = 10000
  u = np.random.normal(0,1,A.shape[0])
  p_k0 = 1
  alph = 1
  omg = 1
  v = p = s = np.zeros(A.shape[0])
  r = b - A @ u
  r1 = r 
  for i in range(n_iter):
    p_k = (np.conj(r1) @ r) 
    b_k = ( p_k / p_k0 ) * ( alph / omg )
    p = r + ( b_k *  ( p  - omg  *  v)) 
    v = A @ p 
    alph = p_k / (np.conj(r1) @ v) 
    s_k = r  - alph * v 
    t_k = A @ s_k 
    omg = (t_k @ s_k) / (t_k @ t_k)   
    u_res = u + (omg * s_k) + (alph * p) 
    r = s_k - (omg * t_k) 
    dif = u_res - u
    if np.amax((np.sqrt(dif @ dif)) / (np.sqrt(b @ b))) < eps:
        break
    p_k0 = p_k
    u = u_res.copy()
    return u_res

def main():
  a = 0 
  b = 1 
  n = 3
  h = H(a,b,n)
  A,X = solution(a,b,n,h)
  print( 'iteration : ')
  print(iteration(A,X))
  print( 'bisjoint_gradient_method : ')
  print(bisjoint_gradient_method(A,X))
  test(A,X)
if __name__ == '__main__' :
    main()
